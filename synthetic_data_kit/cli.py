# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# CLI Logic for synthetic-data-kit

import os
from pathlib import Path
from typing import Optional

import requests
import typer
from rich.console import Console
from rich.table import Table

from synthetic_data_kit.core.context import AppContext
from synthetic_data_kit.server.app import run_server
from synthetic_data_kit.utils.config import (
    get_llm_provider,
    get_openai_config,
    get_path_config,
    get_vllm_config,
    load_config,
)
from synthetic_data_kit.utils.provider import check_vllm_server, resolve_provider

# Initialize Typer app
app = typer.Typer(
    name="synthetic-data-kit",
    help="A toolkit for preparing synthetic datasets for fine-tuning LLMs",
    add_completion=True,
)
console = Console()

# Create app context
ctx = AppContext()


# Define global options
@app.callback()
def callback(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
):
    """
    Global options for the Synthetic Data Kit CLI
    """
    if config:
        ctx.config_path = config
    ctx.config = load_config(ctx.config_path)


@app.command("system-check")
def system_check(
    api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL to check"),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Provider to check ('vllm' or 'api-endpoint')"
    ),
):
    """Check connectivity to the selected LLM provider."""
    # Determine provider and API base
    selected_provider, api_base, _ = resolve_provider(
        ctx.config,
        provider=provider,
        api_base=api_base,
    )

    if selected_provider == "api-endpoint":
        cfg = get_openai_config(ctx.config)
        api_base = api_base or cfg.get("api_base")
        api_key = os.environ.get("API_ENDPOINT_KEY") or cfg.get("api_key")
        model = cfg.get("model")

        with console.status("Checking API endpoint access..."):
            try:
                from openai import OpenAI
            except ImportError:
                console.print(
                    "OpenAI package not installed. Run 'pip install openai>=1.0.0'",
                    style="red",
                )
                return 1

            try:
                client = (
                    OpenAI(api_key=api_key, base_url=api_base)
                    if api_base
                    else OpenAI(api_key=api_key)
                )
                client.models.list()
                console.print("API endpoint access confirmed", style="green")
                return 0
            except Exception as e:
                console.print(f"Error connecting to API endpoint: {e}", style="red")
                return 1
    else:
        # Default to vLLM
        vllm_config = get_vllm_config(ctx.config)
        port = vllm_config.get("port", 8000)

        with console.status(f"Checking vLLM server at {api_base}..."):
            if check_vllm_server(api_base):
                console.print(f"vLLM server is running at {api_base}", style="green")
                return 0

            console.print(f"vLLM server is not available at {api_base}", style="red")
            console.print(
                f"Start with: vllm serve {model or vllm_config.get('model')} --port {port}",
                style="yellow",
            )
            return 1


@app.command()
def ingest(
    input: str = typer.Argument(..., help="File or URL to parse"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the output"
    ),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Custom output filename"),
):
    """
    Parse documents (PDF, HTML, YouTube, DOCX, PPT, TXT) into clean text.
    """
    from synthetic_data_kit.core.ingest import process_file

    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "parsed")

    try:
        with console.status(f"Processing {input}..."):
            output_path = process_file(input, output_dir, name, ctx.config)
        console.print(f" Text successfully extracted to [bold]{output_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"Error: {e}", style="red")
        return 1


@app.command()
def create(
    input: str = typer.Argument(..., help="File to process"),
    content_type: str = typer.Option(
        "qa", "--type", help="Type of content to generate [qa|summary|cot|cot-enhance]"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the output"
    ),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="VLLM API base URL"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    num_pairs: Optional[int] = typer.Option(
        None, "--num-pairs", "-n", help="Target number of QA pairs or CoT examples to generate"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """
    Generate content from text using local LLM inference.

    Content types:
    - qa: Generate question-answer pairs from text (use --num-pairs to specify how many)
    - summary: Generate a summary of the text
    - cot: Generate Chain of Thought reasoning examples from text (use --num-pairs to specify how many)
    - cot-enhance: Enhance existing tool-use conversations with Chain of Thought reasoning
      (use --num-pairs to limit the number of conversations to enhance, default is to enhance all)
      (for cot-enhance, the input must be a JSON file with either:
       - A single conversation in 'conversations' field
       - An array of conversation objects, each with a 'conversations' field
       - A direct array of conversation messages)
    """
    from synthetic_data_kit.core.create import process_file

    provider, api_base, model = resolve_provider(
        ctx.config,
        provider=None,
        api_base=api_base,
        model=model,
    )
    console.print(f"Using {provider} provider", style="green")

    if provider == "vllm" and not check_vllm_server(api_base):
        console.print(f"Error: VLLM server not available at {api_base}", style="red")
        console.print("Please start the VLLM server with:", style="yellow")
        console.print(f"vllm serve {model}", style="bold blue")
        return 1

    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "generated")

    try:
        with console.status(f"Generating {content_type} content from {input}..."):
            output_path = process_file(
                input,
                output_dir,
                ctx.config_path,
                api_base,
                model,
                content_type,
                num_pairs,
                verbose,
                provider=provider,  # Pass the provider parameter
            )
        if output_path:
            console.print(f" Content saved to [bold]{output_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"Error: {e}", style="red")
        return 1


@app.command("curate")
def curate(
    input: str = typer.Argument(..., help="Input file to clean"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    threshold: Optional[float] = typer.Option(
        None, "--threshold", "-t", help="Quality threshold (1-10)"
    ),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="VLLM API base URL"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """
    Clean and filter content based on quality.
    """
    from synthetic_data_kit.core.curate import curate_qa_pairs

    provider, api_base, model = resolve_provider(
        ctx.config,
        provider=None,
        api_base=api_base,
        model=model,
    )

    if provider == "vllm" and not check_vllm_server(api_base):
        console.print(f"Error: VLLM server not available at {api_base}", style="red")
        console.print("Please start the VLLM server with:", style="yellow")
        console.print(f"vllm serve {model}", style="bold blue")
        return 1

    # Get default output path from config if not provided
    if not output:
        cleaned_dir = get_path_config(ctx.config, "output", "cleaned")
        os.makedirs(cleaned_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input))[0]
        output = os.path.join(cleaned_dir, f"{base_name}_cleaned.json")

    try:
        with console.status(f"Cleaning content from {input}..."):
            result_path = curate_qa_pairs(
                input,
                output,
                threshold,
                api_base,
                model,
                ctx.config_path,
                verbose,
                provider=provider,  # Pass the provider parameter
            )
        console.print(f" Cleaned content saved to [bold]{result_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"Error: {e}", style="red")
        return 1


@app.command("save-as")
def save_as(
    input: str = typer.Argument(..., help="Input file to convert"),
    format: Optional[str] = typer.Option(
        None, "--format", "-f", help="Output format [jsonl|alpaca|ft|chatml]"
    ),
    storage: str = typer.Option(
        "json", "--storage", help="Storage format [json|hf]", show_default=True
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """
    Convert to different formats for fine-tuning.

    The --format option controls the content format (how the data is structured).
    The --storage option controls how the data is stored (JSON file or HF dataset).

    When using --storage hf, the output will be a directory containing a Hugging Face
    dataset in Arrow format, which is optimized for machine learning workflows.
    """
    from synthetic_data_kit.core.save_as import convert_format

    # Get format from args or config
    if not format:
        format_config = ctx.config.get("format", {})
        format = format_config.get("default", "jsonl")

    # Set default output path if not provided
    if not output:
        final_dir = get_path_config(ctx.config, "output", "final")
        os.makedirs(final_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input))[0]

        if storage == "hf":
            # For HF datasets, use a directory name
            output = os.path.join(final_dir, f"{base_name}_{format}_hf")
        else:
            # For JSON files, use appropriate extension
            if format == "jsonl":
                output = os.path.join(final_dir, f"{base_name}.jsonl")
            else:
                output = os.path.join(final_dir, f"{base_name}_{format}.json")

    try:
        with console.status(f"Converting {input} to {format} format with {storage} storage..."):
            output_path = convert_format(input, output, format, ctx.config, storage_format=storage)

        if storage == "hf":
            console.print(
                f" Converted to {format} format and saved as HF dataset to [bold]{output_path}[/bold]",
                style="green",
            )
        else:
            console.print(
                f" Converted to {format} format and saved to [bold]{output_path}[/bold]",
                style="green",
            )
        return 0
    except Exception as e:
        console.print(f"Error: {e}", style="red")
        return 1


@app.command("server")
def server(
    host: str = typer.Option("127.0.0.1", "--host", help="Host address to bind the server to"),
    port: int = typer.Option(5000, "--port", "-p", help="Port to run the server on"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Run the server in debug mode"),
):
    """
    Start a web interface for the Synthetic Data Kit.

    This launches a web server that provides a UI for all SDK functionality,
    including generating and curating QA pairs, as well as viewing
    and managing generated files.
    """
    provider = get_llm_provider(ctx.config)
    console.print(f"Starting web server with {provider} provider...", style="green")
    console.print(f"Web interface available at: http://{host}:{port}", style="bold green")
    console.print("Press CTRL+C to stop the server.", style="italic")

    # Run the Flask server
    run_server(host=host, port=port, debug=debug)


if __name__ == "__main__":
    app()
