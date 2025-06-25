from pathlib import Path
from typing import Any, Dict, Optional

import typer
import uvicorn

from datacreek.core.create import process_file

app_cli = typer.Typer(help="Command line utilities for managing the datacreek application")


@app_cli.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Run the REST API server."""
    uvicorn.run("datacreek.api:app", host=host, port=port, reload=True)


@app_cli.command()
def generate(
    file: Path = typer.Argument(..., exists=True, help="Input file"),
    output_dir: Path = typer.Option("output", help="Directory for results"),
    content_type: str = typer.Option("qa", help="Type: qa|summary|cot|cot-enhance"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    provider: Optional[str] = typer.Option(None, help="LLM provider"),
    api_base: Optional[str] = typer.Option(None, help="Custom API base"),
    temperature: Optional[float] = typer.Option(None, help="Generation temperature"),
    prompt_file: Optional[Path] = typer.Option(None, help="Override prompt file"),
    num_pairs: Optional[int] = typer.Option(None, help="Number of pairs/examples"),
):
    """Run content generation from the command line."""
    overrides: Dict[str, Any] = {}
    if temperature is not None:
        overrides.setdefault("generation", {})["temperature"] = temperature
    if prompt_file:
        overrides.setdefault("prompts", {})[f"{content_type}_generation"] = prompt_file.read_text()

    process_file(
        file_path=str(file),
        output_dir=str(output_dir),
        api_base=api_base,
        model=model,
        provider=provider,
        content_type=content_type,
        num_pairs=num_pairs,
        config_overrides=overrides if overrides else None,
    )


@app_cli.command()
def init_db_cmd() -> None:
    """Create database tables."""
    from datacreek.db import init_db

    init_db()
    typer.echo("Database initialized")


@app_cli.command()
def test():
    """Run the unit test suite."""
    import pytest

    raise SystemExit(pytest.main(["-q"]))


if __name__ == "__main__":
    app_cli()
