import typer
import uvicorn

app_cli = typer.Typer(help="Command line utilities for managing the datacreek application")

@app_cli.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Run the REST API server."""
    uvicorn.run("datacreek.api:app", host=host, port=port, reload=True)

@app_cli.command()
def test():
    """Run the unit test suite."""
    import pytest
    raise SystemExit(pytest.main(["-q"]))

if __name__ == "__main__":
    app_cli()
