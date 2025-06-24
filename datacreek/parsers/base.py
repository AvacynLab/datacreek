class BaseParser:
    """Abstract parser interface."""

    def parse(self, file_path: str) -> str:
        """Return raw text from the given resource."""
        raise NotImplementedError

    def save(self, content: str, output_path: str) -> None:
        """Persist parsed text to ``output_path``."""
        raise NotImplementedError
