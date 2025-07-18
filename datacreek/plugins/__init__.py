"""Plugins for external storage backends."""

__all__ = ["export_embeddings_pg"]

from .pgvector_export import export_embeddings_pg
