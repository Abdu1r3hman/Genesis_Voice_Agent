"""Lightweight CRM: customer identity/memory + appointment booking (local SQLite)."""

from .store import get_store, CustomerStore

__all__ = ["get_store", "CustomerStore"]
