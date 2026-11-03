"""Append-only JSONL store with resume and integrity checks.

One record per line, written with a single ``write`` and fsynced. There is no
API to delete or overwrite a record. A writer killed mid-record leaves a torn
final line (no trailing newline); opening for write quarantines that fragment
to ``<path>.torn`` and cuts the file back to the last complete record. Every
complete line must parse against the model and keys must be unique, otherwise
opening fails.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class DuplicateKeyError(ValueError):
    """A record with this key already exists."""


class CorruptStoreError(ValueError):
    """A complete line in the store is not a valid record."""


class JsonlStore(Generic[T]):
    def __init__(
        self,
        path: str | os.PathLike,
        model: type[T],
        key: str = "gen_key",
        *,
        readonly: bool = False,
        fsync: bool = True,
    ) -> None:
        self.path = Path(path)
        self._model = model
        self._key = key
        self._readonly = readonly
        self._fsync = fsync
        self._records: dict[str, T] = {}
        self.recovered_bytes = 0
        if not readonly:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        cut = data.rfind(b"\n") + 1  # end of the last complete line
        torn = data[cut:]
        for lineno, raw in enumerate(data[:cut].split(b"\n")[:-1], start=1):
            try:
                record = self._model.model_validate_json(raw)
            except ValidationError as exc:
                raise CorruptStoreError(f"{self.path}:{lineno}: {exc}") from exc
            self._insert(record)
        if torn and not self._readonly:
            with open(self.path.with_name(self.path.name + ".torn"), "ab") as fh:
                fh.write(torn)
            with open(self.path, "r+b") as fh:
                fh.truncate(cut)
            self.recovered_bytes = len(torn)

    def _insert(self, record: T) -> None:
        key = getattr(record, self._key)
        if key in self._records:
            raise DuplicateKeyError(f"duplicate {self._key}: {key}")
        self._records[key] = record

    def append(self, record: T) -> None:
        if self._readonly:
            raise PermissionError("store was opened read-only")
        key = getattr(record, self._key)
        if key in self._records:
            raise DuplicateKeyError(f"duplicate {self._key}: {key}")
        line = record.model_dump_json().encode() + b"\n"
        with open(self.path, "ab") as fh:
            fh.write(line)
            fh.flush()
            if self._fsync:
                os.fsync(fh.fileno())
        self._records[key] = record

    def get(self, key: str) -> T:
        return self._records[key]

    def __contains__(self, key: object) -> bool:
        return key in self._records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[T]:
        return iter(self._records.values())
