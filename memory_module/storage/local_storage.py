from pathlib import Path

import aiofiles
from fastapi import UploadFile

from .storage import Storage


class LocalStorage(Storage):
    def __init__(self, base_path: str):
        self.base_path = Path(base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve_key_path(self, key: str) -> Path:
        if not key:
            raise ValueError("Storage key must be a non-empty string.")

        resolved_path = (self.base_path / key).resolve()
        if self.base_path not in resolved_path.parents and resolved_path != self.base_path:
            raise ValueError("Storage key resolves outside the configured base path.")
        return resolved_path

    async def save(self, file: UploadFile, key: str) -> str:
        file_path = self._resolve_key_path(key)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        await file.seek(0)
        async with aiofiles.open(file_path, "wb") as output_file:
            while chunk := await file.read(1024 * 1024):
                await output_file.write(chunk)
        await file.seek(0)

        return str(file_path)

    async def read(self, key: str) -> bytes:
        file_path = self._resolve_key_path(key)
        if not file_path.exists():
            raise FileNotFoundError(f"Storage key not found: {key}")

        async with aiofiles.open(file_path, "rb") as input_file:
            return await input_file.read()

    async def delete(self, key: str) -> None:
        file_path = self._resolve_key_path(key)
        if not file_path.exists():
            raise FileNotFoundError(f"Storage key not found: {key}")

        file_path.unlink()
