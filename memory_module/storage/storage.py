from abc import ABC, abstractmethod

from fastapi import UploadFile


class Storage(ABC):
    @abstractmethod
    async def save(self, file: UploadFile, key: str) -> str:
        pass

    @abstractmethod
    async def read(self, key: str) -> bytes:
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        pass
