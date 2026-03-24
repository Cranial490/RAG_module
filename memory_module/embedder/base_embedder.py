from abc import ABC, abstractmethod
from typing import Union, List

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> Union[List[float]]:
        pass