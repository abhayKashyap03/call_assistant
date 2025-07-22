from abc import ABC, abstractmethod
from pydantic import BaseModel


class Document(BaseModel):
    content: str
    metadata: dict = {}


class BaseExtractor(ABC):
    """
    Base class for extractors.
    All extractors should inherit from this class.
    """

    @abstractmethod
    def extract(self, src: str) -> Document:
        """
        Extracts information from the given text.

        Args:
            src (str): The source to extract information from.

        Returns:
            Document: Document containing the extracted information.
        """
        pass
