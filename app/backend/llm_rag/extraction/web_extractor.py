from app.backend.llm_rag.extraction.base_extractor import BaseExtractor, Document
import trafilatura


class WebExtractor(BaseExtractor):
    """
    Extractor for web content.
    Inherits from BaseExtractor.
    """

    def extract(self, src: str) -> Document:
        """
        Extracts information from the given web content.

        Args:
            src (str): The source URL or HTML content to extract information from.

        Returns:
            Document: Document containing the extracted information.
        """
        response = trafilatura.fetch_url(src)
        if not response:
            raise ValueError(f"Failed to fetch content from {src}")
        content = trafilatura.extract(response, output_format='json', with_metadata=True)
        if not content:
            raise ValueError(f"No content extracted from {src}")
        return Document(
            content=content['raw_text'],
            metadata={"source": content['source']}
        )
