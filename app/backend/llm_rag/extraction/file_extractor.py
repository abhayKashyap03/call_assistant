from app.backend.llm_rag.extraction.base_extractor import BaseExtractor, Document
import pymupdf
import os


class FileExtractor(BaseExtractor):
    """
    Extractor for document content.
    Inherits from BaseExtractor.
    """
    def _load_files(self, src: str):
        """
        Loads files from the given source path.

        Args:
            src (str): The source path to load files from.

        Returns:
            pymupdf.Document: Document containing the loaded files.
        """
        assert os.path.exists(src) and os.path.isabs(src), f"Source does not exist or is not absolute path: {src}"
        if os.path.isdir(src):
            files = [os.path.join(src, file) for file in os.listdir(src)]
            docs = pymupdf.open(files[0])
            for file in files[1:]:
                docs.insert_file(pymupdf.open(file))
        else:
            docs = pymupdf.open(src)
        return docs
    
    def extract(self, src: str) -> Document:
        """
        Extracts information from the given document content.

        Args:
            src (str): The source document content to extract information from.

        Returns:
            Document: Document containing the extracted information.
        """
        docs = self._load_files(src)
        for page in docs:
            text = page.get_text()
            if text:
                return Document(
                    title=os.path.basename(src),
                    content=text,
                    metadata={"source": src}
                )
        raise ValueError(f"No content extracted from {src}")
        
