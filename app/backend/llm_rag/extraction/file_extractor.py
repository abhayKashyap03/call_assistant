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
        Extracts text, images, and tables from the given document content.

        Args:
            src (str): The source document content to extract information from.

        Returns:
            Document: Document containing the extracted information.
        """
        docs = self._load_files(src)
        all_text = []
        all_tables = []
        all_images = []
        content_parts = []
        for i, page in enumerate(docs):
            # Extract text
            text = page.get_text()
            if text:
                content_parts.append(text)
                all_text.append(text)
            # Extract tables
            try:
                tables = page.find_tables()
                for table in tables:
                    try:
                        table_data = table.extract()
                        content_parts.append(f"[TABLE page={i}] {table_data}")
                        all_tables.append(table_data)
                    except Exception as e:
                        print(f"Error extracting table on page {i}: {e}")
            except Exception as e:
                print(f"Error finding tables on page {i}: {e}")
            # Extract images (add placeholders to content)
            try:
                images = page.get_images(full=True)
                if images:
                    for img_idx, img_info in enumerate(images):
                        img_str = f"[IMAGE page={i} idx={img_idx} info={img_info}]"
                        content_parts.append(img_str)
                        all_images.append(img_info)
            except Exception as e:
                print(f"Error extracting images on page {i}: {e}")
        if not content_parts:
            raise ValueError(f"No content extracted from {src}")
        content = "\n\n".join(content_parts)
        metadata = {
            "source": src,
            "num_images": len(all_images),
            "num_tables": len(all_tables)
        }
        return Document(
            title=os.path.basename(src),
            content=content,
            metadata=metadata
        )
        
