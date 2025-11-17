from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

from config.settings import config

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Create hierarchical chunks from documents"""
    
    def __init__(self):
        self.config = config.chunking
        self.logger = logging.getLogger(__name__)
        
        # Initialize text splitters for different document types
        self.paper_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.paper_chunk_size,
            chunk_overlap=self.config.paper_chunk_overlap,
            separators=self.config.separators,
            length_function=len
        )
        
        self.news_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.news_chunk_size,
            chunk_overlap=self.config.news_chunk_overlap,
            separators=self.config.separators,
            length_function=len
        )
    
    def create_chunks(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create multi-level chunks for better retrieval
        
        Args:
            document: Processed document
            
        Returns:
            List of chunk objects
        """
        chunks = []
        doc_type = document['doc_type']
        
        summary_text = f"{document['title']}. {document['content'][:200]}"
        summary_chunk = {
            'chunk_id': f"{document['doc_id']}_summary",
            'doc_id': document['doc_id'],
            'doc_type': doc_type,
            'chunk_type': 'summary',
            'text': summary_text,
            'chunk_index': 0,
            'metadata': document['metadata']
        }
        chunks.append(summary_chunk)
        
        if doc_type == 'research_paper':
            splitter = self.paper_splitter
        else:
            splitter = self.news_splitter
        
        content_chunks = splitter.split_text(document['content'])
        
        for idx, chunk_text in enumerate(content_chunks):
            chunk = {
                'chunk_id': f"{document['doc_id']}_chunk_{idx}",
                'doc_id': document['doc_id'],
                'doc_type': doc_type,
                'chunk_type': 'detail',
                'text': chunk_text,
                'chunk_index': idx + 1,   
                'metadata': document['metadata']
            }
            chunks.append(chunk)
        
        self.logger.debug(
            f"Created {len(chunks)} chunks for document {document['doc_id']}"
        )
        return chunks
    