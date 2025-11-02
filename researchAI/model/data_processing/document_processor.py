from typing import Dict, List, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Process and normalize documents from different sources
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_arxiv_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process arXiv paper into standardized format
        
        Args:
            paper: Raw paper data from your data pipeline
            
        Returns:
            Processed paper document
        """
        try:
            content = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
            
            return {
                'doc_id': paper['arxiv_id'],
                'doc_type': 'research_paper',
                'title': paper['title'],
                'content': content,  
                'metadata': {
                    'arxiv_id': paper['arxiv_id'],
                    'title': paper['title'],  
                    'authors': paper.get('authors', ''),
                    'author_count': paper.get('author_count', 0),
                    'published_date': paper['published_date'],
                    'updated_date': paper.get('updated_date', paper['published_date']),
                    'categories': paper.get('all_categories', []),
                    'primary_category': paper.get('primary_category', 'artificial_intelligence'),
                    'relevance_score': paper.get('overall_relevance', 0.0),
                    'category_scores': paper.get('category_scores', {}),
                    'pdf_url': paper.get('pdf_url', ''),
                    'html_url': paper.get('html_url', ''),
                    'source': 'arXiv'  
                }
            }
        except Exception as e:
            self.logger.error(f"Error processing arXiv paper {paper.get('arxiv_id', 'unknown')}: {e}")
            return None
    
    def process_news_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process news article into standardized format
        
        Args:
            article: Raw article data from your data pipeline
            
        Returns:
            Processed news document
        """
        try:
            content = f"Title: {article['title']}\n\n{article.get('description', '')}"
            
            return {
                'doc_id': article['article_id'],
                'doc_type': 'news_article',
                'title': article['title'],
                'content': content, 
                'metadata': {
                    'article_id': article['article_id'],
                    'title': article['title'], 
                    'source_name': article.get('source_name', 'Unknown'),
                    'author': article.get('author', 'Unknown'),
                    'published_at': article['published_at'],
                    'url': article.get('url', ''),
                    'image_url': article.get('image_url', ''),
                    'categories': article.get('all_categories', []),
                    'primary_category': article.get('primary_category', 'general'),
                    'relevance_score': article.get('overall_relevance', 0.0),
                    'category_scores': article.get('category_scores', {}),
                    'source': 'News'  
                }
            }
        except Exception as e:
            self.logger.error(f"Error processing news article {article.get('article_id', 'unknown')}: {e}")
            return None
    
    def process_batch(self, documents: List[Dict[str, Any]], doc_type: str) -> List[Dict[str, Any]]:
        """
        Process a batch of documents
        
        Args:
            documents: List of raw documents
            doc_type: 'paper' or 'news'
            
        Returns:
            List of processed documents
        """
        processed_docs = []
        
        for doc in documents:
            if doc_type == 'paper':
                processed = self.process_arxiv_paper(doc)
            elif doc_type == 'news':
                processed = self.process_news_article(doc)
            else:
                self.logger.warning(f"Unknown document type: {doc_type}")
                continue
            
            if processed:
                processed_docs.append(processed)
        
        self.logger.info(f"Processed {len(processed_docs)}/{len(documents)} {doc_type} documents")
        return processed_docs
    
    def filter_by_relevance(self, documents: List[Dict[str, Any]], 
                           min_relevance: float = 0.3) -> List[Dict[str, Any]]:
        """
        Filter documents by relevance score
        
        Args:
            documents: List of processed documents
            min_relevance: Minimum relevance score threshold
            
        Returns:
            Filtered documents
        """
        filtered = [
            doc for doc in documents 
            if doc['metadata'].get('relevance_score', 0) >= min_relevance
        ]
        
        self.logger.info(
            f"Filtered {len(documents)} documents to {len(filtered)} "
            f"(min_relevance={min_relevance})"
        )
        return filtered
    
    def filter_by_date(self, documents: List[Dict[str, Any]], 
                       days_back: int = 365) -> List[Dict[str, Any]]:
        """
        Filter documents by recency
        
        Args:
            documents: List of processed documents
            days_back: Number of days to look back
            
        Returns:
            Filtered documents
        """

        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        filtered = []
        
        for doc in documents:
            date_str = doc['metadata'].get('published_date') or doc['metadata'].get('published_at')
            if not date_str:
                continue
            
            try:
                # Parse ISO format date
                doc_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                if doc_date >= cutoff_date:
                    filtered.append(doc)
            except Exception as e:
                self.logger.warning(f"Error parsing date {date_str}: {e}")
        
        self.logger.info(
            f"Filtered {len(documents)} documents to {len(filtered)} "
            f"(last {days_back} days)"
        )
        return filtered