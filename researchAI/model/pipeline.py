from typing import List, Dict, Any, Optional
import time
import logging
from data_processing.document_processor import DocumentProcessor
from data_processing.chunking import DocumentChunker
from data_processing.embedding import HybridEmbedder
from retrieval.retriever import HybridRetriever
from generation.generator import ResponseGenerator, StreamingGenerator
from generation.response_validator import ResponseValidator
from evaluation.model_bias_detector import RAGBiasDetector
from evaluation.metrics import RAGMetrics
from evaluation.experiment_tracker import ExperimentTracker
from utils.logger import setup_logging
logger = logging.getLogger(__name__)


class TechTrendsRAGPipeline:
    """Complete RAG pipeline for technology trends"""
    
    def __init__(self, enable_tracking: bool = True):
        """
        Initialize RAG pipeline
        
        Args:
            enable_cache: Enable response caching
            enable_tracking: Enable MLflow experiment tracking
        """
        setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Tech Trends RAG Pipeline")
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.chunker = DocumentChunker()
        self.embedder = HybridEmbedder()
        self.retriever = HybridRetriever(self.embedder)
        self.generator = ResponseGenerator()
        self.streaming_generator = StreamingGenerator()
        self.validator = ResponseValidator()
        self.fairness_detector = RAGBiasDetector()
        self.metrics_calculator = RAGMetrics()
        self.tracker = ExperimentTracker() if enable_tracking else None
        self.logger.info(f"Tracker is {'enabled' if self.tracker else 'not enabled!'}")

        self.logger.info("Pipeline initialized successfully")


    
    def index_documents(self, papers: List[Dict[str, Any]] = None,
                       news: List[Dict[str, Any]] = None):
        """
        Index documents into the RAG system
        
        Args:
            papers: List of arXiv papers
            news: List of news articles
        """
        self.logger.info("Starting document indexing")
        self.logger.info(f"Indexing document: arxiv papers")
        # Process and index papers
        if papers:
            for paper in papers:
            # Process document
                processed_doc = self.doc_processor.process_arxiv_paper(paper)
                
                # Create chunks
                doc_chunks = self.chunker.create_chunks(processed_doc)
                
                # Generate embeddings
                doc_chunks = self.embedder.embed_chunks(doc_chunks)
                
                # Index
                self.retriever.index_documents(doc_chunks, 'paper')
                
                self.logger.info(f"Indexed {len(doc_chunks)} chunks")
        
        # Process and index news
        if news:
            self.logger.info(f"Indexing document: news articles")
            for news in news:
        
            # Process document
                processed_doc = self.doc_processor.process_news_article(news)
                
                # Create chunks
                doc_chunks = self.chunker.create_chunks(processed_doc)
                
                # Generate embeddings
                doc_chunks = self.embedder.embed_chunks(doc_chunks)
                
                # Index
                self.retriever.index_documents(doc_chunks, 'news')
                
                self.logger.info(f"Indexed {len(doc_chunks)} chunks")
        
        # Save indexes
        self.retriever.save_indexes()
        self.logger.info("Document indexing completed")
    
    def load_indexes(self) -> bool:
        """
        Load pre-built indexes
        
        Returns:
            True if successful
        """
        self.logger.info("Loading indexes")
        success = self.retriever.load_indexes()
        
        if success:
            stats = self.retriever.get_index_stats()
            self.logger.info(f"Indexes loaded: {stats}")
        
        return success

    def query(self, query: str, 
         filters: Optional[Dict[str, Any]] = None,
         enable_streaming: bool = False) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            query: User query
            filters: Optional filters (categories, date range, etc.)
            enable_streaming: Enable streaming response
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        
        self.logger.info(f"Processing query: {query}")
        
        # Start tracking if enabled
        if self.tracker:
            self.tracker.start_run()
            self.tracker.log_config()
        
        # Step 1: Retrieval
        self.logger.info("Retrieving relevant documents")
        retrieved_docs = self.retriever.retrieve(
            query=query,
            filters=filters
        )
        
        if not retrieved_docs:
            return {
                'query': query,
                'response': "I couldn't find relevant information to answer your question.",
                'sources': [],
                'error': 'no_results'
            }
        
        self.logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Generation
        self.logger.info("Generating response")
        if enable_streaming:
            return self.streaming_generator.generate_response_stream(
                query=query,
                retrieved_docs=retrieved_docs
            )
        else:
            result = self.generator.generate_response(
                query=query,
                retrieved_docs=retrieved_docs
            )
        
        # Step 3: Validation
        self.logger.info("Validating response")
        validation = self.validator.validate(
            query=query,
            response=result['response'],
            retrieved_docs=retrieved_docs
        )
        
        # Step 4: Calculate metrics
        response_time = time.time() - start_time
        metrics = self.metrics_calculator.calculate_end_to_end_metrics(
            query=query,
            response=result['response'],
            retrieved_docs=retrieved_docs,
            response_time=response_time
        )

        # Step 5: Lightweight fairness check with Fairlearn
        self.logger.info("Checking retrieval fairness with Fairlearn")

        # Normalize retrieval scores to [0, 1] range for fairness evaluation
        normalized_retrieval_score = self._normalize_retrieval_score(retrieved_docs)

        # Prepare single-query fairness check data
        fairness_evaluation_data = {
            'query': query,
            'retrieved_docs': retrieved_docs,
            'response': result['response'],
            'performance_metrics': {
                'retrieval_score': normalized_retrieval_score,
                'response_quality': validation['overall_score'],
                'source_diversity': self._calculate_source_diversity(retrieved_docs),
                'response_time': response_time
            }
        }

        # Run Fairlearn-based single-query fairness check
        fairness_report = self.fairness_detector.evaluate_single_query_fairness_with_fairlearn(
            fairness_evaluation_data
        )
        
        # Compile final result
        final_result = {
            'query': query,
            'response': result['response'],
            'sources': result['sources'],
            'num_sources': result['num_sources'],
            'retrieved_docs': retrieved_docs,
            'bias_report': fairness_report,
            'validation': validation,
            'metrics': metrics,
            'response_time': response_time,
            'from_cache': False
        }
        
        # Track experiment
        if self.tracker:
            self.tracker.log_query_result(
                query=query,
                response=result['response'],
                metrics=metrics,
                retrieved_docs=retrieved_docs,
                bias_report=fairness_report
            )
            self.tracker.end_run()
        
        self.logger.info(f"Query completed in {response_time:.2f}s")
        return final_result

    def _normalize_retrieval_score(self, retrieved_docs: List[Dict[str, Any]]) -> float:
        """
        Normalize retrieval scores to [0, 1] range
        
        Args:
            retrieved_docs: Retrieved documents with scores
            
        Returns:
            Normalized average score in [0, 1]
        """
        if not retrieved_docs:
            return 0.0
        
        scores = [doc.get('score', 0.0) for doc in retrieved_docs]
        
        if not scores:
            return 0.0
        
        min_score = min(scores)
        max_score = max(scores)
        
        # Handle case where all scores are the same
        if max_score == min_score:
            return 0.5
        
        # Min-max normalization to [0, 1]
        normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
        
        return sum(normalized_scores) / len(normalized_scores)
    

    def _calculate_source_diversity(self, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate source diversity score"""
        if not retrieved_docs:
            return 0.0
        
        sources = set()
        for doc in retrieved_docs:
            source = doc.get('metadata', {}).get('source_name') or \
                    doc.get('metadata', {}).get('arxiv_id', 'unknown')
            sources.add(source)
        
        return len(sources) / len(retrieved_docs)