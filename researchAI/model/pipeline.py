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
    
    def __init__(self, enable_tracking: bool = True, enable_monitoring: bool = True):
        
        """
        Initialize RAG pipeline
        
        Args:
            enable_tracking: Enable MLflow experiment tracking
            enable_monitoring: Enable hybrid monitoring (local + GCP)
        """
        setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Tech Trends RAG Pipeline")
        
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

        self.monitoring = None
        if enable_monitoring:
            self._initialize_monitoring()
    
    def _initialize_monitoring(self):
        """Initialize GCP monitoring via HybridMonitor"""
        try:
            from monitoring import HybridMonitor
            import os

            project_id = os.getenv("GCP_PROJECT_ID")

            self.monitoring = HybridMonitor(
                project_id=project_id,
                model_name="techtrends-rag",
                enable_gcp=True,
            )

            status = self.monitoring.get_status()
            mode = status.get("mode", "unknown")
            gcp_status = status.get("gcp_monitor", {})

            self.logger.info(f"✓ Monitoring initialized (mode={mode})")
            self.logger.info(
                "  ✓ GCP Cloud Monitoring: %s (project_id=%s)",
                "Active" if gcp_status.get("available") else "Unavailable",
                gcp_status.get("project_id", "not_set"),
            )

        except ImportError as e:
            self.logger.warning(f"Hybrid monitor not available: {e}")
            self.logger.info("Continuing without monitoring")
            self.monitoring = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize monitoring: {e}")
            self.monitoring = None

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
        if papers:
            for paper in papers:
                processed_doc = self.doc_processor.process_arxiv_paper(paper)
                
                doc_chunks = self.chunker.create_chunks(processed_doc)
                
                doc_chunks = self.embedder.embed_chunks(doc_chunks)
                
                self.retriever.index_documents(doc_chunks, 'paper')
                
                self.logger.info(f"Indexed {len(doc_chunks)} chunks")
        
        if news:
            self.logger.info(f"Indexing document: news articles")
            for news in news:
        
                processed_doc = self.doc_processor.process_news_article(news)
                
                doc_chunks = self.chunker.create_chunks(processed_doc)
                
                doc_chunks = self.embedder.embed_chunks(doc_chunks)
                
                self.retriever.index_documents(doc_chunks, 'news')
                
                self.logger.info(f"Indexed {len(doc_chunks)} chunks")
        
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

        if self.tracker:
            try:
                self.tracker.start_run()
                self.tracker.log_config()
            except Exception as e:
                self.logger.warning(f"Failed to start MLflow run: {e}")
            # Try to end any existing run first
            try:
                import mlflow
                if mlflow.active_run() is not None:
                    mlflow.end_run()
            except:
                pass
            # Try starting again
            try:
                self.tracker.start_run()
                self.tracker.log_config()
            except Exception as e2:
                self.logger.error(f"Failed to start MLflow run after cleanup: {e2}")
                # Continue without tracking rather than failing the query
                self.tracker = None
        # if self.tracker:
        #     self.tracker.start_run()
        #     self.tracker.log_config()
        
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
        
        self.logger.info("Validating response")
        validation = self.validator.validate(
            query=query,
            response=result['response'],
            retrieved_docs=retrieved_docs
        )
        
        response_time = time.time() - start_time
        metrics = self.metrics_calculator.calculate_end_to_end_metrics(
            query=query,
            response=result['response'],
            retrieved_docs=retrieved_docs,
            response_time=response_time
        )

        self.logger.info("Checking retrieval fairness with Fairlearn")

        normalized_retrieval_score = self._normalize_retrieval_score(retrieved_docs)

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

        fairness_report = self.fairness_detector.evaluate_single_query_fairness_with_fairlearn(
            fairness_evaluation_data
        )
        
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
        
        if self.tracker:
            self.tracker.log_query_result(
                query=query,
                response=result['response'],
                metrics=metrics,
                retrieved_docs=retrieved_docs,
                bias_report=fairness_report
            )
            self.tracker.end_run()
        
         # Log to monitoring
        if self.monitoring:
            try:
                self.monitoring.log_query(final_result)
            except Exception as e:
                self.logger.warning(f"Failed to log to monitoring: {e}")
        
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
        
        if max_score == min_score:
            return 0.5
        
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
    
    def push_to_artifact_registry(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        repository: Optional[str] = None,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        bias_report: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Push model to Google Cloud Artifact Registry
        
        Args:
            project_id: GCP project ID (uses config if not provided)
            location: GCP location (uses config if not provided)
            repository: Repository name (uses config if not provided)
            version: Version name (optional, uses timestamp if not provided)
            metrics: Model performance metrics
            bias_report: Bias evaluation report
            description: Optional description
            
        Returns:
            Artifact Registry path where model was uploaded
        """
        from deployment.artifact_registry_pusher import ArtifactRegistryPusher
        from config.settings import config
        
        project_id = project_id or config.gcp.project_id
        location = location or config.gcp.location
        repository = repository or config.gcp.artifact_repository
        
        if not project_id:
            raise ValueError("GCP project_id is required. Set via --project-id or GCP_PROJECT_ID env var")
        
        self.logger.info(
            f"Pushing model to Artifact Registry: "
            f"{location}-generic.pkg.dev/{project_id}/{repository}"
        )
        
        pusher = ArtifactRegistryPusher(
            project_id=project_id,
            location=location,
            repository=repository
        )
        
        artifact_path = pusher.push(
            version=version,
            metrics=metrics,
            bias_report=bias_report,
            description=description
        )
        
        self.logger.info(f"Model successfully pushed to: {artifact_path}")
        return artifact_path
    
    def update_indexes(self, papers: List[Dict[str, Any]] = None,
                   news: List[Dict[str, Any]] = None):
        """
        Update existing indexes with new documents
        
        Args:
            papers: List of new arXiv papers
            news: List of new news articles
        """
        self.logger.info("Starting incremental index update")
        
        indexes_loaded = self.load_indexes()
        
        if not indexes_loaded:
            self.logger.warning("No existing indexes found. Creating new indexes...")
            self.index_documents(papers=papers, news=news)
            return
        
        if papers:
            self.logger.info(f"Updating indexes with {len(papers)} new papers")
            for paper in papers:
                processed_doc = self.doc_processor.process_arxiv_paper(paper)
                
                doc_chunks = self.chunker.create_chunks(processed_doc)
                
                doc_chunks = self.embedder.embed_chunks(doc_chunks)
                
                self.retriever.update_indexes(doc_chunks, 'paper')
                
                self.logger.info(f"Updated with {len(doc_chunks)} paper chunks")
        
        if news:
            self.logger.info(f"Updating indexes with {len(news)} new articles")
            for article in news:
                processed_doc = self.doc_processor.process_news_article(article)
                
                doc_chunks = self.chunker.create_chunks(processed_doc)
                
                doc_chunks = self.embedder.embed_chunks(doc_chunks)
                
                self.retriever.update_indexes(doc_chunks, 'news')
                
                self.logger.info(f"Updated with {len(doc_chunks)} news chunks")
        
        self.retriever.save_indexes()
        self.logger.info("Index update completed and saved")