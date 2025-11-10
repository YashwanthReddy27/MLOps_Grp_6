import json
import logging
from pathlib import Path
import argparse
import os

from pipeline import TechTrendsRAGPipeline
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


def load_data(papers_path: str = None, news_path: str = None) -> tuple:
    """
    Load data from JSON files
    
    Args:
        papers_path: Path to papers JSON
        news_path: Path to news JSON
        
    Returns:
        Tuple of (papers, news)
    """
    papers = None
    news = None
    
    if papers_path and Path(papers_path).exists():
        logger.info(f"Loading papers from {papers_path}")
        with open(papers_path, 'r') as f:
            data = json.load(f)
            papers = data.get('papers', [])
        logger.info(f"Loaded {len(papers)} papers")
    
    if news_path and Path(news_path).exists():
        logger.info(f"Loading news from {news_path}")
        with open(news_path, 'r') as f:
            data = json.load(f)
            news = data.get('articles', [])
        logger.info(f"Loaded {len(news)} news articles")
    
    return papers, news


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description='Build RAG indexes and push to Artifact Registry'
    )
    parser.add_argument('--papers', type=str, help='Path to papers JSON')
    parser.add_argument('--news', type=str, help='Path to news JSON')
    parser.add_argument('--no-tracking', action='store_true', help='Disable MLflow tracking')
    
    # Artifact Registry options
    parser.add_argument(
        '--push-to-registry',
        action='store_true',
        help='Push model to Artifact Registry after training'
    )
    parser.add_argument(
        '--project-id',
        type=str,
        help='GCP project ID (default: from config/env)'
    )
    parser.add_argument(
        '--location',
        type=str,
        default='us-central1',
        help='GCP location (default: us-central1)'
    )
    parser.add_argument(
        '--repository',
        type=str,
        default='rag-models',
        help='Artifact Registry repository name (default: rag-models)'
    )
    parser.add_argument(
        '--version',
        type=str,
        help='Model version (default: auto-generated timestamp)'
    )
    parser.add_argument(
        '--description',
        type=str,
        help='Version description'
    )
    parser.add_argument(
        '--force-push',
        action='store_true',
        help='Push even if validation fails'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("STARTING RAG INDEX BUILDING")
    logger.info("=" * 80)
    
    # Load data
    papers, news = load_data(args.papers, args.news)
    
    if not papers and not news:
        logger.error("No data provided. Use --papers and/or --news arguments.")
        return
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = TechTrendsRAGPipeline(
        enable_tracking=not args.no_tracking
    )
    
    # Index documents
    logger.info("Indexing documents...")
    pipeline.index_documents(papers=papers, news=news)
    
    # Test query
    logger.info("=" * 80)
    logger.info("RUNNING TEST QUERY")
    logger.info("=" * 80)
    test_result = pipeline.query("What are the latest trends in AI?")
    
    logger.info(f"Response: {test_result['response'][:200]}...")
    logger.info(f"Sources: {test_result['num_sources']}")
    logger.info(f"Validation score: {test_result['validation']['overall_score']:.2f}")
    logger.info(
        f"Fairness score: {test_result['bias_report'].get('overall_fairness_score', 0.0):.3f}"
    )
    
    # Push to Artifact Registry if requested
    if args.push_to_registry:
        validation_score = test_result['validation']['overall_score']
        should_push = args.force_push or validation_score >= 0.7
        
        if should_push:
            logger.info("=" * 80)
            logger.info("PUSHING MODEL TO ARTIFACT REGISTRY")
            logger.info("=" * 80)
            
            try:
                # Get project ID
                project_id = args.project_id or os.getenv('GCP_PROJECT_ID')
                if not project_id:
                    logger.error(
                        "GCP project ID required. Use --project-id or set GCP_PROJECT_ID env var"
                    )
                    return
                
                artifact_path = pipeline.push_to_artifact_registry(
                    project_id=project_id,
                    location=args.location,
                    repository=args.repository,
                    version=args.version,
                    metrics=test_result['metrics'],
                    bias_report=test_result['bias_report'],
                    description=args.description
                )
                
                logger.info("=" * 80)
                logger.info(" MODEL SUCCESSFULLY PUSHED TO ARTIFACT REGISTRY")
                logger.info(f"  Location: {artifact_path}")
                logger.info(f"  Version: {args.version or 'auto-generated'}")
                logger.info("=" * 80)
                
            except Exception as e:
                logger.error("=" * 80)
                logger.error(" FAILED TO PUSH MODEL")
                logger.error(f"  Error: {e}")
                logger.error("=" * 80)
        else:
            logger.warning("=" * 80)
            logger.warning(f"VALIDATION SCORE TOO LOW ({validation_score:.2f})")
            logger.warning("Model not pushed to Artifact Registry. Use --force-push to override.")
            logger.warning("=" * 80)
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()