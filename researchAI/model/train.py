import json
import logging
from pathlib import Path
import argparse

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
    parser = argparse.ArgumentParser(description='Build RAG indexes')
    parser.add_argument('--papers', type=str, help='Path to papers JSON')
    parser.add_argument('--news', type=str, help='Path to news JSON')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--no-tracking', action='store_true', help='Disable MLflow tracking')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting RAG index building")
    
    # Load data
    papers, news = load_data(args.papers, args.news)
    
    if not papers and not news:
        logger.error("No data provided. Use --papers and/or --news arguments.")
        return
    
    # Initialize pipeline
    pipeline = TechTrendsRAGPipeline(
        enable_cache=not args.no_cache,
        enable_tracking=not args.no_tracking
    )
    
    # Index documents
    pipeline.index_documents(papers=papers, news=news)
    
    # Test query
    logger.info("Testing with sample query")
    test_result = pipeline.query("What are the latest trends in AI?")
    
    logger.info(f"Test query completed")
    logger.info(f"Response: {test_result['response'][:200]}...")
    logger.info(f"Sources: {test_result['num_sources']}")
    logger.info(f"Validation score: {test_result['validation']['overall_score']:.2f}")
    
    logger.info("Index building completed successfully")


if __name__ == "__main__":
    main()