"""
Script to update RAG indexes with new data from cleaned folder
"""
import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any
import sys

from pipeline import TechTrendsRAGPipeline
from utils.logger import setup_logging
from datetime import datetime
logger = logging.getLogger(__name__)


class IndexUpdater:
    """Handles updating RAG indexes with new cleaned data"""
    
    def __init__(self, 
                 cleaned_dir: str = "./data/cleaned",
                 processed_dir: str = "./data/processed"):
        """
        Initialize Index Updater
        
        Args:
            cleaned_dir: Directory containing cleaned data files
            processed_dir: Directory to move processed files
        """
        self.cleaned_dir = Path(cleaned_dir)
        self.processed_dir = Path(processed_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create directories if they don't exist
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline
        self.pipeline = TechTrendsRAGPipeline(enable_tracking=True)
        
    def read_cleaned_files(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Path]]:
        """
        Read all JSON files from cleaned directory
        
        Returns:
            Tuple of (papers_list, news_list, file_paths)
        """
        papers = []
        news = []
        file_paths = []
        
        # Get all JSON files in cleaned directory
        json_files = list(self.cleaned_dir.glob("*.json"))
        
        if not json_files:
            self.logger.warning("No files found in cleaned directory")
            return papers, news, file_paths
        
        self.logger.info(f"Found {len(json_files)} files in cleaned directory")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Determine file type based on filename or content structure
                filename = file_path.name.lower()
                
                if 'arxiv' in filename or 'paper' in filename:
                    # ArXiv papers file
                    if 'papers' in data:
                        papers.extend(data['papers'])
                        self.logger.info(f"Loaded {len(data['papers'])} papers from {file_path.name}")
                    else:
                        self.logger.warning(f"No 'papers' key found in {file_path.name}")
                
                elif 'news' in filename or 'tech' in filename:
                    # News articles file
                    if 'articles' in data:
                        news.extend(data['articles'])
                        self.logger.info(f"Loaded {len(data['articles'])} articles from {file_path.name}")
                    else:
                        self.logger.warning(f"No 'articles' key found in {file_path.name}")
                
                else:
                    # Try to infer from content structure
                    if 'papers' in data:
                        papers.extend(data['papers'])
                        self.logger.info(f"Inferred papers from {file_path.name}")
                    elif 'articles' in data:
                        news.extend(data['articles'])
                        self.logger.info(f"Inferred news from {file_path.name}")
                    else:
                        self.logger.warning(f"Could not determine data type for {file_path.name}")
                
                file_paths.append(file_path)
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON from {file_path.name}: {e}")
            except Exception as e:
                self.logger.error(f"Error reading {file_path.name}: {e}")
        
        return papers, news, file_paths
    
    def move_to_processed(self, file_paths: List[Path]):
        """
        Move processed files to processed directory
        
        Args:
            file_paths: List of file paths to move
        """
        if not file_paths:
            return
        
        self.logger.info(f"Moving {len(file_paths)} files to processed directory")
        
        for file_path in file_paths:
            try:
                destination = self.processed_dir / file_path.name
                
                # If file exists in processed, rename with timestamp
                if destination.exists():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    stem = destination.stem
                    suffix = destination.suffix
                    destination = self.processed_dir / f"{stem}_{timestamp}{suffix}"
                
                shutil.move(str(file_path), str(destination))
                self.logger.info(f"Moved {file_path.name} to {destination}")
                
            except Exception as e:
                self.logger.error(f"Error moving {file_path.name}: {e}")
    
    def update_indexes(self):
        """
        Main function to update indexes with new data
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING INDEX UPDATE PROCESS")
        self.logger.info("=" * 80)
        
        # Step 1: Read cleaned files
        self.logger.info("\nStep 1: Reading cleaned data files...")
        papers, news, file_paths = self.read_cleaned_files()
        
        # Check if any data was found
        if not papers and not news:
            self.logger.warning("⚠️  NO DATA FOUND TO INDEX")
            self.logger.warning("Cleaned folder is empty or contains no valid data files")
            return False
        
        self.logger.info(f"✓ Found {len(papers)} papers and {len(news)} articles")
        
        # Step 2: Load existing indexes
        self.logger.info("\nStep 2: Loading existing indexes...")
        if not self.pipeline.load_indexes():
            self.logger.error("Failed to load existing indexes")
            self.logger.info("Creating new indexes instead...")
            # If no indexes exist, use index_documents instead
            try:
                self.pipeline.index_documents(papers=papers if papers else None, 
                                             news=news if news else None)
                self.logger.info("✓ New indexes created successfully")
            except Exception as e:
                self.logger.error(f"Failed to create indexes: {e}")
                return False
        else:
            self.logger.info("✓ Existing indexes loaded successfully")
            
            # Step 3: Update indexes with new data
            self.logger.info("\nStep 3: Updating indexes with new data...")
            try:
                self.pipeline.update_indexes(papers=papers if papers else None, 
                                            news=news if news else None)
                self.logger.info("✓ Indexes updated successfully")
            except Exception as e:
                self.logger.error(f"Failed to update indexes: {e}")
                return False
        
        # Step 4: Move processed files
        self.logger.info("\nStep 4: Moving processed files...")
        self.move_to_processed(file_paths)
        self.logger.info("✓ Files moved to processed directory")
        
        # Step 5: Get index statistics
        self.logger.info("\nStep 5: Index statistics:")
        try:
            stats = self.pipeline.retriever.get_index_stats()
            self.logger.info(f"Papers index: {stats['papers']['dense']['total_vectors']} vectors")
            self.logger.info(f"News index: {stats['news']['dense']['total_vectors']} vectors")
        except Exception as e:
            self.logger.warning(f"Could not retrieve index stats: {e}")
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("✅ INDEX UPDATE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80)
        
        return True


def main():
    """Main execution function"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Create updater instance
        updater = IndexUpdater()
        
        # Run update process
        success = updater.update_indexes()
        
        if success:
            logger.info("Index update completed successfully")
            sys.exit(0)
        else:
            logger.error("Index update failed or no data found")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Critical error during index update: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()