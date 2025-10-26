"""
Complete News Enrichment Module
Combines content fetching, enrichment pipeline, and Airflow task functions
Location: /opt/airflow/dags/common/news_enrichment.py
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import time
import logging
from datetime import datetime
from urllib.parse import urlparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

# Import utilities from your common modules
from common.data_enrichment import DataEnricher, CategoryManager
from common.file_management import FileManager
from common.data_cleaning import TextCleaner
from common.deduplication import DeduplicationManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONTENT FETCHER CLASS
# ============================================================================

class ArticleContentFetcher:
    """Fetches and extracts full article content from URLs"""
    
    def __init__(self, timeout: int = 10, max_workers: int = 5):
        """
        Initialize content fetcher
        
        Args:
            timeout: Request timeout in seconds
            max_workers: Max parallel workers for fetching
        """
        self.timeout = timeout
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def extract_article_content(self, url: str) -> Dict[str, Optional[str]]:
        """
        Extract article content from URL
        
        Args:
            url: Article URL
            
        Returns:
            Dictionary with extracted content
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "noscript"]):
                script.decompose()
            
            # Try multiple extraction strategies
            content = self._extract_by_article_tags(soup)
            if not content:
                content = self._extract_by_common_patterns(soup)
            if not content:
                content = self._extract_by_paragraphs(soup)
            
            # Extract additional metadata
            metadata = self._extract_metadata(soup)
            
            return {
                'full_content': content,
                'word_count': len(content.split()) if content else 0,
                'extracted_at': datetime.now().isoformat(),
                'extraction_method': 'web_scraping',
                **metadata
            }
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return {
                'full_content': None,
                'word_count': 0,
                'extraction_error': str(e),
                'extracted_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return {
                'full_content': None,
                'word_count': 0,
                'extraction_error': str(e),
                'extracted_at': datetime.now().isoformat()
            }
    
    def _extract_by_article_tags(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract content using article/main tags"""
        article = soup.find('article') or soup.find('main')
        if article:
            paragraphs = article.find_all('p')
            if paragraphs:
                content = ' '.join([p.get_text().strip() for p in paragraphs])
                return self._clean_text(content)
        return None
    
    def _extract_by_common_patterns(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract using common content div patterns"""
        content_indicators = [
            {'class': re.compile(r'(article|content|post|entry|story|body)', re.I)},
            {'id': re.compile(r'(article|content|post|entry|story|body)', re.I)}
        ]
        
        for indicator in content_indicators:
            containers = soup.find_all('div', indicator)
            if containers:
                best_container = max(containers, key=lambda x: len(x.find_all('p')))
                paragraphs = best_container.find_all('p')
                if paragraphs:
                    content = ' '.join([p.get_text().strip() for p in paragraphs])
                    return self._clean_text(content)
        return None
    
    def _extract_by_paragraphs(self, soup: BeautifulSoup) -> Optional[str]:
        """Fallback: Extract all substantial paragraphs"""
        paragraphs = soup.find_all('p')
        substantial_paragraphs = [
            p.get_text().strip() for p in paragraphs 
            if len(p.get_text().strip()) > 50
        ]
        
        if substantial_paragraphs:
            content = ' '.join(substantial_paragraphs)
            return self._clean_text(content)
        return None
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract additional metadata from the page"""
        metadata = {}
        
        # Extract author
        author_meta = soup.find('meta', {'name': 'author'}) or \
                     soup.find('meta', {'property': 'article:author'})
        if author_meta:
            metadata['extracted_author'] = author_meta.get('content')
        
        # Extract publish date
        date_meta = soup.find('meta', {'property': 'article:published_time'}) or \
                   soup.find('meta', {'name': 'publish_date'})
        if date_meta:
            metadata['extracted_publish_date'] = date_meta.get('content')
        
        # Extract keywords
        keywords_meta = soup.find('meta', {'name': 'keywords'})
        if keywords_meta:
            metadata['extracted_keywords'] = keywords_meta.get('content')
        
        # Extract enhanced description
        desc_meta = soup.find('meta', {'name': 'description'}) or \
                   soup.find('meta', {'property': 'og:description'})
        if desc_meta:
            metadata['extracted_description'] = desc_meta.get('content')
        
        return metadata
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(Advertisement|ADVERTISEMENT)', '', text)
        text = re.sub(r'(Read more|Continue reading)\.?$', '', text, flags=re.I)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()
    
    def fetch_multiple(self, articles: List[Dict]) -> List[Dict]:
        """
        Fetch content for multiple articles in parallel
        
        Args:
            articles: List of article dictionaries with 'url' field
            
        Returns:
            List of enriched articles
        """
        enriched_articles = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_article = {
                executor.submit(self.extract_article_content, article['url']): article 
                for article in articles if article.get('url')
            }
            
            for future in as_completed(future_to_article):
                article = future_to_article[future]
                try:
                    content_data = future.result()
                    enriched_article = {**article, **content_data}
                    
                    # Use extracted description if longer
                    if content_data.get('extracted_description'):
                        orig_desc_len = len(article.get('description', ''))
                        new_desc_len = len(content_data['extracted_description'])
                        if new_desc_len > orig_desc_len:
                            enriched_article['enhanced_description'] = content_data['extracted_description']
                    
                    enriched_articles.append(enriched_article)
                    
                except Exception as e:
                    logger.error(f"Error processing article {article.get('title', 'Unknown')}: {e}")
                    article['enrichment_failed'] = True
                    enriched_articles.append(article)
                
                # Rate limiting
                time.sleep(0.5)
        
        return enriched_articles


# ============================================================================
# ENRICHMENT PIPELINE CLASS
# ============================================================================

class NewsEnrichmentPipeline:
    """Complete enrichment pipeline for News API data"""
    
    def __init__(self):
        self.fetcher = ArticleContentFetcher()
        self.enricher = DataEnricher()
        self.category_manager = CategoryManager()
    
    def enrich_news_batch(self, news_data: Dict, categories_config: Dict) -> Dict:
        """
        Enrich a batch of news articles
        
        Args:
            news_data: News API response data
            categories_config: Categories configuration for classification
            
        Returns:
            Enriched news data
        """
        articles = news_data.get('articles', [])
        
        logger.info(f"Starting enrichment for {len(articles)} articles...")
        
        # Step 1: Fetch full content for all articles
        enriched_articles = self.fetcher.fetch_multiple(articles)
        
        # Step 2: Further enrich each article
        for article in enriched_articles:
            # Combine text for categorization
            text_for_categorization = f"{article.get('title', '')} {article.get('description', '')} {article.get('full_content', '')}"
            
            # Categorize based on full content
            categorization = self.category_manager.categorize_content(
                text_for_categorization,
                categories_config
            )
            article.update(categorization)
            
            # Extract keywords from full content
            if article.get('full_content'):
                article['extracted_keywords'] = self.enricher.extract_keywords(
                    article['full_content'],
                    min_length=4,
                    max_keywords=15
                )
            
            # Add timestamps
            article = self.enricher.add_timestamps(article)
            
            # Calculate content quality score
            article['content_quality_score'] = self._calculate_quality_score(article)
        
        # Update the original data
        news_data['articles'] = enriched_articles
        news_data['enrichment_stats'] = self._calculate_enrichment_stats(enriched_articles)
        
        return news_data
    
    def _calculate_quality_score(self, article: Dict) -> float:
        """Calculate quality score based on available content"""
        score = 0.0
        
        if article.get('full_content'):
            word_count = article.get('word_count', 0)
            if word_count > 500:
                score += 0.4
            elif word_count > 200:
                score += 0.3
            elif word_count > 100:
                score += 0.2
            else:
                score += 0.1
        
        if article.get('enhanced_description'):
            score += 0.2
        
        if article.get('extracted_author'):
            score += 0.1
        if article.get('extracted_keywords'):
            score += 0.1
        if article.get('extracted_publish_date'):
            score += 0.1
        
        if article.get('overall_relevance', 0) > 0.5:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_enrichment_stats(self, articles: List[Dict]) -> Dict:
        """Calculate statistics about enrichment process"""
        total = len(articles)
        
        return {
            'total_articles': total,
            'successfully_enriched': sum(1 for a in articles if a.get('full_content')),
            'failed_enrichment': sum(1 for a in articles if a.get('enrichment_failed')),
            'average_word_count': sum(a.get('word_count', 0) for a in articles) / total if total > 0 else 0,
            'articles_with_enhanced_description': sum(1 for a in articles if a.get('enhanced_description')),
            'average_quality_score': sum(a.get('content_quality_score', 0) for a in articles) / total if total > 0 else 0,
            'category_distribution': self.category_manager.get_category_distribution(articles)
        }


# ============================================================================
# AIRFLOW TASK FUNCTIONS
# ============================================================================

def enrich_news_articles(**context):
    """
    Airflow task to enrich news articles with full content from URLs
    Reads from fetch_tech_news output and enriches with web scraping
    """
    try:
        ti = context['task_instance']
        
        # Get the result from fetch_tech_news
        extract_result = ti.xcom_pull(task_ids='extract_tech_news', key='result')
        
        if not extract_result:
            logger.warning("No result from extract_tech_news task")
            ti.xcom_push(key='enriched_result', value={
                'filename': None,
                'article_count': 0,
                'message': 'No data from previous task'
            })
            return 0
        
        logger.info(f"Extract result: {extract_result}")
        
        # Check if we have articles
        if not extract_result.get('filename') or extract_result.get('article_count', 0) == 0:
            logger.warning(f"No articles to enrich: {extract_result.get('message', 'No new articles')}")
            ti.xcom_push(key='enriched_result', value={
                'filename': None,
                'article_count': 0,
                'message': extract_result.get('message', 'No new articles found')
            })
            return 0
        
        # Read the news data from file
        news_file = extract_result['filename']
        logger.info(f"Reading news data from: {news_file}")
        
        try:
            with open(news_file, 'r') as f:
                news_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error reading file {news_file}: {e}")
            ti.xcom_push(key='enriched_result', value={
                'filename': None,
                'article_count': 0,
                'error': str(e)
            })
            return 0
        
        articles = news_data.get('articles', [])
        logger.info(f"Found {len(articles)} articles to enrich")
        
        if not articles:
            ti.xcom_push(key='enriched_result', value={
                'filename': news_file,
                'article_count': 0,
                'message': 'No articles to enrich'
            })
            return 0
        
        # Get keywords configuration
        keywords_config = news_data.get('keywords_config')
        if not keywords_config:
            from news_api_pipeline import NEWS_KEYWORDS
            keywords_config = NEWS_KEYWORDS
        
        # Enrich articles
        try:
            logger.info("Starting article enrichment...")
            pipeline = NewsEnrichmentPipeline()
            enriched_data = pipeline.enrich_news_batch(news_data, keywords_config)
            
            stats = enriched_data.get('enrichment_stats', {})
            logger.info(f"Enrichment complete: {stats}")
            
        except Exception as e:
            logger.error(f"Enrichment error: {e}", exc_info=True)
            enriched_data = news_data
            enriched_data['enrichment_error'] = str(e)
        
        # Save enriched data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'/opt/airflow/data/enriched/tech_news_enriched_{timestamp}.json'
        
        Path('/opt/airflow/data/enriched').mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(enriched_data, f, indent=2, default=str)
            logger.info(f"Enriched data saved to {output_path}")
            
            ti.xcom_push(key='enriched_result', value={
                'filename': output_path,
                'article_count': len(enriched_data.get('articles', [])),
                'enrichment_stats': enriched_data.get('enrichment_stats', {})
            })
            
            return len(enriched_data.get('articles', []))
            
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            ti.xcom_push(key='enriched_result', value={
                'filename': news_file,
                'article_count': len(articles),
                'error': str(e)
            })
            return len(articles)
        
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        context['task_instance'].xcom_push(key='enriched_result', value={
            'filename': None,
            'article_count': 0,
            'error': str(e)
        })
        return 0


def extract_keywords_and_categorize_enriched(**context):
    """
    Categorize articles using enriched content with full article text
    Works with both enriched and original data
    """
    try:
        file_mgr = FileManager()
        cleaner = TextCleaner()
        ti = context['task_instance']
        
        # Try enriched data first, fallback to original
        enriched_result = ti.xcom_pull(task_ids='enrich_article_content', key='enriched_result')
        
        data_file = None
        using_enriched = False
        
        if enriched_result and enriched_result.get('filename'):
            data_file = enriched_result['filename']
            using_enriched = True
            logger.info(f"Using enriched data from: {data_file}")
        else:
            extract_result = ti.xcom_pull(task_ids='extract_tech_news', key='result')
            if extract_result and extract_result.get('filename'):
                data_file = extract_result['filename']
                logger.info(f"Using original data from: {data_file}")
            else:
                logger.warning("No data to categorize")
                ti.xcom_push(key='categorized_result', value={
                    'filename': None,
                    'article_count': 0,
                    'category_distribution': {}
                })
                return 0
        
        # Load and process data
        data = file_mgr.load_json(data_file)
        articles = data.get('articles', [])
        
        if not articles:
            ti.xcom_push(key='categorized_result', value={
                'filename': None,
                'article_count': 0,
                'category_distribution': {}
            })
            return 0
        
        keywords_config = data.get('keywords_config')
        if not keywords_config:
            from news_api_pipeline import NEWS_KEYWORDS
            keywords_config = NEWS_KEYWORDS
        
        logger.info(f"Categorizing {len(articles)} articles (enriched: {using_enriched})")
        
        categorized_articles = []
        
        for article in articles:
            title_clean = cleaner.clean_text(article.get('title', ''))
            desc_clean = cleaner.clean_text(article.get('description', ''))
            
            if not title_clean:
                continue
            
            # Use full content if available
            if article.get('full_content'):
                full_content_preview = article['full_content'][:3000]
                combined_text = f"{title_clean} {desc_clean} {full_content_preview}"
            else:
                combined_text = f"{title_clean} {desc_clean}"
            
            categorization = CategoryManager.categorize_content(combined_text, keywords_config)
            
            categorized = {
                'article_id': DeduplicationManager('news').generate_hash(title_clean, article.get('url', '')),
                'title': title_clean,
                'description': desc_clean,
                'enhanced_description': article.get('enhanced_description', desc_clean),
                'url': article.get('url'),
                'source_name': article.get('source', {}).get('name') if isinstance(article.get('source'), dict) else article.get('source_name'),
                'author': article.get('extracted_author') or article.get('author') or 'Unknown',
                'published_at': article.get('publishedAt') or article.get('published_at'),
                'image_url': article.get('urlToImage') or article.get('image_url'),
                'primary_category': categorization['primary_category'],
                'all_categories': categorization['all_categories'],
                'category_scores': categorization['category_scores'],
                'overall_relevance': categorization['overall_relevance'],
                'processed_at': datetime.now().isoformat(),
                'full_content': article.get('full_content', '')[:10000] if article.get('full_content') else None,
                'word_count': article.get('word_count', 0),
                'content_quality_score': article.get('content_quality_score', 0),
                'extracted_keywords': article.get('extracted_keywords', [])
            }
            
            categorized_articles.append(categorized)
        
        categorized_articles.sort(key=lambda x: x['overall_relevance'], reverse=True)
        
        # Calculate statistics
        category_counts = {}
        enrichment_success = sum(1 for a in categorized_articles if a.get('full_content'))
        
        for article in categorized_articles:
            cat = article['primary_category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        logger.info(f"Categorized {len(categorized_articles)} articles ({enrichment_success} enriched)")
        
        # Save categorized data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        categorized_file = f'/opt/airflow/data/cleaned/tech_news_categorized_{timestamp}.json'
        
        categorized_data = {
            'keywords_config': keywords_config,
            'total_articles': len(categorized_articles),
            'enriched_count': enrichment_success,
            'category_distribution': category_counts,
            'processed_at': datetime.now().isoformat(),
            'articles': categorized_articles
        }
        
        file_mgr.save_json(categorized_data, categorized_file)
        
        ti.xcom_push(key='categorized_result', value={
            'filename': categorized_file,
            'article_count': len(categorized_articles),
            'enriched_count': enrichment_success,
            'category_distribution': category_counts
        })
        
        logger.info(f"Saved to {categorized_file}")
        
        return len(categorized_articles)
        
    except Exception as e:
        logger.error(f"Categorization error: {e}", exc_info=True)
        raise