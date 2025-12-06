from datetime import datetime, timedelta
from pathlib import Path
import requests
import logging
from bs4 import BeautifulSoup
import yaml

import re
from airflow.models import Variable

# Import common utilities from modular structure
from common.deduplication import DeduplicationManager
from common.data_cleaning import TextCleaner
from common.data_enrichment import CategoryManager
from common.file_management import FileManager
from common.database_utils import DatabaseManager
from text_summarizer.summarize_text import Summarize

class NewsAPIPipeline:
    """
    News API Pipeline to fetch, process, categorize, and store tech news articles.
    """
    def __init__(self):
        with open("/home/airflow/gcs/dags/common/config/secrets.yaml") as f:
            self.config = yaml.safe_load(f)
        # === Logging setup ===
        base_log_dir = Path("/home/airflow/gcs/logs")
        base_log_dir.mkdir(parents=True, exist_ok=True)  # ensure directory exists
        log_file_path = base_log_dir / f"{__name__}_{datetime.now().strftime('%Y-%m-%d')}.log"
        logging.basicConfig(filename=log_file_path, level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.deduplication_manager = DeduplicationManager('news')
        self.text_cleaner = TextCleaner()
        self.category_manager = CategoryManager()
        self.file_manager = FileManager()
        self.database_manager = DatabaseManager()
        self.text_summarizer = Summarize()
        # Default arguments for the DAG
        self.default_args = {
            'owner': 'airflow',
            'depends_on_past': False,
            'email_on_failure': True,
            'email_on_retry': True,
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
        }
        # CONFIGURATION: Define your preferred keywords and categories
        self.NEWS_KEYWORDS = {
            'artificial_intelligence': {
                'keywords': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks', 'GPT', 'LLM', 'AI', 'generative AI', 'OpenAI', 'Claude', 'ChatGPT', 'transformer', 'reinforcement learning'],
                'weight': 1.0
            },
            'language_models': {
                'keywords': ['large language model', 'foundation model', 'GPT-4', 'GPT-5', 'Gemini', 'Llama', 'Mistral', 'prompt engineering', 'fine-tuning', 'RLHF', 'instruction tuning', 'context window', 'reasoning model', 'pre-training', 'BERT'],
                'weight': 0.95
            },
            'multimodal_ai': {
                'keywords': ['multimodal AI', 'multimodal learning', 'vision-language model', 'text-to-image', 'text-to-video', 'DALL-E', 'Midjourney', 'Stable Diffusion', 'Sora', 'image generation', 'video generation', 'speech synthesis', 'text-to-speech', 'cross-modal retrieval'],
                'weight': 0.9
            },
            'efficient_ml': {
                'keywords': ['model compression', 'quantization', 'pruning', 'knowledge distillation', 'low-rank adaptation', 'LoRA', 'QLoRA', 'parameter-efficient fine-tuning', 'PEFT', 'mixture of experts', 'MoE', 'model optimization', 'edge AI'],
                'weight': 0.9
            },
            'ai_agents': {
                'keywords': ['AI agent', 'autonomous agent', 'agentic AI', 'agentic LLM', 'multi-agent system', 'agent framework', 'AutoGPT', 'agent orchestration', 'tool use', 'function calling', 'agent memory', 'planning agent', 'collaborative agents'],
                'weight': 0.9
            },
            'reasoning_planning': {
                'keywords': ['reasoning', 'chain-of-thought', 'CoT', 'tree of thoughts', 'planning', 'problem solving', 'mathematical reasoning', 'logical reasoning', 'commonsense reasoning', 'zero-shot', 'few-shot learning', 'in-context learning'],
                'weight': 0.9
            },
            'diffusion_generative': {
                'keywords': ['diffusion model', 'denoising diffusion', 'DDPM', 'DDIM', 'score-based model', 'latent diffusion', 'consistency model', 'flow matching', 'generative adversarial network', 'GAN', 'StyleGAN', 'image-to-image translation'],
                'weight': 0.85
            },
            'retrieval_augmentation': {
                'keywords': ['retrieval augmented generation', 'RAG', 'dense retrieval', 'semantic search', 'vector search', 'vector database', 'embedding model', 'FAISS', 'approximate nearest neighbor', 'knowledge retrieval', 'semantic indexing'],
                'weight': 0.85
            },
            'rl_agents': {
                'keywords': ['reinforcement learning', 'deep RL', 'policy gradient', 'Q-learning', 'DQN', 'PPO', 'SAC', 'multi-agent RL', 'inverse RL', 'offline RL', 'model-based RL', 'world model', 'reward modeling'],
                'weight': 0.85
            },
            'ai_safety_alignment': {
                'keywords': ['AI safety', 'AI alignment', 'AI ethics', 'responsible AI', 'AI governance', 'AI regulation', 'explainable AI', 'interpretability', 'XAI', 'bias detection', 'fairness', 'AI risks', 'existential risk', 'mechanistic interpretability'],
                'weight': 0.85
            },
            'computer_vision': {
                'keywords': ['computer vision', 'object detection', 'image segmentation', 'semantic segmentation', 'instance segmentation', 'panoptic segmentation', 'image classification', 'facial recognition', 'YOLO', 'SAM', 'vision transformer', 'ViT', '3D vision', '3D reconstruction', 'depth estimation', 'NeRF', 'action recognition', 'optical flow'],
                'weight': 0.8
            },
            'nlp_techniques': {
                'keywords': ['natural language processing', 'NLP', 'text generation', 'language understanding', 'named entity recognition', 'NER', 'question answering', 'summarization', 'machine translation', 'sentiment analysis', 'information extraction'],
                'weight': 0.8
            },
            'mlops_infrastructure': {
                'keywords': ['MLOps', 'LLMOps', 'AgentOps', 'model deployment', 'model serving', 'AI infrastructure', 'AI pipelines', 'model monitoring', 'drift detection', 'continuous evaluation', 'distributed training', 'model parallelism'],
                'weight': 0.8
            },
            'robotics_embodied': {
                'keywords': ['robotics', 'embodied AI', 'humanoid robot', 'robotic automation', 'robot learning', 'manipulation', 'grasping', 'navigation', 'sim-to-real', 'imitation learning', 'demonstration learning', 'motion planning', 'robotic vision', 'dynamics model', 'Boston Dynamics', 'Tesla Bot'],
                'weight': 0.75
            },
            'graph_knowledge': {
                'keywords': ['graph neural network', 'GNN', 'graph transformer', 'message passing', 'node embedding', 'link prediction', 'graph representation learning', 'knowledge graph', 'relational reasoning', 'semantic retrieval'],
                'weight': 0.75
            },
            'continual_meta_learning': {
                'keywords': ['continual learning', 'lifelong learning', 'catastrophic forgetting', 'meta-learning', 'transfer learning', 'domain adaptation', 'multi-task learning', 'curriculum learning', 'self-supervised learning', 'semi-supervised learning'],
                'weight': 0.7
            },
            'federated_privacy': {
                'keywords': ['federated learning', 'differential privacy', 'privacy-preserving ML', 'privacy-preserving AI', 'secure aggregation', 'homomorphic encryption', 'synthetic data', 'data poisoning', 'backdoor attack'],
                'weight': 0.7
            },
            'ai_hardware': {
                'keywords': ['AI chip', 'GPU', 'TPU', 'neural processing unit', 'NPU', 'AI accelerator', 'Nvidia H100', 'neuromorphic computing', 'quantum AI', 'inference optimization', 'tinyML'],
                'weight': 0.7
            },
            'neurosymbolic': {
                'keywords': ['neurosymbolic', 'neuro-symbolic AI', 'symbolic reasoning', 'logic programming', 'program synthesis', 'differentiable programming', 'probabilistic programming', 'causal inference'],
                'weight': 0.65
            },
            'medical_scientific_ai': {
                'keywords': ['medical AI', 'healthcare AI', 'medical imaging', 'disease diagnosis', 'drug discovery', 'protein folding', 'genomics', 'clinical decision support', 'radiology AI', 'scientific machine learning', 'physics-informed neural network', 'PINN', 'scientific AI'],
                'weight': 0.6
            },
            'time_series': {
                'keywords': ['time series', 'forecasting', 'temporal modeling', 'sequential data', 'recurrent neural network', 'RNN', 'LSTM', 'GRU', 'temporal convolution', 'attention mechanism'],
                'weight': 0.55
            }
        }


    def fetch_tech_news(self, **context):
        """
        Fetches tech news based on predefined keywords from NEWS_KEYWORDS configuration.
        Fetches news for ALL categories and tags each article with matched categories.
        """
        try:
            # Initialize common utilities
            file_mgr = self.file_manager
            dedup = self.deduplication_manager
            
            # Ensure directories exist
            file_mgr.ensure_directories()
            
            # Build search query from all keywords
            all_keywords = []
            for category_data in self.NEWS_KEYWORDS.values():
                all_keywords.extend(category_data['keywords'])
            
            # Create OR query for NewsAPI
            search_query = ' OR '.join([f'"{kw}"' for kw in all_keywords[:10]])  # Limit to avoid too long queries

            self.logger.info(f"[EXTRACT] Fetching news with keywords: {search_query}")

            # NewsAPI endpoint and parameters
            url = 'https://newsapi.org/v2/everything'
            params = {
                'apiKey': self.config['news_api'],
                'q': search_query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 100,  # Increased to get more articles
                'domains': "techcrunch.com,theverge.com,wired.com,arstechnica.com,zdnet.com,bleepingcomputer.com,securityweek.com,thehackernews.com,venturebeat.com,thenextweb.com",
                'from': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')  # Last 3 days
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])

            self.logger.info(f"[EXTRACT] Total articles fetched: {len(articles)}")

            # Filter duplicates using common utility
            new_articles, new_hashes = dedup.filter_duplicates(
                articles,
                lambda article: dedup.generate_hash(
                    article.get('title', ''),
                    article.get('url', '')
                )
            )

            self.logger.info(f"[EXTRACT] New unique articles: {len(new_articles)}")
            self.logger.info(f"[EXTRACT] Duplicates filtered: {len(articles) - len(new_articles)}")

            if new_articles:
                # Get current timestamp for the filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'/home/airflow/gcs/data/raw/tech_news_{timestamp}.json'
                
                # Prepare output data
                output_data = {
                    'status': data.get('status'),
                    'totalResults': len(new_articles),
                    'keywords_config': self.NEWS_KEYWORDS,
                    'fetchedAt': datetime.now().isoformat(),
                    'articles': new_articles
                }
                
                # Save using common utility
                file_mgr.save_json(output_data, filename)
                
                # Update hashes using common utility
                dedup.update_hashes(new_hashes)

                self.logger.info(f"[EXTRACT] Successfully saved {len(new_articles)} new articles to {filename}")

                # Push result to XCom
                context['task_instance'].xcom_push(key='result', value={
                    'filename': filename,
                    'article_count': len(new_articles),
                    'keywords_used': list(self.NEWS_KEYWORDS.keys())
                })
            else:
                self.logger.info("[EXTRACT] No new articles found. All articles were duplicates.")
                context['task_instance'].xcom_push(key='result', value={
                    'filename': None,
                    'article_count': 0,
                    'message': 'No new articles found'
                })
            
            return len(new_articles)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[EXTRACT] Error fetching news data: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"[EXTRACT] Response status: {e.response.status_code}")
                self.logger.error(f"[EXTRACT] Response body: {e.response.text}")
            raise
        except Exception as e:
            self.logger.error(f"[EXTRACT] Unexpected error: {str(e)}")
            raise


    def extract_keywords_and_categorize(self, **context):
        """
        NEW TASK: Extract keywords from articles and categorize them based on NEWS_KEYWORDS config.
        Uses CategoryManager from common utilities.
        """
        try:
            # Initialize common utilities
            file_mgr = self.file_manager
            cleaner = self.text_cleaner
            
            # Get data from previous task
            result = context['task_instance'].xcom_pull(task_ids='extract_tech_news', key='result')
            
            if not result or not result.get('filename'):
                self.logger.info("[CATEGORIZE] No data to process")
                return 0
            
            raw_file = result['filename']
            self.logger.info(f"[CATEGORIZE] Processing file: {raw_file}")

            # Load raw data
            raw_data = file_mgr.load_json(raw_file)
            articles = raw_data.get('articles', [])
            
            # Process and categorize articles using CategoryManager
            categorized_articles = []
            
            for article in articles:
                # Clean text using common utility
                title_clean = cleaner.clean_text(article.get('title', ''))
                desc_clean = cleaner.clean_text(article.get('description', ''))
                
                if not title_clean:
                    continue  # Skip invalid articles
                
                combined_text = title_clean + ' ' + desc_clean
                
                # Use CategoryManager to categorize
                categorization = self.category_manager.categorize_content(combined_text, self.NEWS_KEYWORDS)
                
                enriched = {
                    'article_id': self.deduplication_manager.generate_hash(title_clean, article.get('url', '')),
                    'title': title_clean,
                    'description': self.extract_article_content(url=article.get('url')) if self.extract_article_content(url=article.get('url')) else desc_clean,
                    'url': article.get('url'),
                    'source_name': article.get('source', {}).get('name'),
                    'author': article.get('author') or 'Unknown',
                    'published_at': article.get('publishedAt'),
                    'primary_category': categorization['primary_category'],
                    'all_categories': categorization['all_categories'],
                    'category_scores': categorization['category_scores'],
                    'overall_relevance': categorization['overall_relevance'],
                    'processed_at': datetime.now().isoformat()
                }
                
                categorized_articles.append(enriched)
            
            # Sort by overall relevance
            categorized_articles.sort(key=lambda x: x['overall_relevance'], reverse=True)

            self.logger.info(f"[CATEGORIZE] Categorized {len(categorized_articles)} articles")

            # Print category distribution
            category_counts = {}
            for article in categorized_articles:
                cat = article['primary_category']
                category_counts[cat] = category_counts.get(cat, 0) + 1

            self.logger.info("[CATEGORIZE] Category distribution:")
            for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  - {cat}: {count} articles")

            # Save categorized data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            categorized_file = f'/home/airflow/gcs/data/cleaned/tech_news_categorized_{timestamp}.json'
            
            categorized_data = {
                'keywords_config': self.NEWS_KEYWORDS,
                'total_articles': len(categorized_articles),
                'category_distribution': category_counts,
                'processed_at': datetime.now().isoformat(),
                'articles': categorized_articles
            }
            
            file_mgr.save_json(categorized_data, categorized_file)
            
            # Push to XCom
            context['task_instance'].xcom_push(key='categorized_result', value={
                'filename': categorized_file,
                'article_count': len(categorized_articles),
                'category_distribution': category_counts
            })

            self.logger.info(f"[CATEGORIZE] Saved categorized data to {categorized_file}")

            return len(categorized_articles)
            
        except Exception as e:
            self.logger.error(f"[CATEGORIZE] Error: {str(e)}")
            raise

    def extract_article_content(self, url: str):
        """
        Extract article content from URL
        
        Args:
            url: Article URL
            
        Returns:
            Dictionary with extracted content
        """
        try:
            session = requests.Session()
            session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            response = session.get(url, timeout=30)
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

            summarized_content = self.text_summarizer.summarize_news_descriptions(content)
            self.logger.info("Summarization complete!")
            return summarized_content
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing {url}: {e}")
            return None

    def _extract_by_article_tags(self, soup):
        """Extract content using article/main tags"""
        article = soup.find('article') or soup.find('main')
        if article:
            paragraphs = article.find_all('p')
            if paragraphs:
                content = ' '.join([p.get_text().strip() for p in paragraphs])
                return self._clean_text(content)
        return None
    
    def _extract_by_common_patterns(self, soup):
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
    
    def _extract_by_paragraphs(self, soup):
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
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(Advertisement|ADVERTISEMENT)', '', text)
        text = re.sub(r'(Read more|Continue reading)\.?$', '', text, flags=re.I)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()
