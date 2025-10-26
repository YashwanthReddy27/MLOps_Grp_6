from datetime import datetime, timedelta
import os
import json
import requests
import logging
from airflow.models import Variable

# Import common utilities from modular structure
from common.deduplication import DeduplicationManager
from common.data_cleaning import TextCleaner
from common.data_enrichment import CategoryManager
from common.file_management import FileManager
from common.database_utils import DatabaseManager

class NewsAPIPipeline:
    """
    News API Pipeline to fetch, process, categorize, and store tech news articles.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.deduplication_manager = DeduplicationManager('news')
        self.text_cleaner = TextCleaner()
        self.category_manager = CategoryManager()
        self.file_manager = FileManager()
        self.database_manager = DatabaseManager()
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
                'apiKey': os.getenv('NEWS_API_KEY'),
                'q': search_query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 100,  # Increased to get more articles
                'domains': 'techcrunch.com,theverge.com,wired.com,arstechnica.com,zdnet.com,bleepingcomputer.com,securityweek.com,thehackernews.com,venturebeat.com,thenextweb.com',
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
                filename = f'/opt/airflow/data/raw/tech_news_{timestamp}.json'
                
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
                categorization = self.CategoryManager.categorize_content(combined_text, self.NEWS_KEYWORDS)
                
                enriched = {
                    'article_id': self.deduplication_manager.generate_hash(title_clean, article.get('url', '')),
                    'title': title_clean,
                    'description': desc_clean,
                    'url': article.get('url'),
                    'source_name': article.get('source', {}).get('name'),
                    'author': article.get('author') or 'Unknown',
                    'published_at': article.get('publishedAt'),
                    'image_url': article.get('urlToImage'),
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
            categorized_file = f'/opt/airflow/data/cleaned/tech_news_categorized_{timestamp}.json'
            
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


    def load_to_postgresql(self, **context):
        """
        Load categorized articles to PostgreSQL with multi-category support.
        """
        try:
            db = self.database_manager
            file_mgr = self.file_manager
            
            categorized_result = context['ti'].xcom_pull(task_ids='extract_keywords_and_categorize', key='categorized_result')
            
            # Check if we have new articles
            if not categorized_result or not categorized_result.get('filename'):
                self.logger.info("[DB] No new articles to load")
                
                # Return existing articles from database
                try:
                    existing_articles = db.execute_query("""
                        SELECT article_id, title, description, url, source_name, 
                            primary_category, all_categories, overall_relevance, published_at
                        FROM tech_news_articles
                        ORDER BY overall_relevance DESC, published_at DESC
                        LIMIT 50
                    """)
                    
                    if existing_articles:
                        self.logger.info(f"[DB] Found {len(existing_articles)} existing articles in database")
                        
                        context['ti'].xcom_push(key='result', value={
                            'status': 'no_new_articles',
                            'message': 'All articles were duplicates. Returning previously fetched articles.',
                            'article_count': len(existing_articles),
                            'articles': existing_articles
                        })
                        return len(existing_articles)
                    else:
                        self.logger.info("[DB] No existing articles found in database")
                        context['ti'].xcom_push(key='result', value={
                            'status': 'no_articles',
                            'message': 'No articles found',
                            'article_count': 0
                        })
                        return 0
                        
                except Exception as e:
                    self.logger.error(f"[DB] Error querying existing articles: {e}")
                    context['ti'].xcom_push(key='result', value={
                        'status': 'no_articles',
                        'message': 'No new articles and database query failed',
                        'article_count': 0
                    })
                    return 0
            
            # Load new categorized articles
            categorized_file = categorized_result['filename']
            self.logger.info(f"[DB] Loading new data from: {categorized_file}")
            
            categorized_data = file_mgr.load_json(categorized_file)
            articles = categorized_data.get('articles', [])
            
            # Create table with multi-category support
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS tech_news_articles (
                article_id VARCHAR(32) PRIMARY KEY,
                title VARCHAR(500) NOT NULL,
                description TEXT,
                url VARCHAR(500),
                source_name VARCHAR(100),
                author VARCHAR(200),
                published_at TIMESTAMP,
                image_url VARCHAR(500),
                primary_category VARCHAR(100),
                all_categories TEXT[],
                category_scores JSONB,
                overall_relevance DECIMAL(5,3),
                processed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_primary_category ON tech_news_articles(primary_category);
            CREATE INDEX IF NOT EXISTS idx_all_categories ON tech_news_articles USING GIN(all_categories);
            CREATE INDEX IF NOT EXISTS idx_relevance ON tech_news_articles(overall_relevance);
            CREATE INDEX IF NOT EXISTS idx_published ON tech_news_articles(published_at);
            """
            
            db.create_table_if_not_exists('tech_news_articles', create_table_sql)
            
            # Prepare and insert records
            db_records = []
            for article in articles:
                db_records.append({
                    'article_id': article['article_id'],
                    'title': article['title'][:500],
                    'description': article.get('description', '')[:1000] if article.get('description') else None,
                    'url': article.get('url', '')[:500] if article.get('url') else None,
                    'source_name': article.get('source_name', '')[:100] if article.get('source_name') else None,
                    'author': article.get('author', 'Unknown')[:200],
                    'published_at': article.get('published_at'),
                    'image_url': article.get('image_url', '')[:500] if article.get('image_url') else None,
                    'primary_category': article.get('primary_category', 'general_tech'),
                    'all_categories': article.get('all_categories', []),
                    'category_scores': json.dumps(article.get('category_scores', {})),
                    'overall_relevance': article.get('overall_relevance', 0.0),
                    'processed_at': article.get('processed_at'),
                })
            
            count = db.upsert_records('tech_news_articles', db_records, 'article_id')
            self.logger.info(f"[DB] Successfully loaded {count} NEW articles to PostgreSQL")
            
            # Get category distribution from database
            category_dist = db.execute_query("""
                SELECT primary_category, COUNT(*) as count
                FROM tech_news_articles
                GROUP BY primary_category
                ORDER BY count DESC
            """)
            
            context['ti'].xcom_push(key='result', value={
                'status': 'success',
                'message': f'Successfully loaded {count} new articles',
                'article_count': count,
                'category_distribution': {row[0]: row[1] for row in category_dist}
            })
            
            return count
            
        except Exception as e:
            self.logger.error(f"[DB] Error: {e}")
            import traceback
            traceback.print_exc()
            return 0


    def cleanup_old_files(self, **context):
        """
        Clean up old files using common utility.
        """
        try:
            file_mgr = self.file_manager
            
            # Get retention period from Airflow Variables (default 7 days)
            retention_days = int(Variable.get("news_retention_days", default_var="7"))

            self.logger.info(f"[CLEANUP] Starting cleanup with {retention_days} day retention")

            # Clean up raw files
            deleted_raw = file_mgr.cleanup_old_files(
                '/opt/airflow/data/raw',
                'tech_news_',
                days=retention_days
            )
            
            # Clean up categorized files
            deleted_categorized = file_mgr.cleanup_old_files(
                '/opt/airflow/data/cleaned',
                'tech_news_categorized_',
                days=retention_days
            )
            
            total_deleted = deleted_raw + deleted_categorized

            self.logger.info(f"[CLEANUP] Summary: Deleted {total_deleted} files total")
            self.logger.info(f"[CLEANUP]   - Raw files: {deleted_raw}")
            self.logger.info(f"[CLEANUP]   - Categorized files: {deleted_categorized}")

            # Push cleanup results to XCom
            context['task_instance'].xcom_push(key='cleanup_result', value={
                'deleted_files': total_deleted,
                'deleted_raw': deleted_raw,
                'deleted_categorized': deleted_categorized,
                'retention_days': retention_days
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"[CLEANUP] Error during cleanup: {str(e)}")
            # Don't raise - cleanup failure shouldn't fail the entire DAG
            return False
