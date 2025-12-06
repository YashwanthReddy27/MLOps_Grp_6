"""
arXiv AI Research Pipeline
Fetches and categorizes AI research papers from arXiv using the same keywords as news pipeline
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
import requests
import logging
import xml.etree.ElementTree as ET
from airflow.models import Variable


# Import common utilities from modular structure
from common.deduplication import DeduplicationManager
from common.data_cleaning import TextCleaner
from common.data_enrichment import CategoryManager
from common.data_validator import DataValidator
from common.database_utils import DatabaseManager
from common.file_management import FileManager

class ArxivPipeline:

    def __init__(self):
        self.deduplication_manager = DeduplicationManager('arxiv', base_dir = '/home/airflow/gcs/data')
        self.text_cleaner = TextCleaner()
        self.category_manager = CategoryManager()
        self.data_validator = DataValidator()
        self.database_manager = DatabaseManager()
        self.file_manager = FileManager()
        # Default arguments for the DAG
        self.default_args = {
            'owner': 'airflow',
            'depends_on_past': False,
            'email_on_failure': True,
            'email_on_retry': True,
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
        }

        # === Logging setup ===
        base_log_dir = Path("/home/airflow/gcs/logs")
        base_log_dir.mkdir(parents=True, exist_ok=True)  # ensure directory exists
        log_file_path = base_log_dir / f"{__name__}_{datetime.now().strftime('%Y-%m-%d')}.log"
        
        logging.basicConfig(filename=log_file_path, level=logging.INFO)
        self.logger = logging.getLogger(__name__)


        # CONFIGURATION: AI research keywords (aligned with news pipeline)
        self.ARXIV_KEYWORDS = {
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
        # arXiv API configuration
        self.ARXIV_API_BASE = 'http://export.arxiv.org/api/query'
        self.ARXIV_CATEGORIES = [
            'cs.AI',  # Artificial Intelligence
            'cs.LG',  # Machine Learning
            'cs.CL',  # Computation and Language
            'cs.CV',  # Computer Vision
            'cs.NE',  # Neural and Evolutionary Computing
        ]


    def fetch_arxiv_papers(self,**context):
        """
        Fetch AI research papers from arXiv based on categories and keywords.
        Uses common utilities for deduplication and file management.
        """
        try:
            # Initialize common utilities
            file_mgr = self.file_manager
            dedup = self.deduplication_manager

            
            # Ensure directories exist
            file_mgr.ensure_directories()

            self.logger.info(f"[EXTRACT] Fetching arXiv papers for AI research")

            # Build search query for arXiv categories
            category_query = ' OR '.join([f'cat:{cat}' for cat in self.ARXIV_CATEGORIES])

            # Calculate date range (last 7 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # arXiv API parameters
            params = {
                'search_query': category_query,
                'start': 0,
                'max_results': 100,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }

            self.logger.info(f"[EXTRACT] Query: {category_query}")

            self.logger.info(f"[EXTRACT] Date range: {start_date.date()} to {end_date.date()}")

            response = requests.get(self.ARXIV_API_BASE, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom',
                        'arxiv': 'http://arxiv.org/schemas/atom'}
            
            entries = root.findall('atom:entry', namespace)
            
            print(f"[EXTRACT] Total papers fetched: {len(entries)}")
            
            # Parse entries
            papers = []
            for entry in entries:
                try:
                    # Extract paper details
                    arxiv_id = entry.find('atom:id', namespace).text.split('/abs/')[-1]
                    title = entry.find('atom:title', namespace).text.strip()
                    summary = entry.find('atom:summary', namespace).text.strip()
                    published = entry.find('atom:published', namespace).text
                    updated = entry.find('atom:updated', namespace).text
                    
                    # Authors
                    authors = []
                    for author in entry.findall('atom:author', namespace):
                        name = author.find('atom:name', namespace).text
                        authors.append(name)
                    
                    # Categories
                    categories = []
                    for category in entry.findall('atom:category', namespace):
                        cat = category.get('term')
                        categories.append(cat)
                    
                    # Primary category
                    primary_category_elem = entry.find('arxiv:primary_category', namespace)
                    primary_category = primary_category_elem.get('term') if primary_category_elem is not None else categories[0] if categories else 'unknown'
                    
                    # PDF link
                    pdf_link = None
                    for link in entry.findall('atom:link', namespace):
                        if link.get('title') == 'pdf':
                            pdf_link = link.get('href')
                            break
                    
                    # HTML link
                    html_link = entry.find('atom:id', namespace).text
                    
                    paper = {
                        'arxiv_id': arxiv_id,
                        'title': title,
                        'abstract': summary,
                        'authors': authors,
                        'published_date': published,
                        'updated_date': updated,
                        'categories': categories,
                        'primary_category': primary_category,
                        'pdf_url': pdf_link,
                        'html_url': html_link
                    }
                    
                    papers.append(paper)
                    
                except Exception as e:
                    self.logger.error(f"[EXTRACT] Error parsing entry: {e}")
                    continue
            
            self.logger.info(f"[EXTRACT] Successfully parsed {len(papers)} papers")

            # Filter duplicates using common utility
            new_papers, new_hashes = dedup.filter_duplicates(
                papers,
                lambda paper: dedup.generate_hash(
                    paper.get('arxiv_id', ''),
                    paper.get('title', '')
                )
            )
            self.logger.info(f"[EXTRACT] New unique papers: {len(new_papers)}")
            self.logger.info(f"[EXTRACT] Duplicates filtered: {len(papers) - len(new_papers)}")

            if new_papers:
                # Get current timestamp for the filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'/home/airflow/gcs/data/raw/arxiv_papers_{timestamp}.json'
                
                # Prepare output data
                output_data = {
                    'status': 'success',
                    'totalResults': len(new_papers),
                    'categories': self.ARXIV_CATEGORIES,
                    'keywords_config': self.ARXIV_KEYWORDS,
                    'fetchedAt': datetime.now().isoformat(),
                    'papers': new_papers
                }
                
                # Save using common utility
                file_mgr.save_json(output_data, filename)
                
                # Update hashes using common utility
                dedup.update_hashes(new_hashes)

                self.logger.info(f"[EXTRACT] Successfully saved {len(new_papers)} new papers to {filename}")
                
                # Push result to XCom
                context['task_instance'].xcom_push(key='result', value={
                    'filename': filename,
                    'paper_count': len(new_papers)
                })
            else:
                self.logger.info("[EXTRACT] No new papers found. All papers were duplicates.")
                context['task_instance'].xcom_push(key='result', value={
                    'filename': None,
                    'paper_count': 0,
                    'message': 'No new papers found'
                })
            
            return len(new_papers)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[EXTRACT] Error fetching arXiv data: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"[EXTRACT] Unexpected error: {str(e)}")
            raise


    def process_and_categorize_papers(self, **context):
        """
        Process arXiv papers: clean LaTeX, categorize by AI keywords.
        Uses CategoryManager from common utilities.
        """
        try:
            # Initialize common utilities
            file_mgr = self.file_manager
            cleaner = self.text_cleaner
            validator = self.data_validator
            
            # Get data from previous task
            result = context['task_instance'].xcom_pull(task_ids='fetch_arxiv_papers', key='result')
            
            if not result or not result.get('filename'):
                self.logger.info("[PROCESS] No data to process")
                return 0
            
            raw_file = result['filename']
            self.logger.info(f"[PROCESS] Processing file: {raw_file}")

            # Load raw data
            raw_data = file_mgr.load_json(raw_file)
            papers = raw_data.get('papers', [])
            
            # Process and categorize papers
            processed_papers = []
            
            for paper in papers:
                # Clean and validate arXiv ID
                arxiv_id_raw = paper.get('arxiv_id', '')
                arxiv_id = validator.clean_arxiv_id(arxiv_id_raw)
                
                if not validator.validate_arxiv_id(arxiv_id):
                    self.logger.info(f"[PROCESS] Invalid arXiv ID after cleaning: {arxiv_id_raw} -> {arxiv_id}")
                    continue
                
                # Clean text using common utility (remove LaTeX from abstract)
                title_clean = cleaner.remove_latex(paper.get('title', ''))
                title_clean = cleaner.clean_text(title_clean)
                
                abstract_clean = cleaner.remove_latex(paper.get('abstract', ''))
                abstract_clean = cleaner.clean_text(abstract_clean)
                
                if not title_clean or not abstract_clean:
                    continue  # Skip invalid papers
                
                # Combine title and abstract for categorization
                combined_text = title_clean + ' ' + abstract_clean
                
                # Use CategoryManager to categorize
                categorization = CategoryManager.categorize_content(combined_text, self.ARXIV_KEYWORDS)
                
                # Format authors
                authors_list = paper.get('authors', [])
                authors_str = ', '.join(authors_list[:5])  # First 5 authors
                if len(authors_list) > 5:
                    authors_str += f' et al. ({len(authors_list)} total)'
                
                processed = {
                    'arxiv_id': arxiv_id,  # Use cleaned ID
                    'arxiv_id_with_version': arxiv_id_raw,  # Keep original for reference
                    'title': title_clean,
                    'abstract': abstract_clean,
                    'authors': authors_str,
                    'author_count': len(authors_list),
                    'published_date': paper.get('published_date'),
                    'updated_date': paper.get('updated_date'),
                    'arxiv_categories': paper.get('categories', []),
                    'primary_arxiv_category': paper.get('primary_category'),
                    'pdf_url': paper.get('pdf_url'),
                    'html_url': paper.get('html_url'),
                    'primary_category': categorization['primary_category'],
                    'all_categories': categorization['all_categories'],
                    'category_scores': categorization['category_scores'],
                    'overall_relevance': categorization['overall_relevance'],
                    'processed_at': datetime.now().isoformat()
                }
                
                processed_papers.append(processed)
            
            # Sort by relevance score
            processed_papers.sort(key=lambda x: x['overall_relevance'], reverse=True)

            self.logger.info(f"[PROCESS] Processed {len(processed_papers)} papers")

            # Print category distribution
            category_counts = CategoryManager.get_category_distribution(processed_papers)
            self.logger.info("[PROCESS] Category distribution:")
            for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  - {cat}: {count} papers")
            
            # Save processed data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            processed_file = f'/home/airflow/gcs/data/cleaned/arxiv_papers_processed_{timestamp}.json'
            
            processed_data = {
                'keywords_config': self.ARXIV_KEYWORDS,
                'total_papers': len(processed_papers),
                'category_distribution': category_counts,
                'processed_at': datetime.now().isoformat(),
                'papers': processed_papers
            }
            
            file_mgr.save_json(processed_data, processed_file)
            
            # Push to XCom
            context['task_instance'].xcom_push(key='processed_result', value={
                'filename': processed_file,
                'paper_count': len(processed_papers),
                'category_distribution': category_counts
            })
            
            print(f"[PROCESS] Saved processed data to {processed_file}")
            
            return len(processed_papers)
            
        except Exception as e:
            print(f"[PROCESS] Error: {str(e)}")
            raise
