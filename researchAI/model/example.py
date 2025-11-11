"""
Example usage of the RAG pipeline with Fairness Detection and Artifact Registry
"""
import json
import os
import argparse
from pipeline import TechTrendsRAGPipeline

def print_separator(title=""):
    """Print a nice separator"""
    print("\n" + "="*80)
    if title:
        print(f"{title}")
        print("="*80)

# def print_fairness_report(bias_report):
#     """Print detailed fairness report"""
#     print("\n" + "-"*80)
#     print("FAIRNESS DETECTION REPORT")
#     print("-"*80)
    
#     # Check if this is the new fairness detector format or old bias detector format
#     if 'overall_fairness_score' in bias_report:
#         # New fairness detector format
#         fairness_score = bias_report['overall_fairness_score']
#         print(f"Overall Fairness Score: {fairness_score:.3f}/1.000")
        
#         # Fairness assessment
#         if fairness_score >= 0.8:
#             print("Status: ✅ EXCELLENT - Strong fairness characteristics")
#         elif fairness_score >= 0.6:
#             print("Status: ⚠️  MODERATE - Some bias detected, monitoring recommended")
#         elif fairness_score >= 0.4:
#             print("Status: 🟡 CONCERNING - Significant bias, action needed")
#         else:
#             print("Status: 🔴 CRITICAL - Severe bias, immediate intervention required")
        
#         # Diversity metrics if available
#         diversity_metrics = bias_report.get('diversity_metrics', {})
#         if diversity_metrics:
#             print(f"\nDiversity Metrics:")
#             print(f"  Source Diversity: {diversity_metrics.get('source_diversity', 0.0):.3f}")
#             print(f"  Category Diversity: {diversity_metrics.get('category_diversity', 0.0):.3f}")
#             print(f"  Unique Sources: {diversity_metrics.get('num_unique_sources', 0)}")
#             print(f"  Unique Categories: {diversity_metrics.get('num_unique_categories', 0)}")
    
#     elif 'overall_bias_score' in bias_report:
#         # Legacy bias detector format
#         bias_score = bias_report['overall_bias_score']
#         print(f"Overall Bias Score: {bias_score:.3f}")
#         print(f"Needs Diversification: {bias_report.get('needs_diversification', False)}")
        
#         # Source bias
#         if 'source_bias' in bias_report:
#             source_bias = bias_report['source_bias']
#             print(f"\nSource Diversity: {source_bias.get('source_diversity', 0.0):.3f}")
#             print(f"Source Bias Detected: {source_bias.get('is_biased', False)}")

#     print("-"*80)

def print_fairness_report(bias_report):
    """Print detailed fairness report with Fairlearn metrics"""
    print("\n" + "-"*80)
    print("FAIRNESS DETECTION REPORT (with Fairlearn)")
    print("-"*80)
    
    eval_type = bias_report.get('evaluation_type', 'batch')
    
    if eval_type == 'single_query_fairlearn':
        # Fairlearn-enhanced single-query report
        fairness_score = bias_report['overall_fairness_score']
        print(f"Overall Fairness Score: {fairness_score:.3f}/1.000")
        
        # Fairness assessment
        if fairness_score >= 0.8:
            print("Status: ✅ EXCELLENT - Strong fairness characteristics")
        elif fairness_score >= 0.6:
            print("Status: ⚠️  MODERATE - Some concerns, monitoring recommended")
        elif fairness_score >= 0.4:
            print("Status: 🟡 CONCERNING - Attention needed")
        else:
            print("Status: 🔴 CRITICAL - Immediate review required")
    
    print("-"*80)

def run_queries(pipeline, queries):
    """Run queries and display results"""
    for query in queries:
        print(f"Query: {query}\n")
        
        result = pipeline.query(query=query)
        
        # Display response
        print(f"Response:\n{result['response']}\n")
        
        # Display sources
        print(f"Sources ({len(result['sources'])}):")
        for source in result['sources']:  # Show all sources
            print(f"  [{source['number']}] {source.get('title', 'Untitled')}")
            print(f"      {source.get('source', 'Unknown')} - {str(source.get('date',''))[:10]}")
            if source.get('url'):
                print(f"      {source['url']}")
        
        # Display metrics
        print(f"\nPerformance Metrics:")
        print(f"  Response Time: {result['response_time']:.2f}s")
        print(f"  Validation Score: {result['validation']['overall_score']:.2f}")
        print(f"  Number of Sources: {result['num_sources']}")
        
        # Display fairness report
        print_fairness_report(result['bias_report'])

def main():
    parser = argparse.ArgumentParser(description='RAG Pipeline Example with Artifact Registry')
    
    # Data loading options
    parser.add_argument('--papers', type=str, help='Path to papers JSON file')
    parser.add_argument('--news', type=str, help='Path to news JSON file')
    parser.add_argument('--load-existing', action='store_true', 
                       help='Load existing indexes instead of building new ones')
    
    # Artifact Registry options
    parser.add_argument('--push-to-registry', action='store_true',
                       help='Push model to Artifact Registry after queries')
    parser.add_argument('--project-id', type=str,
                       help='GCP project ID (default: from config/env)')
    parser.add_argument('--location', type=str, default='us-central1',
                       help='GCP location (default: us-central1)')
    parser.add_argument('--repository', type=str, default='rag-models',
                       help='Artifact Registry repository (default: rag-models)')
    parser.add_argument('--version', type=str,
                       help='Model version (default: auto-generated)')
    parser.add_argument('--description', type=str,
                       help='Version description')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print_separator("INITIALIZING RAG PIPELINE")
    print("Setting up Tech Trends RAG Pipeline with Fairness Detection...")
    
    pipeline = TechTrendsRAGPipeline(
        enable_tracking=True
    )
    
    # Load or build indexes
    if args.load_existing:
        print("\nLoading pre-built indexes...")
        if pipeline.load_indexes():
            print("SUCCESS: Indexes loaded successfully!!")
        else:
            print("No existing indexes found. Please build indexes first.")
            return
    else:
        print("\nBuilding new indexes...")
        
        # Get data paths
        papers_path = args.papers or 'path'
        news_path = args.news or 'path'
        
        # Check if paths are placeholders
        if papers_path == 'datafile_path_here' or news_path == 'datafile_path_here':
            print("  Please provide data file paths using --papers and --news arguments")
            print("  Example: python example.py --papers data/papers.json --news data/news.json")
            return
        
        # Load data
        try:
            with open(papers_path, 'r') as f:
                papers_data = json.load(f)
                papers = papers_data.get('papers', [])
            print(f"Loaded {len(papers)} papers")
        except FileNotFoundError:
            print(f"Error: Papers file not found: {papers_path}")
            return
        
        try:
            with open(news_path, 'r') as f:
                news_data = json.load(f)
                news = news_data.get('articles', [])
            print(f"Loaded {len(news)} news articles")
        except FileNotFoundError:
            print(f"Error: News file not found: {news_path}")
            return
        
        # Build indexes (using subset for demo if too large)
        max_docs = 100
        if len(papers) > max_docs or len(news) > max_docs:
            print(f"\nNote: Using first {max_docs} documents from each source for demo")
            papers = papers[:max_docs]
            news = news[:max_docs]
        
        pipeline.index_documents(papers=papers, news=news)
        print("SUCCESS: Indexes built successfully!!")
    
    # Run example queries
    print_separator("RUNNING EXAMPLE QUERIES")
    
    queries = [
        "What are the latest developments in reinforcement learning for large language models?",
        "How is retrieval augmented generation being used in modern AI systems?"
    ]
    
    run_queries(pipeline, queries)
    
    # Push to Artifact Registry if requested
    if args.push_to_registry:
        print_separator("PUSHING TO ARTIFACT REGISTRY")
        
        # Get project ID
        project_id = args.project_id or os.getenv('GCP_PROJECT_ID')
        if not project_id:
            print("Error: GCP project ID required. Use --project-id or set GCP_PROJECT_ID env var")
            return
        
        # Get a sample query result for metrics
        print("Running validation query to collect metrics...")
        test_result = pipeline.query("What are the latest trends in AI?")
        
        validation_score = test_result['validation']['overall_score']
        fairness_score = test_result['bias_report'].get('overall_fairness_score', 0.0)
        
        print(f"Validation Score: {validation_score:.2f}")
        print(f"Fairness Score: {fairness_score:.3f}")
        
        if validation_score:
            try:
                print("\nPushing to Artifact Registry...")
                
                artifact_path = pipeline.push_to_artifact_registry(
                    project_id=project_id,
                    location=args.location,
                    repository=args.repository,
                    version=args.version,
                    metrics=test_result['metrics'],
                    bias_report=test_result['bias_report'],
                    description=args.description or f"RAG model with validation score {validation_score:.2f}"
                )
                
                print_separator("PUSH SUCCESSFUL")
                print(f"Artifact Path: {artifact_path}")
                print(f"Version: {args.version or 'auto-generated'}")
                
            except Exception as e:
                print_separator("PUSH FAILED")
                print(f"Error: {e}")
        else:
            print(f"\n Validation score too low ({validation_score:.2f}). Not pushing to registry.")
            print("   Improve model or use a lower threshold.")
    
    print_separator("EXAMPLE COMPLETED")

if __name__ == "__main__":
    main()