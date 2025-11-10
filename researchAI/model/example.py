"""
Example usage of the RAG pipeline with Fairness Detection
"""
import json
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
    for query in queries:
        print(f"Query: {query}\n")
        
        result = pipeline.query(query=query)
        
        # Display response
        print(f"Response:\n{result['response']}\n")
        
        # Display sources
        print(f"Sources ({len(result['sources'])}):")
        for source in result['sources']:  # Show top 5 sources
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
    # Initialize pipeline
    print_separator("INITIALIZING RAG PIPELINE")
    print("Setting up Tech Trends RAG Pipeline with Fairness Detection...")
    
    pipeline = TechTrendsRAGPipeline(
        enable_tracking=True
    )
    
    # Load pre-built indexes or build new ones
    print("\nLoading pre-built indexes...")
    if pipeline.load_indexes():
        print("✅ Indexes loaded successfully")
    else:
        print("⚠️  No existing indexes found. Building new indexes...")
        
        # Load your data
        papers_path = 'data_file_path'
        news_path = 'data_file_path'
        
        with open(papers_path, 'r') as f:
            papers_data = json.load(f)
            papers = papers_data.get('papers', [])
        with open(news_path, 'r') as f:
            news_data = json.load(f)
            news = news_data.get('articles', [])
        
        # Build indexes (using subset for demo)
        pipeline.index_documents(papers=papers[:100], news=news[:100])
        print("✅ Indexes built successfully")
    
    run_queries(pipeline, [
        "What are the latest developments in reinforcement learning for large language models?"
        # "How is retrieval augmented generation being used in modern AI systems?"
    ])

if __name__ == "__main__":
    main()