"""
Example script demonstrating GCP monitoring setup and usage.

This replaces the older `monioring_check.py` and uses the unified
`monitoring.py` module.
"""

import argparse
import json
import os
import time
from pathlib import Path
from monitoring import add_monitoring_to_pipeline  # type: ignore

# Import your actual pipeline class
from pipeline import TechTrendsRAGPipeline


def build_pipeline() -> TechTrendsRAGPipeline:
    """
    Construct a TechTrendsRAGPipeline instance.
    Adapt this function if your pipeline requires extra arguments.
    """
    pipeline = TechTrendsRAGPipeline(enable_monitoring=False)
    return pipeline


def run_demo(project_id: str | None = None) -> None:
    print("=== Monitoring demo ===")

    pipeline = build_pipeline()

    monitor = add_monitoring_to_pipeline(
        pipeline,
        project_id=project_id,
        model_name="techtrends-rag",
        enable_gcp=True,
    )

    status = monitor.get_status()
    print("Monitoring status:", json.dumps(status, indent=2))

    if not status["gcp_monitor"]["available"]:
        print("✗ GCP monitoring not available, exiting.")
        return

    # Run a few demo queries to generate metrics
    queries = [
        "What are the latest developments in large language models?",
        "How is reinforcement learning used in robotics?",
        "What are recent advances in computer vision?",
    ]

    for q in queries:
        print(f"\nRunning query: {q}")
        start = time.time()
        result = pipeline.query(q)
        elapsed = time.time() - start
        print(f"Query completed in {elapsed:.2f}s")
        # The monitoring wrapper will log automatically

    # After running some queries, set Evidently baseline
    print("\nSetting Evidently baseline from collected features...")
    monitor.set_baseline()

    # Check health
    print("\nFetching monitoring health...")
    health = monitor.check_health()
    print(json.dumps(health, indent=2))

    # Check retraining decision
    print("\nChecking retraining recommendation...")
    decision = monitor.should_trigger_retraining()
    print(json.dumps(decision, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitoring demo")
    parser.add_argument(
        "--project-id",
        help="GCP project id (falls back to env GCP_PROJECT_ID)",
        default=None,
    )
    args = parser.parse_args()

    project_id = args.project_id or os.getenv("GCP_PROJECT_ID")
    if not project_id:
        print("ERROR: GCP project id not provided (use --project-id or GCP_PROJECT_ID env).")
        raise SystemExit(1)

    run_demo(project_id=project_id)


if __name__ == "__main__":
    main()