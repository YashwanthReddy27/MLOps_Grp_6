import json
import logging
import subprocess
import tarfile
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class ArtifactRegistryPusher:
    """Push RAG artifacts to Google Cloud Artifact Registry"""
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        repository: str = "rag-models"
    ):
        """
        Initialize Artifact Registry pusher
        
        Args:
            project_id: GCP project ID
            location: GCP region (e.g., 'us-central1')
            repository: Artifact Registry repository name
        """
        self.project_id = project_id
        self.location = location
        self.repository = repository
        self.logger = logging.getLogger(__name__)
        
        try:
            result = subprocess.run(
                ['gcloud', 'version'],
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info("gcloud CLI is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("gcloud not found. Please install Google Cloud SDK")
            raise RuntimeError(
                "gcloud is required. Install from: https://cloud.google.com/sdk/docs/install"
            )
        
        subprocess.run(
            ['gcloud', 'config', 'set', 'project', project_id],
            capture_output=True,
            text=True
        )
        
        self.logger.info(
            f"Initialized Artifact Registry pusher: "
            f"{location}-generic.pkg.dev/{project_id}/{repository}"
        )
    
    def push(
        self,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        bias_report: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Push RAG model to Artifact Registry
        
        Args:
            version: Version name (default: timestamp)
            metrics: Model performance metrics
            bias_report: Bias evaluation report
            description: Optional description
            
        Returns:
            Artifact Registry path
        """
        if version is None:
            version = datetime.now().strftime("v%Y%m%d-%H%M%S")
        
        version = self._clean_version(version)
        
        self.logger.info(f"Pushing model version: {version}")
        
        tarball_path = self._create_tarball(version, metrics, bias_report)
        
        artifact_path = self._upload_to_registry(
            tarball_path=tarball_path,
            version=version,
            description=description
        )
        
        if Path(tarball_path).exists():
            Path(tarball_path).unlink()
        
        self.logger.info(f"✓ Model pushed to: {artifact_path}")
        return artifact_path
    
    def _clean_version(self, version: str) -> str:
        """
        Clean version string to meet Artifact Registry requirements
        Version must be lowercase alphanumeric with hyphens/underscores
        """
        cleaned = version.lower()
        cleaned = cleaned.replace('.', '-')
        cleaned = cleaned.replace(':', '-')
        cleaned = cleaned.replace(' ', '-')
        
        import re
        cleaned = re.sub(r'[^a-z0-9\-_]', '', cleaned)
        
        return cleaned
    
    def _create_tarball(
        self,
        version: str,
        metrics: Optional[Dict[str, Any]],
        bias_report: Optional[Dict[str, Any]]
    ) -> str:
        """
        Create a tarball with all RAG artifacts
        
        Args:
            version: Version string
            metrics: Model metrics
            bias_report: Bias report
            
        Returns:
            Path to created tarball
        """
        self.logger.info("Creating artifact package...")
        
        temp_dir = Path("./artifact_packages")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        tarball_path = temp_dir / f"rag-model-{version}.tar.gz"
        
        with tarfile.open(tarball_path, "w:gz") as tar:
            added_items = []
            
            faiss_dir = Path("./faiss_indexes")
            if faiss_dir.exists():
                tar.add(faiss_dir, arcname="faiss_indexes")
                added_items.append("faiss_indexes")
                self.logger.info("  Added FAISS indexes")
            else:
                self.logger.warning("  FAISS indexes not found")
            
            bm25_dir = Path("./bm25_indexes")
            if bm25_dir.exists():
                tar.add(bm25_dir, arcname="bm25_indexes")
                added_items.append("bm25_indexes")
                self.logger.info("  Added BM25 indexes")
            else:
                self.logger.warning("  BM25 indexes not found")
            
            mlruns_dir = Path("./mlruns")
            if mlruns_dir.exists():
                tar.add(mlruns_dir, arcname="mlruns")
                added_items.append("mlruns")
                self.logger.info("  Added MLflow runs")
            else:
                self.logger.warning("  MLflow runs not found")
            
            settings_file = Path("config/settings.py")
            if settings_file.exists():
                tar.add(settings_file, arcname="config/settings.py")
                added_items.append("config")
                self.logger.info("  Added settings.py")
            else:
                self.logger.warning("  settings.py not found")
            
            metadata = {
                "version": version,
                "created_at": datetime.now().isoformat(),
                "project_id": self.project_id,
                "location": self.location,
                "repository": self.repository,
                "metrics": metrics or {},
                "bias_report": bias_report or {},
                "components": {
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "llm_model": "gemini-2.0-flash-exp",
                    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "indexes": ["faiss", "bm25"]
                },
                "included_artifacts": added_items
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(metadata, f, indent=2)
                metadata_path = f.name
            
            tar.add(metadata_path, arcname="metadata.json")
            added_items.append("metadata")
            self.logger.info("  Added metadata.json")
            
            Path(metadata_path).unlink()
        
        file_size_mb = tarball_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"Created tarball: {tarball_path} ({file_size_mb:.2f} MB)")
        
        return str(tarball_path)
    
    def _upload_to_registry(
        self,
        tarball_path: str,
        version: str,
        description: Optional[str]
    ) -> str:
        """
        Upload tarball to Artifact Registry
        
        Args:
            tarball_path: Path to tarball
            version: Version string
            description: Optional description
            
        Returns:
            Artifact path in registry
        """
        self.logger.info("Uploading to Artifact Registry...")
        
        package_name = "rag-model"
        
        artifact_path = (
            f"{self.location}-generic.pkg.dev/"
            f"{self.project_id}/{self.repository}/{package_name}:{version}"
        )
        
        cmd = [
            "gcloud", "artifacts", "generic", "upload",
            "--project", self.project_id,
            "--location", self.location,
            "--repository", self.repository,
            "--package", package_name,
            "--version", version,
            "--source", tarball_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            self.logger.info(f"Upload successful")
            return artifact_path
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Upload failed: {e.stderr}")
            raise RuntimeError(f"Failed to upload to Artifact Registry: {e.stderr}")
