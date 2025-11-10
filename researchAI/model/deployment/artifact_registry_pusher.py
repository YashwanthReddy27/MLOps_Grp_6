"""
Push RAG model artifacts to Google Cloud Artifact Registry
"""
import os
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
        
        # Verify gcloud is available
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
        
        # Set gcloud project
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
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")
        
        # Clean version string (Artifact Registry requirements)
        version = self._clean_version(version)
        
        self.logger.info(f"Pushing model version: {version}")
        
        # Create tarball
        tarball_path = self._create_tarball(version, metrics, bias_report)
        
        # Upload to Artifact Registry
        artifact_path = self._upload_to_registry(
            tarball_path=tarball_path,
            version=version,
            description=description
        )
        
        # Cleanup
        if Path(tarball_path).exists():
            Path(tarball_path).unlink()
        
        self.logger.info(f"✓ Model pushed to: {artifact_path}")
        return artifact_path
    
    def _clean_version(self, version: str) -> str:
        """
        Clean version string to meet Artifact Registry requirements
        Version must be lowercase alphanumeric with hyphens/underscores
        """
        # Replace dots and other chars with hyphens
        cleaned = version.lower()
        cleaned = cleaned.replace('.', '-')
        cleaned = cleaned.replace(':', '-')
        cleaned = cleaned.replace(' ', '-')
        
        # Remove any other special characters
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
        
        # Create temp directory for tarball
        temp_dir = Path("./artifact_packages")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        tarball_path = temp_dir / f"rag-model-{version}.tar.gz"
        
        with tarfile.open(tarball_path, "w:gz") as tar:
            # Track what was added
            added_items = []
            
            # 1. Add FAISS indexes
            faiss_dir = Path("./faiss_indexes")
            if faiss_dir.exists():
                tar.add(faiss_dir, arcname="faiss_indexes")
                added_items.append("faiss_indexes")
                self.logger.info("  Added FAISS indexes")
            else:
                self.logger.warning("  FAISS indexes not found")
            
            # 2. Add BM25 indexes
            bm25_dir = Path("./bm25_indexes")
            if bm25_dir.exists():
                tar.add(bm25_dir, arcname="bm25_indexes")
                added_items.append("bm25_indexes")
                self.logger.info("  Added BM25 indexes")
            else:
                self.logger.warning("  BM25 indexes not found")
            
            # 3. Add MLflow runs
            mlruns_dir = Path("./mlruns")
            if mlruns_dir.exists():
                tar.add(mlruns_dir, arcname="mlruns")
                added_items.append("mlruns")
                self.logger.info("  Added MLflow runs")
            else:
                self.logger.warning("  MLflow runs not found")
            
            # 4. Add settings.py
            settings_file = Path("config/settings.py")
            if settings_file.exists():
                tar.add(settings_file, arcname="config/settings.py")
                added_items.append("config")
                self.logger.info("  Added settings.py")
            else:
                self.logger.warning("  settings.py not found")
            
            # 5. Create and add metadata
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
            
            # Save metadata to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(metadata, f, indent=2)
                metadata_path = f.name
            
            tar.add(metadata_path, arcname="metadata.json")
            added_items.append("metadata")
            self.logger.info("  Added metadata.json")
            
            # Cleanup temp metadata file
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
        
        # Construct artifact path
        artifact_path = (
            f"{self.location}-generic.pkg.dev/"
            f"{self.project_id}/{self.repository}/{package_name}:{version}"
        )
        
        # Build gcloud command
        cmd = [
            "gcloud", "artifacts", "generic", "upload",
            "--project", self.project_id,
            "--location", self.location,
            "--repository", self.repository,
            "--package", package_name,
            "--version", version,
            "--source", tarball_path
        ]
        
        # # Add description if provided
        # if description:
        #     cmd.extend(["--description", description])
        
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
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all versions in Artifact Registry
        
        Returns:
            List of version information
        """
        try:
            cmd = [
                "gcloud", "artifacts", "versions", "list",
                "--project", self.project_id,
                "--location", self.location,
                "--repository", self.repository,
                "--package", "rag-model",
                "--format", "json"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            versions = json.loads(result.stdout)
            
            # Parse and format version info
            formatted_versions = []
            for v in versions:
                formatted_versions.append({
                    "name": v.get("name", "").split("/")[-1],
                    "create_time": v.get("createTime", ""),
                    "update_time": v.get("updateTime", ""),
                    "description": v.get("description", "")
                })
            
            return sorted(
                formatted_versions,
                key=lambda x: x["create_time"],
                reverse=True
            )
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to list versions: {e.stderr}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse version list: {e}")
            return []
    
    def download(self, version: str, output_dir: str = "./downloaded_model"):
        """
        Download a specific version from Artifact Registry
        
        Args:
            version: Version to download
            output_dir: Local directory to save artifacts
        """
        # Clean version
        version = self._clean_version(version)
        
        self.logger.info(f"Downloading version: {version}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download tarball
        tarball_name = f"rag-model-{version}.tar.gz"
        tarball_path = output_path / tarball_name
        
        cmd = [
            "gcloud", "artifacts", "generic", "download",
            "--project", self.project_id,
            "--location", self.location,
            "--repository", self.repository,
            "--package", "rag-model",
            "--version", version,
            "--destination", str(tarball_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"Downloaded tarball to: {tarball_path}")
            
            # Extract tarball
            with tarfile.open(tarball_path, "r:gz") as tar:
                tar.extractall(output_path)
            
            self.logger.info(f"✓ Extracted to: {output_path}")
            
            # Cleanup tarball
            tarball_path.unlink()
            
            # List extracted contents
            extracted_items = [item.name for item in output_path.iterdir()]
            self.logger.info(f"Extracted items: {extracted_items}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Download failed: {e.stderr}")
            raise RuntimeError(f"Failed to download from Artifact Registry: {e.stderr}")
    
    def delete_version(self, version: str):
        """
        Delete a specific version from Artifact Registry
        
        Args:
            version: Version to delete
        """
        # Clean version
        version = self._clean_version(version)
        
        self.logger.warning(f"Deleting version: {version}")
        
        cmd = [
            "gcloud", "artifacts", "versions", "delete",
            version,
            "--project", self.project_id,
            "--location", self.location,
            "--repository", self.repository,
            "--package", "rag-model",
            "--quiet"  # Skip confirmation
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f" Deleted version: {version}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Delete failed: {e.stderr}")
            raise RuntimeError(f"Failed to delete version: {e.stderr}")
    
    def get_version_info(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific version
        
        Args:
            version: Version to query
            
        Returns:
            Version metadata or None
        """
        # Clean version
        version = self._clean_version(version)
        
        try:
            # Download just to read metadata
            with tempfile.TemporaryDirectory() as tmpdir:
                tarball_path = Path(tmpdir) / f"rag-model-{version}.tar.gz"
                
                cmd = [
                    "gcloud", "artifacts", "generic", "download",
                    "--project", self.project_id,
                    "--location", self.location,
                    "--repository", self.repository,
                    "--package", "rag-model",
                    "--version", version,
                    "--destination", str(tarball_path)
                ]
                
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Extract and read metadata
                with tarfile.open(tarball_path, "r:gz") as tar:
                    try:
                        metadata_member = tar.getmember("metadata.json")
                        metadata_file = tar.extractfile(metadata_member)
                        if metadata_file:
                            metadata = json.load(metadata_file)
                            return metadata
                    except KeyError:
                        self.logger.warning("metadata.json not found in tarball")
                        return None
            
        except Exception as e:
            self.logger.error(f"Failed to get version info: {e}")
            return None