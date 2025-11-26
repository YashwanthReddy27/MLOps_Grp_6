import json
import logging
import subprocess
import tarfile
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from config.settings import config
import shutil

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


        # Cache the current latest version
        self.current_version = self._fetch_latest_version()
        self.logger.info(f"Current latest version in registry: {self.current_version}")
                
        self.logger.info(
            f"Initialized Artifact Registry pusher: "
            f"{location}-generic.pkg.dev/{project_id}/{repository}"
        )

    def _fetch_latest_version(self) -> Optional[str]:
        """
        Fetch the current latest version from registry (cached in __init__)
        
        Returns:
            Latest version string or None if no versions exist
        """
        try:
            versions = self.list_versions(limit=1)
            
            if not versions:
                self.logger.info("No existing versions in registry")
                return None
            
            latest = versions[0]['version']
            self.logger.debug(f"Fetched latest version: {latest}")
            return latest
            
        except Exception as e:
            self.logger.warning(f"Could not fetch latest version: {e}")
            return None

    def _get_next_version(self) -> str:
        """
        Get the next semantic version based on cached current version
        Increments: 1.0 → 1.1 → ... → 1.9 → 2.0
        
        Returns:
            Next version string (e.g., "1.1", "1.2", "2.0")
        """
        if self.current_version is None:
            self.logger.info("No existing versions. Starting with 1.0")
            return "1.0"
        
        import re
        match = re.search(r'(\d+)[.-](\d+)', self.current_version)
        
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            
            # Increment logic: after 1.9 → 2.0
            if minor >= 9:
                new_version = f"{major + 1}.0"
            else:
                new_version = f"{major}.{minor + 1}"
            
            self.logger.info(f"Next version: {new_version} (current: {self.current_version})")
            return new_version
        else:
            from datetime import datetime
            fallback = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.logger.warning(f"Could not parse version format. Using timestamp: {fallback}")
            return fallback

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
            version = self._get_next_version()
            # version = datetime.now().strftime("v%Y%m%d-%H%M%S")
        
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
        
        self.current_version = version
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

    def pull_latest(self, destination_dir: str = "./model") -> Dict[str, Any]:
        """
        Pull the latest artifact version from Artifact Registry
        
        Args:
            destination_dir: Directory to extract artifacts to (default: ./model)
            
        Returns:
            Dictionary with metadata about pulled artifact
        """
        self.logger.info("Pulling latest artifact from Artifact Registry...")
        
        try:
            # Use cached version from __init__
            if self.current_version is None:
                self.logger.error("No versions found in Artifact Registry")
                raise RuntimeError("No artifacts found in registry")

            latest_version = self.current_version
            self.logger.info(f"Using cached latest version: {latest_version}")

            package_name = "rag-model"
            
            # Download the artifact
            temp_dir = Path(tempfile.mkdtemp())
            download_path = temp_dir / f"rag-model-{latest_version}.tar.gz"
            
            download_cmd = [
                "gcloud", "artifacts", "generic", "download",
                "--project", self.project_id,
                "--location", self.location,
                "--repository", self.repository,
                "--package", package_name,
                "--version", latest_version,
                "--destination", str(temp_dir)
            ]
            
            self.logger.info("Downloading artifact...")
            subprocess.run(
                download_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Find the downloaded file (gcloud may create a subdirectory)
            downloaded_files = list(temp_dir.rglob("*.tar.gz"))
            if not downloaded_files:
                raise RuntimeError("Downloaded artifact not found")
            
            tarball_path = downloaded_files[0]
            self.logger.info(f"Artifact downloaded to: {tarball_path}")
            
            # Extract the artifact
            metadata = self._extract_artifact(tarball_path, destination_dir)
            
            # Cleanup temp directory
            shutil.rmtree(temp_dir)
            
            self.logger.info(f"✅ Latest artifact pulled and extracted to: {destination_dir}")
            
            return {
                'version': latest_version,
                'destination': destination_dir,
                'metadata': metadata,
                'success': True
            }
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to pull artifact: {e.stderr}")
            raise RuntimeError(f"Failed to pull from Artifact Registry: {e.stderr}")
        except Exception as e:
            self.logger.error(f"Error pulling artifact: {e}")
            raise
    
    def pull_specific_version(
        self, 
        version: str, 
        destination_dir: str = "./model"
    ) -> Dict[str, Any]:
        """
        Pull a specific artifact version from Artifact Registry
        
        Args:
            version: Specific version to pull
            destination_dir: Directory to extract artifacts to
            
        Returns:
            Dictionary with metadata about pulled artifact
        """
        self.logger.info(f"Pulling artifact version {version} from Artifact Registry...")
        
        try:
            package_name = "rag-model"
            version = self._clean_version(version)
            
            # Download the artifact
            temp_dir = Path(tempfile.mkdtemp())
            
            download_cmd = [
                "gcloud", "artifacts", "generic", "download",
                "--project", self.project_id,
                "--location", self.location,
                "--repository", self.repository,
                "--package", package_name,
                "--version", version,
                "--destination", str(temp_dir)
            ]
            
            self.logger.info(f"Downloading artifact version {version}...")
            subprocess.run(
                download_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Find the downloaded file
            downloaded_files = list(temp_dir.rglob("*.tar.gz"))
            if not downloaded_files:
                raise RuntimeError("Downloaded artifact not found")
            
            tarball_path = downloaded_files[0]
            self.logger.info(f"Artifact downloaded to: {tarball_path}")
            
            # Extract the artifact
            metadata = self._extract_artifact(tarball_path, destination_dir)
            
            # Cleanup temp directory
            shutil.rmtree(temp_dir)
            
            self.logger.info(f"✅ Artifact version {version} pulled and extracted to: {destination_dir}")
            
            return {
                'version': version,
                'destination': destination_dir,
                'metadata': metadata,
                'success': True
            }
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to pull artifact: {e.stderr}")
            raise RuntimeError(f"Failed to pull from Artifact Registry: {e.stderr}")
        except Exception as e:
            self.logger.error(f"Error pulling artifact: {e}")
            raise
    
    def _extract_artifact(
        self, 
        tarball_path: Path, 
        destination_dir: str
    ) -> Dict[str, Any]:
        """
        Extract artifact tarball to destination directory
        
        Args:
            tarball_path: Path to the tarball file
            destination_dir: Directory to extract to
            
        Returns:
            Metadata from the artifact
        """
        self.logger.info(f"Extracting artifact to {destination_dir}...")
        
        dest_path = Path(destination_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        metadata = None
        
        with tarfile.open(tarball_path, "r:gz") as tar:
            # Extract metadata first
            try:
                metadata_member = tar.getmember("metadata.json")
                metadata_file = tar.extractfile(metadata_member)
                metadata = json.load(metadata_file)
                self.logger.info(f"Loaded metadata: version={metadata.get('version')}")
            except KeyError:
                self.logger.warning("No metadata.json found in artifact")
            
            # Extract FAISS indexes
            faiss_members = [m for m in tar.getmembers() if m.name.startswith("faiss_indexes/")]
            if faiss_members:
                faiss_dest = dest_path / "faiss_indexes"
                faiss_dest.mkdir(parents=True, exist_ok=True)
                
                for member in faiss_members:
                    # Remove the faiss_indexes/ prefix from the path
                    member.name = member.name.replace("faiss_indexes/", "")
                    if member.name:  # Skip if name becomes empty
                        tar.extract(member, path=faiss_dest)
                
                self.logger.info(f"✅ Extracted FAISS indexes to {faiss_dest}")
            else:
                self.logger.warning("No FAISS indexes found in artifact")
            
            # Extract BM25 indexes
            bm25_members = [m for m in tar.getmembers() if m.name.startswith("bm25_indexes/")]
            if bm25_members:
                bm25_dest = dest_path / "bm25_indexes"
                bm25_dest.mkdir(parents=True, exist_ok=True)
                
                for member in bm25_members:
                    # Remove the bm25_indexes/ prefix from the path
                    member.name = member.name.replace("bm25_indexes/", "")
                    if member.name:  # Skip if name becomes empty
                        tar.extract(member, path=bm25_dest)
                
                self.logger.info(f"✅ Extracted BM25 indexes to {bm25_dest}")
            else:
                self.logger.warning("No BM25 indexes found in artifact")
            
            # Extract MLflow runs (optional)
            mlruns_members = [m for m in tar.getmembers() if m.name.startswith("mlruns/")]
            if mlruns_members:
                mlruns_dest = dest_path / "mlruns"
                mlruns_dest.mkdir(parents=True, exist_ok=True)
                
                for member in mlruns_members:
                    member.name = member.name.replace("mlruns/", "")
                    if member.name:
                        tar.extract(member, path=mlruns_dest)
                
                self.logger.info(f"✅ Extracted MLflow runs to {mlruns_dest}")
            
            # Extract config (optional)
            config_members = [m for m in tar.getmembers() if m.name.startswith("config/")]
            if config_members:
                config_dest = dest_path / "config"
                config_dest.mkdir(parents=True, exist_ok=True)
                
                for member in config_members:
                    member.name = member.name.replace("config/", "")
                    if member.name:
                        tar.extract(member, path=config_dest)
                
                self.logger.info(f"✅ Extracted config to {config_dest}")
        
        return metadata
    
    def list_versions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List available versions in Artifact Registry
        
        Args:
            limit: Maximum number of versions to return
            
        Returns:
            List of version information dictionaries
        """
        self.logger.info("Listing artifact versions...")
        
        try:
            package_name = "rag-model"
            list_cmd = [
                "gcloud", "artifacts", "versions", "list",
                "--project", self.project_id,
                "--location", self.location,
                "--repository", self.repository,
                "--package", package_name,
                "--format", "json",
                "--sort-by", "~createTime",
                "--limit", str(limit)
            ]
            
            result = subprocess.run(
                list_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            versions = json.loads(result.stdout)
            
            version_info = []
            for v in versions:
                version_info.append({
                    'version': v['name'].split('/')[-1],
                    'created': v.get('createTime', 'Unknown'),
                    'updated': v.get('updateTime', 'Unknown')
                })
            
            self.logger.info(f"Found {len(version_info)} versions")
            return version_info
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to list versions: {e.stderr}")
            return []