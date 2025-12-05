"""
API Client for communicating with backend
"""
import requests
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class APIClient:
    """Client for RAG Pipeline API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def _make_request(self, method: str, endpoint: str, 
                     data: Optional[Dict] = None, 
                     params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make HTTP request to API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            
        Returns:
            Response data or None on error
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, timeout=30)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, params=params, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to {url}")
            return None
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {url}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            if hasattr(e.response, 'json'):
                try:
                    error_data = e.response.json()
                    logger.error(f"Error details: {error_data}")
                except:
                    pass
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    
    def health_check(self) -> Optional[Dict]:
        """
        Check API health status
        
        Returns:
            Health status or None
        """
        return self._make_request('GET', '/api/health')
    
    def query(self, query: str, filters: Optional[Dict] = None, 
             enable_streaming: bool = False) -> Optional[Dict]:
        """
        Send query to RAG pipeline
        
        Args:
            query: User query
            filters: Optional filters
            enable_streaming: Enable streaming response
            
        Returns:
            Query response or None
        """
        data = {
            'query': query,
            'filters': filters,
            'enable_streaming': enable_streaming
        }
        
        return self._make_request('POST', '/api/query', data=data)
    
    def get_metrics(self) -> Optional[Dict]:
        """
        Get system metrics
        
        Returns:
            Metrics data or None
        """
        return self._make_request('GET', '/api/metrics')