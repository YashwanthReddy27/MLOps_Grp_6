version: '3.8'

services:
  # Backend API Service
  backend:
    build:
      context: ./model
      dockerfile: Dockerfile
    container_name: rag-backend
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - GCP_PROJECT_ID=${GCP_PROJECT_ID}
      - MLFLOW_TRACKING_URI=./mlruns
      - LOG_LEVEL=INFO
    volumes:
      # Mount code for development
      - ./model:/app
      # Persist indexes
      - ./faiss_indexes:/app/faiss_indexes
      - ./bm25_indexes:/app/bm25_indexes
      # Persist MLflow runs
      - ./mlruns:/app/mlruns
      # Persist logs
      - ./logs:/app/logs
    command: python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    networks:
      - rag-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Frontend Streamlit Service
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: rag-frontend
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://backend:8000
    volumes:
      # Mount code for development
      - ./frontend:/app
    command: streamlit run app.py --server.port=8501 --server.address=0.0.0.0
    depends_on:
      - backend
    networks:
      - rag-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

networks:
  rag-network:
    driver: bridge

volumes:
  faiss_indexes:
  bm25_indexes:
  mlruns:
  logs: