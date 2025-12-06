terraform import google_composer_environment.example_environment "projects/mlops-gcp-lab1/locations/us-east1/environments/rag-environment"
terraform import google_service_account.custom_service_account "projects/mlops-gcp-lab1/serviceAccounts/terraform-sa@mlops-gcp-lab1.iam.gserviceaccount.com"
terraform import google_storage_bucket.composer_logs_bucket "projects/mlops-gcp-lab1/locations/us-east1/buckets/composer-logs-bucket-mlops-gcp-lab1"
terraform import google_logging_project_sink.composer_logs_sink "projects/mlops-gcp-lab1/locations/us-east1/sinks/composer-logs-sink"