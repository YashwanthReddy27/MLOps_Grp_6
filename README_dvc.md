# DVC Setup and Configuration Documentation

## 1. Initial Setup

Prerequisites Installed:

- DVC: 3.63.0
- Python: 3.10.10
- Remote Storage: Google Cloud Storage (GCS)

## 2. Configuration

### Remote Storage Setup

Two Google Cloud Storage remotes are configured for production and backup.

#### Primary Remote (Production)

- Name: mygcp
- URL: gs://citewise-dvc-store-8232761765
- Project: applied-light-453519-q3
- Usage: Main remote storage for production data

#### Secondary Remote (Backup)

- Name: gcs
- URL: gs://citewise-dvc-store-54873845389/resilienceai-datapipeline
- Project: resilienceai-datapipeline
- Usage: Backup storage for critical datasets

### Configuration File

Location: `.dvc/config`

Configuration:

```
[core]
    # Primary remote for production data
    remote = mygcp

# Secondary remote (backup)
['remote "gcs"']
    url = gs://citewise-dvc-store-54873845389/resilienceai-datapipeline
    projectname = resilienceai-datapipeline  # Backup project

# Primary remote (production)
['remote "mygcp"']
    url = gs://citewise-dvc-store-8232761765
    projectname = applied-light-453519-q3  # Production project
```

## 3. Basic DVC Commands

### Initialize DVC

```bash
dvc init
```

### Add Remote Storage

```bash
# Primary remote
dvc remote add -d mygcp gs://citewise-dvc-store-8232761765

# Secondary remote
dvc remote add gcs gs://citewise-dvc-store-54873845389/resilienceai-datapipeline
```

### Track Data

```bash
# Track a directory
dvc add data/raw

# Track a single file
dvc add data/raw/dataset.json
```

### Push/Pull Data

```bash
# Push data to the default remote
dvc push

# Pull from specific remote (backup)
dvc pull -r gcs
```

### Version Control

```bash
# Track DVC metadata with Git
git add data/raw.dvc .dvc/config
git commit -m "Add initial data version"
```

## 4. Best Practices

### Git Integration

- Always commit .dvc and .dvc/config files to Git.
- Never commit raw or large data files to Git.
- Use .dvcignore to exclude large or temporary files.

### Remote Management

- Regularly push data to both primary (production) and secondary (backup) remotes.
- Monitor remote storage usage in the GCS console.
- Ensure service accounts have Storage Object Admin or Editor permissions.

### Cache Management

- Cache Type: hardlink (efficient for NTFS file systems)
- Cache Location: C:\.dvc\cache

## 5. Troubleshooting

### Common Issues and Solutions

#### 1. Authentication Errors

- Ensure proper Google Cloud credentials are configured.
- Verify IAM roles for the active service account.

#### 2. Push/Pull Failures

- Check network connectivity and GCS bucket permissions.
- Ensure enough local disk space is available.

#### 3. Cache Issues

```bash
# Clean and rebuild cache
dvc gc -f
dvc checkout
```

## 6. Maintenance

### Regular Maintenance Commands

```bash
# Check DVC status
dvc status

# Clean unused cache files
dvc gc

# List all configured remotes
dvc remote list

# Check DVC version and environment info
dvc doctor
```

## 7. Recommendations for Future Enhancements

To further improve automation, reproducibility, and continuous data versioning:

### Automate Versioning with Pipelines:

- Implement dvc.yaml stages for fetching, cleaning, and processing data.
- Use Airflow or Cron jobs to automate daily DVC version commits (dvc add → dvc commit → dvc push).

### Integrate Continuous Versioning:

- Automatically track new data as it arrives using event triggers or scheduled tasks.
- Combine DVC with CI/CD tools (GitHub Actions, GitLab CI, Jenkins) for automated reproducibility checks.

### Enhance Monitoring and Logging:

- Track DVC push/pull operations in Cloud Logging or Datadog.
- Maintain a data lineage report for transparency in versioned datasets.

### Data Validation and Quality Control:

- Integrate data validation stages in the DVC pipeline to ensure dataset consistency before pushing to remote storage.