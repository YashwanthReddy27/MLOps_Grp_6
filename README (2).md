# DVC Data Versioning Setup

> Data Version Control (DVC) configuration for managing JSON datasets with Google Cloud Storage

[![DVC](https://img.shields.io/badge/DVC-3.63.0-945DD6?style=flat-square&logo=dvc)](https://dvc.org/)
[![Python](https://img.shields.io/badge/Python-3.10.10-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![GCS](https://img.shields.io/badge/Storage-Google%20Cloud-4285F4?style=flat-square&logo=google-cloud)](https://cloud.google.com/storage)

## 📋 Overview

This repository uses DVC (Data Version Control) to track and version large JSON datasets efficiently. DVC works alongside Git to provide version control for data files while keeping the repository lightweight.

**Key Features:**
- ✅ Dual remote storage (Production + Backup)
- ✅ Google Cloud Storage integration
- ✅ JSON data versioning and tracking
- ✅ Automated data pipeline support
- ✅ Version-controlled datasets

## 🚀 Quick Start

### Prerequisites

```bash
# Install DVC
pip install dvc[gs]==3.63.0

# Verify installation
dvc version
```

### Initial Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo>

# Pull the latest data
dvc pull
```

## 📁 Project Structure

```
.
├── .dvc/
│   ├── config          # DVC configuration
│   └── .gitignore
├── data/
│   ├── raw/            # Raw JSON datasets (tracked by DVC)
│   └── processed/      # Processed JSON data (tracked by DVC)
├── .dvcignore
└── README.md
```

## 🔧 Configuration

### Remote Storage

Two GCS buckets are configured for redundancy:

| Remote | Type | Bucket | Project |
|--------|------|--------|---------|
| **mygcp** | Production | `citewise-dvc-store-8232761765` | applied-light-453519-q3 |
| **gcs** | Backup | `citewise-dvc-store-54873845389` | resilienceai-datapipeline |

### Authentication

Ensure you have Google Cloud credentials configured:

```bash
# Set up application default credentials
gcloud auth application-default login

# Or set service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

## 💻 Common Commands

### Tracking JSON Data

```bash
# Track a single JSON file
dvc add data/raw/dataset.json

# Track multiple JSON files
dvc add data/raw/*.json

# Track a directory of JSON files
dvc add data/raw

# Commit the changes
git add data/raw/dataset.json.dvc .gitignore
git commit -m "Add new JSON dataset"

# Push data to remote storage
dvc push
```

### Retrieving Data

```bash
# Pull all data from default remote
dvc pull

# Pull from backup remote
dvc pull -r gcs

# Pull specific file
dvc pull data/raw/dataset.json.dvc
```

### Checking Status

```bash
# Check DVC status
dvc status

# List all remotes
dvc remote list

# Verify setup
dvc doctor

# List tracked JSON files
dvc list . data/raw --dvc-only
```

## 🔄 Workflow

### Adding New JSON Data

1. Add the JSON file to your project
2. Track it with DVC: `dvc add data/raw/new_data.json`
3. Commit the `.dvc` file: `git add data/raw/new_data.json.dvc`
4. Push data to remote: `dvc push`
5. Push metadata to Git: `git push`

### Collaborating

1. Pull latest code: `git pull`
2. Pull latest data: `dvc pull`
3. Make your changes
4. Track and push updates following the workflow above

### Working with Large JSON Files

```bash
# Verify JSON integrity before tracking
python -m json.tool data/raw/large_file.json > /dev/null

# Track the file
dvc add data/raw/large_file.json

# Monitor storage usage
du -sh .dvc/cache
```

## 🛠️ Maintenance

### Cache Management

```bash
# Clean unused cache files
dvc gc

# Check cache status
dvc cache dir

# Force cleanup and checkout
dvc gc -f
dvc checkout
```

### JSON Data Validation

```bash
# Validate JSON syntax (Python)
python -c "import json; json.load(open('data/raw/file.json'))"

# Validate JSON syntax (jq)
jq empty data/raw/file.json
```

### Troubleshooting

**Authentication Issues:**
```bash
# Re-authenticate with GCP
gcloud auth application-default login
```

**Storage Issues:**
```bash
# Check remote connectivity
dvc remote list
dvc status -r mygcp
```

**Cache Issues:**
```bash
# Rebuild cache
dvc gc -f
dvc checkout
```

**Large JSON Files:**
```bash
# Check file size before tracking
ls -lh data/raw/*.json

# Monitor cache growth
du -sh .dvc/cache
```

## 📚 Best Practices

### General Best Practices
- ✅ **Always** commit `.dvc` files to Git
- ❌ **Never** commit raw JSON data files to Git
- ✅ Push to both production and backup remotes regularly
- ✅ Use meaningful commit messages
- ✅ Document dataset changes in commit messages
- ✅ Keep `.dvcignore` updated

### JSON-Specific Best Practices
- ✅ Validate JSON syntax before tracking with DVC
- ✅ Use consistent naming conventions for JSON files
- ✅ Consider compressing large JSON files
- ✅ Store raw JSON in `data/raw/` and processed in `data/processed/`
- ✅ Document JSON schema changes in commit messages
- ✅ Use `jq` or Python's json module for JSON manipulation
- ✅ Monitor JSON file sizes and cache growth

## 🔐 Security

- Service accounts require **Storage Object Admin** role
- Keep GCS credentials secure and never commit them
- Use environment variables for sensitive configuration
- Regularly audit IAM permissions
- Validate JSON data before pushing to avoid data leaks

## 📖 Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [DVC with Google Cloud Storage](https://dvc.org/doc/user-guide/data-management/remote-storage/google-cloud-storage)
- [Full Setup Documentation](./docs/DVC_Setup_and_Configuration.md)
- [JSON Best Practices](https://www.json.org/json-en.html)
- [jq Manual](https://stedolan.github.io/jq/manual/)

## 🤝 Contributing

When contributing JSON data or changes:

1. Ensure DVC is properly installed
2. Validate JSON syntax before adding
3. Pull latest data before making changes
4. Test your changes locally
5. Update documentation if adding new datasets
6. Follow the standard workflow for commits

## 📝 Support

For issues or questions:
- Check the [troubleshooting section](#-maintenance)
- Review the [full documentation](./docs/DVC_Setup_and_Configuration.md)
- Validate JSON files before reporting issues
- Contact the data engineering team

## 📄 License

[Your License Here]

---

**Last Updated:** October 2025  
**DVC Version:** 3.63.0  
**Python Version:** 3.10.10  
**Data Format:** JSON