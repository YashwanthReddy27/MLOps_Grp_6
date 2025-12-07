# Data Pipeline Deployment Guide

## Overview

This document outlines the deployment architecture, infrastructure setup, and GitHub Actions automation for the **researchAI** data pipeline on **Google Cloud Platform (GCP)**.

---

## Cloud Infrastructure

### **Cloud Provider**

* **GCP (Google Cloud Platform)**

### **Prerequisites**

* A **pre-created GCP project**
* Region: **us-east1**
* Required cloud resources:

  * Storage Bucket
  * Log Router
  * Cloud Composer (Airflow V3 – GCP Managed)
* IAM Requirements:

  * Service account with authentication keys
  * Appropriate IAM roles for Terraform, Cloud Composer, Storage, and Logging

---

## Infrastructure Deployment (Terraform)

### **Infrastructure as Code Tooling**

* **Terraform**
* Executed inside **GitHub Actions Linux environment**
* Workflow file: `.github/workflows/deploy cloud_composer.yml`
* Terraform files:

  * `researchAI/k8s/terraform.sh`
  * `researchAI/k8s/main.tf`

### **Workflow Description: deploy cloud_composer.yml**

This workflow prepares Terraform to manage existing and new GCP resources.

#### **Key Functionality**

1. Imports existing GCP resources into Terraform state, including:

   * IAM API
   * Service Accounts
   * Cloud Composer API
   * Log Routers
   * Cloud Composer Environment
2. Runs `terraform.sh`, which:

   * Initializes Terraform
   * Generates `terraform.tfstate` by importing GCP resources
3. Uses the Terraform configuration in `main.tf` to:

   * Run `terraform plan` to preview changes
   * Run `terraform apply` to apply modifications to the GCP environment

This ensures all infrastructure becomes Terraform-managed and kept in sync with desired configuration.

---

## Data Sync Between GitHub and Cloud Composer

Cloud Composer stores code and data in Google Cloud Storage. This section describes how DAGs, data, and logs are synchronized.

### **Storage Structure**

* **DAGs** → Stored in Cloud Composer bucket under `dags/`
* **Pipeline data** → Stored in `data/`
* **Logs** → Managed by GCP (not stored in composer bucket)

  * A **custom Log Router** forwards a copy of logs to a user-owned storage bucket

### **DVC Tracking**

Both data and logs are tracked using **DVC**, enabling versioned storage and reproducibility.
Automation is handled by: `.github/workflows/sync cloud composer.yml`

---

## Workflow Description: sync cloud_composer.yml

This workflow handles the sync between Cloud Composer and the repo.

### **Steps Performed**

1. Authenticaes with gcloud CLI to the gcp project.
2. Checkout the code to current directory
3. downloads the dags and data files from Cloud Composer bucket into a temp location
4. Deletes the old dags on Cloud Composer bucket dag folder
5. Copies the new dags fromr repo on to Cloud Composer bucket dag folder
6. Move to the dvc tracked location on repo and fetch the existing data files and log files from remote
7. Copies the Cloud Composer bucket's data files and log files into DVC feteched data and logs folders.
8. DVC the new data and logs into remote by updating data.dvc and logs.dvc files.
9. Commits the new .dvc files to repo on GitHub 

---

## GitHub Actions Automation Summary

Two automated workflows orchestrate the pipeline lifecycle:

### **1. Infrastructure Deployment**

* File: `.github/workflows/deploy cloud_composer.yml`
* Responsibility:

  * Deploy and manage GCP infrastructure via Terraform

### **2. Data & DAG Sync**

* File: `.github/workflows/sync cloud composer.yml`
* Responsibility:

  * Synchronize DAGs, data, and logs between Composer and the GitHub repo
  * Maintain versioned data/logs via DVC
* Schedule: Runs every 6 hours

---

## GitHub Workflows (Summary)

### **Infrastructure Deployment Workflow**

Automates provisioning and updating of GCP resources via Terraform.

* Authenticates using a GCP service account.
* Initializes Terraform and imports existing GCP resources.
* Applies infrastructure changes defined in `researchAI/k8s/main.tf`.
* Verifies Cloud Composer environment deployment.

### **Cloud Composer Sync Workflow**

Keeps Cloud Composer and GitHub repository fully synchronized.

* Authenticates to GCP and retrieves Composer bucket details.
* Downloads DAGs, data, and logs from Composer storage.
* Replaces old DAGs in Composer with updated repo DAGs.
* Pulls DVC data, merges new data/logs, and pushes updates.
* Commits updated DVC metadata back to GitHub.


### ** Flow Diagram **

                         ┌──────────────────────────┐
                         │        GitHub Repo       │
                         │  • DAGs (Airflow)        │
                         │  • DVC-tracked data/logs │
                         └───────────┬──────────────┘
                                     │
                                     │ GitHub Actions
                                     ▼
        ┌───────────────────────────────────────────────────────────┐
        │                    GitHub Actions (CI/CD)                 │
        │  Workflows:                                               │
        │   • deploy_cloud_composer.yml                             │
        │   • sync_cloud_composer.yml                               │
        │                                                           │
        │  - Runs terraform.sh                                      │
        │  - Imports existing GCP resources                         │
        │  - Manages terraform.tfstate                              │
        │  - Applies main.tf to update GCP infra                    │
        │  - Syncs DAGs → Composer bucket                           │
        │  - Syncs data/logs ↔ DVC                                  │
        └───────────────┬───────────────────────────────────────────┘
                        │
                        │ Terraform Apply
                        ▼
        ┌───────────────────────────────────────────────────────────┐
        │                           GCP                             │
        │                                                           │
        │   ┌──────────────────────────────────────────────────┐    │
        │   │                 Cloud Composer (Airflow V3)      │    │
        │   │  • Composer-managed DAG Runner                   │    │ 
        │   │  • Uses GCS Bucket for DAGs & Data               │    │
        │   └───────────┬───────────────────────┬──────────────┘    │
        │               │                       │                   │
        │               │ DAG Sync              │     Logs Output   │
        │               ▼                       ▼                   │
        │   ┌─────────────────────┐   ┌───────────────────────┐     │
        │   │ Composer GCS Bucket │   │ Cloud Logging (GCP)   │     │
        │   │  • /dags/ folder    │   │  (Managed)            │     │
        │   │  • /data/ folder    │   └──────────┬────────────┘     │
        │   └───────────┬─────────┘              │                  │
        │               │                        │ Log Router Sink  │
        │               │                        ▼                  │
        │   ┌───────────────────────────┐   ┌────────────────────┐  │
        │   │   DVC Data/Logs Repo      │   │ Custom Storage     │  │
        │   │ (Fetched in workflow)     │   │ Bucket (Log Sink)  │  │
        │   └───────────┬───────────────┘   └────────────────────┘  │
        │               |                                           |
        |               │ DVC Push Updates                          │
        └───────────────▼───────────────────────────────────────────┘
                        GitHub Repo updated with new .dvc files
