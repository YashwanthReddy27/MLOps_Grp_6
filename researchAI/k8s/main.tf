provider "google-beta" {
  project = "karthikdevfest"
  region  = "us-east1"
}

resource "google_project_service" "composer_api" {
  project                             = "karthikdevfest"
  provider                            = google-beta
  service                             = "composer.googleapis.com"
  disable_on_destroy                  = false
  check_if_service_has_usage_on_destroy = true
}

# ───────────────────────────────────────────────
# LOOKUP EXISTING SERVICE ACCOUNT (if present)
# ───────────────────────────────────────────────
data "google_service_account" "existing" {
  project    = "karthikdevfest"
  account_id = "terraform-sa"
}

resource "google_service_account" "custom_service_account" {
  account_id   = "terraform-sa"
  display_name = "terraform Service Account"
  project      = "karthikdevfest"

  # Only create the service account if it doesn't already exist
  lifecycle {
    prevent_destroy = false
    ignore_changes  = [display_name]
  }
  
}

# ───────────────────────────────────────────────
# IAM BINDING USING LOCAL VALUE
# ───────────────────────────────────────────────
resource "google_project_iam_member" "custom_service_account" {
  project = "karthikdevfest"
  member  = "serviceAccount:terraform-sa@karthikdevfest.iam.gserviceaccount.com"
  role    = "roles/composer.worker"

  lifecycle {
    ignore_changes = [member]  # Prevent unnecessary recreation if the member email is unchanged
  }
}



# ───────────────────────────────────────────────
# COMPOSER ENVIRONMENT USING SAME SA
# ───────────────────────────────────────────────
resource "google_composer_environment" "example_environment" {
  name     = "rag-environment"
  provider = google-beta
  region   = "us-east1"
  project  = "karthikdevfest"

  depends_on = [
    google_project_iam_member.custom_service_account,
    google_project_service.composer_api
  ]
 
  config {
    software_config {
      image_version = "composer-3-airflow-3.1.0-build.2"

      pypi_packages = {
        pandas                        = ""
        scikit-learn                  = ""
        torch                         = ""
        transformers                  = ""
        sentencepiece                 = ""
        kneed                         = ""
        requests                      = ""
        apache-airflow-providers-postgres = ""
        great-expectations            = ">=0.18.8"
        sqlalchemy                    = ""
        beautifulsoup4                = ""
        lxml                          = ""
        html5lib                      = ""
        fairlearn                     = ""
        dvc                           = "[gs]"
      }
    }

    node_config {
      service_account = "terraform-sa@karthikdevfest.iam.gserviceaccount.com"
    }
  }

}
