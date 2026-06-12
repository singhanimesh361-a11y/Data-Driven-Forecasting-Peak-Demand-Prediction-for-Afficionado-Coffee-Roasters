# ============================================================================
# Afficionado Demand Intelligence Platform — Terraform Configuration
# ============================================================================
# Provisions: Cloud SQL (PostgreSQL 15), GCS bucket, Cloud Run service,
#             Cloud Scheduler, IAM bindings.
#
# Usage:
#   terraform init
#   terraform plan -var="project_id=my-project" -var="db_password=secret"
#   terraform apply
# ============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "adip-terraform-state"
    prefix = "terraform/state"
  }
}

# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# ---------------------------------------------------------------------------
# Enable required APIs
# ---------------------------------------------------------------------------

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudscheduler.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudbuild.googleapis.com",
  ])

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# Service Account for ADIP
# ---------------------------------------------------------------------------

resource "google_service_account" "adip" {
  account_id   = "adip-service"
  display_name = "ADIP Service Account"
  description  = "Service account for the Afficionado Demand Intelligence Platform"
}

# ---------------------------------------------------------------------------
# Cloud SQL — PostgreSQL 15
# ---------------------------------------------------------------------------

resource "google_sql_database_instance" "adip_db" {
  name             = "adip-postgres-${var.environment}"
  database_version = "POSTGRES_15"
  region           = var.region

  deletion_protection = var.environment == "production" ? true : false

  settings {
    tier              = var.db_tier
    availability_type = var.environment == "production" ? "REGIONAL" : "ZONAL"
    disk_size         = var.db_disk_size_gb
    disk_type         = "PD_SSD"
    disk_autoresize   = true

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.adip_vpc.id
    }

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = var.environment == "production"
      transaction_log_retention_days = 7

      backup_retention_settings {
        retained_backups = 14
        retention_unit   = "COUNT"
      }
    }

    maintenance_window {
      day          = 7  # Sunday
      hour         = 4  # 04:00 UTC
      update_track = "stable"
    }

    database_flags {
      name  = "max_connections"
      value = "100"
    }

    database_flags {
      name  = "log_min_duration_statement"
      value = "1000"
    }

    insights_config {
      query_insights_enabled  = true
      record_application_tags = true
      record_client_address   = true
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_sql_database" "adip" {
  name     = "adip"
  instance = google_sql_database_instance.adip_db.name
}

resource "google_sql_user" "adip" {
  name     = "adip_app"
  instance = google_sql_database_instance.adip_db.name
  password = var.db_password
}

# ---------------------------------------------------------------------------
# VPC (for Cloud SQL private IP)
# ---------------------------------------------------------------------------

resource "google_compute_network" "adip_vpc" {
  name                    = "adip-vpc-${var.environment}"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "adip_subnet" {
  name          = "adip-subnet-${var.environment}"
  ip_cidr_range = "10.0.0.0/24"
  network       = google_compute_network.adip_vpc.id
  region        = var.region

  private_ip_google_access = true
}

resource "google_compute_global_address" "private_ip" {
  name          = "adip-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.adip_vpc.id
}

resource "google_service_networking_connection" "private_vpc" {
  network                 = google_compute_network.adip_vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip.name]
}

# ---------------------------------------------------------------------------
# GCS Bucket — forecast artefact storage
# ---------------------------------------------------------------------------

resource "google_storage_bucket" "adip_data" {
  name     = var.bucket_name
  location = var.region

  uniform_bucket_level_access = true
  force_destroy               = var.environment != "production"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  lifecycle_rule {
    condition {
      age                = 30
      num_newer_versions = 3
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    app         = "adip"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# Grant service account access to the bucket
resource "google_storage_bucket_iam_member" "adip_bucket_admin" {
  bucket = google_storage_bucket.adip_data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.adip.email}"
}

# ---------------------------------------------------------------------------
# Artifact Registry — Docker images
# ---------------------------------------------------------------------------

resource "google_artifact_registry_repository" "adip" {
  location      = var.region
  repository_id = "adip-registry"
  format        = "DOCKER"
  description   = "Container images for ADIP"

  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"

    most_recent_versions {
      keep_count = 10
    }
  }

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# Cloud Run — Streamlit Dashboard
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "adip_app" {
  name     = "adip-app-${var.environment}"
  location = var.region

  template {
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    service_account = google_service_account.adip.email

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/adip-registry/adip:latest"

      ports {
        container_port = 8501
      }

      resources {
        limits = {
          cpu    = var.cloud_run_cpu
          memory = var.cloud_run_memory
        }
        cpu_idle = true
      }

      env {
        name  = "ADIP_ENV"
        value = var.environment
      }

      env {
        name  = "STREAMLIT_SERVER_HEADLESS"
        value = "true"
      }

      env {
        name  = "STREAMLIT_SERVER_PORT"
        value = "8501"
      }

      env {
        name  = "GCS_BUCKET"
        value = google_storage_bucket.adip_data.name
      }

      env {
        name = "DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_url.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        http_get {
          path = "/_stcore/health"
        }
        initial_delay_seconds = 30
        period_seconds        = 10
        failure_threshold     = 5
      }

      liveness_probe {
        http_get {
          path = "/_stcore/health"
        }
        period_seconds    = 30
        failure_threshold = 3
      }
    }

    vpc_access {
      connector = google_vpc_access_connector.adip.id
      egress    = "PRIVATE_RANGES_ONLY"
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [google_project_service.apis]
}

# Allow unauthenticated access to Cloud Run
resource "google_cloud_run_v2_service_iam_member" "public" {
  location = google_cloud_run_v2_service.adip_app.location
  name     = google_cloud_run_v2_service.adip_app.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------------------------------------------------------------------------
# VPC Connector (Cloud Run → Cloud SQL)
# ---------------------------------------------------------------------------

resource "google_vpc_access_connector" "adip" {
  name          = "adip-vpc-connector"
  region        = var.region
  network       = google_compute_network.adip_vpc.name
  ip_cidr_range = "10.8.0.0/28"

  min_instances = 2
  max_instances = 3

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# Secret Manager — Database URL
# ---------------------------------------------------------------------------

resource "google_secret_manager_secret" "db_url" {
  secret_id = "adip-database-url"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "db_url" {
  secret      = google_secret_manager_secret.db_url.id
  secret_data = "postgresql://adip_app:${var.db_password}@${google_sql_database_instance.adip_db.private_ip_address}:5432/adip"
}

resource "google_secret_manager_secret_iam_member" "adip_db_access" {
  secret_id = google_secret_manager_secret.db_url.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.adip.email}"
}

# ---------------------------------------------------------------------------
# Cloud Scheduler — Nightly forecast refresh
# ---------------------------------------------------------------------------

resource "google_cloud_scheduler_job" "nightly_forecast" {
  name             = "adip-nightly-forecast-${var.environment}"
  description      = "Trigger nightly forecast generation at 02:00 UTC"
  schedule         = "0 2 * * *"
  time_zone        = "UTC"
  attempt_deadline = "600s"

  retry_config {
    retry_count          = 3
    min_backoff_duration = "30s"
    max_backoff_duration = "300s"
    max_doublings        = 3
  }

  http_target {
    http_method = "POST"
    uri         = "${google_cloud_run_v2_service.adip_app.uri}/api/forecast/refresh"

    oidc_token {
      service_account_email = google_service_account.adip.email
    }
  }

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# IAM Bindings
# ---------------------------------------------------------------------------

# Cloud SQL Client
resource "google_project_iam_member" "adip_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.adip.email}"
}

# Cloud Run Admin (for CI/CD deployments)
resource "google_project_iam_member" "adip_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.adip.email}"
}

# Logging Writer
resource "google_project_iam_member" "adip_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.adip.email}"
}

# Monitoring Metric Writer
resource "google_project_iam_member" "adip_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.adip.email}"
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "cloud_run_url" {
  description = "URL of the deployed ADIP Cloud Run service"
  value       = google_cloud_run_v2_service.adip_app.uri
}

output "cloud_sql_connection" {
  description = "Cloud SQL instance connection name"
  value       = google_sql_database_instance.adip_db.connection_name
}

output "gcs_bucket" {
  description = "GCS bucket for forecast artefacts"
  value       = google_storage_bucket.adip_data.name
}

output "service_account_email" {
  description = "ADIP service account email"
  value       = google_service_account.adip.email
}

output "artifact_registry" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.adip.repository_id}"
}
