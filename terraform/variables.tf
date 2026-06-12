# ============================================================================
# Afficionado Demand Intelligence Platform — Terraform Variables
# ============================================================================

# ---------------------------------------------------------------------------
# Required Variables
# ---------------------------------------------------------------------------

variable "project_id" {
  description = "Google Cloud project ID"
  type        = string

  validation {
    condition     = length(var.project_id) > 0
    error_message = "project_id must not be empty."
  }
}

variable "region" {
  description = "Google Cloud region for all resources"
  type        = string
  default     = "us-central1"

  validation {
    condition     = can(regex("^[a-z]+-[a-z]+[0-9]$", var.region))
    error_message = "region must be a valid GCP region (e.g., us-central1)."
  }
}

variable "db_password" {
  description = "Password for the Cloud SQL database user (adip_app)"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.db_password) >= 16
    error_message = "db_password must be at least 16 characters for security."
  }
}

variable "bucket_name" {
  description = "Name of the GCS bucket for forecast artefact storage"
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$", var.bucket_name))
    error_message = "bucket_name must be a valid GCS bucket name."
  }
}

# ---------------------------------------------------------------------------
# Optional Variables — Environment
# ---------------------------------------------------------------------------

variable "environment" {
  description = "Deployment environment (production, staging, development)"
  type        = string
  default     = "production"

  validation {
    condition     = contains(["production", "staging", "development"], var.environment)
    error_message = "environment must be one of: production, staging, development."
  }
}

# ---------------------------------------------------------------------------
# Optional Variables — Cloud SQL
# ---------------------------------------------------------------------------

variable "db_tier" {
  description = "Cloud SQL machine tier"
  type        = string
  default     = "db-custom-2-4096"
}

variable "db_disk_size_gb" {
  description = "Cloud SQL disk size in GB"
  type        = number
  default     = 20

  validation {
    condition     = var.db_disk_size_gb >= 10 && var.db_disk_size_gb <= 1000
    error_message = "db_disk_size_gb must be between 10 and 1000."
  }
}

# ---------------------------------------------------------------------------
# Optional Variables — Cloud Run
# ---------------------------------------------------------------------------

variable "min_instances" {
  description = "Minimum number of Cloud Run instances"
  type        = number
  default     = 1

  validation {
    condition     = var.min_instances >= 0 && var.min_instances <= 10
    error_message = "min_instances must be between 0 and 10."
  }
}

variable "max_instances" {
  description = "Maximum number of Cloud Run instances"
  type        = number
  default     = 3

  validation {
    condition     = var.max_instances >= 1 && var.max_instances <= 100
    error_message = "max_instances must be between 1 and 100."
  }
}

variable "cloud_run_cpu" {
  description = "CPU allocation per Cloud Run instance"
  type        = string
  default     = "2"
}

variable "cloud_run_memory" {
  description = "Memory allocation per Cloud Run instance"
  type        = string
  default     = "2Gi"
}
