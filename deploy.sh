#!/usr/bin/env bash
# ============================================================================
# Afficionado Demand Intelligence Platform — Cloud Deployment Script
# ============================================================================
# Deploys ADIP to Google Cloud Run with Artifact Registry.
#
# Prerequisites:
#   - gcloud CLI installed and configured
#   - Docker installed
#   - Sufficient IAM permissions (roles/run.admin, roles/artifactregistry.writer)
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh                     # Deploy with defaults
#   ./deploy.sh --project my-proj   # Override project ID
#   ./deploy.sh --region us-east1   # Override region
# ============================================================================

set -euo pipefail
IFS=$'\n\t'

# ---------------------------------------------------------------------------
# Configuration (override via env vars or CLI flags)
# ---------------------------------------------------------------------------
PROJECT_ID="${GCP_PROJECT_ID:-afficionado-demand-intel}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="adip-app"
REPO_NAME="adip-registry"
IMAGE_NAME="adip"
TAG="${DEPLOY_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')}"
FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}"

MIN_INSTANCES=1
MAX_INSTANCES=3
MEMORY="2Gi"
CPU="2"
PORT=8501
TIMEOUT=300

# Colours for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
log_info()    { echo -e "${BLUE}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC}    $(date '+%H:%M:%S') $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*"; }

die() { log_error "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)  PROJECT_ID="$2"; shift 2 ;;
        --region)   REGION="$2";     shift 2 ;;
        --tag)      TAG="$2";        shift 2 ;;
        --dry-run)  DRY_RUN=true;    shift   ;;
        -h|--help)
            echo "Usage: $0 [--project PROJECT_ID] [--region REGION] [--tag TAG] [--dry-run]"
            exit 0
            ;;
        *) die "Unknown argument: $1" ;;
    esac
done

FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}"

log_info "============================================================"
log_info "ADIP Cloud Deployment"
log_info "============================================================"
log_info "Project:  ${PROJECT_ID}"
log_info "Region:   ${REGION}"
log_info "Image:    ${FULL_IMAGE}"
log_info "Service:  ${SERVICE_NAME}"
log_info "============================================================"

# ---------------------------------------------------------------------------
# Step 1: Authenticate with Google Cloud
# ---------------------------------------------------------------------------
log_info "Step 1/6 — Authenticating with Google Cloud..."

if ! gcloud auth print-access-token &>/dev/null; then
    log_warn "Not authenticated — initiating login..."
    gcloud auth login --brief
fi

gcloud config set project "${PROJECT_ID}" --quiet
gcloud config set run/region "${REGION}" --quiet

log_ok "Authenticated as $(gcloud config get account)"

# ---------------------------------------------------------------------------
# Step 2: Enable required APIs
# ---------------------------------------------------------------------------
log_info "Step 2/6 — Enabling required Google Cloud APIs..."

APIS=(
    "run.googleapis.com"
    "artifactregistry.googleapis.com"
    "cloudbuild.googleapis.com"
    "cloudscheduler.googleapis.com"
    "sqladmin.googleapis.com"
    "secretmanager.googleapis.com"
)

for api in "${APIS[@]}"; do
    if gcloud services list --enabled --filter="name:${api}" --format="value(name)" | grep -q "${api}"; then
        log_ok "  ${api} (already enabled)"
    else
        gcloud services enable "${api}" --quiet
        log_ok "  ${api} (enabled)"
    fi
done

# ---------------------------------------------------------------------------
# Step 3: Create Artifact Registry repository (if not exists)
# ---------------------------------------------------------------------------
log_info "Step 3/6 — Setting up Artifact Registry..."

if gcloud artifacts repositories describe "${REPO_NAME}" \
    --location="${REGION}" --format="value(name)" &>/dev/null; then
    log_ok "Repository ${REPO_NAME} already exists"
else
    gcloud artifacts repositories create "${REPO_NAME}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="ADIP container images" \
        --quiet
    log_ok "Created repository ${REPO_NAME}"
fi

# Configure Docker to use gcloud as credential helper
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ---------------------------------------------------------------------------
# Step 4: Build and push Docker image
# ---------------------------------------------------------------------------
log_info "Step 4/6 — Building Docker image..."

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    log_warn "DRY RUN — skipping docker build"
else
    docker build \
        --platform linux/amd64 \
        --tag "${FULL_IMAGE}" \
        --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="${TAG}" \
        .

    log_ok "Image built: ${FULL_IMAGE}"

    log_info "Pushing image to Artifact Registry..."
    docker push "${FULL_IMAGE}"
    docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
    log_ok "Image pushed successfully"
fi

# ---------------------------------------------------------------------------
# Step 5: Deploy to Cloud Run
# ---------------------------------------------------------------------------
log_info "Step 5/6 — Deploying to Cloud Run..."

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    log_warn "DRY RUN — skipping Cloud Run deployment"
else
    gcloud run deploy "${SERVICE_NAME}" \
        --image="${FULL_IMAGE}" \
        --platform=managed \
        --region="${REGION}" \
        --port="${PORT}" \
        --min-instances="${MIN_INSTANCES}" \
        --max-instances="${MAX_INSTANCES}" \
        --memory="${MEMORY}" \
        --cpu="${CPU}" \
        --timeout="${TIMEOUT}" \
        --allow-unauthenticated \
        --set-env-vars="ADIP_ENV=production,STREAMLIT_SERVER_HEADLESS=true,STREAMLIT_SERVER_PORT=${PORT}" \
        --labels="app=adip,env=production,version=${TAG}" \
        --quiet

    SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
        --region="${REGION}" \
        --format="value(status.url)")

    log_ok "Deployed to: ${SERVICE_URL}"

    # Health check
    log_info "Running post-deploy health check..."
    for i in $(seq 1 10); do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/_stcore/health" || echo "000")
        if [[ "${HTTP_CODE}" == "200" ]]; then
            log_ok "Health check passed (attempt ${i})"
            break
        fi
        log_warn "Attempt ${i}: HTTP ${HTTP_CODE} — retrying in 5s..."
        sleep 5
    done

    if [[ "${HTTP_CODE}" != "200" ]]; then
        log_error "Health check failed after 10 attempts"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 6: Register Antigravity agent (if manifest exists)
# ---------------------------------------------------------------------------
log_info "Step 6/6 — Registering Antigravity agent..."

if [[ -f "antigravity-agent.yaml" ]]; then
    if command -v antigravity &>/dev/null; then
        antigravity agent register \
            --manifest=antigravity-agent.yaml \
            --project="${PROJECT_ID}" \
            --region="${REGION}" \
            --endpoint="${SERVICE_URL:-https://${SERVICE_NAME}-${PROJECT_ID}.${REGION}.run.app}"
        log_ok "Agent registered with Antigravity platform"
    else
        log_warn "antigravity CLI not found — skipping agent registration"
        log_warn "Install: pip install antigravity-cli && antigravity agent register --manifest=antigravity-agent.yaml"
    fi
else
    log_warn "antigravity-agent.yaml not found — skipping registration"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
log_info "============================================================"
log_ok   "DEPLOYMENT COMPLETE"
log_info "============================================================"
log_info "Service:    ${SERVICE_NAME}"
log_info "Image:      ${FULL_IMAGE}"
log_info "URL:        ${SERVICE_URL:-N/A}"
log_info "Dashboard:  ${SERVICE_URL:-N/A}"
log_info "Health:     ${SERVICE_URL:-N/A}/_stcore/health"
log_info "============================================================"
