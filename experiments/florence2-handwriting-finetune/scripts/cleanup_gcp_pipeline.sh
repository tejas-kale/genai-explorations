#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# cleanup_gcp_pipeline.sh
#
# Tears down the per-run resources created by run_gcp_pipeline.sh for a
# given RUN_ID: the GPU VM, IAP firewall rule, Cloud NAT + router, subnet,
# VPC, per-run service account, and the KMS key version used to encrypt the
# boot disk. Run with the same RUN_ID/PROJECT/REGION/ZONE (or the same
# individual resource-name overrides) used for the matching run.
#
# Every `gcloud ... delete` below is followed by `|| true` so that a
# resource that was never created (e.g. a run that failed partway through
# run_gcp_pipeline.sh) or was already deleted doesn't abort the rest of the
# cleanup - this script is meant to be safe to re-run.
#
# SECURITY / COST NOTE: deleting these resources is what actually stops
# billing and closes the network/IAM surface opened for the run; a run that
# is never cleaned up leaves a billed GPU VM (or, if the VM itself was
# already deleted manually, an unused VPC/SA/NAT) running indefinitely.
#
# KNOWN LIMITATION: GCP KMS keyrings (and the keys inside them) cannot be
# deleted - only individual key *versions* can be destroyed, which is what
# the last line does. The keyring and key resource names therefore persist
# forever per run; this is an accepted, unavoidable cost of the per-run KMS
# keyring design (see run_gcp_pipeline.sh header for details).
# ---------------------------------------------------------------------------

PROJECT=${PROJECT:?}
REGION=${REGION:-europe-west4}
ZONE=${ZONE:-europe-west4-a}
RUN_ID=${RUN_ID:?}
SA=${SA:-${RUN_ID}-sa}
NETWORK=${NETWORK:-${RUN_ID}-vpc}
SUBNET=${SUBNET:-${RUN_ID}-subnet}
ROUTER=${ROUTER:-${RUN_ID}-router}
NAT=${NAT:-${RUN_ID}-nat}
VM=${VM:-${RUN_ID}-vm}
FW=${FW:-${RUN_ID}-iap-ssh}
KMS_KEYRING=${KMS_KEYRING:-${RUN_ID}-ring}
KMS_KEY=${KMS_KEY:-${RUN_ID}-boot}

gcloud config set project "$PROJECT"

# --- VM: delete first so nothing is still using the network/disk/SA below --
gcloud compute instances delete "$VM" --zone="$ZONE" --quiet || true

# --- Network: firewall rule, then NAT, then router, then subnet, then VPC --
# (deletion order matters: each resource depends on the one after it, so we
# tear down in the reverse order run_gcp_pipeline.sh created them in)
gcloud compute firewall-rules delete "$FW" --quiet || true
gcloud compute routers nats delete "$NAT" --router="$ROUTER" --region="$REGION" --quiet || true
gcloud compute routers delete "$ROUTER" --region="$REGION" --quiet || true
gcloud compute networks subnets delete "$SUBNET" --region="$REGION" --quiet || true
gcloud compute networks delete "$NETWORK" --quiet || true

# --- IAM: remove the per-run VM service account -----------------------------
gcloud iam service-accounts delete "$SA@$PROJECT.iam.gserviceaccount.com" --quiet || true

# --- KMS: destroy the key version (the closest thing to "delete" for KMS) --
# The keyring/key resource names themselves are permanent in GCP and cannot
# be removed by this or any script; destroying version 1 (the only version
# this pipeline ever creates) revokes the ability to decrypt the boot disk,
# which is the actual security-relevant cleanup step.
gcloud kms keys versions destroy 1 --key="$KMS_KEY" --keyring="$KMS_KEYRING" --location="$REGION" --quiet || true
