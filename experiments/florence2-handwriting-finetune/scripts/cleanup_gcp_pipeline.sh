#!/usr/bin/env bash
set -euo pipefail

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
gcloud compute instances delete "$VM" --zone="$ZONE" --quiet || true
gcloud compute firewall-rules delete "$FW" --quiet || true
gcloud compute routers nats delete "$NAT" --router="$ROUTER" --region="$REGION" --quiet || true
gcloud compute routers delete "$ROUTER" --region="$REGION" --quiet || true
gcloud compute networks subnets delete "$SUBNET" --region="$REGION" --quiet || true
gcloud compute networks delete "$NETWORK" --quiet || true
gcloud iam service-accounts delete "$SA@$PROJECT.iam.gserviceaccount.com" --quiet || true
gcloud kms keys versions destroy 1 --key="$KMS_KEY" --keyring="$KMS_KEYRING" --location="$REGION" --quiet || true
