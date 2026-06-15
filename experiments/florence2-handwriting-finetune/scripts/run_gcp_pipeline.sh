#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:?}
REGION=${REGION:-europe-west4}
ZONE=${ZONE:-europe-west4-a}
RUN_ID=${RUN_ID:-hw-$(date -u +%Y%m%d%H%M%S)}
MACHINE=${MACHINE:-g2-standard-8}
ACCELERATOR=${ACCELERATOR:-type=nvidia-l4,count=1}
IMAGE_FAMILY=${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}
IMAGE_PROJECT=${IMAGE_PROJECT:-deeplearning-platform-release}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_DIR=${DATA_DIR:-$ROOT/data}
UPLOAD=${UPLOAD:-$ROOT/upload_sanitised}
OUTPUT=${OUTPUT:-$ROOT/experiments}
BOOT_DISK_GB=${BOOT_DISK_GB:-200}
KMS_KEYRING=${KMS_KEYRING:-${RUN_ID}-ring}
KMS_KEY=${KMS_KEY:-${RUN_ID}-boot}
SA=${SA:-${RUN_ID}-sa}
NETWORK=${NETWORK:-${RUN_ID}-vpc}
SUBNET=${SUBNET:-${RUN_ID}-subnet}
ROUTER=${ROUTER:-${RUN_ID}-router}
NAT=${NAT:-${RUN_ID}-nat}
VM=${VM:-${RUN_ID}-vm}
FW=${FW:-${RUN_ID}-iap-ssh}
PYTHON=${PYTHON:-python3}

gcloud config set project "$PROJECT"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
$PYTHON "$ROOT/src/prepare_upload.py" --input "$DATA_DIR" --output "$UPLOAD"

gcloud compute networks create "$NETWORK" --subnet-mode=custom

gcloud compute networks subnets create "$SUBNET" --network="$NETWORK" --range=10.10.0.0/24 --region="$REGION" --enable-private-ip-google-access

gcloud compute firewall-rules create "$FW" --network="$NETWORK" --allow=tcp:22 --source-ranges=35.235.240.0/20

gcloud compute routers create "$ROUTER" --network="$NETWORK" --region="$REGION"
gcloud compute routers nats create "$NAT" --router="$ROUTER" --region="$REGION" --nat-all-subnet-ip-ranges --auto-allocate-nat-external-ips

gcloud kms keyrings create "$KMS_KEYRING" --location="$REGION"
gcloud kms keys create "$KMS_KEY" --location="$REGION" --keyring="$KMS_KEYRING" --purpose=encryption

gcloud iam service-accounts create "$SA" --display-name="$RUN_ID handwriting VM"
SA_EMAIL="$SA@$PROJECT.iam.gserviceaccount.com"
KMS_RESOURCE="projects/$PROJECT/locations/$REGION/keyRings/$KMS_KEYRING/cryptoKeys/$KMS_KEY"
gcloud kms keys add-iam-policy-binding "$KMS_KEY" --location="$REGION" --keyring="$KMS_KEYRING" --member="serviceAccount:service-$PROJECT_NUMBER@compute-system.iam.gserviceaccount.com" --role=roles/cloudkms.cryptoKeyEncrypterDecrypter || true

gcloud compute instances create "$VM" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --accelerator="$ACCELERATOR" \
  --maintenance-policy=TERMINATE \
  --provisioning-model=STANDARD \
  --network-interface="subnet=$SUBNET,no-address" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="${BOOT_DISK_GB}GB" \
  --boot-disk-kms-key="$KMS_RESOURCE" \
  --service-account="$SA_EMAIL" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --shielded-secure-boot \
  --shielded-vtpm \
  --shielded-integrity-monitoring \
  --metadata=enable-oslogin=TRUE,block-project-ssh-keys=TRUE,install-nvidia-driver=True

SSH_ARGS=(--tunnel-through-iap --zone="$ZONE")
if [[ -n "${SSH_KEY_FILE:-}" ]]; then
  SSH_ARGS+=(--ssh-key-file="$SSH_KEY_FILE")
fi

gcloud compute scp --recurse "${SSH_ARGS[@]}" "$UPLOAD" "$VM:~/input"
gcloud compute scp "${SSH_ARGS[@]}" "$ROOT/src/gcp/remote_transcribe.py" "$ROOT/requirements-remote.txt" "$ROOT/models.json" "$VM:~/"

gcloud compute ssh "$VM" "${SSH_ARGS[@]}" --command='set -euo pipefail
sudo apt-get update
sudo apt-get install -y python3.10-venv python3-pip
python3 -m venv ~/venv
. ~/venv/bin/activate
pip install -U pip
pip install -r ~/requirements-remote.txt
mkdir -p ~/experiments
python remote_transcribe.py --input input --output experiments --model microsoft/Florence-2-base --max-new-tokens 512 --attempts 3
pip install -U transformers
python - <<"PY"
import json, subprocess
for m in json.load(open("models.json"))[1:]:
    cmd=["python", "remote_transcribe.py", "--input", "input", "--output", "experiments", "--model", m["model_id"], "--max-new-tokens", str(m["max_new_tokens"]), "--attempts", "3"]
    if m["load_in_4bit"]:
        cmd.append("--load-in-4bit")
    subprocess.run(cmd, check=True)
PY'

gcloud compute scp --recurse "${SSH_ARGS[@]}" "$VM:~/experiments" "$OUTPUT"

echo "Outputs copied to $OUTPUT"
echo "Run scripts/cleanup_gcp_pipeline.sh with the same RUN_ID to delete VM, disk, network, NAT, service account, and KMS key version."
