#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# run_gcp_pipeline.sh
#
# Provisions a fresh, single-use, locked-down GCP environment; ships the
# sanitised handwriting scans plus the transcription code up to a GPU VM;
# runs every model listed in models.json against every image with three
# OCR prompts each; and copies the resulting .txt/.json transcripts back
# into experiments/. Everything this script creates is meant to be deleted
# again by cleanup_gcp_pipeline.sh once the run finishes.
#
# SECURITY POSTURE (why the infra looks the way it does)
#   - No external IP on the VM (--no-address) + IAP-only SSH firewall rule
#     + Cloud NAT for egress: the VM never has a public interface, so the
#     scanned handwriting (which may contain personal data) is never
#     reachable from, nor directly exposed to, the open internet. The only
#     inbound path is Google's Identity-Aware Proxy TCP tunnel, which is
#     authenticated via IAM, not a routable public IP. Outbound internet
#     access needed to install packages / pull model weights goes out
#     through Cloud NAT instead of a public IP on the instance.
#   - CMEK boot disk (--boot-disk-kms-key): the boot disk is encrypted with
#     a customer-managed key created for this run rather than a
#     Google-managed key, so the encryption key (and the ability to revoke
#     it) stays under our control rather than Google's.
#   - Per-run service account, VPC/subnet, firewall, router/NAT, and KMS
#     keyring (all named with $RUN_ID): every run gets its own throwaway
#     identity and network. This contains blast radius (a compromised VM
#     or SA can only touch this run's resources) and makes teardown a
#     matter of deleting one RUN_ID's worth of resources with no risk of
#     collateral damage to other runs or shared infra.
#
# COST PROFILE
#   L4 GPU on a g2-standard-8, on-demand billing. Cost only accrues while
#   the VM exists (roughly: VM boot + apt/pip installs + model download +
#   inference time for however many models are configured). Deleting the
#   VM via cleanup_gcp_pipeline.sh stops billing for compute; the boot
#   disk, NAT, and KMS keyring are also deleted/destroyed by that script
#   (see caveat about keyrings below).
#
# PROS / CONS
#   + Reproducible: every run gets an identical, from-scratch environment.
#   + Isolated: no shared long-lived VM or credentials to leak or drift.
#   + Cheap per run: nothing persists (and therefore bills) once torn down.
#   - Slow cold start: every run pays for VM boot, NVIDIA driver install,
#     Python venv setup, pip installs, and a fresh model download - there
#     is no warm cache between runs.
#   - KMS keyrings accumulate: GCP KMS keyrings (and the keys inside them)
#     cannot be deleted, only individual key *versions* can be destroyed.
#     cleanup_gcp_pipeline.sh destroys the key version, but the empty
#     keyring/key resource is permanent, so the project slowly accumulates
#     one keyring per run forever.
#   - Quota/region pitfalls: L4 GPUs are only available in a subset of
#     regions/zones, and require GPU quota pre-approved in that region; a
#     run will fail at instance-create time if the project lacks quota or
#     the chosen zone doesn't stock L4s.
# ---------------------------------------------------------------------------

# --- Configuration (env-overridable) ---------------------------------------
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

# --- Sanitise images locally before anything ever leaves the machine -------
# prepare_upload.py strips EXIF/GPS metadata, converts HEIC->JPEG, and
# downsizes images (see src/prepare_upload.py). Nothing is uploaded to GCP
# until this has run, so the only thing that reaches the VM is a clean JPEG.
gcloud config set project "$PROJECT"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
$PYTHON "$ROOT/src/prepare_upload.py" --input "$DATA_DIR" --output "$UPLOAD"

# --- Network: custom VPC + private subnet, no default routes to the world --
# A custom-mode VPC/subnet dedicated to this run (rather than the project's
# default network) so the VM's networking is fully scoped to $RUN_ID and can
# be torn down without touching anything else in the project.
gcloud compute networks create "$NETWORK" --subnet-mode=custom

gcloud compute networks subnets create "$SUBNET" --network="$NETWORK" --range=10.10.0.0/24 --region="$REGION" --enable-private-ip-google-access

# Firewall: allow SSH only from Google's IAP TCP-forwarding range
# (35.235.240.0/20). The VM has no external IP, so this is the only path in,
# and it's gated further by IAM permissions on the IAP tunnel itself.
gcloud compute firewall-rules create "$FW" --network="$NETWORK" --allow=tcp:22 --source-ranges=35.235.240.0/20

# Cloud Router + NAT: gives the address-less VM outbound internet access
# (apt packages, pip installs, HuggingFace model downloads) without ever
# assigning it a public IP.
gcloud compute routers create "$ROUTER" --network="$NETWORK" --region="$REGION"
gcloud compute routers nats create "$NAT" --router="$ROUTER" --region="$REGION" --nat-all-subnet-ip-ranges --auto-allocate-nat-external-ips

# --- KMS: per-run key for the CMEK-encrypted boot disk ----------------------
# NOTE: keyrings and keys cannot be deleted in GCP, only key *versions* can
# be destroyed (see cleanup_gcp_pipeline.sh). This is why a fresh keyring
# per run leaves permanent, empty debris behind - a known/accepted cost of
# this design (see header notes above).
gcloud kms keyrings create "$KMS_KEYRING" --location="$REGION"
gcloud kms keys create "$KMS_KEY" --location="$REGION" --keyring="$KMS_KEYRING" --purpose=encryption

# --- IAM: per-run service account for the VM, plus the grant GCE itself needs
# The VM runs as its own single-purpose service account (least privilege,
# easy to revoke). Separately, the *Compute Engine service agent*
# (service-<project-number>@compute-system...) - not the VM's own SA - is
# the identity that actually performs the disk encrypt/decrypt operations
# under the hood, so it needs the encrypter/decrypter grant on the key.
# `|| true`: this binding is not strictly idempotent/critical-path-safe
# across retries, so failures here are swallowed; if disk creation later
# fails with a KMS permission error, check this binding manually.
gcloud iam service-accounts create "$SA" --display-name="$RUN_ID handwriting VM"
SA_EMAIL="$SA@$PROJECT.iam.gserviceaccount.com"
KMS_RESOURCE="projects/$PROJECT/locations/$REGION/keyRings/$KMS_KEYRING/cryptoKeys/$KMS_KEY"
gcloud kms keys add-iam-policy-binding "$KMS_KEY" --location="$REGION" --keyring="$KMS_KEYRING" --member="serviceAccount:service-$PROJECT_NUMBER@compute-system.iam.gserviceaccount.com" --role=roles/cloudkms.cryptoKeyEncrypterDecrypter || true

# --- VM: L4 GPU instance from a Deep Learning VM image ----------------------
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
# --maintenance-policy=TERMINATE: instances with an attached GPU cannot be
# live-migrated during host maintenance, so they must be stopped/terminated
# instead - this is required for any GPU-attached instance, not optional.
# --network-interface=...,no-address: omits the ephemeral external IP that
# gcloud would otherwise assign by default; combined with the firewall rule
# above, this is what makes the VM unreachable except via the IAP tunnel.
# enable-oslogin + block-project-ssh-keys: SSH identity/authorization is
# managed through IAM/OS Login rather than project-wide SSH metadata keys.

# --- scp: ship sanitised images + run code up to the VM ---------------------
# --tunnel-through-iap: routes the SSH/SCP connection through the IAP TCP
# tunnel instead of a direct connection to a public IP (the VM has none).
# This is the client-side counterpart to the IAP firewall rule created above.
SSH_ARGS=(--tunnel-through-iap --zone="$ZONE")
if [[ -n "${SSH_KEY_FILE:-}" ]]; then
  SSH_ARGS+=(--ssh-key-file="$SSH_KEY_FILE")
fi

gcloud compute scp --recurse "${SSH_ARGS[@]}" "$UPLOAD" "$VM:~/input"
gcloud compute scp "${SSH_ARGS[@]}" "$ROOT/src/gcp/remote_transcribe.py" "$ROOT/requirements-remote.txt" "$ROOT/models.json" "$VM:~/"

# --- run: set up the venv on the VM, then transcribe with every model ------
# models.json[0] (Florence-2-base) is run first against the requirements-
# remote.txt-pinned transformers version. `transformers` is then upgraded
# in place before looping over the remaining models.json entries (chat-style
# Qwen/Gemma vision models), because those newer models need a newer
# transformers than the version Florence-2-base was validated against.
# Running Florence first, then upgrading, avoids needing two separate
# venvs/requirements files for one run.
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

# --- fetch: copy every model's transcripts back to the local machine -------
gcloud compute scp --recurse "${SSH_ARGS[@]}" "$VM:~/experiments" "$OUTPUT"

echo "Outputs copied to $OUTPUT"
echo "Run scripts/cleanup_gcp_pipeline.sh with the same RUN_ID to delete VM, disk, network, NAT, service account, and KMS key version."
