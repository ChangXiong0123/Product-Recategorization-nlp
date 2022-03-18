#!/bin/bash

set -exo pipefail

# readonly PACKAGES=$(/usr/share/google/get_metadata_value attributes/PIP_PACKAGES || true)
readonly bucket=$(/usr/share/google/get_metadata_value attributes/bucket)


function err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
  exit 1
}

function run_with_retry() {
  local -r cmd=("$@")
  for ((i = 0; i < 10; i++)); do
    if "${cmd[@]}"; then
      return 0
    fi
    sleep 5
  done
  err "Failed to run command: ${cmd[*]}"
}

function install_pip() {
  if command -v pip >/dev/null; then
    echo "pip is already installed."
    return 0
  fi

  if command -v easy_install >/dev/null; then
    echo "Installing pip with easy_install..."
    run_with_retry easy_install pip
    return 0
  fi

  echo "Installing python-pip..."
  run_with_retry apt update
  run_with_retry apt install python-pip -y
}

function main() {
  # if [[ -z "${PACKAGES}" ]]; then
  #   echo "ERROR: Must specify PIP_PACKAGES metadata key"
  #   exit 1
  # fi

  install_pip
  gsutil cp gs://"${bucket}"/dags/lmedia/dataproc/nlp/packages/spark-nlp-3.3.0.tar.gz .
  run_with_retry pip install spark-nlp-3.3.0.tar.gz
}

main