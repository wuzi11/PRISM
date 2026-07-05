#!/usr/bin/env bash
# Build LLM / zero / random pathway priors for any h5ad with obs["Group"].
#
# Required: DATA_PATH (train or combined AnnData)
# Optional: TEST_ADATA_PATH, PRIOR_PATH, PRIOR_MODE, ENV_FILE
#
# Examples:
#   DATA_PATH=datasets/sci_plex_train_drug_split_0.h5ad \
#   TEST_ADATA_PATH=datasets/sci_plex_test_drug_split_0.h5ad \
#   PRIOR_PATH=prism/resources/deepseek/llm_pathway_priors_sci_plex.json \
#   bash prism/shell/build_llm_priors.sh
#
#   DATA_PATH=datasets_lincs/lincs_l1000_train_drug_split_0.h5ad \
#   TEST_ADATA_PATH=datasets_lincs/lincs_l1000_test_drug_split_0.h5ad \
#   PRIOR_PATH=prism/results/priors/lincs_drug_split_0.json \
#   CONCURRENCY=10 bash prism/shell/build_llm_priors.sh
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prism_init.sh"

usage() {
  cat <<EOF
Usage: DATA_PATH=/path/to/train.h5ad [options] $0

Build pathway direction priors (llm / zero / random) from AnnData Group labels.

Required:
  DATA_PATH          Primary h5ad (must contain obs['Group'])

Optional:
  TEST_ADATA_PATH    Extra h5ad(s), comma-separated (e.g. test set)
  PRIOR_PATH         Output JSON (default: prism/results/priors/<data_basename>.json)
  LOG_PATH           Build log (default: same dir as PRIOR_PATH, .build.log suffix)
  PRIOR_MODE         llm | zero | random | auto (default: auto)
  ENV_FILE           LLM API env file
  PRISM_ENV_FILE     Alias for ENV_FILE
  CONCURRENCY        Parallel LLM requests (default: 4, llm mode only)
  SAVE_EVERY         Checkpoint every N queries (default: 10)
  REPLACE_ZERO       Re-query all-zero priors when resuming (default: true)
  RESUME             Resume from existing PRIOR_PATH (default: true)
EOF
}

DATA_PATH="${DATA_PATH:-}"
TEST_ADATA_PATH="${TEST_ADATA_PATH:-}"
ENV_FILE="${PRISM_ENV_FILE:-${ENV_FILE:-$(resolve_prism_llm_env)}}"
PRIOR_MODE="${PRIOR_MODE:-auto}"

if [[ -z "${DATA_PATH}" ]]; then
  usage >&2
  echo "ERROR: DATA_PATH is required." >&2
  exit 1
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "ERROR: DATA_PATH not found: ${DATA_PATH}" >&2
  exit 1
fi

_data_base="$(basename "${DATA_PATH}" .h5ad)"
PRIOR_PATH="${PRIOR_PATH:-${PRISM_RESULTS}/priors/${_data_base}.json}"
LOG_PATH="${LOG_PATH:-${PRIOR_PATH%.json}.build.log}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
  echo "Loaded env from ${ENV_FILE}"
fi

if [[ "${PRIOR_MODE}" == "auto" ]]; then
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    PRIOR_MODE="llm"
  else
    PRIOR_MODE="zero"
    echo "OPENAI_API_KEY not set -> using zero pathway priors (set API key or PRIOR_MODE=llm)"
  fi
fi

if [[ "${PRIOR_MODE}" == "llm" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: PRIOR_MODE=llm requires OPENAI_API_KEY (source ${ENV_FILE})." >&2
  exit 1
fi

mkdir -p "$(dirname "${PRIOR_PATH}")" "$(dirname "${LOG_PATH}")"

if [[ -f "${PRIOR_PATH}" && ! -f "${PRIOR_PATH}.bak" && "${PRIOR_MODE}" == "llm" ]]; then
  cp "${PRIOR_PATH}" "${PRIOR_PATH}.bak"
  echo "Backed up existing priors to ${PRIOR_PATH}.bak"
fi

CONCURRENCY="${CONCURRENCY:-4}"
SAVE_EVERY="${SAVE_EVERY:-10}"
REPLACE_ZERO="${REPLACE_ZERO:-true}"
RESUME="${RESUME:-true}"

_extra_args=()
if [[ -n "${TEST_ADATA_PATH}" ]]; then
  _extra_args+=(--extra_adata_paths "${TEST_ADATA_PATH}")
fi

echo "Building pathway priors (mode=${PRIOR_MODE})"
echo "  train  : ${DATA_PATH}"
if [[ -n "${TEST_ADATA_PATH}" ]]; then
  echo "  extra  : ${TEST_ADATA_PATH}"
fi
echo "  output : ${PRIOR_PATH}"
echo "  log    : ${LOG_PATH}"
if [[ "${PRIOR_MODE}" == "llm" ]]; then
  echo "  model  : ${LLM_MODEL:-deepseek-v4-flash}"
  echo "  workers: ${CONCURRENCY}"
fi

_llm_args=()
if [[ "${PRIOR_MODE}" == "llm" ]]; then
  _llm_args+=(
    --replace_zero "${REPLACE_ZERO}"
    --resume "${RESUME}"
    --concurrency "${CONCURRENCY}"
    --save_every "${SAVE_EVERY}"
    --env_file "${ENV_FILE}"
  )
  if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
    _llm_args+=(--llm_api_url "${OPENAI_BASE_URL}")
  fi
  if [[ -n "${LLM_MODEL:-}" ]]; then
    _llm_args+=(--llm_model "${LLM_MODEL}")
  fi
fi

python "${PRISM_SCRIPTS}/build_llm_pathway_priors.py" \
  --adata_path "${DATA_PATH}" \
  "${_extra_args[@]}" \
  --output_path "${PRIOR_PATH}" \
  --mode "${PRIOR_MODE}" \
  "${_llm_args[@]}" \
  2>&1 | tee -a "${LOG_PATH}"
