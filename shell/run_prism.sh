#!/usr/bin/env bash
# Unified Prism pipeline: build priors (optional) -> train -> infer.
#
# Examples:
#   DATASET=sciplex SPLIT=drug_split_0 PRIOR=deepseek bash prism/shell/run_prism.sh
#   DATASET=sciplex PRIOR=qwen bash prism/shell/run_prism.sh
#   DATASET=sciplex PRIOR=zero bash prism/shell/run_prism.sh
#   DATASET=lincs SPLIT=drug_split_1 PRIOR=zero bash prism/shell/run_prism.sh
#   DATASET=lincs PRIOR=deepseek BUILD_PRIORS=true CONCURRENCY=10 bash prism/shell/run_prism.sh
#   PRIOR_PATH=/custom/priors.json DATA_PATH=/custom/train.h5ad ... bash prism/shell/run_prism.sh
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prism_init.sh"

usage() {
  cat <<EOF
Usage: bash $0

Prism full pipeline with selectable dataset and pathway priors.

Dataset (pick one):
  DATASET            sciplex | lincs (default: sciplex)
  SPLIT              e.g. drug_split_0, drug_split_1 (default: drug_split_0)

Priors:
  PRIOR              deepseek | qwen | zero | random (default: deepseek)
                     Uses pre-built JSON under prism/resources/ when available.
  PRIOR_PATH         Override prior JSON path (skips PRIOR preset)
  BUILD_PRIORS       true | false | auto (default: auto)
                     auto = build only if PRIOR_PATH missing or PRIOR needs generation
  PRIOR_MODE         llm | zero | random (only when building; overrides PRIOR for build)

Training:
  RUN_NAME           Checkpoint / results run name (auto-derived if unset)
  LR_ANNEAL_STEPS    Training steps (sciplex default: 5000, lincs: 50000)
  BATCH_SIZE         Batch size (sciplex: 2048, lincs: 4096)
  MICROBATCH         Microbatch for lincs-style training (default: 2048)

Paths (override dataset presets):
  DATA_PATH, CONTROL_DATA_PATH, TEST_ADATA_PATH, TEST_CONTROL_ADATA_PATH
  ENV_FILE / PRISM_ENV_FILE   LLM API env file (default: config/llm.env)
EOF
}

DATASET="${DATASET:-sciplex}"
SPLIT="${SPLIT:-drug_split_0}"
PRIOR="${PRIOR:-deepseek}"
BUILD_PRIORS="${BUILD_PRIORS:-auto}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# --- dataset paths ---
case "${DATASET}" in
  sciplex)
    _data_root="${DATASETS_DIR}"
    _prefix="sci_plex"
    LR_ANNEAL_STEPS="${LR_ANNEAL_STEPS:-5000}"
    BATCH_SIZE="${BATCH_SIZE:-2048}"
    LOG_INTERVAL="${LOG_INTERVAL:-100}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
    MICROBATCH="${MICROBATCH:-${BATCH_SIZE}}"
    ;;
  lincs)
    _data_root="${DATASETS_LINCS_DIR}"
    _prefix="lincs_l1000"
    LR_ANNEAL_STEPS="${LR_ANNEAL_STEPS:-50000}"
    BATCH_SIZE="${BATCH_SIZE:-4096}"
    LOG_INTERVAL="${LOG_INTERVAL:-500}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
    MICROBATCH="${MICROBATCH:-2048}"
    ;;
  *)
    echo "ERROR: DATASET must be sciplex or lincs (got: ${DATASET})" >&2
    usage >&2
    exit 1
    ;;
esac

DATA_PATH="${DATA_PATH:-${_data_root}/${_prefix}_train_${SPLIT}.h5ad}"
CONTROL_DATA_PATH="${CONTROL_DATA_PATH:-${_data_root}/${_prefix}_train_${SPLIT}_control.h5ad}"
TEST_ADATA_PATH="${TEST_ADATA_PATH:-${_data_root}/${_prefix}_test_${SPLIT}.h5ad}"
TEST_CONTROL_ADATA_PATH="${TEST_CONTROL_ADATA_PATH:-${_data_root}/${_prefix}_test_${SPLIT}_control.h5ad}"

# --- prior path & env ---
_need_build=false
_build_mode=""

if [[ -n "${PRIOR_PATH:-}" ]]; then
  : # user override
else
  case "${PRIOR}" in
    deepseek)
      ENV_FILE="$(resolve_prism_llm_env)"
      if [[ "${DATASET}" == "sciplex" ]]; then
        PRIOR_PATH="${PRISM_RESOURCES}/deepseek/llm_pathway_priors_sci_plex.json"
      else
        PRIOR_PATH="${PRISM_RESULTS}/priors/lincs_${SPLIT}_deepseek.json"
      fi
      _build_mode="llm"
      ;;
    qwen)
      ENV_FILE="$(resolve_prism_llm_env)"
      if [[ "${DATASET}" == "sciplex" ]]; then
        PRIOR_PATH="${PRISM_RESOURCES}/qwen/llm_pathway_priors_sci_plex.json"
      else
        PRIOR_PATH="${PRISM_RESULTS}/priors/lincs_${SPLIT}_qwen.json"
      fi
      _build_mode="llm"
      ;;
    zero)
      ENV_FILE=""
      if [[ "${DATASET}" == "sciplex" ]]; then
        PRIOR_PATH="${PRISM_RESOURCES}/llm_pathway_priors_sci_plex_zero.json"
      else
        PRIOR_PATH="${PRISM_RESULTS}/priors/lincs_${SPLIT}_zero.json"
      fi
      _build_mode="zero"
      ;;
    random)
      ENV_FILE=""
      if [[ "${DATASET}" == "sciplex" ]]; then
        PRIOR_PATH="${PRISM_RESOURCES}/llm_pathway_priors_sci_plex_random.json"
      else
        PRIOR_PATH="${PRISM_RESULTS}/priors/lincs_${SPLIT}_random.json"
      fi
      _build_mode="random"
      ;;
    *)
      echo "ERROR: PRIOR must be deepseek | qwen | zero | random (got: ${PRIOR})" >&2
      usage >&2
      exit 1
      ;;
  esac
fi

if [[ -n "${PRIOR_MODE:-}" ]]; then
  _build_mode="${PRIOR_MODE}"
fi

# --- load LLM env when needed ---
if [[ -f "${ENV_FILE:-}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
  echo "Loaded env from ${ENV_FILE}"
fi

# --- decide whether to build priors ---
case "${BUILD_PRIORS}" in
  true|1|yes)
    _need_build=true
    ;;
  false|0|no)
    _need_build=false
    ;;
  auto)
    if [[ ! -f "${PRIOR_PATH}" ]]; then
      _need_build=true
    elif [[ "${PRIOR}" == "deepseek" || "${PRIOR}" == "qwen" ]]; then
      _need_build=false
    elif [[ "${PRIOR}" == "zero" || "${PRIOR}" == "random" ]]; then
      _need_build=false
    fi
    ;;
  *)
    echo "ERROR: BUILD_PRIORS must be true | false | auto" >&2
    exit 1
    ;;
esac

if [[ "${_need_build}" == "true" && "${_build_mode}" == "llm" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: BUILD_PRIORS requires OPENAI_API_KEY for PRIOR=${PRIOR} (source ${ENV_FILE})." >&2
  exit 1
fi

if [[ "${DATASET}" == "lincs" && "${_need_build}" == "true" && "${_build_mode}" == "llm" ]]; then
  echo "WARNING: LINCS has ~90k drug-cell pairs; LLM prior build is very slow."
fi

# --- run dirs ---
RUN_NAME="${RUN_NAME:-prism_${DATASET}_${SPLIT}_${PRIOR}}"
CKPT_DIR="${CKPT_DIR:-${PRISM_CHECKPOINTS}/${RUN_NAME}}"
PRED_DIR="${PRED_DIR:-${PRISM_RESULTS}/${RUN_NAME}/predictions}"
LOGGER_SUBDIR="${LOGGER_SUBDIR:-prism_${DATASET}_${SPLIT}_${PRIOR}}"

mkdir -p "$(dirname "${PRIOR_PATH}")" "${CKPT_DIR}" "${PRED_DIR}" "${PRISM_LOGS}"

echo "========== Prism pipeline =========="
echo "  dataset : ${DATASET} (${SPLIT})"
echo "  prior   : ${PRIOR} -> ${PRIOR_PATH}"
echo "  build   : ${_need_build} (mode=${_build_mode})"
echo "  train   : ${DATA_PATH}"
echo "  test    : ${TEST_ADATA_PATH}"
echo "  ckpt    : ${CKPT_DIR}"
echo "  preds   : ${PRED_DIR}"
echo "  steps   : ${LR_ANNEAL_STEPS}"
echo "==================================="

# --- step 1: priors ---
if [[ "${_need_build}" == "true" ]]; then
  echo "========== Step 1: build pathway priors (${_build_mode}) =========="
  if [[ -z "${CONCURRENCY:-}" ]]; then
    if [[ "${DATASET}" == "lincs" && "${_build_mode}" == "llm" ]]; then
      CONCURRENCY=10
    else
      CONCURRENCY=4
    fi
  fi
  DATA_PATH="${DATA_PATH}" \
  TEST_ADATA_PATH="${TEST_ADATA_PATH}" \
  PRIOR_PATH="${PRIOR_PATH}" \
  PRIOR_MODE="${_build_mode}" \
  ENV_FILE="${ENV_FILE:-}" \
  PRISM_ENV_FILE="${ENV_FILE:-}" \
  CONCURRENCY="${CONCURRENCY}" \
  SAVE_EVERY="${SAVE_EVERY:-10}" \
  bash "${PRISM_SHELL}/build_llm_priors.sh"
else
  echo "========== Step 1: skip prior build (using ${PRIOR_PATH}) =========="
  if [[ ! -f "${PRIOR_PATH}" ]]; then
    echo "ERROR: Prior file not found: ${PRIOR_PATH}" >&2
    echo "Set BUILD_PRIORS=true to generate it." >&2
    exit 1
  fi
fi

# --- step 2: train ---
echo "========== Step 2: train Prism (${LR_ANNEAL_STEPS} steps) =========="
python "${PRISM_SCRIPTS}/train_prism.py" \
  --logger_path "${PRISM_LOGS}/logger_files/${LOGGER_SUBDIR}" \
  --data_path "${DATA_PATH}" \
  --control_data_path "${CONTROL_DATA_PATH}" \
  --pathway_prior_path "${PRIOR_PATH}" \
  --resume_checkpoint "${CKPT_DIR}" \
  --gene_size "${GENE_SIZE:-200}" \
  --output_dim "${OUTPUT_DIM:-200}" \
  --batch_size "${BATCH_SIZE}" \
  --microbatch "${MICROBATCH}" \
  --lr_anneal_steps "${LR_ANNEAL_STEPS}" \
  --log_interval "${LOG_INTERVAL}" \
  --save_interval "${SAVE_INTERVAL}" \
  --pathway_loss_weight "${PATHWAY_LOSS_WEIGHT:-1}"

# --- step 3: infer ---
echo "========== Step 3: inference =========="
python "${PRISM_SCRIPTS}/infer_prism.py" \
  --model_path "${CKPT_DIR}/model.pt" \
  --pathway_prior_path "${PRIOR_PATH}" \
  --test_adata_path "${TEST_ADATA_PATH}" \
  --control_adata_path "${TEST_CONTROL_ADATA_PATH}" \
  --output_dir "${PRED_DIR}"

echo "Done."
echo "  priors : ${PRIOR_PATH}"
echo "  ckpt   : ${CKPT_DIR}/model.pt"
echo "  preds  : ${PRED_DIR}/perturbed_expression_prism.npy"
