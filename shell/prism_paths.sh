#!/usr/bin/env bash
# Prism package path helpers. Source via prism_init.sh.

: "${PRISM_PKG:?Source prism_init.sh first}"

PROJECT_ROOT="${PROJECT_ROOT:-${PRISM_PKG}}"

PRISM_SCRIPTS="${PRISM_PKG}/scripts"
PRISM_SHELL="${PRISM_PKG}/shell"
PRISM_CONFIG="${PRISM_PKG}/config"
PRISM_RESOURCES="${PRISM_PKG}/resources"
PRISM_CHECKPOINTS="${PRISM_PKG}/checkpoints"
PRISM_RESULTS="${PRISM_PKG}/results"
PRISM_LOGS="${PRISM_PKG}/logs"
PRISM_LLM_ENV="${PRISM_CONFIG}/llm.env"

DATASETS_DIR="${DATASETS_DIR:-${PRISM_PKG}/datasets}"
DATASETS_LINCS_DIR="${DATASETS_LINCS_DIR:-${PRISM_PKG}/datasets_lincs}"

export PRISM_PKG PRISM_SCRIPTS PRISM_SHELL PRISM_CONFIG PRISM_RESOURCES
export PRISM_CHECKPOINTS PRISM_RESULTS PRISM_LOGS PRISM_LLM_ENV
export DATASETS_DIR DATASETS_LINCS_DIR

resolve_prism_llm_env() {
  if [[ -n "${PRISM_ENV_FILE:-}" && -f "${PRISM_ENV_FILE}" ]]; then
    echo "${PRISM_ENV_FILE}"
  elif [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then
    echo "${ENV_FILE}"
  elif [[ -f "${PRISM_LLM_ENV}" ]]; then
    echo "${PRISM_LLM_ENV}"
  else
    echo "${PRISM_LLM_ENV}"
  fi
}
