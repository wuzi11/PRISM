#!/usr/bin/env bash
# Standalone Prism environment init.
# Source from prism/shell/*.sh scripts.

_prism_shell="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_PKG="$(cd "${_prism_shell}/.." && pwd)"
_PRISM_PARENT="$(dirname "${PRISM_PKG}")"
PROJECT_ROOT="${PROJECT_ROOT:-${PRISM_PKG}}"

export PRISM_PKG PROJECT_ROOT
export PYTHONPATH="${_PRISM_PARENT}${PYTHONPATH:+:${PYTHONPATH}}"

# shellcheck disable=SC1091
source "${_prism_shell}/prism_paths.sh"

cd "${_PRISM_PARENT}"
unset _prism_shell _PRISM_PARENT
