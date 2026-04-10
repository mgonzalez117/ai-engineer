#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[vllm-entrypoint] $*"
}

mask_secret() {
  local secret="${1:-}"
  if [[ -z "$secret" ]]; then
    echo "<empty>"
    return
  fi

  if ((${#secret} <= 8)); then
    echo "********"
    return
  fi

  echo "${secret:0:4}***${secret: -4}"
}

require_env() {
  local var_name="$1"
  local value="${!var_name:-}"
  if [[ -z "$value" ]]; then
    log "ERROR: Missing required env var: ${var_name}"
    exit 1
  fi
}

require_env "MODEL_NAME"
require_env "VLLM_API_KEY"

MAX_LORAS="${MAX_LORAS:-4}"
if ! [[ "$MAX_LORAS" =~ ^[0-9]+$ ]] || [[ "$MAX_LORAS" -lt 1 ]]; then
  log "ERROR: MAX_LORAS must be a positive integer, got: ${MAX_LORAS}"
  exit 1
fi

log "Starting vLLM OpenAI server with"
log "MODEL_NAME=${MODEL_NAME}"
log "VLLM_API_KEY=$(mask_secret "$VLLM_API_KEY")"
log "MAX_LORAS=${MAX_LORAS}"
log "LORA_MODULES=${LORA_MODULES:-<empty>}"

args=(
  --host "0.0.0.0"
  --port "${PORT:-8000}"
  --model "$MODEL_NAME"
  --api-key "$VLLM_API_KEY"
)

if [[ -n "${LORA_MODULES:-}" ]]; then
  normalized_lora_modules="${LORA_MODULES//,/ }"
  read -r -a lora_modules_array <<< "${normalized_lora_modules}"

  if ((${#lora_modules_array[@]} == 0)); then
    log "ERROR: LORA_MODULES is set but empty after parsing."
    exit 1
  fi

  if ((${#lora_modules_array[@]} > MAX_LORAS)); then
    log "ERROR: Number of LoRA modules (${#lora_modules_array[@]}) exceeds MAX_LORAS (${MAX_LORAS})."
    exit 1
  fi

  args+=(--enable-lora --max-loras "$MAX_LORAS" --lora-modules "${lora_modules_array[@]}")
else
  log "LoRA disabled (LORA_MODULES not provided)."
fi

exec python -m vllm.entrypoints.openai.api_server "${args[@]}" "$@"