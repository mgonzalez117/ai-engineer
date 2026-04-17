#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[vllm-entrypoint] $*" >&2
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

resolve_shell_placeholder() {
  local raw_value="$1"

  if [[ "$raw_value" =~ ^\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}$ ]]; then
    local ref_var="${BASH_REMATCH[1]}"
    local with_default="${BASH_REMATCH[2]-}"
    local default_value="${BASH_REMATCH[3]-}"
    local resolved="${!ref_var-}"

    if [[ -z "$resolved" && -n "$with_default" ]]; then
      resolved="$default_value"
    fi

    log "Resolved placeholder value from ${raw_value} to ${resolved:-<empty>}"
    printf '%s' "$resolved"
    return
  fi

  printf '%s' "$raw_value"
}

resolve_python_bin() {
  if command -v python >/dev/null 2>&1; then
    printf '%s' "python"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "python3"
    return
  fi

  return 1
}

sanitize_lora_name() {
  local raw_name="$1"
  local sanitized
  sanitized="$(printf '%s' "$raw_name" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_.-]+/-/g; s/^-+//; s/-+$//')"
  printf '%s' "$sanitized"
}

require_env "MODEL_NAME"
require_env "VLLM_API_KEY"

MAX_LORAS="${MAX_LORAS:-4}"
if ! [[ "$MAX_LORAS" =~ ^[0-9]+$ ]] || [[ "$MAX_LORAS" -lt 1 ]]; then
  log "ERROR: MAX_LORAS must be a positive integer, got: ${MAX_LORAS}"
  exit 1
fi

PYTHON_BIN="$(resolve_python_bin || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  log "ERROR: Neither 'python' nor 'python3' was found in PATH."
  exit 1
fi

raw_lora_modules="${LORA_MODULES:-}"
resolved_lora_modules="$(resolve_shell_placeholder "$raw_lora_modules")"

log "Starting vLLM OpenAI server with:"
log "MODEL_NAME=${MODEL_NAME}"
log "VLLM_API_KEY=$(mask_secret "$VLLM_API_KEY")"
log "MAX_LORAS=${MAX_LORAS}"
log "PYTHON_BIN=${PYTHON_BIN}"
log "LORA_MODULES=${resolved_lora_modules:-<empty>}"

args=(
  --host "0.0.0.0"
  --port "${PORT:-8000}"
  --model "$MODEL_NAME"
  --api-key "$VLLM_API_KEY"
)

if [[ -n "$resolved_lora_modules" ]]; then
  lora_modules_count=0
  lora_modules_args=()

  if [[ "$resolved_lora_modules" =~ ^[[:space:]]*\{.*\}[[:space:]]*$ ]] || [[ "$resolved_lora_modules" =~ ^[[:space:]]*\[.*\][[:space:]]*$ ]]; then
    lora_modules_args+=("$resolved_lora_modules")
    lora_modules_count=1
  else
    normalized_lora_modules="${resolved_lora_modules//,/ }"
    read -r -a lora_modules_array <<< "${normalized_lora_modules}"

    if ((${#lora_modules_array[@]} == 0)); then
      log "ERROR: LORA_MODULES is set but empty after parsing."
      exit 1
    fi

    for i in "${!lora_modules_array[@]}"; do
      lora_item="${lora_modules_array[$i]}"

      if [[ "$lora_item" == *=* ]]; then
        lora_modules_args+=("$lora_item")
      else
        raw_name="${lora_item##*/}"
        lora_name="$(sanitize_lora_name "$raw_name")"
        if [[ -z "$lora_name" ]]; then
          lora_name="adapter$((i + 1))"
        fi
        normalized_item="${lora_name}=${lora_item}"
        log "Auto-normalized bare LoRA value '${lora_item}' to '${normalized_item}'"
        lora_modules_args+=("$normalized_item")
      fi
    done

    lora_modules_count=${#lora_modules_args[@]}
  fi

  if ((lora_modules_count > MAX_LORAS)); then
    log "ERROR: Number of LoRA modules (${lora_modules_count}) exceeds MAX_LORAS (${MAX_LORAS})."
    exit 1
  fi

  args+=(--enable-lora --max-loras "$MAX_LORAS" --lora-modules "${lora_modules_args[@]}")
else
  log "LoRA disabled (LORA_MODULES not provided)."
fi

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server "${args[@]}" "$@"