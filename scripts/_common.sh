#!/usr/bin/env bash
# Common sweep config sourced by every method script.
#
# All sweep scripts iterate (model x prune_budget) and run main.py.
# Resume-safe: experiments whose 12 expected eval files already exist
# are skipped.
#
# Override via environment variables before invocation:
#   REPO_DIR     repo root (default: parent dir of scripts/)
#   CACHE_DIR    HuggingFace cache (default: $REPO_DIR/llm_weights)
#   OUTROOT      results root (default: results)
#   CONDA_ENV    conda env (default: psamp)
#   CUDA_VISIBLE_DEVICES   GPU index (default: 0)

set -e

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/llm_weights}"
OUTROOT="${OUTROOT:-results}"
CONDA_ENV="${CONDA_ENV:-psamp}"
cd "$REPO_DIR"

if [ -z "${CONDA_BASE:-}" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [ -z "$CONDA_BASE" ]; then
    for c in /opt/conda "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
      [ -d "$c" ] && CONDA_BASE="$c" && break
    done
  fi
fi
# shellcheck disable=SC1091
[ -n "$CONDA_BASE" ] && source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export HF_DATASETS_TRUST_REMOTE_CODE=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Models and prune budgets from the LoRP paper (Locality-Aware Redundancy
# Pruning). Format:  hf_id : shortname : "budget budget ..."
MODELS_SPEC=(
  "meta-llama/Llama-3.1-8B:llama3.1:7 9"                    # N=32
  "allenai/Olmo-3-1025-7B:olmo3:7 9"                        # N=32
  "Qwen/Qwen3-8B:qwen3-8b:7 9"                              # N=36
  "Qwen/Qwen3-14B:qwen3-14b:11 13 15"                       # N=40
  "mistralai/Mistral-Nemo-Base-2407:mistral-nemo:11 13 15"  # N=40
)

EVAL_TASKS="arc_easy,arc_challenge,hellaswag,winogrande,boolq,openbookqa,rte,copa,race"
EXPECTED=(ppl_c4.txt ppl_wiki.txt ppl_ptb.txt arc_easy.txt arc_challenge.txt \
          hellaswag.txt winogrande.txt boolq.txt openbookqa.txt rte.txt copa.txt race.txt)

mkdir -p "$OUTROOT"

run_method() {
  local METHOD="$1"
  echo "================================================================"
  echo "[$METHOD SWEEP]  outroot=$OUTROOT  cache=$CACHE_DIR  conda=$CONDA_ENV"
  echo "================================================================"

  for spec in "${MODELS_SPEC[@]}"; do
    IFS=':' read -r model shortname prunes_str <<< "$spec"
    read -r -a prunes <<< "$prunes_str"

    for prune in "${prunes[@]}"; do
      exp="${shortname}_${prune}_${METHOD}"
      dir="${OUTROOT}/${exp}"
      mkdir -p "$dir"

      local all_done=true
      for f in "${EXPECTED[@]}"; do
        [ -s "$dir/$f" ] || { all_done=false; break; }
      done
      if [ "$all_done" = "true" ]; then
        echo "[SKIP] $exp"
        continue
      fi

      echo "[RUN]  $exp  ($model)"
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      python main.py \
        --model "$model" \
        --cache_dir "$CACHE_DIR" \
        --pruning_method "$METHOD" \
        --calibration_data c4 \
        --train_size 128 \
        --val_size 16 \
        --seqlen 2048 \
        --ppl_seqlen 2048 \
        --total_num_prune "$prune" \
        --eval_ppl \
        --eval_tasks "$EVAL_TASKS" \
        --outdir "$OUTROOT" \
        --exp_name "$exp" 2>&1 | tee "${dir}/run.log"
    done
  done

  echo "[DONE] $METHOD sweep"
}

run_dense() {
  echo "================================================================"
  echo "[dense SWEEP]  outroot=$OUTROOT  cache=$CACHE_DIR  conda=$CONDA_ENV"
  echo "================================================================"

  for spec in "${MODELS_SPEC[@]}"; do
    IFS=':' read -r model shortname _ <<< "$spec"
    exp="${shortname}_dense"
    dir="${OUTROOT}/${exp}"
    mkdir -p "$dir"

    local all_done=true
    for f in "${EXPECTED[@]}"; do
      [ -s "$dir/$f" ] || { all_done=false; break; }
    done
    if [ "$all_done" = "true" ]; then
      echo "[SKIP] $exp"
      continue
    fi

    echo "[RUN]  $exp  ($model)"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python main.py \
      --model "$model" \
      --cache_dir "$CACHE_DIR" \
      --pruning_method none \
      --total_num_prune 0 \
      --eval_ppl \
      --eval_tasks "$EVAL_TASKS" \
      --outdir "$OUTROOT" \
      --exp_name "$exp" 2>&1 | tee "${dir}/run.log"
  done

  echo "[DONE] dense sweep"
}
