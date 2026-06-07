# LoRP: Locality-Aware Redundancy Pruning for LLM Depth Compression

Training-free, **one-shot, no-recovery** depth pruning for large language models.
LoRP adapts its pruning pattern to each architecture by measuring how
inter-layer redundancy is distributed across depth — *localized* in some
families (Llama, OLMo, Mistral) and more *globally distributed* in others
(Qwen) — and allocating the prune budget accordingly.

<img width="100%" height="246" alt="Image" src="https://github.com/user-attachments/assets/866dfc9a-d87d-42ee-9f26-24d12f14b6f0" />

The key quantity is the **Representation Locality Score (RLS)**, a closed-form,
hyperparameter-free property of the inter-layer cosine geometry:

```
RLS(S) = − log₂( S̄_off ),     S̄_off = (2 / N(N−1)) · Σ_{i<j} S_ij
```

where `S ∈ ℝ^{N×N}` is the pairwise per-token cosine-similarity matrix between
transformer-block hidden states, captured on a small calibration corpus.
A **higher** RLS means similarity decays faster across depth (localized
redundancy); a **lower** RLS means more globally distributed redundancy.

This repository implements LoRP and reproduces every baseline used in the
paper. Everything is **one-shot, training-free, and recovery-free** — no LoRA,
no distillation, no fine-tuning.

---

## Method (LoRP)

1. **Similarity capture.** Forward-hook every block, ℓ2-normalize token hidden
   states (eq. 3), and average token-wise cosine similarity into `S` (eq. 4).
2. **Representation Locality Score.** `RLS = −log₂(S̄_off)` (eqs. 5–6).
3. **RLS-guided granularity K.** Choose the clustering granularity from RLS:

   | RLS range | K | example models |
   |---|---|---|
   | `RLS ≥ 1.0` | 2 | Llama-3.1-8B (1.149) |
   | `0.7 ≤ RLS < 1.0` | 3 | OLMo-3-7B (0.941), Mistral-Nemo-12B (0.926) |
   | `RLS < 0.7` | 4 | Qwen3-8B (0.685), Qwen3-14B (0.644) |

   (override with `--num_clusters K`).
4. **Layer clustering.** Spectral clustering on the affinity `A = (S+1)/2`
   (eq. 7) into `K` representational phases, re-indexed by depth.
5. **Two-stage redundancy-aware allocation.**
   - *Stage 1 — coverage-aware init:* prune the most-redundant eligible layer of
     each cluster (eqs. 8–9), spreading decisions across phases.
   - *Stage 2 — residual allocation:* repeatedly prune the most-redundant layer
     from the cluster with the largest residual intra-cluster redundancy
     (eqs. 10–11) until the budget is met.

   Boundary layers `{0, N−1}` are protected. Architectures with localized
   redundancy keep pruning within one cluster; distributed ones spread across
   clusters.

---

## Repository layout

```
.
├── main.py                 Pruning + evaluation entry point
├── lib/
│   ├── data.py             Calibration data loaders (C4 / WikiText-2 / PTB)
│   └── layer_select.py     LoRP and all baseline selectors
├── scripts/
│   ├── _common.sh          Shared model/budget sweep config
│   ├── lorp.sh             LoRP (ours)
│   ├── streamline.sh       LLM-Streamline
│   ├── shortgpt.sh         ShortGPT
│   └── dense.sh            Dense reference
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt   # torch, transformers, datasets, lm-eval, scikit-learn, numpy, tqdm
```

Spectral clustering uses `scikit-learn`. Downstream evaluation uses
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
All experiments in the paper run on a single NVIDIA A40 48GB GPU (fp16).

---

## Usage

```bash
python main.py \
    --model meta-llama/Llama-3.1-8B \
    --pruning_method lorp \
    --total_num_prune 7 \
    --calibration_data c4 --train_size 128 --seqlen 2048 \
    --eval_ppl \
    --eval_tasks arc_easy,arc_challenge,hellaswag,winogrande,boolq,openbookqa,rte,copa,race \
    --exp_name llama3.1_7_lorp
```

Per-task accuracy and per-dataset perplexity are written as individual `.txt`
files under `results/<exp_name>/`.

| Flag | Meaning |
|---|---|
| `--pruning_method` | `lorp` \| `streamline` \| `shortgpt` \| `none` |
| `--total_num_prune` | number of blocks to remove (e.g. 7, 9, 11, 13, 15) |
| `--num_clusters` | LoRP only: force K (default = RLS-guided) |
| `--calibration_data` | `c4` (default) \| `wikitext2` \| `ptb` |
| `--eval_ppl` | report WikiText-2 / C4 / PTB perplexity |
| `--eval_tasks` | lm-eval-harness task list (empty disables) |

### Reproduce the full sweep

Each script sweeps the paper's five models × prune budgets and is resume-safe
(skips runs whose 12 eval files already exist):

```bash
bash scripts/dense.sh        # dense references
bash scripts/lorp.sh         # LoRP (ours)
bash scripts/shortgpt.sh
bash scripts/streamline.sh
```

Override the GPU / cache / conda env via environment variables, e.g.
`CUDA_VISIBLE_DEVICES=1 CONDA_ENV=myenv CACHE_DIR=/data/hf bash scripts/lorp.sh`.

---

## Results (from the paper)

### Perplexity (↓), no recovery — average of WikiText-2 / C4 / PTB (Table 2)

| Model | Lp/Lt | ShortGPT | LLM-Streamline | **LoRP (ours)** |
|---|---|---|---|---|
| LLaMA-3.1-8B | 7/32 | 67.21 | ≥2000 | **41.95** |
| OLMo-3-7B | 7/32 | 30.02 | 30.02 | **24.82** |
| Qwen3-8B | 7/36 | 262.67 | 229.11 | **27.92** |
| Qwen3-14B | 11/40 | 276.50 | ≥2000 | **37.44** |
| Mistral-Nemo-12B | 11/40 | 357.93 | 357.93 | **50.86** |

### Zero-shot accuracy (↑), 9 commonsense tasks, AVG (Table 3)

| Model | Lp/Lt | ShortGPT | LLM-Streamline | **LoRP (ours)** |
|---|---|---|---|---|
| LLaMA-3.1-8B | 7/32 | 57.10 | 42.59 | **57.21** |
| OLMo-3-7B | 7/32 | 58.73 | 58.73 | **60.38** |
| Qwen3-8B | 7/36 | 51.50 | 54.07 | **56.65** |
| Qwen3-14B | 11/40 | 50.53 | 46.51 | **52.84** |
| Mistral-Nemo-12B | 11/40 | 54.49 | 54.49 | **55.08** |

LoRP gives the lowest average perplexity and the highest (or competitive)
downstream accuracy across architectures, with the largest gains on Qwen
models whose redundancy is most globally distributed. Depth pruning also
translates into near-linear inference speedups (Qwen3-14B: 1.36×–1.56× at
Lp=11–15, −24%…−33% peak memory; paper Appendix A).


<img width="100%" height="438" alt="Image" src="https://github.com/user-attachments/assets/8999c747-edab-46e3-9484-7dab6426226c" />

---

## Citation

```bibtex
@article{yun2026lorp,
  title   = {Locality-Aware Redundancy Pruning for LLM Depth Compression},
  author  = {Yun, Vincent-Daniel and Kim, Youngrae and Lim, Woosang and
             Heo, Youngjin and Kim, Minkyu and Lee, Sunwoo},
  journal = {arXiv preprint arXiv:2605.27786},
  year    = {2026}
}
```

### Baselines

- **ShortGPT** — Men et al., Block Influence: https://github.com/sramshetty/ShortGPT
- **LLM-Streamline** — Chen et al.: https://github.com/RUCKBReasoning/LLM-Streamline

All models and datasets are used under their respective licenses for
non-commercial research only (see the paper, Appendix B).
