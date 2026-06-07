import argparse
import os
import random
from importlib.metadata import version

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.data import get_loaders, test_ppl
from lib.layer_select import select_layer

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

print("torch", version("torch"))
print("transformers", version("transformers"))
print("# of gpus:", torch.cuda.device_count())


def get_llm(model_name, cache_dir="llm_weights"):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="cuda",
        attn_implementation="sdpa",
    )


@torch.no_grad()
def evaluate(model, tokenizer, args, save_folder_path):
    model.config.use_cache = False
    model.config.output_attentions = False
    model.config.output_hidden_states = False
    model.config.return_dict = True
    model.eval()
    ppl_short = {"ptb": "ptb", "wikitext2": "wiki", "c4": "c4"}

    if args.eval_ppl:
        datasets = ["ptb", "wikitext2", "c4"]
        ppl_results = test_ppl(model, tokenizer, datasets, args.ppl_seqlen)
        for dataset, ppl_val in ppl_results.items():
            print(f"{dataset} perplexity: {ppl_val:.2f}")
            with open(os.path.join(save_folder_path,
                                    f"ppl_{ppl_short[dataset]}.txt"), "w") as f:
                f.write(f"{ppl_val:.4f}\n")

    if args.eval_tasks:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table

        task_list = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]
        wrapped = HFLM(pretrained=model, tokenizer=tokenizer,
                       batch_size=8, device="cuda")
        results = lm_eval.simple_evaluate(model=wrapped, tasks=task_list,
                                          batch_size=1, num_fewshot=0)
        print(make_table(results))
        for task in task_list:
            tr = results.get("results", {}).get(task)
            if tr is None:
                continue
            val = tr.get("acc_norm,none", tr.get("acc,none", None))
            if val is None:
                continue
            with open(os.path.join(save_folder_path, f"{task}.txt"), "w") as f:
                f.write(f"{val:.6f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model id, e.g. meta-llama/Llama-3.1-8B")
    parser.add_argument("--cache_dir", type=str, default="llm_weights",
                        help="Local HF cache directory")

    parser.add_argument("--pruning_method", type=str, default="lorp",
                        choices=["none", "lorp", "streamline", "shortgpt"],
                        help=("none      : no pruning, dense evaluation only\n"
                              "lorp      : LoRP — Locality-Aware Redundancy Pruning (ours)\n"
                              "streamline: LLM-Streamline (contiguous block)\n"
                              "shortgpt  : ShortGPT (Block Influence)"))

    parser.add_argument("--total_num_prune", type=int, default=7,
                        help="Number of transformer blocks to remove")
    parser.add_argument("--num_clusters", type=int, default=None,
                        help="LoRP: override the RLS-guided cluster count K "
                             "(default None = choose K from RLS)")
    parser.add_argument("--calibration_data", type=str, default="c4",
                        choices=["wikitext2", "c4", "ptb"])
    parser.add_argument("--train_size", type=int, default=128,
                        help="Number of calibration sequences")
    parser.add_argument("--val_size", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Calibration sequence length")
    parser.add_argument("--ppl_seqlen", type=int, default=2048,
                        help="Sequence length for PPL evaluation")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--eval_ppl", default=False, action="store_true")
    parser.add_argument("--eval_tasks", type=str,
                        default="arc_easy,arc_challenge,hellaswag,winogrande,"
                                "boolq,openbookqa,rte,copa,race",
                        help="Comma-separated lm-eval-harness task list "
                             "(empty string disables downstream eval)")
    parser.add_argument("--outdir", type=str, default="results",
                        help="Output root")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Sub-directory under outdir for this run")

    args = parser.parse_args()
    save_folder_path = os.path.join(args.outdir, args.exp_name)
    os.makedirs(save_folder_path, exist_ok=True)
    print(f"[OUT] {save_folder_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print("=" * 77)
    print(f"model={args.model}  pruning_method={args.pruning_method}  "
          f"k={args.total_num_prune}")
    print("=" * 77)

    print("[LOAD] model")
    model = get_llm(args.model, args.cache_dir).to("cuda")
    model.eval()
    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, legacy=False)
    for p in model.parameters():
        p.requires_grad = False

    if args.pruning_method == "none":
        print("[Dense] evaluating un-pruned model")
        evaluate(model, tokenizer, args, save_folder_path)
        return

    trainloader, _ = get_loaders(
        name=args.calibration_data,
        tokenizer=tokenizer,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        seqlen=args.seqlen,
    )

    init_num_layer = len(model.model.layers)
    before_params = sum(p.numel() for p in model.parameters())

    layer, start_l, end_l, score, t_select, pruned_indices = select_layer(
        model, trainloader, args.total_num_prune, device,
        pruning_method=args.pruning_method,
        num_clusters=args.num_clusters,
    )

    if pruned_indices is not None:
        keep = set(range(len(model.model.layers))) - set(pruned_indices)
        model.model.layers = torch.nn.ModuleList(
            [l for i, l in enumerate(model.model.layers) if i in keep])
    else:
        model.model.layers = torch.nn.ModuleList(
            [l for i, l in enumerate(model.model.layers)
             if i < start_l or i >= end_l])

    after_params = sum(p.numel() for p in model.parameters())
    print(f"Layers: {init_num_layer} -> {len(model.model.layers)}  "
          f"Params: {before_params/1e9:.3f}B -> {after_params/1e9:.3f}B "
          f"(reduced by {100 - 100.0 * after_params / before_params:.2f}%)")

    print("[Pruned] evaluating pruned model (no recovery)")
    evaluate(model, tokenizer, args, save_folder_path)


if __name__ == "__main__":
    main()
