"""Layer-selection criteria for training-free, one-shot depth pruning.

Implements LoRP (ours) and the two training-free baselines compared against it
in "Locality-Aware Redundancy Pruning for LLM Depth Compression":

    lorp        Locality-Aware Redundancy Pruning (ours)
    shortgpt    ShortGPT — Block-Influence score
    streamline  LLM-Streamline — contiguous boundary-similarity block

Every selector returns: (num_to_prune, start_l, end_l, score, pruned_indices)
where `pruned_indices` is an explicit list for non-contiguous methods
(shortgpt, lorp), or `None` for contiguous methods (streamline), in which case
the range [start_l, end_l) is removed.
"""
import time

import numpy as np
import torch
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════════════
# LLM-Streamline — contiguous block maximizing cos(h[s], h[s+L])
# ════════════════════════════════════════════════════════════════════════
def get_pruned_layer_streamline(model, trainloader, num_to_prune, device,
                                latter_barrier=1):
    model = model.to(device)
    num_layers = len(model.model.layers)
    max_start = num_layers - num_to_prune - latter_barrier + 1
    act = [torch.zeros(1).to(device) for _ in range(num_layers)]
    cosine_sim = [torch.zeros(1).to(device) for _ in range(max_start)]

    def hook(module, input, output, layer_name):
        act[layer_name] = input[0]

    handles = [
        layer.register_forward_hook(
            lambda module, input, output, l=l: hook(module, input, output, l))
        for l, layer in enumerate(model.model.layers)]

    num_samples = num_sample = 128
    for _, batch in tqdm(enumerate(trainloader), desc="Selecting (Streamline)",
                         total=num_samples):
        with torch.no_grad():
            try: model(batch[0].to(device))
            except IndexError: pass
        for i in range(1, max_start):
            cosine_sim[i] += torch.cosine_similarity(
                act[i], act[i + num_to_prune], dim=-1).mean()
        num_sample -= 1
        if not num_sample:
            break

    for h in handles: h.remove()

    cosine_sim = [s.item() / num_samples for s in cosine_sim]
    start_l = cosine_sim.index(max(cosine_sim))
    end_l = start_l + num_to_prune
    torch.cuda.empty_cache()
    return num_to_prune, start_l, end_l, max(cosine_sim), None


# ════════════════════════════════════════════════════════════════════════
# ShortGPT — Block-Influence (BI) cosine score
# ════════════════════════════════════════════════════════════════════════
def get_pruned_layer_shortgpt(model, trainloader, num_to_prune, device):
    model = model.to(device)
    num_layers = len(model.model.layers)
    act = [torch.zeros(1).to(device) for _ in range(num_layers)]
    bi_score = [torch.zeros(1).to(device) for _ in range(num_layers)]

    def hook(module, input, output, l):
        act[l] = input[0]

    handles = [
        layer.register_forward_hook(
            lambda module, input, output, l=l: hook(module, input, output, l))
        for l, layer in enumerate(model.model.layers)]

    num_samples = num_sample = 128
    for _, batch in tqdm(enumerate(trainloader), desc="Selecting (ShortGPT)",
                         total=num_samples):
        with torch.no_grad():
            try: model(batch[0].to(device))
            except IndexError: pass
        for i in range(num_layers - 1):
            bi_score[i] += torch.cosine_similarity(act[i], act[i + 1], dim=-1).mean()
        num_sample -= 1
        if not num_sample:
            break

    for h in handles: h.remove()

    # BI = 1 - cos(in, out); higher cosine ⇒ lower influence ⇒ prune first.
    bi_score = [s.item() / num_samples for s in bi_score[:-1]]
    bi_score[0] = -float("inf")  # protect layer 0
    pruned_indices = sorted(
        sorted(range(len(bi_score)), key=lambda i: bi_score[i],
               reverse=True)[:num_to_prune])
    start_l = pruned_indices[0]
    end_l = pruned_indices[-1] + 1
    avg_sim = sum(bi_score[i] for i in pruned_indices) / num_to_prune
    torch.cuda.empty_cache()
    return num_to_prune, start_l, end_l, avg_sim, pruned_indices


# ════════════════════════════════════════════════════════════════════════
# LoRP — Locality-Aware Redundancy Pruning (ours)
# ════════════════════════════════════════════════════════════════════════
def _capture_sim_and_cluster(model, trainloader, device, num_clusters,
                             max_batches, random_state):
    """Capture pairwise per-token cosine similarity matrix S between block
    hidden states and spectral-cluster the affinity A=(S+1)/2 into K phases.

    Returns (sim_matrix [N×N float64], labels [N], cluster_members {k:[layers]}).
    """
    model = model.to(device)
    num_layers = len(model.model.layers)
    act = [None for _ in range(num_layers)]

    def hook(module, input, output, l):
        act[l] = input[0]

    handles = [
        layer.register_forward_hook(
            lambda module, input, output, l=l: hook(module, input, output, l))
        for l, layer in enumerate(model.model.layers)]

    sim_sum = torch.zeros(num_layers, num_layers, device=device, dtype=torch.float32)
    count = 0
    num_sample = max_batches
    for _, batch in tqdm(enumerate(trainloader), desc="Selecting (LoRP)",
                         total=max_batches):
        with torch.no_grad():
            try: model(batch[0].to(device))
            except IndexError: pass
        feats = []
        for l in range(num_layers):
            x = act[l]
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)   # eq. (3)
            feats.append(x.reshape(-1, x.shape[-1]))
        F = torch.stack(feats, dim=0).float()
        S = torch.einsum("ind,jnd->ij", F, F) / F.shape[1]  # eq. (4)
        sim_sum += S
        count += 1
        num_sample -= 1
        if not num_sample:
            break

    for h in handles: h.remove()

    sim_matrix = (sim_sum / count).cpu().numpy().astype(np.float64)
    np.fill_diagonal(sim_matrix, 1.0)

    affinity = (sim_matrix + 1.0) / 2.0                     # eq. (7)
    np.fill_diagonal(affinity, 1.0)
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(n_clusters=num_clusters, affinity="precomputed",
                            random_state=random_state, assign_labels="kmeans")
    labels_raw = sc.fit_predict(affinity)

    # re-index clusters by first-occurrence depth
    remap, nxt = {}, 0
    for l in range(num_layers):
        c = int(labels_raw[l])
        if c not in remap:
            remap[c] = nxt
            nxt += 1
    labels = np.array([remap[int(c)] for c in labels_raw], dtype=int)
    cluster_members = {k: [] for k in range(num_clusters)}
    for l in range(num_layers):
        cluster_members[int(labels[l])].append(l)
    return sim_matrix, labels, cluster_members


def rls_to_k(rls):
    """RLS-guided clustering policy (paper §5.1):
        RLS >= 1.0          -> K = 2   (localized, e.g. Llama-3.1)
        0.7 <= RLS < 1.0    -> K = 3   (e.g. OLMo-3, Mistral-Nemo)
        RLS < 0.7           -> K = 4   (distributed, e.g. Qwen3)
    """
    if rls >= 1.0:
        return 2
    if rls >= 0.7:
        return 3
    return 4


def get_pruned_layer_lorp(model, trainloader, num_to_prune, device,
                          max_batches=128, num_clusters=None,
                          protect_first=True, protect_last=True,
                          random_state=0):
    """LoRP (ours): Locality-Aware Redundancy Pruning.

      1. Capture pairwise per-token cosine similarity matrix S (eqs. 3-4).
      2. Representation Locality Score  RLS = -log2(mean off-diagonal of S) (eqs. 5-6).
      3. RLS-guided granularity K (paper §5.1; override via `num_clusters`).
      4. Spectral-cluster the affinity A=(S+1)/2 into K phases (eq. 7).
      5. Two-stage redundancy-aware allocation:
           Stage 1 — coverage-aware init: most-redundant eligible layer per cluster.
           Stage 2 — residual allocation: repeatedly prune the most-redundant
                     layer from the cluster with the largest residual redundancy.
      6. Boundary layers {0, N-1} are protected.
    """
    num_layers = len(model.model.layers)

    sim_matrix, labels_k2, members_k2 = _capture_sim_and_cluster(
        model, trainloader, device,
        num_clusters=2, max_batches=max_batches, random_state=random_state)

    off = sim_matrix[np.triu_indices(num_layers, k=1)]      # i < j
    off_mean = float(off.mean())                            # eq. (5)
    rls = float(-np.log2(off_mean))                         # eq. (6)
    K = int(num_clusters) if num_clusters else rls_to_k(rls)
    print(f"[LoRP] off_mean={off_mean:.4f}  RLS=-log2(off_mean)={rls:+.4f}  ->  K={K}"
          + ("  (forced)" if num_clusters else ""))

    if K == 2:
        labels, cluster_members = labels_k2, members_k2
    else:
        affinity = (sim_matrix + 1.0) / 2.0
        np.fill_diagonal(affinity, 1.0)
        from sklearn.cluster import SpectralClustering
        sc = SpectralClustering(n_clusters=K, affinity="precomputed",
                                random_state=random_state, assign_labels="kmeans")
        labels_raw = sc.fit_predict(affinity)
        remap, nxt = {}, 0
        for l in range(num_layers):
            c = int(labels_raw[l])
            if c not in remap:
                remap[c] = nxt
                nxt += 1
        labels = np.array([remap[int(c)] for c in labels_raw], dtype=int)
        cluster_members = {kk: [] for kk in range(K)}
        for l in range(num_layers):
            cluster_members[int(labels[l])].append(l)

    protected = set()
    if protect_first: protected.add(0)
    if protect_last:  protected.add(num_layers - 1)

    def member_redundancy(layer_idx, members):              # eq. (8)
        others = [m for m in members if m != layer_idx]
        if not others:
            return -float("inf")
        return float(np.mean([sim_matrix[layer_idx, m] for m in others]))

    def cluster_residual_mean(remaining):                   # eq. (10)
        if len(remaining) < 2:
            return -float("inf")
        s, c = 0.0, 0
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                s += sim_matrix[remaining[i], remaining[j]]
                c += 1
        return s / c

    cluster_candidates = {}
    for kk, members in cluster_members.items():
        eligible = [l for l in members if l not in protected]
        eligible.sort(key=lambda x: -member_redundancy(x, members))
        cluster_candidates[kk] = eligible
    pointers = {kk: 0 for kk in cluster_candidates}

    # Stage 1 — coverage-aware initialization (one layer per cluster)
    pruned_indices, pruned_set = [], set()
    for kk in sorted(cluster_candidates.keys()):
        if len(pruned_indices) >= num_to_prune:
            break
        if pointers[kk] < len(cluster_candidates[kk]):
            cand = cluster_candidates[kk][pointers[kk]]
            pointers[kk] += 1
            pruned_indices.append(cand)
            pruned_set.add(cand)

    # Stage 2 — residual redundancy allocation
    while len(pruned_indices) < num_to_prune:
        best_kk, best_mean = None, -float("inf")
        for kk, members in cluster_members.items():
            if pointers[kk] >= len(cluster_candidates[kk]):
                continue
            remaining = [m for m in members if m not in pruned_set]
            mu = cluster_residual_mean(remaining)
            if mu > best_mean:
                best_mean, best_kk = mu, kk
        if best_kk is None:
            for kk in sorted(cluster_candidates.keys()):
                if pointers[kk] < len(cluster_candidates[kk]):
                    best_kk = kk
                    break
            if best_kk is None:
                break
        cand = cluster_candidates[best_kk][pointers[best_kk]]
        pointers[best_kk] += 1
        pruned_indices.append(cand)
        pruned_set.add(cand)

    pruned_indices = sorted(pruned_indices)
    if not pruned_indices:
        raise RuntimeError("[LoRP] No eligible layers to prune.")
    start_l = pruned_indices[0]
    end_l = pruned_indices[-1] + 1
    avg_sim = float(np.mean([
        member_redundancy(l, cluster_members[int(labels[l])])
        for l in pruned_indices]))

    print(f"[LoRP] K={K}  pruned_indices={pruned_indices}")
    torch.cuda.empty_cache()
    return num_to_prune, start_l, end_l, avg_sim, pruned_indices


# ════════════════════════════════════════════════════════════════════════
def select_layer(model, trainloader, num_to_prune, dev,
                 pruning_method="lorp", num_clusters=None):
    tick = time.time()
    print(f"[select_layer] pruning_method={pruning_method}")

    if pruning_method == "lorp":
        out = get_pruned_layer_lorp(model, trainloader, num_to_prune, dev,
                                    num_clusters=num_clusters)
    elif pruning_method == "shortgpt":
        out = get_pruned_layer_shortgpt(model, trainloader, num_to_prune, dev)
    elif pruning_method == "streamline":
        out = get_pruned_layer_streamline(model, trainloader, num_to_prune, dev)
    else:
        raise NotImplementedError(
            f"pruning_method={pruning_method}; supported: lorp, shortgpt, streamline")

    layer, start_l, end_l, score, pruned_indices = out
    t_select = time.time() - tick
    print(f"Pruned range: layers [{start_l}, {end_l})  (selection took {t_select:.1f}s)")
    return layer, start_l, end_l, score, t_select, pruned_indices
