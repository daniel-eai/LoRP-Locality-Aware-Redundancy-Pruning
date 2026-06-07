import random

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm


def get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    if test_only:
        return testenc
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader, valloader = [], []
    val_ratio = 0.9
    train_max = int(trainenc.input_ids.shape[1] * val_ratio) - seqlen - 1
    for _ in range(train_size):
        i = random.randint(0, train_max)
        inp = trainenc.input_ids[:, i:i + seqlen]
        tar = inp.clone(); tar[:, :-1] = -100
        trainloader.append((inp, tar))
    val_start = int(trainenc.input_ids.shape[1] * val_ratio) - seqlen - 1
    val_end = trainenc.input_ids.shape[1] - seqlen - 1
    for _ in range(val_size):
        i = random.randint(val_start, val_end)
        inp = trainenc.input_ids[:, i:i + seqlen]
        tar = inp.clone(); tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader


def get_ptb(tokenizer, nsamples, val_size, seed, seqlen, test_only):
    valdata = load_dataset("ptb_text_only", "penn_treebank",
                            split="validation", trust_remote_code=True)
    testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
    if test_only:
        return testenc
    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        inp = trainenc.input_ids[:, i:i + seqlen]
        tar = inp.clone(); tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(tokenizer, train_size, val_size, seed, seqlen, test_only):
    traindata = load_dataset("allenai/c4", "en", split="train", streaming=True)\
                    .shuffle(seed=seed, buffer_size=10_000)
    validationdata = load_dataset("allenai/c4", "en", split="validation",
                                    streaming=True)\
                    .shuffle(seed=seed, buffer_size=10_000)
    train_it = iter(traindata)
    val_it = iter(validationdata)

    def next_window(it, rng):
        while True:
            ex = next(it)
            enc = tokenizer(ex["text"], return_tensors="pt")
            L = enc.input_ids.shape[1]
            if L >= seqlen + 1:
                start = rng.randint(0, L - seqlen - 1)
                inp = enc.input_ids[:, start:start + seqlen]
                tar = inp.clone(); tar[:, :-1] = -100
                return inp, tar

    rng_valenc = random.Random(0)
    valenc = []
    for _ in range(256):
        inp, _ = next_window(val_it, rng_valenc)
        valenc.append(inp)
    valenc = torch.hstack(valenc)
    if test_only:
        return valenc

    rng_train = random.Random(seed)
    trainloader = [next_window(train_it, rng_train) for _ in range(train_size)]
    rng_val = random.Random(seed + 1)
    valloader = [next_window(val_it, rng_val) for _ in range(val_size)]
    return trainloader, valloader


def get_loaders(name, tokenizer, train_size=128, val_size=64,
                 seed=0, seqlen=2048, test_only=False):
    if "wikitext2" in name:
        return get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only)
    if "c4" in name:
        return get_c4(tokenizer, train_size, val_size, seed, seqlen, test_only)
    if "ptb" in name:
        return get_ptb(tokenizer, train_size, val_size, seed, seqlen, test_only)
    raise NotImplementedError(f"unknown calibration corpus: {name}")


@torch.no_grad()
def test_ppl(model, tokenizer, datasets=("wikitext2",), ppl_seqlen=2048):
    results = {}
    for dataset in datasets:
        testloader = get_loaders(dataset, tokenizer, seed=0, seqlen=ppl_seqlen,
                                  test_only=True)
        testenc = testloader if "c4" in dataset else testloader.input_ids
        seqlen = ppl_seqlen
        nsamples = testenc.numel() // seqlen

        if hasattr(model, "lm_head"):
            classifier = model.lm_head
        elif hasattr(model.model, "lm_head"):
            classifier = None
        elif hasattr(model, "output"):
            classifier = model.output
        else:
            raise NotImplementedError("unknown LM head structure")

        nlls = []
        for i in tqdm(range(nsamples)):
            batch = testenc[:, i * seqlen:(i + 1) * seqlen].to(model.device)
            outputs = model.model(batch)
            if classifier is not None:
                hidden = outputs[0]
                logits = classifier(hidden.to(classifier.weight.dtype))
            else:
                logits = outputs[0]
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, i * seqlen:(i + 1) * seqlen][:, 1:].to(shift_logits.device)
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            nlls.append(loss.float() * seqlen)
        results[dataset] = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
    return results
