from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from . import config


def train_lora(train: pd.DataFrame, test: pd.DataFrame, base: str | None = None, epochs: int = 3,
               batch_size: int = 16, lr: float = 2e-4, max_len: int = 128, r: int = 16,
               out_dir=None, seed: int = 42, log=print) -> dict:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    base = base or config.LORA_BASE
    out_dir = out_dir or config.LORA_DIR
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lab2id = {l: i for i, l in enumerate(config.LABELS)}

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(config.LABELS), id2label=dict(enumerate(config.LABELS)), label2id=lab2id)
    lcfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=2 * r, lora_dropout=0.1,
                      target_modules=["query", "value"])
    model = get_peft_model(model, lcfg).to(dev)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    def encode(df):
        enc = tok(df["text"].astype(str).tolist(), truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt")
        y = torch.tensor(df["label"].map(lab2id).values)
        return torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"], y)

    tr = torch.utils.data.DataLoader(encode(train), batch_size=batch_size, shuffle=True,
                                     generator=torch.Generator().manual_seed(seed))
    te = torch.utils.data.DataLoader(encode(test), batch_size=64)

    counts = train["label"].value_counts().reindex(config.LABELS).fillna(0).values
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(counts), dtype=torch.float, device=dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=0.01)
    steps = epochs * len(tr)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def evaluate():
        model.eval()
        preds = []
        with torch.no_grad():
            for ids, mask, _ in te:
                with torch.autocast(device_type=dev, enabled=use_amp):
                    logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
                preds.extend(logits.argmax(-1).cpu().tolist())
        model.train()
        y_pred = [config.LABELS[p] for p in preds]
        return y_pred, {"accuracy": accuracy_score(test["label"], y_pred),
                        "macro_f1": f1_score(test["label"], y_pred, average="macro")}

    history, t0 = [], time.time()
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for ids, mask, y in tr:
            opt.zero_grad()
            with torch.autocast(device_type=dev, enabled=use_amp):
                logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
                loss = loss_fn(logits.float(), y.to(dev))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total_loss += loss.item()
        _, m = evaluate()
        history.append({"epoch": ep, "train_loss": total_loss / len(tr), **m})
        log(f"epoch {ep}: loss={history[-1]['train_loss']:.4f} acc={m['accuracy']:.3f} macro-F1={m['macro_f1']:.3f}")

    y_pred, metrics = evaluate()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return {"history": pd.DataFrame(history), "y_pred": y_pred, "metrics": metrics,
            "trainable_params": trainable, "total_params": total, "minutes": (time.time() - t0) / 60}
