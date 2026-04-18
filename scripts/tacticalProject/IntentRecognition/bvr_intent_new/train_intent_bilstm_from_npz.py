# -*- coding: utf-8 -*-
"""
Train BiLSTM intent model from the NPZ dataset prepared by prepare_intent_dataset_13.py.

This version also writes runtime metadata into the checkpoint so the online
adapter can load the model directly for simulation.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None


matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


def set_seed(seed: int = 42):
    import os
    import random

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_runtime_meta(data_dir: Path):
    meta_path = data_dir / "meta.npy"
    if not meta_path.exists():
        return {}

    meta = np.load(meta_path, allow_pickle=True).item()
    runtime_meta = {
        "has_wing_mask": bool(meta.get("has_wing_mask", True)),
        "scale_wing_mask": bool(meta.get("scale_wing_mask", False)),
        "cont_dim_scaled": int(meta.get("cont_dim_scaled", 0)),
    }

    scaler_mean = meta.get("scaler_mean")
    scaler_scale = meta.get("scaler_scale")
    scaler = meta.get("scaler")
    if scaler_mean is None and scaler is not None and hasattr(scaler, "mean_"):
        scaler_mean = scaler.mean_
    if scaler_scale is None and scaler is not None and hasattr(scaler, "scale_"):
        scaler_scale = scaler.scale_

    if scaler_mean is not None:
        runtime_meta["scaler_mean"] = np.asarray(scaler_mean, dtype=np.float32)
    if scaler_scale is not None:
        runtime_meta["scaler_scale"] = np.asarray(scaler_scale, dtype=np.float32)

    return runtime_meta


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class IntentLSTM(nn.Module):
    def __init__(
        self,
        input_dim_num,
        num_classes,
        hidden_dim=64,
        num_layers=1,
        use_status_embed=False,
        status_vocab_size=4,
        status_emb_dim=4,
        bidirectional=False,
        use_self_attn=False,
        attn_dim=None,
        dropout=0.0,
        use_layernorm=False,
    ):
        super().__init__()
        self.use_status = use_status_embed
        self.use_self_attn = use_self_attn
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        feat_in = input_dim_num
        if self.use_status:
            self.status_emb = nn.Embedding(status_vocab_size, status_emb_dim)
            feat_in += status_emb_dim
        else:
            self.status_emb = None

        self.pre_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=feat_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(0.0 if num_layers == 1 else dropout),
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * self.num_directions
        self.ln = nn.LayerNorm(lstm_out_dim) if use_layernorm else nn.Identity()

        if self.use_self_attn:
            if attn_dim is None:
                attn_dim = lstm_out_dim
            self.attn_W = nn.Linear(lstm_out_dim, attn_dim, bias=True)
            self.attn_u = nn.Linear(attn_dim, 1, bias=False)
            self.post_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.fc = nn.Linear(lstm_out_dim, num_classes)
        else:
            self.post_dropout = nn.Identity()
            self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x_num, x_status=None):
        if self.use_status and x_status is not None:
            emb = self.status_emb(x_status)
            x = torch.cat([x_num, emb], dim=-1)
        else:
            x = x_num

        x = self.pre_dropout(x)
        hidden, _ = self.lstm(x)

        if self.use_self_attn:
            mid = torch.tanh(self.attn_W(hidden))
            scores = self.attn_u(mid).squeeze(-1)
            alpha = torch.softmax(scores, dim=1)
            context = torch.sum(hidden * alpha.unsqueeze(-1), dim=1)
            context = self.ln(context)
            context = self.post_dropout(context)
            return self.fc(context)

        last = hidden[:, -1, :]
        last = self.ln(last)
        return self.fc(last)


def make_loader(pack, batch_size, shuffle, use_status_embed):
    x_num = torch.tensor(pack["X_num"], dtype=torch.float32)
    y = torch.tensor(pack["y"], dtype=torch.long)
    if use_status_embed and pack.get("X_st") is not None:
        x_st = torch.tensor(pack["X_st"], dtype=torch.long)
        dataset = TensorDataset(x_num, x_st, y)
    else:
        dataset = TensorDataset(x_num, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_runtime_checkpoint(
    model,
    label2id,
    input_dim_num,
    num_classes,
    window_size,
    use_status_embed,
    status_vocab_size,
    status_emb_dim,
    hidden_dim,
    num_layers,
    bidirectional,
    use_self_attn,
    attn_dim,
    dropout,
    use_layernorm,
    extra_meta=None,
    best_val_macro_f1=None,
    best_epoch=None,
):
    ckpt = {
        "model_state": model.state_dict(),
        "label2id": label2id,
        "input_dim_num": int(input_dim_num),
        "hidden_dim": int(hidden_dim),
        "num_classes": int(num_classes),
        "num_layers": int(num_layers),
        "window_size": int(window_size),
        "use_status_embed": bool(use_status_embed),
        "status_vocab_size": int(status_vocab_size if use_status_embed else 0),
        "status_emb_dim": int(status_emb_dim if use_status_embed else 0),
        "bidirectional": bool(bidirectional),
        "use_self_attn": bool(use_self_attn),
        "attn_dim": attn_dim,
        "dropout": float(dropout),
        "use_layernorm": bool(use_layernorm),
    }
    if best_val_macro_f1 is not None:
        ckpt["best_val_macro_f1"] = float(best_val_macro_f1)
    if best_epoch is not None:
        ckpt["epoch"] = int(best_epoch)
    if extra_meta:
        ckpt.update(extra_meta)
    return ckpt


def train_model(
    train_pack,
    val_pack,
    input_dim_num,
    num_classes,
    out_dir,
    hidden_dim=64,
    lr=5e-4,
    batch_size=64,
    epochs=50,
    num_layers=1,
    use_status_embed=False,
    status_vocab_size=4,
    status_emb_dim=4,
    early_stop_patience=6,
    device="cpu",
    bidirectional=False,
    use_self_attn=False,
    attn_dim=None,
    dropout=0.2,
    use_layernorm=False,
    use_class_weight=False,
    use_focal=False,
    focal_gamma=2.0,
    runtime_checkpoint_kwargs=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(train_pack, batch_size=batch_size, shuffle=True, use_status_embed=use_status_embed)
    val_loader = make_loader(val_pack, batch_size=batch_size, shuffle=False, use_status_embed=use_status_embed)

    class_weights = None
    if use_class_weight:
        classes = np.arange(num_classes)
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=train_pack["y"])
        class_weights = torch.tensor(cw, dtype=torch.float32, device=device)
        print("[info] class_weight:", {int(c): float(w) for c, w in zip(classes, cw)})

    if use_focal:
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        print(f"[info] Using FocalLoss(gamma={focal_gamma}, with_class_weight={class_weights is not None})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = IntentLSTM(
        input_dim_num=input_dim_num,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_status_embed=use_status_embed,
        status_vocab_size=(status_vocab_size if use_status_embed else 0),
        status_emb_dim=status_emb_dim,
        bidirectional=bidirectional,
        use_self_attn=use_self_attn,
        attn_dim=attn_dim,
        dropout=dropout,
        use_layernorm=use_layernorm,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_f1 = -1.0
    patience_left = early_stop_patience
    best_path = out_dir / "intent_lstm_best.pth"
    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_macro_f1": []}

    print(
        f"[info] device={device}, input_dim_num={input_dim_num}, use_status_embed={use_status_embed}, "
        f"hidden_dim={hidden_dim}, num_layers={num_layers}, bidir={bidirectional}, self_attn={use_self_attn}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_cnt = 0

        for batch in train_loader:
            optimizer.zero_grad()
            if use_status_embed and len(batch) == 3:
                xb_num, xb_st, yb = [t.to(device) for t in batch]
                out = model(xb_num, xb_st)
            else:
                xb_num, yb = [t.to(device) for t in batch]
                out = model(xb_num)

            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * yb.size(0)
            total_correct += (out.argmax(1) == yb).sum().item()
            total_cnt += yb.size(0)

        train_loss = total_loss / max(1, total_cnt)
        train_acc = total_correct / max(1, total_cnt)

        model.eval()
        val_total_loss = 0.0
        val_total_correct = 0
        val_cnt = 0
        all_true = []
        all_pred = []

        with torch.no_grad():
            for batch in val_loader:
                if use_status_embed and len(batch) == 3:
                    xb_num, xb_st, yb = [t.to(device) for t in batch]
                    out = model(xb_num, xb_st)
                else:
                    xb_num, yb = [t.to(device) for t in batch]
                    out = model(xb_num)

                loss = criterion(out, yb)
                pred = out.argmax(1)
                val_total_loss += loss.item() * yb.size(0)
                val_total_correct += (pred == yb).sum().item()
                val_cnt += yb.size(0)
                all_true.extend(yb.cpu().numpy())
                all_pred.extend(pred.cpu().numpy())

        val_loss = val_total_loss / max(1, val_cnt)
        val_acc = val_total_correct / max(1, val_cnt)
        val_macro_f1 = f1_score(np.array(all_true), np.array(all_pred), average="macro")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        print(
            f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_macro_f1={val_macro_f1:.3f}"
        )

        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            patience_left = early_stop_patience
            best_ckpt = {
                "model_state": model.state_dict(),
                "best_val_macro_f1": best_f1,
                "epoch": epoch,
            }
            if runtime_checkpoint_kwargs:
                best_ckpt = build_runtime_checkpoint(
                    model=model,
                    label2id=runtime_checkpoint_kwargs["label2id"],
                    input_dim_num=runtime_checkpoint_kwargs["input_dim_num"],
                    num_classes=runtime_checkpoint_kwargs["num_classes"],
                    window_size=runtime_checkpoint_kwargs["window_size"],
                    use_status_embed=runtime_checkpoint_kwargs["use_status_embed"],
                    status_vocab_size=runtime_checkpoint_kwargs["status_vocab_size"],
                    status_emb_dim=runtime_checkpoint_kwargs["status_emb_dim"],
                    hidden_dim=runtime_checkpoint_kwargs["hidden_dim"],
                    num_layers=runtime_checkpoint_kwargs["num_layers"],
                    bidirectional=runtime_checkpoint_kwargs["bidirectional"],
                    use_self_attn=runtime_checkpoint_kwargs["use_self_attn"],
                    attn_dim=runtime_checkpoint_kwargs["attn_dim"],
                    dropout=runtime_checkpoint_kwargs["dropout"],
                    use_layernorm=runtime_checkpoint_kwargs["use_layernorm"],
                    extra_meta=runtime_checkpoint_kwargs.get("extra_meta"),
                    best_val_macro_f1=best_f1,
                    best_epoch=epoch,
                )
            torch.save(
                best_ckpt,
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EarlyStop] no macro-F1 improvement for {early_stop_patience} epochs, stop at epoch {epoch:02d}")
                break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_log.csv", index=False, encoding="utf-8-sig")

    try:
        plt.figure(figsize=(9, 4))
        plt.subplot(1, 2, 1)
        plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
        plt.plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss")
        plt.xlabel("epoch")
        plt.title("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(hist_df["epoch"], hist_df["train_acc"], label="train_acc")
        plt.plot(hist_df["epoch"], hist_df["val_acc"], label="val_acc")
        plt.plot(hist_df["epoch"], hist_df["val_macro_f1"], label="val_macro_f1")
        plt.xlabel("epoch")
        plt.title("Acc / Macro-F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "training_curves.png", dpi=180)
        plt.close()
    except Exception as exc:  # pragma: no cover
        print("[warn] failed to draw training curves:", exc)

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(
            f"[info] reloaded best checkpoint val_macro_f1={ckpt.get('best_val_macro_f1', -1):.4f}, "
            f"epoch={ckpt.get('epoch', '?')}"
        )
    else:
        print("[warn] best checkpoint not found, continue with current weights")

    return model, best_path


def evaluate(model, test_pack, label2id, out_dir, device="cpu", use_status_embed=False):
    x_num = torch.tensor(test_pack["X_num"], dtype=torch.float32)
    y = torch.tensor(test_pack["y"], dtype=torch.long)
    if use_status_embed and test_pack.get("X_st") is not None:
        x_st = torch.tensor(test_pack["X_st"], dtype=torch.long)
        dataset = TensorDataset(x_num, x_st, y)
    else:
        dataset = TensorDataset(x_num, y)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if use_status_embed and len(batch) == 3:
                xb_num, xb_st, yb = [t.to(device) for t in batch]
                out = model(xb_num, xb_st)
            else:
                xb_num, yb = [t.to(device) for t in batch]
                out = model(xb_num)
            preds = out.argmax(1).cpu().numpy()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds)

    id2label = {v: k for k, v in label2id.items()}
    labels = [id2label[i] for i in range(len(id2label))]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n=== test macro-F1: {macro_f1:.4f} ===")
    print("\n=== classification report ===")
    print(classification_report(y_true, y_pred, target_names=labels, digits=3))

    report_dict = classification_report(y_true, y_pred, target_names=labels, digits=3, output_dict=True)
    pd.DataFrame(report_dict).T.to_csv(Path(out_dir) / "classification_report.csv", encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    plt.figure(figsize=(8, 6))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    else:
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "confusion_matrix.png", dpi=180)
    plt.close()


def load_split(data_dir: Path, name: str):
    path = data_dir / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing split file: {path}")

    arr = np.load(path, allow_pickle=True)
    pack = {
        "X_num": arr["X_num"],
        "y": arr["y"],
        "label2id": arr["label2id"].item(),
        "X_st": arr["X_st"] if "X_st" in arr.files else None,
    }
    return pack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="directory containing train/val/test.npz")
    ap.add_argument("--out_dir", required=True, help="directory to write model and logs")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-status-embed", action="store_true")
    ap.add_argument("--status-emb-dim", type=int, default=4)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--use-self-attn", action="store_true")
    ap.add_argument("--attn-dim", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--use-layernorm", action="store_true")
    ap.add_argument("--use-class-weight", action="store_true")
    ap.add_argument("--use-focal", action="store_true")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    args = ap.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    runtime_meta = load_runtime_meta(data_dir)

    train_pack = load_split(data_dir, "train")
    val_pack = load_split(data_dir, "val")
    test_pack = load_split(data_dir, "test")

    label2id = train_pack["label2id"]
    input_dim_num = train_pack["X_num"].shape[-1]
    num_classes = len(label2id)
    window_size = train_pack["X_num"].shape[1]
    use_status_embed = args.use_status_embed and train_pack.get("X_st") is not None
    status_vocab_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime_checkpoint_kwargs = {
        "label2id": label2id,
        "input_dim_num": input_dim_num,
        "num_classes": num_classes,
        "window_size": window_size,
        "use_status_embed": use_status_embed,
        "status_vocab_size": status_vocab_size,
        "status_emb_dim": args.status_emb_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "bidirectional": args.bidirectional,
        "use_self_attn": args.use_self_attn,
        "attn_dim": args.attn_dim,
        "dropout": args.dropout,
        "use_layernorm": args.use_layernorm,
        "extra_meta": runtime_meta,
    }

    model, best_path = train_model(
        train_pack=train_pack,
        val_pack=val_pack,
        input_dim_num=input_dim_num,
        num_classes=num_classes,
        out_dir=out_dir,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_layers=args.num_layers,
        use_status_embed=use_status_embed,
        status_vocab_size=status_vocab_size,
        status_emb_dim=args.status_emb_dim,
        early_stop_patience=6,
        device=device,
        bidirectional=args.bidirectional,
        use_self_attn=args.use_self_attn,
        attn_dim=args.attn_dim,
        dropout=args.dropout,
        use_layernorm=args.use_layernorm,
        use_class_weight=args.use_class_weight,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        runtime_checkpoint_kwargs=runtime_checkpoint_kwargs,
    )

    best_runtime_ckpt = None
    if best_path.exists():
        best_state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model_state"])
        best_runtime_ckpt = build_runtime_checkpoint(
            model=model,
            label2id=label2id,
            input_dim_num=input_dim_num,
            num_classes=num_classes,
            window_size=window_size,
            use_status_embed=use_status_embed,
            status_vocab_size=status_vocab_size,
            status_emb_dim=args.status_emb_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional,
            use_self_attn=args.use_self_attn,
            attn_dim=args.attn_dim,
            dropout=args.dropout,
            use_layernorm=args.use_layernorm,
            extra_meta=runtime_meta,
            best_val_macro_f1=best_state.get("best_val_macro_f1"),
            best_epoch=best_state.get("epoch"),
        )
        torch.save(best_runtime_ckpt, best_path)
        print(f"[info] rewrote best checkpoint with runtime metadata: {best_path}")

    final_ckpt = build_runtime_checkpoint(
        model=model,
        label2id=label2id,
        input_dim_num=input_dim_num,
        num_classes=num_classes,
        window_size=window_size,
        use_status_embed=use_status_embed,
        status_vocab_size=status_vocab_size,
        status_emb_dim=args.status_emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        use_self_attn=args.use_self_attn,
        attn_dim=args.attn_dim,
        dropout=args.dropout,
        use_layernorm=args.use_layernorm,
        extra_meta=runtime_meta,
        best_val_macro_f1=(best_runtime_ckpt or {}).get("best_val_macro_f1"),
        best_epoch=(best_runtime_ckpt or {}).get("epoch"),
    )
    final_path = out_dir / "intent_bilstm_attn.pth"
    torch.save(final_ckpt, final_path)
    print(f"[info] saved final checkpoint: {final_path}")

    evaluate(model, test_pack, label2id, out_dir=out_dir, device=device, use_status_embed=use_status_embed)


if __name__ == "__main__":
    main()
