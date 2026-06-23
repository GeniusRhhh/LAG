# -*- coding: utf-8 -*-
"""
train_intent_bilstm_from_npz.py
---------------------------------
功能：从 prepare_intent_dataset.py 生成的 train/val/test.npz 加载数据，
      训练 BiLSTM-Self-Attention 模型，保存最佳模型与日志。
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ---------------- Utils ----------------
def set_seed(seed: int = 42):
    import random, os
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------- Focal Loss ----------------
class FocalLoss(nn.Module):
    """多类 Focal Loss（支持 alpha=class_weights）"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Tensor[num_classes] or None
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ---------------- 模型（增强版：可选 BiLSTM + Self-Attention） ----------------
class IntentLSTM(nn.Module):
    """
    增强版基线：
    - 可选雷达状态嵌入（与原来一致）
    - 可选双向 LSTM（--bidirectional）
    - 可选 Self-Attention 汇聚（--use-self-attn, --attn-dim）
    - 可选 LayerNorm + Dropout（--dropout）
    """
    def __init__(self, input_dim_num, num_classes,
                 hidden_dim=64, num_layers=1,
                 use_status_embed=False, status_vocab_size=4, status_emb_dim=4,
                 bidirectional=False, use_self_attn=False, attn_dim=None,
                 dropout=0.0, use_layernorm=False):
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
            bidirectional=bidirectional
        )

        lstm_out_dim = hidden_dim * self.num_directions
        self.use_layernorm = use_layernorm
        self.ln = nn.LayerNorm(lstm_out_dim) if use_layernorm else nn.Identity()

        if self.use_self_attn:
            if attn_dim is None:
                attn_dim = lstm_out_dim
            self.attn_W = nn.Linear(lstm_out_dim, attn_dim, bias=True)
            self.attn_u = nn.Linear(attn_dim, 1, bias=False)
            self.post_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.fc = nn.Linear(lstm_out_dim, num_classes)
        else:
            self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x_num, x_status=None):
        if self.use_status and x_status is not None:
            E = self.status_emb(x_status)             # [B, T, E]
            x = torch.cat([x_num, E], dim=-1)         # [B, T, F_num+E]
        else:
            x = x_num

        x = self.pre_dropout(x)
        H, _ = self.lstm(x)                           # H: [B, T, D]

        if self.use_self_attn:
            M = torch.tanh(self.attn_W(H))            # [B, T, A]
            scores = self.attn_u(M).squeeze(-1)       # [B, T]
            alpha = torch.softmax(scores, dim=1)      # [B, T]
            context = torch.sum(H * alpha.unsqueeze(-1), dim=1)  # [B, D]
            context = self.ln(context)
            context = self.post_dropout(context)
            logits = self.fc(context)
        else:
            last = H[:, -1, :]                        # [B, D]
            last = self.ln(last)
            logits = self.fc(last)
        return logits


# ---------------- 训练 ----------------
def train_model(train_pack, val_pack, input_dim_num, num_classes, out_dir,
                hidden_dim=64, lr=5e-4, batch_size=64, epochs=50, num_layers=1,
                use_status_embed=False, status_vocab_size=4, status_emb_dim=4,
                early_stop_patience=6, device="cpu",
                # 模型增强
                bidirectional=False, use_self_attn=False, attn_dim=None,
                dropout=0.2, use_layernorm=False,
                # 不均衡控制
                use_class_weight=False, use_focal=False, focal_gamma=2.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 构建 DataLoader
    def make_loader(pack, shuffle):
        Xn = torch.tensor(pack["X_num"], dtype=torch.float32)
        y = torch.tensor(pack["y"], dtype=torch.long)
        if use_status_embed and ("X_st" in pack and pack["X_st"] is not None):
            Xs = torch.tensor(pack["X_st"], dtype=torch.long)
            ds = TensorDataset(Xn, Xs, y)
        else:
            ds = TensorDataset(Xn, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_pack, shuffle=True)
    val_loader = make_loader(val_pack, shuffle=False)

    # 类别权重 / 损失
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

    # 模型/优化器
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
        use_layernorm=use_layernorm
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_f1 = -1.0
    patience_left = early_stop_patience
    best_path = out_dir / "intent_lstm_best.pth"
    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_macro_f1": []}

    print(f"[info] device={device}, input_dim_num={input_dim_num}, use_status_embed={use_status_embed}, "
          f"hidden_dim={hidden_dim}, num_layers={num_layers}, bidir={bidirectional}, self_attn={use_self_attn}")

    for epoch in range(1, epochs + 1):
        # ---- train
        model.train()
        total_loss, total_correct, total_cnt = 0.0, 0, 0
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

        # ---- val
        model.eval()
        val_total_loss, val_total_correct, val_cnt = 0.0, 0, 0
        all_true, all_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                if use_status_embed and len(batch) == 3:
                    xb_num, xb_st, yb = [t.to(device) for t in batch]
                    out = model(xb_num, xb_st)
                else:
                    xb_num, yb = [t.to(device) for t in batch]
                    out = model(xb_num)
                loss = criterion(out, yb)
                val_total_loss += loss.item() * yb.size(0)
                pred = out.argmax(1)
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

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_macro_f1={val_macro_f1:.3f}")

        # Early stop on best macro-F1
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            patience_left = early_stop_patience
            torch.save({
                "model_state": model.state_dict(),
                "best_val_macro_f1": best_f1,
                "epoch": epoch
            }, best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EarlyStop] 连续 {early_stop_patience} 轮未提升，提前停止在第 {epoch:02d} 轮。")
                break

    # 保存训练曲线
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    try:
        plt.figure(figsize=(9, 4))
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
        plt.plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss")
        plt.xlabel("epoch")
        plt.title("Loss")
        plt.legend()
        # Acc/F1
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
        print("📈 训练曲线已保存:", out_dir / "training_curves.png")
    except Exception as e:
        print("[WARN] 曲线绘制失败：", e)

    # 加载最佳权重
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"🏆 已加载最佳（val_macro_f1={ckpt.get('best_val_macro_f1', -1):.4f}, epoch={ckpt.get('epoch', '?')})")
    else:
        print("⚠ 未找到最佳模型文件，返回当前权重。")
    return model


# ---------------- 测试评估 ----------------
def evaluate(model, test_pack, label2id, out_dir, device="cpu", use_status_embed=False):
    Xn = torch.tensor(test_pack["X_num"], dtype=torch.float32)
    y = torch.tensor(test_pack["y"], dtype=torch.long)
    if use_status_embed and ("X_st" in test_pack and test_pack["X_st"] is not None):
        Xs = torch.tensor(test_pack["X_st"], dtype=torch.long)
        ds = TensorDataset(Xn, Xs, y)
    else:
        ds = TensorDataset(Xn, y)
    loader = DataLoader(ds, batch_size=128, shuffle=False)

    y_true, y_pred = [], []
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

    labels = list(label2id.keys())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n=== 测试集 Macro-F1: {macro_f1:.4f} ===")
    print("\n=== 测试集分类报告 ===")
    print(classification_report(y_true, y_pred, target_names=labels, digits=3))
    # === 保存分类报告为 CSV 表格 ===
    report_dict = classification_report(
        y_true, y_pred, target_names=labels, digits=3, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).T
    csv_path = Path(out_dir) / "classification_report.csv"
    report_df.to_csv(csv_path, encoding="utf-8-sig")
    print("📄 分类报告已保存为 CSV:", csv_path)


    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("混淆矩阵")
    plt.tight_layout()
    fig_path = Path(out_dir) / "confusion_matrix.png"
    plt.savefig(fig_path, dpi=180)
    plt.close()
    print("📊 混淆矩阵已保存:", fig_path)


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="包含 train/val/test.npz 的目录")
    ap.add_argument("--out_dir", required=True, help="模型与日志输出目录（可与 data_dir 相同）")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    # 状态嵌入（是否启用）
    ap.add_argument("--use-status-embed", action="store_true",
                    help="若 npz 中包含 X_st，则使用状态嵌入")
    ap.add_argument("--status-emb-dim", type=int, default=4)
    # 模型增强
    ap.add_argument("--bidirectional", action="store_true", help="使用双向LSTM")
    ap.add_argument("--use-self-attn", action="store_true", help="启用Self-Attention时间汇聚")
    ap.add_argument("--attn-dim", type=int, default=None, help="注意力中间维度（默认等于LSTM输出维）")
    ap.add_argument("--dropout", type=float, default=0.2, help="Dropout（默认0.2）")
    ap.add_argument("--use-layernorm", action="store_true", help="池化后加LayerNorm")
    # 不均衡
    ap.add_argument("--use-class-weight", action="store_true")
    ap.add_argument("--use-focal", action="store_true")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    args = ap.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    # 读取数据
    def load_split(name):
        path = data_dir / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"未找到 {path}")
        arr = np.load(path, allow_pickle=True)
        pack = {
            "X_num": arr["X_num"],
            "y": arr["y"],
            "label2id": arr["label2id"].item()
        }
        if "X_st" in arr.files:
            pack["X_st"] = arr["X_st"]
        else:
            pack["X_st"] = None
        return pack

    train_pack = load_split("train")
    val_pack = load_split("val")
    test_pack = load_split("test")

    label2id = train_pack["label2id"]
    input_dim_num = train_pack["X_num"].shape[-1]
    num_classes = len(label2id)

    # 根据数据决定是否可以用状态嵌入
    use_status_embed = args.use_status_embed and ("X_st" in train_pack and train_pack["X_st"] is not None)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 训练
    model = train_model(
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
        status_vocab_size=4,  # 与 prepare 阶段保持一致
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
        focal_gamma=args.focal_gamma
    )

    # 保存最终模型
    ckpt_path = out_dir / "intent_lstm_baseline.pth"
    torch.save({
        "model_state": model.state_dict(),
        "label2id": label2id,
        "input_dim_num": input_dim_num,
        "hidden_dim": args.hidden_dim,
        "num_classes": num_classes,
        "use_status_embed": use_status_embed,
        "status_emb_dim": (args.status_emb_dim if use_status_embed else 0),
        "bidirectional": args.bidirectional,
        "use_self_attn": args.use_self_attn,
        "attn_dim": args.attn_dim,
        "dropout": args.dropout,
        "use_layernorm": args.use_layernorm
    }, ckpt_path)
    print("✅ 模型已保存:", ckpt_path)

    # 测试评估
    evaluate(model, test_pack, label2id, out_dir=out_dir, device=device, use_status_embed=use_status_embed)


if __name__ == "__main__":
    main()
