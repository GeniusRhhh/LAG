# -*- coding: utf-8 -*-
"""
prepare_intent_dataset.py
---------------------------------
功能：从 NEU 特征 CSV 构建滑窗样本，并划分 train/val/test，保存为 npz。
对应原 train_intent_bilstm_atten.py 中：
- set_seed
- parse_perclass_step
- pick_feature_cols
- build_windows
- split_dataset
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------- Utils ----------------
def set_seed(seed: int = 42):
    import random, os
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_perclass_step(s: str):
    """
    将 '防御=4,攻击=2,逃逸=1,协同=1,探测=1,中立=1' 解析为 dict
    """
    if not s or not s.strip():
        return None
    mapping = {}
    for kv in s.split(","):
        kv = kv.strip()
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                mapping[k.strip()] = float(v.strip())
            except:
                pass
    return mapping if mapping else None


# ---------------- 选择特征列 ----------------
def pick_feature_cols(df_columns, include_status=False):
    """
    对齐 make_intent_features_NEU_v2.py 的输出：
    连续特征（8）：Enemy_speed_m_s, Own_speed_m_s, rel_alt_m, range_m,
                 enemy_enter_angle_deg, bearing_deg, bearing_rate_deg_s, closure_rate_m_s
    可选：Status_code（整数索引，结合 --use-status-embed 使用）
    """
    num_cols = [
        "Enemy_speed_m_s", "Own_speed_m_s", "rel_alt_m", "range_m",
        "enemy_enter_angle_deg", "bearing_deg", "bearing_rate_deg_s", "closure_rate_m_s"
    ]
    for c in num_cols:
        if c not in df_columns:
            raise ValueError(f"缺少连续特征列：{c}（请先用 make_intent_features_NEU_v2.py 生成）")
    status_ok = "Status_code" in df_columns
    return num_cols, (status_ok if include_status else False)


# ---------------- 数据构建：滑窗 ----------------
def build_windows(input_csv, T=12, S=1, label_col="Intent_Label",
                  include_status_embed=False, perclass_step=None):
    """
    - 严格排序分组；组内标准化；滑窗末帧打标签
    - perclass_step: dict[str,float]，如 {"防御":4,"攻击":2,"逃逸":1,"协同":1,"探测":1,"中立":1}
      若不为 None，则以“窗口末帧真实标签”的步长覆盖默认 S（step 至少取 1）
    """
    df = pd.read_csv(input_csv)
    if label_col not in df.columns:
        raise ValueError(f"CSV 中缺少标签列 {label_col}（如需训练请确保未使用 --no-labels）")

    # 排序：Dataset_Token → Agent_ID → Target_ID → Time_s
    sort_cols = ["Agent_ID", "Target_ID", "Time_s"]
    if "Dataset_Token" in df.columns:
        sort_cols = ["Dataset_Token"] + sort_cols
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # 分组键
    group_cols = ["Agent_ID", "Target_ID"]
    if "Dataset_Token" in df.columns:
        group_cols = ["Dataset_Token"] + group_cols

    # 特征列
    num_cols, use_status = pick_feature_cols(df.columns, include_status=include_status_embed)

    # 标签映射（保持字典序，便于复现）
    labels_sorted = sorted(df[label_col].dropna().unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}
    df["_yid_"] = df[label_col].map(label2id).astype("Int64")

    # 滑窗
    Xnum_list, Xst_list, y_list = [], [], []
    scaler = StandardScaler()

    for _, g in df.groupby(group_cols, sort=False):
        g = g.reset_index(drop=True)
        # 连续特征 -> 组内标准化
        num_data = g[num_cols].to_numpy(dtype=np.float32)
        num_data = scaler.fit_transform(num_data)
        num_data = np.nan_to_num(num_data, nan=0.0, posinf=1e6, neginf=-1e6)

        # 状态索引（不标准化）
        st_seq = None
        if use_status:
            st_seq = g["Status_code"].fillna(0).astype(int).to_numpy()

        y_seq = g["_yid_"].to_numpy(dtype=np.int64)
        y_txt = g[label_col].to_numpy()

        i, L = 0, len(g)
        while i + T <= L:
            end = i + T
            Xnum_list.append(num_data[i:end])
            if use_status:
                Xst_list.append(st_seq[i:end])
            y_list.append(y_seq[end - 1])

            step = S
            if perclass_step is not None:
                step = perclass_step.get(str(y_txt[end - 1]), S)  # 用末帧标签名查
            i += max(1, int(round(step)))

    X_num = np.stack(Xnum_list) if Xnum_list else np.zeros((0, T, len(num_cols)), np.float32)
    X_st = (np.stack(Xst_list) if (use_status and Xst_list) else None)
    y = np.array(y_list, dtype=np.int64)

    # status_vocab_size 原脚本写死 4，这里保持一致
    status_vocab_size = 4 if use_status else 0

    return X_num, X_st, y, label2id, len(num_cols), status_vocab_size


# ---------------- 划分并保存 ----------------
def split_and_save(X_num, X_st, y, label2id, out_dir,
                   train_ratio=0.7, val_ratio=0.15, seed=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    N = len(y)
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)

    X_num = X_num[idx]
    y = y[idx]
    if X_st is not None:
        X_st = X_st[idx]

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    n_test = N - n_train - n_val
    print(f"训练集: {n_train}, 验证集: {n_val}, 测试集: {n_test}")

    splits = {}
    for name, sl in [
        ("train", slice(0, n_train)),
        ("val", slice(n_train, n_train + n_val)),
        ("test", slice(n_train + n_val, None))
    ]:
        pack = {"X_num": X_num[sl], "y": y[sl], "label2id": label2id}
        if X_st is not None:
            pack["X_st"] = X_st[sl]
        splits[name] = pack
        np.savez_compressed(out_dir / f"{name}.npz", **pack)

    print("✅ 数据集已保存到:", out_dir)

    # 额外保存一个 meta.npy，方便训练脚本读取一些元信息
    meta = {
        "label2id": label2id,
        "input_dim_num": X_num.shape[-1] if X_num.ndim == 3 else 0,
        "use_status_embed": (X_st is not None),
    }
    np.save(out_dir / "meta.npy", meta, allow_pickle=True)
    print("✅ 元信息已保存:", out_dir / "meta.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 CSV（由 make_intent_features_NEU_v2.py 生成）")
    ap.add_argument("--out_dir", required=True, help="输出目录（train/val/test.npz + meta.npy）")
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--S", type=int, default=1)
    ap.add_argument("--label-col", type=str, default="Intent_Label")
    ap.add_argument("--seed", type=int, default=42)
    # 状态嵌入
    ap.add_argument("--use-status-embed", action="store_true", help="同时构造 Status_code 序列（用于后续嵌入）")
    # 按类步长
    ap.add_argument("--perclass-step", type=str, default="",
                    help="按类步长，如 '防御=4,攻击=2,逃逸=1,协同=1,探测=1,中立=1'")
    args = ap.parse_args()

    set_seed(args.seed)
    step_map = parse_perclass_step(args.perclass_step)

    # Step 1: 构建滑窗
    X_num, X_st, y, label2id, input_dim_num, status_vocab = build_windows(
        args.input,
        T=args.T,
        S=args.S,
        label_col=args.label_col,
        include_status_embed=args.use_status_embed,
        perclass_step=step_map
    )
    print(f"[info] 样本数={len(y)}, 特征维度={input_dim_num}, 使用状态嵌入={args.use_status_embed}")

    # Step 2: 划分并保存
    split_and_save(X_num, X_st, y, label2id, out_dir=args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
