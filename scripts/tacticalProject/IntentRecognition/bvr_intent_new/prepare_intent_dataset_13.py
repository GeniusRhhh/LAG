# -*- coding: utf-8 -*-
"""
prepare_intent_dataset_13.py
-------------------------------------------------
- 从 features CSV 构建滑窗样本
- 随机划分 train/val/test
- StandardScaler 仅用 train 拟合，再用于 val/test
- 连续特征：8维基础 + 4维僚机几何 = 12维
- 缺失指示：wing_valid（建议加入 X_num 作为额外1维；可开关）
- 状态嵌入：Status_code -> X_st
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int = 42):
    import random, os
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_perclass_step(s: str):
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


def pick_feature_cols(df_columns, include_status=False, include_wing_mask=True):
    # 12维连续特征
    num_cols = [
        "Enemy_speed_m_s", "Own_speed_m_s", "rel_alt_m", "range_m",
        "enemy_enter_angle_deg", "bearing_deg", "bearing_rate_deg_s", "closure_rate_m_s",
        "dist_enemy_wing_m", "distdiff_to_own_m", "angle_enemy_wing_from_own_deg", "alt_diff_enemy_wing_m",
    ]
    for c in num_cols:
        if c not in df_columns:
            raise ValueError(f"缺少连续特征列：{c}（请先生成）")

    # 缺失掩码
    wing_mask_col = None
    if include_wing_mask:
        if "wing_valid" not in df_columns:
            raise ValueError("缺少 wing_valid 列（使用带 wing_valid 的特征脚本导出）")
        wing_mask_col = "wing_valid"

    status_ok = "Status_code" in df_columns
    use_status = (status_ok if include_status else False)

    return num_cols, wing_mask_col, use_status


def build_windows_raw(input_csv, T=12, S=1, label_col="Intent_Label",
                      include_status_embed=False, perclass_step=None,
                      include_wing_mask=True):
    """
    构建滑窗
    """
    df = pd.read_csv(input_csv)
    if label_col not in df.columns:
        raise ValueError(f"CSV 中缺少标签列 {label_col}")

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
    num_cols, wing_mask_col, use_status = pick_feature_cols(
        df.columns,
        include_status=include_status_embed,
        include_wing_mask=include_wing_mask
    )

    # 标签映射
    labels_sorted = sorted(df[label_col].dropna().unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}
    df["_yid_"] = df[label_col].map(label2id).astype("Int64")

    Xnum_list, Xst_list, y_list = [], [], []
    for _, g in df.groupby(group_cols, sort=False):
        g = g.reset_index(drop=True)

        # 连续特征 raw
        num_data = g[num_cols].to_numpy(dtype=np.float32)
        num_data = np.nan_to_num(num_data, nan=0.0, posinf=1e6, neginf=-1e6)

        # wing_valid（0/1）
        wing_mask = None
        if wing_mask_col is not None:
            wing_mask = g[wing_mask_col].fillna(0).astype(np.float32).to_numpy()

        # 状态索引序列
        st_seq = None
        if use_status:
            st_seq = g["Status_code"].fillna(0).astype(int).to_numpy()

        y_seq = g["_yid_"].to_numpy(dtype=np.int64)
        y_txt = g[label_col].to_numpy()

        i, L = 0, len(g)
        while i + T <= L:
            end = i + T
            xw = num_data[i:end]  # [T, 12]
            if wing_mask is not None:
                # 把 wing_valid 作为额外1维拼进去（推荐做法）
                wm = wing_mask[i:end].reshape(T, 1)  # [T,1]
                xw = np.concatenate([xw, wm], axis=1)  # [T, 13]

            Xnum_list.append(xw)

            if use_status:
                Xst_list.append(st_seq[i:end])
            y_list.append(y_seq[end - 1])

            step = S
            if perclass_step is not None:
                step = perclass_step.get(str(y_txt[end - 1]), S)
            i += max(1, int(round(step)))

    X_num = np.stack(Xnum_list) if Xnum_list else np.zeros((0, T, 13 if include_wing_mask else 12), np.float32)
    X_st = (np.stack(Xst_list) if (use_status and Xst_list) else None)
    y = np.array(y_list, dtype=np.int64)

    status_vocab_size = 4 if use_status else 0
    input_dim_num = X_num.shape[-1] if X_num.ndim == 3 else 0
    return X_num, X_st, y, label2id, input_dim_num, status_vocab_size


def split_and_scale_and_save(X_num, X_st, y, label2id, out_dir,
                             train_ratio=0.7, val_ratio=0.15, seed=42,
                             has_wing_mask=True,
                             scale_wing_mask=False):
    """
    - 先 shuffle 划分
    - 再用 train 拟合 scaler
    - 再 transform val/test
    - 可选择 wing_valid 是否参与标准化（推荐：不标准化）
    """
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

    # -------- train-only scaler --------
    # X_num: [N, T, D]
    D = X_num.shape[-1]
    # 默认：最后一维是 wing_valid（0/1），不建议标准化
    if has_wing_mask:
        if scale_wing_mask:
            cont_dim = D
            mask_dim = 0
        else:
            cont_dim = D - 1
            mask_dim = 1
    else:
        cont_dim = D
        mask_dim = 0

    scaler = StandardScaler()
    train_flat = X_num[:n_train, :, :cont_dim].reshape(-1, cont_dim)
    scaler.fit(train_flat)

    def transform_block(Xb):
        Xc = Xb[:, :, :cont_dim]
        Xc2 = scaler.transform(Xc.reshape(-1, cont_dim)).reshape(Xc.shape)
        if mask_dim == 1:
            Xm = Xb[:, :, cont_dim:]  # [N,T,1]
            Xout = np.concatenate([Xc2, Xm], axis=2)
        else:
            Xout = Xc2
        Xout = np.nan_to_num(Xout, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        return Xout

    X_num_scaled = transform_block(X_num)

    # -------- save splits --------

    for name, sl in [
        ("train", slice(0, n_train)),
        ("val", slice(n_train, n_train + n_val)),
        ("test", slice(n_train + n_val, None))
    ]:
        pack = {"X_num": X_num_scaled[sl], "y": y[sl], "label2id": label2id}
        if X_st is not None:
            pack["X_st"] = X_st[sl]
        np.savez_compressed(out_dir / f"{name}.npz", **pack)


    meta = {
        "label2id": label2id,
        "input_dim_num": int(X_num_scaled.shape[-1]),
        "use_status_embed": bool(X_st is not None),
        "scaler": scaler,
        "has_wing_mask": bool(has_wing_mask),
        "scale_wing_mask": bool(scale_wing_mask),
        "cont_dim_scaled": int(cont_dim),
    }
    np.save(out_dir / "meta.npy", meta, allow_pickle=True)
    print("✅ 数据集已保存到:", out_dir)
    print("✅ 元信息已保存:", out_dir / "meta.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 CSV")
    ap.add_argument("--out_dir", required=True, help="输出目录（train/val/test.npz + meta.npy）")
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--S", type=int, default=1)
    ap.add_argument("--label-col", type=str, default="Intent_Label")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use-status-embed", action="store_true", help="构造 Status_code 序列（用于嵌入）")
    ap.add_argument("--perclass-step", type=str, default="", help="按类步长，如 '防御=4,攻击=2,...'")

    # wing mask
    ap.add_argument("--no-wing-mask", action="store_true", help="不使用 wing_valid 掩码（不推荐）")
    ap.add_argument("--scale-wing-mask", action="store_true", help="让 wing_valid 也参与标准化（不推荐）")

    args = ap.parse_args()

    set_seed(args.seed)
    step_map = parse_perclass_step(args.perclass_step)

    X_num, X_st, y, label2id, input_dim_num, status_vocab = build_windows_raw(
        args.input,
        T=args.T,
        S=args.S,
        label_col=args.label_col,
        include_status_embed=args.use_status_embed,
        perclass_step=step_map,
        include_wing_mask=(not args.no_wing_mask)
    )
    print(f"[info] 样本数={len(y)}, X_num维度={input_dim_num}, 使用状态嵌入={args.use_status_embed}, wing_mask={not args.no_wing_mask}")

    split_and_scale_and_save(
        X_num, X_st, y, label2id,
        out_dir=args.out_dir,
        seed=args.seed,
        has_wing_mask=(not args.no_wing_mask),
        scale_wing_mask=args.scale_wing_mask
    )


if __name__ == "__main__":
    main()