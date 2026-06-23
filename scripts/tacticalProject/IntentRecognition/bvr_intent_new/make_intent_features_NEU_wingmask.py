# -*- coding: utf-8 -*-
"""
make_intent_features_NEU_wingmask.py
-----------------------------------
在原 8 维 NEU 特征基础上，新增 4 维僚机几何特征：
  - dist_enemy_wing_m
  - distdiff_to_own_m
  - angle_enemy_wing_from_own_deg
  - alt_diff_enemy_wing_m

缺失处理：
  - 若 Wing_X/Y/Z 任一缺失，则 wing_valid=0，四个僚机特征全部置 0
  - 否则 wing_valid=1，正常计算

雷达状态：
  - Status -> Status_code (embedding 用)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def decompose_velocity_NEU(speed, heading_deg, pitch_deg):
    hr = np.deg2rad(heading_deg)
    pr = np.deg2rad(pitch_deg)
    vh = speed * np.cos(pr)
    vn = vh * np.cos(hr)
    ve = vh * np.sin(hr)
    vz = speed * np.sin(pr)
    return vn, ve, vz


def horiz_angle_between_NEU(vn, ve, ln, le, eps=1e-8):
    dot  = vn * ln + ve * le
    magv = np.sqrt(vn ** 2 + ve ** 2)
    magl = np.sqrt(ln ** 2 + le ** 2)
    cosang = dot / ((magv * magl) + eps)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def angular_diff_signed(a, b):
    return (a - b + 180) % 360 - 180


def _norm_status(s: str) -> str:
    if pd.isna(s):
        return "unknown"
    s = str(s).strip().lower().replace(" ", "_")
    if s not in {"search", "lock_on", "track"}:
        s = "unknown"
    return s


def _normalize_intent_string(x):
    if pd.isna(x):
        return x
    s = str(x).strip().lower()
    return s.replace(" ", "_")


def build_features(input_csv, output_csv, keep_labels=True, encoding="utf-8-sig"):
    df = pd.read_csv(input_csv)

    # 确保存在 Status 列
    if "Status" not in df.columns:
        df["Status"] = np.nan

    # ---- 必要列检查（敌机/我机坐标 + 时间）----
    required_cols = ["X_m","Y_m","Z_m","Tgt_X_m","Tgt_Y_m","Tgt_Z_m","Time_s"]
    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        raise ValueError(f"输入CSV缺少必要列: {miss}")

    # ---- 僚机坐标列检查 ----
    wing_cols = ["Wing_X_m", "Wing_Y_m", "Wing_Z_m"]
    miss_w = [c for c in wing_cols if c not in df.columns]
    if miss_w:
        raise ValueError(f"输入CSV缺少僚机坐标列: {miss_w}（无法计算僚机特征）")

    # 数值列 -> 数值
    num_cols = [
        "X_m","Tgt_X_m","Y_m","Tgt_Y_m","Z_m","Tgt_Z_m",
        "Velocity_m_s","Tgt_Velocity_m_s",
        "Heading_deg","Tgt_Heading_deg",
        "Pitch_deg","Tgt_Pitch_deg",
        "Roll_deg","Tgt_Roll_deg","Time_s",
        "Wing_X_m","Wing_Y_m","Wing_Z_m",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 雷达状态标准化 + 编码
    df["Status"] = df["Status"].apply(_norm_status)
    vocab = ["unknown","search","lock_on","track"]
    status_to_id = {s: i for i, s in enumerate(vocab)}
    df["Status_code"] = df["Status"].map(status_to_id).astype("Int64")

    # 敌/我速度
    df["Enemy_speed_m_s"] = df["Velocity_m_s"]
    df["Own_speed_m_s"]   = df["Tgt_Velocity_m_s"]

    # 相对位置（敌 - 我）
    dx = df["X_m"] - df["Tgt_X_m"]
    dy = df["Y_m"] - df["Tgt_Y_m"]
    dz = df["Z_m"] - df["Tgt_Z_m"]
    df["rel_alt_m"] = dz
    df["range_m"]   = np.sqrt(dx**2 + dy**2 + dz**2)

    # 速度分解
    evn, eve, evz = decompose_velocity_NEU(
        df["Velocity_m_s"].fillna(0),
        df["Heading_deg"].fillna(0),
        df["Pitch_deg"].fillna(0),
    )
    ovn, ove, ovz = decompose_velocity_NEU(
        df["Tgt_Velocity_m_s"].fillna(0),
        df["Tgt_Heading_deg"].fillna(0),
        df["Tgt_Pitch_deg"].fillna(0),
    )

    # 接近速率（closure_rate：正=接近）
    los_n, los_e, los_z = -dx, -dy, -dz  # 敌→我
    los_norm = np.sqrt(los_n**2 + los_e**2 + los_z**2) + 1e-8
    un, ue, uz = los_n/los_norm, los_e/los_norm, los_z/los_norm
    rel_vn, rel_ve, rel_vz = evn-ovn, eve-ove, evz-ovz
    df["closure_rate_m_s"] = - (rel_vn*un + rel_ve*ue + rel_vz*uz)

    # 方位角（敌→我，水平）
    bearing_rad = np.arctan2(-dy, -dx)
    df["bearing_deg"] = (np.degrees(bearing_rad) % 360)

    # 敌机水平进入角
    df["enemy_enter_angle_deg"] = horiz_angle_between_NEU(evn, eve, los_n, los_e)

    # ---- 方位角速率（按 token/敌机/目标/时间分组差分）----
    sort_cols = ["Agent_ID","Target_ID","Time_s"]
    if "Dataset_Token" in df.columns:
        sort_cols = ["Dataset_Token"] + sort_cols

    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    group_keys = ["Agent_ID","Target_ID"]
    if "Dataset_Token" in df.columns:
        group_keys = ["Dataset_Token"] + group_keys

    df["bearing_prev"] = df.groupby(group_keys, observed=True)["bearing_deg"].shift(1)
    df["time_prev"]    = df.groupby(group_keys, observed=True)["Time_s"].shift(1)
    df["dbearing_deg"] = np.where(
        df["bearing_prev"].notna(),
        angular_diff_signed(df["bearing_deg"], df["bearing_prev"]),
        np.nan
    )
    df["dt_s"] = df["Time_s"] - df["time_prev"]
    df["bearing_rate_deg_s"] = df["dbearing_deg"] / df["dt_s"].replace(0, np.nan)

    # =========================================================
    # 僚机几何特征（新增4维） + 缺失掩码 wing_valid
    # =========================================================
    wx, wy, wz = df["Wing_X_m"], df["Wing_Y_m"], df["Wing_Z_m"]

    # wing_valid：三列都不缺才算有效
    wing_valid = (~wx.isna()) & (~wy.isna()) & (~wz.isna())
    df["wing_valid"] = wing_valid.astype(np.int64)

    # 为了计算方便：缺失先填 0，但最后用 wing_valid 把特征整体置 0
    wx0 = wx.fillna(0.0)
    wy0 = wy.fillna(0.0)
    wz0 = wz.fillna(0.0)
    m = df["wing_valid"].to_numpy(dtype=np.float32)  # 0/1

    # 1) 敌-僚机距离
    dx_ew = (df["X_m"] - wx0).to_numpy(dtype=np.float32)
    dy_ew = (df["Y_m"] - wy0).to_numpy(dtype=np.float32)
    dz_ew = (df["Z_m"] - wz0).to_numpy(dtype=np.float32)
    dist_ew = np.sqrt(dx_ew**2 + dy_ew**2 + dz_ew**2) * m
    df["dist_enemy_wing_m"] = dist_ew

    # 2) 到我机距离差（敌 vs 僚）
    dx_eo = (df["X_m"] - df["Tgt_X_m"]).to_numpy(dtype=np.float32)
    dy_eo = (df["Y_m"] - df["Tgt_Y_m"]).to_numpy(dtype=np.float32)
    dz_eo = (df["Z_m"] - df["Tgt_Z_m"]).to_numpy(dtype=np.float32)
    dist_eo = np.sqrt(dx_eo**2 + dy_eo**2 + dz_eo**2)

    dx_wo = (wx0 - df["Tgt_X_m"]).to_numpy(dtype=np.float32)
    dy_wo = (wy0 - df["Tgt_Y_m"]).to_numpy(dtype=np.float32)
    dz_wo = (wz0 - df["Tgt_Z_m"]).to_numpy(dtype=np.float32)
    dist_wo = np.sqrt(dx_wo**2 + dy_wo**2 + dz_wo**2)

    df["distdiff_to_own_m"] = (dist_eo - dist_wo) * m

    # 3) 我机视角夹角：angle(own->enemy, own->wing)
    # 向量：R_oe = enemy - own ; R_ow = wing - own
    r1x, r1y, r1z = dx_eo, dy_eo, dz_eo
    r2x, r2y, r2z = dx_wo, dy_wo, dz_wo
    dot = r1x*r2x + r1y*r2y + r1z*r2z
    n1 = np.sqrt(r1x**2 + r1y**2 + r1z**2) + 1e-8
    n2 = np.sqrt(r2x**2 + r2y**2 + r2z**2) + 1e-8
    cosang = np.clip(dot/(n1*n2), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang)) * m
    df["angle_enemy_wing_from_own_deg"] = ang

    # 4) 敌-僚机高度差（带符号）
    df["alt_diff_enemy_wing_m"] = ((df["Z_m"].fillna(0.0) - wz0).to_numpy(dtype=np.float32) * m)

    # 数值安全（把 inf 变 nan，再填 0）
    for c in ["dist_enemy_wing_m","distdiff_to_own_m","angle_enemy_wing_from_own_deg","alt_diff_enemy_wing_m"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 可选：意图标签（你已改成 侦察/攻击/防御/规避/撤退/其他）
    if keep_labels and ("Action_Intent" in df.columns):
        df["Intent_Label_raw"] = df["Action_Intent"]
        mapping = {
            "search":"侦察",
            "lock_on":"攻击","attack_maneuver":"攻击",
            "evasive_maneuver":"防御","defensive_positioning":"防御","defensive_maneuver":"防御",
            "escape":"撤退","coordination":"规避",
        }
        df["_norm_intent"] = df["Action_Intent"].apply(_normalize_intent_string)
        df["Intent_Label"] = df["_norm_intent"].map(mapping)


        df = df[df["Intent_Label"].notna()].reset_index(drop=True)

        df.drop(columns=["_norm_intent"], inplace=True)

    # 导出列
    out_cols = []
    if "Dataset_Token" in df.columns:
        out_cols.append("Dataset_Token")
    out_cols += ["Agent_ID","Target_ID","Time_s"]

    feature_cols = [
        "Enemy_speed_m_s","Own_speed_m_s","rel_alt_m","range_m",
        "enemy_enter_angle_deg","bearing_deg","bearing_rate_deg_s","closure_rate_m_s",
        "Status","Status_code",
        # 新增 4 维僚机几何
        "dist_enemy_wing_m",
        "distdiff_to_own_m",
        "angle_enemy_wing_from_own_deg",
        "alt_diff_enemy_wing_m",
        # 缺失掩码（建议保留）
        "wing_valid",
    ]
    out_cols += feature_cols

    if keep_labels and ("Action_Intent" in df.columns):
        out_cols += ["Intent_Label_raw","Intent_Label"]

    # 最终输出
    df_out = df[out_cols].copy()

    if "Dataset_Token" in df_out.columns:
        df_out = df_out.sort_values(["Dataset_Token","Agent_ID","Target_ID","Time_s"], kind="mergesort").reset_index(drop=True)
    else:
        df_out = df_out.sort_values(["Agent_ID","Target_ID","Time_s"], kind="mergesort").reset_index(drop=True)

    df_out.to_csv(output_csv, index=False, encoding=encoding)
    print("✅ Saved features to:", output_csv)
    print("[INFO] Radar Status vocab -> index:", {s:i for i,s in enumerate(vocab)})
    print("[INFO] wing_valid=1 ratio:", float(df_out["wing_valid"].mean()))

    # 类别统计（若存在 Intent_Label）
    if "Intent_Label" in df_out.columns:
        print("\n=== Intent_Label 类别分布 ===")
        counts = df_out["Intent_Label"].value_counts(dropna=False)
        percents = counts / counts.sum() * 100
        stats_df = pd.DataFrame({"数量": counts, "占比(%)": percents.round(2)})
        print(stats_df)
        print("================================")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="合并后的 CSV（单场或全局）")
    ap.add_argument("--output", required=True, help="输出特征 CSV")
    ap.add_argument("--no-labels", action="store_true", help="不输出标签列，即使输入有 Action_Intent")
    ap.add_argument("--encoding", default="utf-8-sig")
    args = ap.parse_args()

    build_features(
        input_csv=args.input,
        output_csv=args.output,
        keep_labels=(not args.no_labels),
        encoding=args.encoding
    )