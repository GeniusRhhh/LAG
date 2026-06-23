# -*- coding: utf-8 -*-
"""
make_intent_features_NEU.py
---------------------------
根据汇总表，生成 NEU 坐标系下的压缩特征表。

输出特征（共9项 + 1项数值编码）：
  1) Enemy_speed_m_s
  2) Own_speed_m_s
  3) rel_alt_m
  4) range_m
  5) enemy_enter_angle_deg
  6) bearing_deg
  7) bearing_rate_deg_s
  8) closure_rate_m_s
  9) Status         (标准化后的文本类别: search/lock_on/track/unknown)
  + Status_code     (供神经网络 embedding 使用的整数索引)

基础标识列：
  [可选] Dataset_Token, Agent_ID, Target_ID, Time_s

可选输出：
  Intent_Label_raw, Intent_Label（若输入有 Action_Intent；--no-labels 可关闭）

坐标/角度约定：
- NEU：X=North, Y=East, Z=Up
- heading: 0°=北, 顺时针为正
- pitch: 抬头为正
- LOS: 敌 → 我
- closure_rate: 正=接近，负=远离
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号
# ----------------- 工具函数 -----------------
def decompose_velocity_NEU(speed, heading_deg, pitch_deg):
    """NEU坐标分解：返回 (vn, ve, vz)
    heading: 0°=北, 顺时针为正; pitch: 抬头为正
    """
    hr = np.deg2rad(heading_deg)
    pr = np.deg2rad(pitch_deg)
    vh = speed * np.cos(pr)
    vn = vh * np.cos(hr)
    ve = vh * np.sin(hr)
    vz = speed * np.sin(pr)
    return vn, ve, vz

def horiz_angle_between_NEU(vn, ve, ln, le, eps=1e-8):
    """水平进入角（0~180°）：速度水平分量 vs 水平LOS 的夹角"""
    dot  = vn*ln + ve*le
    magv = np.sqrt(vn**2 + ve**2)
    magl = np.sqrt(ln**2 + le**2)
    cosang = dot / ((magv * magl) + eps)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def angular_diff_signed(a, b):
    """角度差 a-b ∈ (-180,180]，用于方位角差分防跳变"""
    return (a - b + 180) % 360 - 180

def _norm_status(s: str) -> str:
    if pd.isna(s): return "unknown"
    s = str(s).strip().lower().replace(" ", "_")
    if s not in {"search","lock_on","track"}:
        s = "unknown"
    return s

def _normalize_intent_string(x):
    if pd.isna(x): return x
    s = str(x).strip().lower()
    return s.replace(" ", "_")

# ----------------- 主流程 -----------------
def build_features(input_csv, output_csv, keep_labels=True, encoding="utf-8-sig"):
    df = pd.read_csv(input_csv)

    # 确保存在 Status 列
    if "Status" not in df.columns:
        df["Status"] = np.nan

    # 数值列 -> 数值
    num_cols = [
        "X_m","Tgt_X_m","Y_m","Tgt_Y_m","Z_m","Tgt_Z_m",
        "Velocity_m_s","Tgt_Velocity_m_s",
        "Heading_deg","Tgt_Heading_deg",
        "Pitch_deg","Tgt_Pitch_deg",
        "Roll_deg","Tgt_Roll_deg","Time_s"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 雷达状态标准化 + 编码
    df["Status"] = df["Status"].apply(_norm_status)
    vocab = ["unknown","search","lock_on","track"]
    status_to_id = {s:i for i,s in enumerate(vocab)}
    df["Status_code"] = df["Status"].map(status_to_id).astype("Int64")

    # 敌/我速度
    df["Enemy_speed_m_s"] = df["Velocity_m_s"]
    df["Own_speed_m_s"]   = df["Tgt_Velocity_m_s"]

    # 相对位置
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

    # 方位角速率
    sort_cols = ["Agent_ID","Target_ID","Time_s"]
    if "Dataset_Token" in df.columns:
        sort_cols = ["Dataset_Token"] + sort_cols
    agent_order = None
    if "Agent_ID" in df.columns:
        agent_order = pd.CategoricalDtype(categories=["B0100","B0200"], ordered=True)
        df["Agent_ID"] = df["Agent_ID"].astype(agent_order)

    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    group_keys = ["Agent_ID","Target_ID"]
    if "Dataset_Token" in df.columns:
        group_keys = ["Dataset_Token"] + group_keys

    df["bearing_prev"] = df.groupby(group_keys, observed=True)["bearing_deg"].shift(1)
    df["time_prev"]    = df.groupby(group_keys, observed=True)["Time_s"].shift(1)
    df["dbearing_deg"] = np.where(df["bearing_prev"].notna(),
                                  angular_diff_signed(df["bearing_deg"], df["bearing_prev"]),
                                  np.nan)
    df["dt_s"] = df["Time_s"] - df["time_prev"]
    df["bearing_rate_deg_s"] = df["dbearing_deg"] / df["dt_s"].replace(0, np.nan)

    # 可选：意图标签
    if keep_labels and ("Action_Intent" in df.columns):
        df["Intent_Label_raw"] = df["Action_Intent"]
        mapping = {
            "search":"探测","formation_maintain":"中立",
            "lock_on":"攻击","attack_maneuver":"攻击",
            "evasive_maneuver":"防御","defensive_positioning":"防御","defensive_maneuver":"防御",
            "escape":"逃逸","coordination":"协同",
        }
        df["_norm_intent"] = df["Action_Intent"].apply(_normalize_intent_string)
        df["Intent_Label"] = df["_norm_intent"].map(mapping).fillna("其他")
        df.drop(columns=["_norm_intent"], inplace=True)

    # 导出列
    out_cols = []
    if "Dataset_Token" in df.columns: out_cols.append("Dataset_Token")
    out_cols += ["Agent_ID","Target_ID","Time_s"]

    feature_cols = [
        "Enemy_speed_m_s","Own_speed_m_s","rel_alt_m","range_m",
        "enemy_enter_angle_deg","bearing_deg","bearing_rate_deg_s","closure_rate_m_s",
        "Status","Status_code"
    ]
    out_cols += feature_cols

    if keep_labels and ("Action_Intent" in df.columns):
        out_cols += ["Intent_Label_raw","Intent_Label"]

    df_out = df[out_cols].copy()
    if agent_order is not None:
        df_out["Agent_ID"] = df_out["Agent_ID"].astype(agent_order)

    # 终排序
    # 终排序（按 Dataset_Token → Agent_ID → Target_ID → Time_s）
    if "Dataset_Token" in df_out.columns:
        df_out = df_out.sort_values(["Dataset_Token", "Agent_ID", "Target_ID", "Time_s"], kind="mergesort").reset_index(
            drop=True)
    else:
        df_out = df_out.sort_values(["Agent_ID", "Target_ID", "Time_s"], kind="mergesort").reset_index(drop=True)

    # ===== 保存 =====
    df_out.to_csv(output_csv, index=False, encoding=encoding)
    print("✅ Saved features to:", output_csv)
    print("[INFO] Radar Status vocab -> index:", {s:i for i,s in enumerate(vocab)})

    # ===== 类别统计与绘图（若存在 Intent_Label） =====
    if "Intent_Label" in df_out.columns:
        print("\n=== Intent_Label 类别分布 ===")
        counts = df_out["Intent_Label"].value_counts(dropna=False)
        percents = counts / counts.sum() * 100
        stats_df = pd.DataFrame({"数量": counts, "占比(%)": percents.round(2)})
        print(stats_df)
        print("================================")

        # 保存分布柱状图（无交互环境更安全）
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 5))
            counts.plot(kind="bar", edgecolor="black", ax=ax)
            ax.set_title("意图类别分布")
            ax.set_xlabel("类别")
            ax.set_ylabel("样本数量")
            ax.tick_params(axis="x", rotation=30)

            total = counts.sum()
            for i, v in enumerate(counts):
                pct = v / total * 100
                ax.text(i, v + max(1, total * 0.01), f"{pct:.1f}%", ha="center", fontsize=10)

            plt.tight_layout()
            img_path = str(output_csv).rsplit(".", 1)[0] + "_label_dist.png"
            fig.savefig(img_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print("📊 已保存类别分布图：", img_path)
        except Exception as e:
            print("[WARN] 绘图失败:", e)
    else:
        print("[INFO] 未找到 Intent_Label 列，跳过类别统计与绘图。")

# ----------------- CLI -----------------
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
