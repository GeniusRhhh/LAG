from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


BASE_DIR = Path(r"D:\Pycharm\LAG\scripts\tacticalProject\cap\docs")
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 1.2

    n001ve_x = [0, 50, 120, 170, 180, 200, 230]
    n001ve_y = [0.95, 0.88, 0.68, 0.50, 0.32, 0.0, 0.0]
    apg68_x = [0, 50, 90, 130, 180, 200, 230]
    apg68_y = [0.97, 0.92, 0.84, 0.76, 0.58, 0.0, 0.0]

    fig, ax = plt.subplots(figsize=(13.6, 6.0))

    ax.axvspan(0, 50, facecolor="#EAF4EA", alpha=0.80, zorder=0)
    ax.axvspan(50, 130, facecolor="#EEF4FB", alpha=0.82, zorder=0)
    ax.axvspan(130, 180, facecolor="#FFF3E3", alpha=0.82, zorder=0)
    ax.axvspan(180, 200, facecolor="#FDECEC", alpha=0.90, zorder=0)
    ax.axvspan(200, 245, facecolor="#F4F4F4", alpha=0.95, zorder=0)

    ax.step(n001ve_x, n001ve_y, where="post", color="#C0504D", linewidth=3.2, label="N001VE", zorder=4)
    ax.step(apg68_x, apg68_y, where="post", color="#1F77B4", linewidth=3.2, label="AN/APG-68(V)9", zorder=4)

    for x in [50, 90, 120, 130, 170, 180, 200]:
        ax.axvline(x, color="#666666", linestyle="--", linewidth=1.0, alpha=0.60, zorder=1)
    ax.axvline(200, color="#333333", linestyle="--", linewidth=1.5, alpha=0.95, zorder=2)

    label_box = dict(facecolor="white", alpha=0.88, edgecolor="none", pad=0.18)
    seg_red = dict(facecolor="white", alpha=0.88, edgecolor="#C0504D", linewidth=0.5, pad=0.16)
    seg_blue = dict(facecolor="white", alpha=0.88, edgecolor="#1F77B4", linewidth=0.5, pad=0.16)

    ax.text(25, 0.072, "近距高概率区", ha="center", va="bottom", fontsize=13, color="#2F5D2F", bbox=label_box)
    ax.text(90, 0.072, "中距主工作区", ha="center", va="bottom", fontsize=13, color="#355C8A", bbox=label_box)
    ax.text(155, 0.072, "远距衰减区", ha="center", va="bottom", fontsize=13, color="#A26A22", bbox=label_box)

    ax.text(16, 0.895, "0-50 km: 0.95", color="#C0504D", fontsize=10.5, ha="left", va="top", bbox=seg_red)
    ax.text(57, 0.840, "50-120 km: 0.88", color="#C0504D", fontsize=10.5, ha="left", va="top", bbox=seg_red)
    ax.text(122, 0.650, "120-170 km: 0.68", color="#C0504D", fontsize=10.5, ha="left", va="top", bbox=seg_red)
    ax.text(164, 0.470, "170-180 km: 0.50", color="#C0504D", fontsize=10.5, ha="left", va="top", bbox=seg_red)
    ax.text(181.0, 0.285, "180-200 km: 0.32", color="#C0504D", fontsize=10.5, ha="left", va="top", bbox=seg_red)

    ax.text(16, 0.985, "0-50 km: 0.97", color="#1F77B4", fontsize=10.5, ha="left", va="bottom", bbox=seg_blue)
    ax.text(57, 0.935, "50-90 km: 0.92", color="#1F77B4", fontsize=10.5, ha="left", va="bottom", bbox=seg_blue)
    ax.text(94.0, 0.875, "90-130 km: 0.84", color="#1F77B4", fontsize=10.5, ha="left", va="bottom", bbox=seg_blue)
    ax.text(131.0, 0.790, "130-180 km: 0.76", color="#1F77B4", fontsize=10.5, ha="left", va="bottom", bbox=seg_blue)
    ax.text(181.0, 0.610, "180-200 km: 0.58", color="#1F77B4", fontsize=10.5, ha="left", va="bottom", bbox=seg_blue)

    # 只保留一个“超限区”标题，下面是两行说明。
    ax.text(214.5, 0.090, "超限区", ha="left", va="center", fontsize=12.0, color="#666666",
            bbox=dict(facecolor="white", alpha=0.92, edgecolor="none", pad=0.12))
    ax.text(214.5, 0.045, "200 km 之后", ha="left", va="bottom", fontsize=9.4, color="#444444",
            bbox=dict(facecolor="white", alpha=0.92, edgecolor="none", pad=0.12))
    ax.text(214.5, 0.014, "P = 0", ha="left", va="bottom", fontsize=9.4, color="#444444",
            bbox=dict(facecolor="white", alpha=0.92, edgecolor="none", pad=0.12))

    ax.set_title("雷达基础探测概率与距离关系", fontsize=16, pad=14)
    ax.set_xlabel("距离 / km", fontsize=13)
    ax.set_ylabel(r"基础探测概率 $P_{\mathrm{base}}$", fontsize=13)

    ax.set_xlim(0, 245)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.tick_params(axis="both", which="minor", length=0)
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, color="#C8C8C8", alpha=0.75, zorder=0)

    leg = ax.legend(loc="upper right", frameon=True, framealpha=0.96, edgecolor="#BBBBBB", fontsize=11)
    leg.get_frame().set_facecolor("white")

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    png_path = FIG_DIR / "radar_detection_probability_curve.png"
    svg_path = FIG_DIR / "radar_detection_probability_curve.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {svg_path}")


if __name__ == "__main__":
    main()
