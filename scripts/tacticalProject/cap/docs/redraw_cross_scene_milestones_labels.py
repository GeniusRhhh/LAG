from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib import font_manager


DOC_DIR = Path(r"D:\Pycharm\LAG\scripts\tacticalProject\cap\docs")
ASSET_DIR = DOC_DIR / "第六章仿真验证文档_0522定样本重写_assets"
ASSET_PATH = ASSET_DIR / "fig6_25_cross_scene_milestones.png"
TMP_PATH = DOC_DIR / "_tmp_cross_scene_milestones.png"


def pick_font() -> str:
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            return name
    return "sans-serif"


def main() -> None:
    font_name = pick_font()
    plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False

    scene_order = ["场景一", "场景二", "场景三"]
    scene_y = {"场景一": 2, "场景二": 1, "场景三": 0}

    scene_data = {
        "场景一": {
            "max_time": 1200.0,
            "milestones": [
                ("首次稳定交战就绪", 104.4, "#43aa8b", "o"),
                ("首次发射门通过", 266.6, "#f8961e", "s"),
                ("首次规避", 322.0, "#d1495b", "D"),
                ("首次截获重组", 446.6, "#2a9d8f", "^"),
                ("首次敌机被击落", 333.8, "#577590", "P"),
            ],
        },
        "场景二": {
            "max_time": 1200.0,
            "milestones": [
                ("首次稳定交战就绪", 54.8, "#43aa8b", "o"),
                ("首次发射门通过", 236.2, "#f8961e", "s"),
                ("首次规避", 279.0, "#d1495b", "D"),
                ("首次截获重组", 390.2, "#2a9d8f", "^"),
                ("首次敌机被击落", 311.2, "#577590", "P"),
            ],
        },
        "场景三": {
            "max_time": 1200.0,
            "milestones": [
                ("首次稳定交战就绪", 14.8, "#43aa8b", "o"),
                ("首次发射门通过", 64.0, "#f8961e", "s"),
                ("首次规避", 131.8, "#d1495b", "D"),
                ("首次截获重组", 261.4, "#2a9d8f", "^"),
                ("首次敌机被击落", 191.2, "#577590", "P"),
            ],
        },
    }

    fig, ax = plt.subplots(figsize=(13.4, 6.4), dpi=220)
    ax.set_facecolor("#fbf8f1")
    fig.patch.set_facecolor("#fbf8f1")

    for spine in ax.spines.values():
        spine.set_visible(False)

    for scene in scene_order:
        y = scene_y[scene]
        ax.hlines(y, 0.0, scene_data[scene]["max_time"], color="#d9d2c3", linewidth=1.3, zorder=1)
        for label, value, color, marker in scene_data[scene]["milestones"]:
            ax.scatter(
                [value],
                [y],
                s=105,
                marker=marker,
                color=color,
                edgecolors="white",
                linewidths=1.2,
                zorder=4,
            )
            ax.text(
                value,
                y + 0.12,
                label,
                fontsize=9.4,
                color=color,
                ha="center",
                va="bottom",
                rotation=25,
            )

    ax.set_title("三场景关键里程碑横向对比图", fontsize=16, color="#3d405b", pad=14, weight="bold")
    ax.set_xlabel("仿真时间 / s", fontsize=11, color="#3d405b")
    ax.set_xlim(0.0, 1200.0)
    ax.set_ylim(-0.5, 2.6)
    ax.set_yticks([scene_y[scene] for scene in scene_order])
    ax.set_yticklabels(scene_order, fontsize=11, color="#3d405b")
    ax.tick_params(axis="x", colors="#3d405b", labelsize=10)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#d9d2c3", linestyle="--", linewidth=0.7, alpha=0.75)

    fig.tight_layout()
    fig.savefig(TMP_PATH, bbox_inches="tight")
    plt.close(fig)
    shutil.copyfile(TMP_PATH, ASSET_PATH)
    TMP_PATH.unlink(missing_ok=True)
    print(str(ASSET_PATH))


if __name__ == "__main__":
    main()
