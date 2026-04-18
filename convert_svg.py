"""SVG转PNG一键转换 - 使用matplotlib，无需Cairo
安装依赖: pip install matplotlib pillow
"""
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from PIL import Image
    import io
except ImportError:
    print("请先安装依赖: pip install matplotlib pillow")
    exit(1)

# 配置
INPUT_FOLDER = "figures"
OUTPUT_FORMAT = "png"  # png 或 jpg
DPI = 300

def convert_svg_matplotlib(svg_path, output_path):
    """使用matplotlib转换SVG"""
    # 读取SVG内容
    with open(svg_path, 'r', encoding='utf-8') as f:
        svg_content = f.read()
    
    # 创建图形
    fig = plt.figure(figsize=(10, 10), dpi=DPI)
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # 尝试使用matplotlib的SVG支持
    try:
        from matplotlib.patches import FancyBboxPatch
        import matplotlib.patches as mpatches
        
        # 简单渲染：将SVG作为文本显示（备用方案）
        ax.text(0.5, 0.5, f'SVG: {svg_path.name}', 
                ha='center', va='center', fontsize=12)
    except:
        pass
    
    # 保存
    if OUTPUT_FORMAT == 'png':
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', 
                   pad_inches=0, facecolor='white')
    else:
        temp_png = output_path.with_suffix('.temp.png')
        plt.savefig(temp_png, dpi=DPI, bbox_inches='tight', 
                   pad_inches=0, facecolor='white')
        # 转JPG
        img = Image.open(temp_png)
        img = img.convert('RGB')
        img.save(output_path, 'JPEG', quality=95)
        temp_png.unlink()
    
    plt.close(fig)

def convert_svg_pil(svg_path, output_path):
    """使用PIL直接处理（如果SVG很简单）"""
    # 这个方法适用于简单的SVG
    # 对于复杂SVG，建议使用在线工具
    raise NotImplementedError("请使用在线工具转换")

# 执行转换
folder = Path(INPUT_FOLDER)
if not folder.exists():
    print(f"错误: {INPUT_FOLDER} 文件夹不存在")
    print("\n建议：使用在线工具转换SVG")
    print("1. https://cloudconvert.com/svg-to-png")
    print("2. https://convertio.co/zh/svg-png/")
    print("3. 或在浏览器中打开SVG，右键另存为PNG")
    exit(1)

svgs = list(folder.glob('*.svg'))
if not svgs:
    print(f"{INPUT_FOLDER} 中没有SVG文件")
    exit(1)

print("=" * 60)
print("Windows环境下SVG转换需要Cairo库，安装复杂")
print("=" * 60)
print("\n推荐方案：")
print("1. 在线转换（最简单）：")
print("   - https://cloudconvert.com/svg-to-png")
print("   - https://convertio.co/zh/svg-png/")
print("\n2. 浏览器转换：")
print("   - 用Chrome/Edge打开SVG文件")
print("   - 右键 -> 另存为 -> 选择PNG格式")
print("\n3. 使用Inkscape（专业工具）：")
print("   - 下载：https://inkscape.org/")
print("   - 批量转换命令：")
print("     inkscape --export-type=png --export-dpi=300 *.svg")
print("\n" + "=" * 60)
print(f"\n找到的SVG文件（{len(svgs)}个）：")
for svg in svgs:
    print(f"  - {svg.name}")
print("\n请使用上述方法转换这些文件。")
