#!/usr/bin/env python3
"""
PyTorch GPU版本安装脚本
自动卸载CPU版本并安装适配你显卡的GPU版本
"""

import subprocess
import sys

def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*80}")
    print(f"🔧 {description}")
    print(f"{'='*80}")
    print(f"执行命令: {cmd}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True,
            capture_output=False
        )
        print(f"✅ {description} - 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - 失败")
        print(f"错误信息: {e}")
        return False

def main():
    print("="*80)
    print("🚀 PyTorch GPU版本安装程序")
    print("="*80)
    print()
    print("检测到的配置:")
    print("  - GPU: NVIDIA GeForce RTX 5060 Ti")
    print("  - CUDA: 13.0")
    print("  - 推荐安装: PyTorch with CUDA 12.1 (向下兼容)")
    print()
    print("安装步骤:")
    print("  1. 卸载现有的CPU版本PyTorch")
    print("  2. 安装GPU版本PyTorch (CUDA 12.1)")
    print("  3. 验证GPU是否可用")
    print()
    
    response = input("是否继续安装？(y/n): ")
    if response.lower() != 'y':
        print("安装已取消")
        return
    
    # 步骤1: 卸载现有版本
    print("\n" + "="*80)
    print("步骤 1/3: 卸载现有PyTorch")
    print("="*80)
    
    uninstall_cmd = f"{sys.executable} -m pip uninstall torch torchvision torchaudio -y"
    run_command(uninstall_cmd, "卸载现有PyTorch")
    
    # 步骤2: 安装GPU版本
    print("\n" + "="*80)
    print("步骤 2/3: 安装GPU版本PyTorch")
    print("="*80)
    print("⚠️  这可能需要几分钟时间，请耐心等待...")
    print()
    
    # 使用CUDA 12.1版本（与CUDA 13.0向下兼容）
    install_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    
    success = run_command(install_cmd, "安装GPU版本PyTorch")
    
    if not success:
        print("\n❌ 安装失败！")
        print("请尝试手动安装:")
        print(f"  {install_cmd}")
        return
    
    # 步骤3: 验证安装
    print("\n" + "="*80)
    print("步骤 3/3: 验证GPU是否可用")
    print("="*80)
    
    verify_script = """
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    
    # 测试GPU操作
    device = torch.device("cuda")
    x = torch.rand(100, 100).to(device)
    y = torch.rand(100, 100).to(device)
    z = torch.matmul(x, y)
    print(f"GPU 测试: 成功在 {z.device} 上执行运算")
else:
    print("警告: GPU 仍然不可用")
"""
    
    verify_cmd = f'{sys.executable} -c "{verify_script}"'
    run_command(verify_cmd, "验证GPU可用性")
    
    print("\n" + "="*80)
    print("✅ 安装完成！")
    print("="*80)
    print()
    print("下一步:")
    print("  1. 重新运行 check_gpu_status.py 验证配置")
    print("  2. 运行你的深度学习训练脚本")
    print()
    print("注意:")
    print("  - 如果PyCharm中仍显示CPU版本，请重启PyCharm")
    print("  - 确保PyCharm使用的是正确的Python解释器")
    print()

if __name__ == "__main__":
    main()
