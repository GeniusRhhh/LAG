#!/usr/bin/env python3
"""
PyTorch Nightly版本安装脚本（支持Python 3.13）
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
    print("🚀 PyTorch Nightly版本安装程序（支持Python 3.13）")
    print("="*80)
    print()
    print("检测到的配置:")
    print(f"  - Python版本: {sys.version}")
    print("  - GPU: NVIDIA GeForce RTX 5060 Ti")
    print("  - CUDA: 13.0")
    print()
    print("⚠️  警告: 你的Python 3.13.5太新了！")
    print("   PyTorch稳定版最高支持Python 3.12")
    print()
    print("解决方案:")
    print("  方案1（推荐）: 降级到Python 3.12并重新创建虚拟环境")
    print("  方案2（临时）: 安装PyTorch nightly开发版（可能不稳定）")
    print()
    
    response = input("选择方案 (1/2): ")
    
    if response == "1":
        print("\n请按以下步骤操作:")
        print("1. 下载并安装Python 3.12: https://www.python.org/downloads/")
        print("2. 删除现有虚拟环境:")
        print("   rmdir /s /q C:\\Users\\ZRF\\PycharmProjects\\LAG\\.venv")
        print("3. 使用Python 3.12创建新虚拟环境:")
        print("   python3.12 -m venv C:\\Users\\ZRF\\PycharmProjects\\LAG\\.venv")
        print("4. 在PyCharm中配置新的解释器")
        print("5. 重新运行 install_pytorch_gpu.py")
        return
    
    elif response == "2":
        print("\n⚠️  安装PyTorch nightly开发版...")
        print("注意: 这是不稳定版本，可能存在bug")
        print()
        
        # 尝试安装nightly版本
        install_cmd = f"{sys.executable} -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121"
        
        success = run_command(install_cmd, "安装PyTorch Nightly")
        
        if success:
            # 验证安装
            verify_script = """
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
    x = torch.rand(100, 100).to(device)
    print(f"GPU 测试: 成功")
"""
            verify_cmd = f'{sys.executable} -c "{verify_script}"'
            run_command(verify_cmd, "验证GPU可用性")
        else:
            print("\n❌ 安装失败！")
            print("建议使用方案1：降级到Python 3.12")
    else:
        print("已取消")

if __name__ == "__main__":
    main()
