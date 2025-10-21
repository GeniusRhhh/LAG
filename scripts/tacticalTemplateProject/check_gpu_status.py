#!/usr/bin/env python3
"""
GPU/CPU 状态检测脚本
用于检测深度学习环境是否正确配置并使用GPU
"""

import sys
import os

def check_pytorch_gpu():
    """检查 PyTorch GPU 可用性"""
    print("=" * 80)
    print("PyTorch GPU 环境检查")
    print("=" * 80)
    
    try:
        import torch
        print(f"✅ PyTorch 已安装")
        print(f"   版本: {torch.__version__}")
        print(f"   Python 版本: {sys.version.split()[0]}")
        
        # 检查 CUDA 可用性
        cuda_available = torch.cuda.is_available()
        print(f"\n{'✅' if cuda_available else '❌'} CUDA 可用性: {cuda_available}")
        if cuda_available:
            # GPU 详细信息
            gpu_count = torch.cuda.device_count()
            print(f"   检测到 {gpu_count} 个 GPU 设备:")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   - GPU {i}: {gpu_name}")
                print(f"     显存: {gpu_memory:.2f} GB")
            
            # 当前默认设备
            current_device = torch.cuda.current_device()
            print(f"\n   当前默认 GPU: {current_device}")
            print(f"   CUDA 版本: {torch.version.cuda}")
            
            # 测试 GPU 操作
            print("\n🧪 测试 GPU 操作...")
            try:
                device = torch.device("cuda")
                
                # 创建张量并移动到GPU
                x = torch.rand(1000, 1000).to(device)
                y = torch.rand(1000, 1000).to(device)
                
                # 执行矩阵乘法
                z = torch.matmul(x, y)
                
                print(f"   ✅ 成功在 GPU 上执行矩阵运算")
                print(f"   张量设备: {z.device}")
                print(f"   张量形状: {z.shape}")
                
                # 清理显存
                del x, y, z
                torch.cuda.empty_cache()
                
                return True
                
            except Exception as e:
                print(f"   ❌ GPU 操作失败: {e}")
                return False
        else:
            print("\n⚠️  未检测到可用的 CUDA GPU")
            print("   可能的原因:")
            print("   1. 未安装 NVIDIA 显卡驱动")
            print("   2. 安装的是 CPU 版本的 PyTorch")
            print("   3. CUDA Toolkit 版本不兼容")
            print("\n   安装 GPU 版本 PyTorch 的命令:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
            
    except ImportError:
        print("❌ PyTorch 未安装")
        print("   请先安装 PyTorch:")
        print("   pip install torch")
        return False
    except Exception as e:
        print(f"❌ 检查过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_tensorflow_gpu():
    """检查 TensorFlow GPU 可用性"""
    print("\n" + "=" * 80)
    print("TensorFlow GPU 环境检查")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow 已安装")
        print(f"   版本: {tf.__version__}")
        
        # 检查 GPU 设备
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ 检测到 {len(gpus)} 个 GPU 设备:")
            for gpu in gpus:
                print(f"   - {gpu.name}")
            
            # 测试 GPU 操作
            print("\n🧪 测试 GPU 操作...")
            try:
                with tf.device('/GPU:0'):
                    a = tf.random.normal([1000, 1000])
                    b = tf.random.normal([1000, 1000])
                    c = tf.matmul(a, b)
                
                print(f"   ✅ 成功在 GPU 上执行矩阵运算")
                print(f"   张量形状: {c.shape}")
                return True
                
            except Exception as e:
                print(f"   ❌ GPU 操作失败: {e}")
                return False
        else:
            print("❌ 未检测到可用的 GPU")
            return False
            
    except ImportError:
        print("ℹ️  TensorFlow 未安装（如果只使用 PyTorch 可忽略）")
        return None
    except Exception as e:
        print(f"❌ 检查过程出错: {e}")
        return False


def check_system_info():
    """检查系统信息"""
    print("\n" + "=" * 80)
    print("系统信息")
    print("=" * 80)
    
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 检查 CUDA 环境变量
    cuda_path = os.environ.get('CUDA_PATH', 'Not set')
    cuda_home = os.environ.get('CUDA_HOME', 'Not set')
    print(f"\nCUDA_PATH: {cuda_path}")
    print(f"CUDA_HOME: {cuda_home}")


def run_performance_test():
    """运行简单的性能测试对比 CPU vs GPU"""
    print("\n" + "=" * 80)
    print("性能测试: CPU vs GPU")
    print("=" * 80)
    
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            print("⚠️  GPU 不可用，跳过性能测试")
            return
        
        # 测试参数
        size = 5000
        iterations = 10
        
        print(f"测试配置: {size}x{size} 矩阵乘法, {iterations} 次迭代\n")
        
        # CPU 测试
        print("🖥️  CPU 测试...")
        device_cpu = torch.device("cpu")
        x_cpu = torch.rand(size, size, device=device_cpu)
        y_cpu = torch.rand(size, size, device=device_cpu)
        
        start = time.time()
        for _ in range(iterations):
            z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start
        
        print(f"   CPU 总耗时: {cpu_time:.3f} 秒")
        print(f"   CPU 平均耗时: {cpu_time/iterations:.3f} 秒/次")
        
        # GPU 测试
        print("\n🎮 GPU 测试...")
        device_gpu = torch.device("cuda")
        x_gpu = torch.rand(size, size, device=device_gpu)
        y_gpu = torch.rand(size, size, device=device_gpu)
        
        # 预热
        for _ in range(3):
            _ = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(iterations):
            z_gpu = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"   GPU 总耗时: {gpu_time:.3f} 秒")
        print(f"   GPU 平均耗时: {gpu_time/iterations:.3f} 秒/次")
        
        # 对比
        speedup = cpu_time / gpu_time
        print(f"\n📊 性能对比:")
        print(f"   GPU 加速比: {speedup:.2f}x")
        
        if speedup > 10:
            print("   ✅ GPU 性能优异！")
        elif speedup > 3:
            print("   ✅ GPU 性能良好")
        elif speedup > 1:
            print("   ⚠️  GPU 性能提升有限，可能存在瓶颈")
        else:
            print("   ❌ GPU 性能异常，请检查配置")
        
        # 清理
        del x_cpu, y_cpu, z_cpu, x_gpu, y_gpu, z_gpu
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🔍 深度学习环境 GPU/CPU 状态检测")
    print("=" * 80)
    print()
    
    # 检查系统信息
    check_system_info()
    
    # 检查 PyTorch
    pytorch_ok = check_pytorch_gpu()
    
    # 检查 TensorFlow (可选)
    tensorflow_ok = check_tensorflow_gpu()
    
    # 性能测试
    if pytorch_ok:
        run_performance_test()
    
    # 总结
    print("\n" + "=" * 80)
    print("检测总结")
    print("=" * 80)
    
    if pytorch_ok:
        print("✅ PyTorch GPU 环境配置正确")
        print("   你的深度学习模型将运行在 GPU 上，享受高性能计算！")
    else:
        print("❌ PyTorch GPU 环境未正确配置")
        print("   你的深度学习模型将运行在 CPU 上，速度会较慢")
        print("\n   建议:")
        print("   1. 检查是否安装了 NVIDIA 显卡驱动")
        print("   2. 重新安装支持 CUDA 的 PyTorch 版本")
        print("   3. 参考官方文档: https://pytorch.org/get-started/locally/")
    
    print("\n" + "=" * 80)
    print("注意: JSBSim 飞行仿真部分始终运行在 CPU 上")
    print("      只有深度学习模型（LSTM等）会使用 GPU 加速")
    print("=" * 80)


if __name__ == "__main__":
    main()
