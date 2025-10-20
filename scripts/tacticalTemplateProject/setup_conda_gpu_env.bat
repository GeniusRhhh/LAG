@echo off
REM Anaconda GPU环境自动配置脚本
REM 创建Python 3.12环境并安装GPU版PyTorch

echo ================================================================================
echo 🐍 Anaconda GPU环境配置脚本
echo ================================================================================
echo.
echo 此脚本将:
echo   1. 创建名为 lag_gpu 的Python 3.12环境
echo   2. 安装GPU版PyTorch (CUDA 12.1)
echo   3. 安装项目依赖
echo   4. 验证GPU配置
echo.
echo 适配配置:
echo   - GPU: NVIDIA GeForce RTX 5060 Ti
echo   - CUDA: 13.0 (驱动) / 12.1 (PyTorch)
echo   - Python: 3.12
echo.
pause

echo.
echo ================================================================================
echo 步骤 1/5: 创建conda环境 (lag_gpu, Python 3.12)
echo ================================================================================
call conda create -n lag_gpu python=3.12 -y
if errorlevel 1 (
    echo ❌ 创建环境失败！
    pause
    exit /b 1
)
echo ✅ 环境创建成功

echo.
echo ================================================================================
echo 步骤 2/5: 激活环境
echo ================================================================================
call conda activate lag_gpu
if errorlevel 1 (
    echo ❌ 激活环境失败！
    pause
    exit /b 1
)
echo ✅ 环境激活成功

echo.
echo ================================================================================
echo 步骤 3/5: 安装GPU版PyTorch (CUDA 12.1)
echo ================================================================================
echo ⚠️  这可能需要5-10分钟，请耐心等待...
echo.
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
if errorlevel 1 (
    echo ❌ PyTorch安装失败！尝试使用pip安装...
    call pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo ❌ pip安装也失败！
        pause
        exit /b 1
    )
)
echo ✅ PyTorch安装成功

echo.
echo ================================================================================
echo 步骤 4/5: 安装项目依赖
echo ================================================================================
call pip install numpy pandas matplotlib scipy
if errorlevel 1 (
    echo ⚠️  部分依赖安装失败，但可以继续
)
echo ✅ 依赖安装完成

echo.
echo ================================================================================
echo 步骤 5/5: 验证GPU配置
echo ================================================================================
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); x = torch.rand(100, 100).to(device); print(f'测试成功: 张量在 {x.device} 上')"

echo.
echo ================================================================================
echo ✅ 安装完成！
echo ================================================================================
echo.
echo 下一步操作:
echo   1. 在PyCharm中配置解释器:
echo      - File ^> Settings ^> Project ^> Python Interpreter
echo      - Add Interpreter ^> Conda Environment
echo      - 选择 Existing environment
echo      - 路径: [你的Anaconda路径]\envs\lag_gpu\python.exe
echo.
echo   2. 验证GPU配置:
echo      conda activate lag_gpu
echo      python check_gpu_status.py
echo.
echo   3. 运行你的项目
echo.
pause
