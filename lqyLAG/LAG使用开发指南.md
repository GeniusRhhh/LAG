# LAG使用开发指南

本指南面向本科毕设同学，详细说明如何在Windows系统上从零开始配置LAG环境，以及如何将自己的MARL算法集成到LAG中进行验证。

## 目录

1. 第一部分：Windows环境配置
   - 1.1 安装Python环境（Anaconda）
   - 1.2 安装Git
   - 1.3 安装GPU支持（可选，NVIDIA显卡用户）
   - 1.4 安装基础依赖包
   - 1.5 安装Shapely
   - 1.6 配置JSBSim子模块
   - 1.7 验证环境配置
   - 1.8 环境配置检查清单
2. 第二部分：PyCharm IDE配置
   - 2.1 下载并安装PyCharm
   - 2.2 配置PyCharm使用Anaconda环境
   - 2.3 PyCharm基本使用
   - 2.4 配置Git集成
3. 第三部分：C++环境配置（可选）
   - 3.1 安装Visual Studio
   - 3.2 安装VSCode
   - 3.3 安装CMake
   - 3.4 编译JSBSim C++库
   - 3.5 在VSCode中创建C++项目
4. 第四部分：算法集成指南
   - 4.1 LAG算法架构概述
   - 4.2 参考现有算法实现
   - 4.3 添加自己的算法
   - 4.4 算法集成检查清单
   - 4.5 评估和可视化
5. 第五部分：常见问题与解决方案
   - 5.1 环境配置问题
   - 5.2 算法集成问题
   - 5.3 训练相关问题
   - 5.4 PyCharm相关问题
6. 附录

## 第一部分：Windows环境配置

### 1.1 安装Python环境

#### 步骤1：下载并安装Anaconda
参考：https://blog.csdn.net/qq_55106902/article/details/147308606?ops_request_misc=%257B%2522request%255Fid%2522%253A%25228dd723a12318b26b108209a55faf565b%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=8dd723a12318b26b108209a55faf565b&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-2-147308606-null-null.142^v102^control&utm_term=anaconda%E5%AE%89%E8%A3%85&spm=1018.2226.3001.4187
1. 访问Anaconda官网
   - 打开浏览器，访问：https://www.anaconda.com/products/distribution
   - 点击Download按钮，选择Windows版本（64-bit）
   - 下载文件大小约500MB，下载时间取决于网络速度

2. 安装Anaconda
   - 双击下载的.exe安装文件
   - 安装过程中需要注意以下选项：
     - 勾选Add Anaconda to my PATH environment variable（非常重要，否则后续无法在命令行使用conda）
     - 选择Just Me（推荐，只给当前用户安装）
     - 安装路径建议使用默认路径：C:\Users\你的用户名\anaconda3
     - 如果C盘空间不足，可以安装到D盘，但路径中不要有中文或空格
   - 点击Install开始安装，可能需要10-20分钟
   - 安装完成后，不要立即点击Finish，先勾选Learn more about Anaconda Cloud，然后点击Finish

3. 验证安装
   - 打开命令提示符（按Win+R键，输入cmd，回车）
   - 输入以下命令验证：
     ```
     conda --version
     ```
   - 如果显示版本号（如conda 23.x.x），说明安装成功
   - 如果显示conda不是内部或外部命令，说明PATH环境变量未正确设置，需要重新安装并勾选Add to PATH选项

#### 步骤2：创建Python虚拟环境

1. 打开Anaconda Prompt
   - 在Windows开始菜单搜索Anaconda Prompt
   - 右键点击，选择以管理员身份运行（推荐，避免权限问题）
   - 如果找不到，也可以使用命令提示符，但需要先激活Anaconda

2. 创建虚拟环境
   - 在Anaconda Prompt中输入以下命令：
     ```
     conda create -n jsbsim python=3.8
     ```
   - 系统会提示是否继续，输入y并回车
   - 等待环境创建完成，约2-5分钟
   - 创建完成后会显示环境路径，通常是C:\Users\你的用户名\anaconda3\envs\jsbsim

3. 激活虚拟环境
   - 输入以下命令激活环境：
     ```
     conda activate jsbsim
     ```
   - 激活成功后，命令提示符前面会显示(jsbsim)
   - 如果激活失败，尝试先执行conda init，然后重新打开Anaconda Prompt

注意：每次使用LAG前，都需要先激活这个环境。如果关闭了命令行窗口，下次使用时需要重新激活。

### 1.2 安装Git
参考：https://blog.csdn.net/fanyun_01/article/details/145350857?ops_request_misc=&request_id=&biz_id=102&utm_term=git%E5%AE%89%E8%A3%85%E9%85%8D%E7%BD%AE&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-145350857.142^v102^control&spm=1018.2226.3001.4187
#### 步骤1：下载Git

1. 访问Git官网
   - 打开浏览器，访问：https://git-scm.com/download/win
   - 点击Download for Windows，会自动下载最新版本的Git for Windows
   - 下载文件大小约50MB

#### 步骤2：安装Git

1. 运行安装程序
   - 双击下载的Git安装程序（通常是Git-x.x.x-64-bit.exe）
   - 安装过程中使用默认选项即可，但需要注意：
     - 选择默认编辑器时，如果不会用Vim，可以选择Notepad++或Visual Studio Code
     - 选择PATH环境时，选择Git from the command line and also from 3rd-party software（推荐）
     - 选择行尾转换时，选择Checkout Windows-style, commit Unix-style line endings（默认）
     - 其他选项保持默认即可

2. 验证安装
   - 打开命令提示符或Anaconda Prompt
   - 输入以下命令：
     ```
     git --version
     ```
   - 如果显示版本号（如git version 2.x.x），说明安装成功

#### 步骤3：配置Git（可选但推荐）

1. 设置用户名和邮箱
   - 在命令行中输入：
     ```
     git config --global user.name "你的名字"
     git config --global user.email "你的邮箱"
     ```
   - 这些信息会在提交代码时使用

### 1.3 安装GPU支持（可选，仅NVIDIA显卡用户）

如果你的电脑有NVIDIA显卡，可以使用GPU加速训练，大幅提升训练速度。这部分是可选的，如果没有GPU或不需要GPU加速，可以跳过直接看1.4节。

#### 步骤1：检查显卡和驱动

1. 检查是否有NVIDIA显卡
   - 右键点击桌面空白处，选择NVIDIA控制面板
   - 或者打开设备管理器，查看显示适配器
   - 如果看到NVIDIA GeForce或NVIDIA Quadro等，说明有NVIDIA显卡

2. 检查显卡驱动版本
   - 打开命令提示符（Win+R，输入cmd）
   - 输入以下命令：
     ```
     nvidia-smi
     ```
   - 如果显示显卡信息，说明驱动已安装
   - 查看右上角的CUDA Version，这是驱动支持的最高CUDA版本（例如11.8、12.1等）
   - 如果显示nvidia-smi不是内部或外部命令，说明没有安装NVIDIA驱动，需要先安装：
     - 访问：https://www.nvidia.com/Download/index.aspx
     - 输入你的显卡型号，下载并安装最新驱动

#### 步骤2：安装CUDA Toolkit

CUDA是NVIDIA提供的并行计算平台，PyTorch需要CUDA来使用GPU。

1. 确定要安装的CUDA版本
   - 查看PyTorch官网支持的CUDA版本：https://pytorch.org/get-started/locally/
   - 目前常用的版本有CUDA 11.8和CUDA 12.1
   - 注意：CUDA版本不能超过nvidia-smi显示的CUDA Version
   - 建议选择CUDA 11.8，因为兼容性更好

2. 下载CUDA Toolkit
   - 访问：https://developer.nvidia.com/cuda-downloads
   - 选择Windows -> x86_64 -> 10/11 -> exe (local)
   - 选择CUDA 11.8或12.1（根据你的选择）
   - 点击下载，文件大小约3GB

3. 安装CUDA Toolkit
   - 双击下载的安装程序
   - 安装类型选择自定义（Custom）
   - 在组件选择页面：
     - 取消勾选Visual Studio Integration（如果不需要）
     - 其他组件保持默认勾选
   - 点击下一步，选择安装路径（建议使用默认路径）
   - 点击安装，等待完成（可能需要10-20分钟）

4. 验证CUDA安装
   - 打开命令提示符
   - 输入：
     ```
     nvcc --version
     ```
   - 如果显示CUDA版本信息，说明安装成功
   - 如果显示不是内部或外部命令，需要将CUDA的bin目录添加到PATH：
     - 默认路径：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
     - 右键此电脑 -> 属性 -> 高级系统设置 -> 环境变量
     - 在系统变量的Path中添加上述路径

#### 步骤3：安装cuDNN

cuDNN是NVIDIA提供的深度神经网络库，用于加速深度学习计算。

1. 下载cuDNN
   - 访问：https://developer.nvidia.com/cudnn
   - 需要注册NVIDIA开发者账号（免费）
   - 登录后，选择Download cuDNN
   - 选择与你的CUDA版本对应的cuDNN版本：
     - CUDA 11.8对应cuDNN 8.9.x
     - CUDA 12.1对应cuDNN 8.9.x
   - 下载Windows版本的zip文件（约500MB）

2. 安装cuDNN
   - 解压下载的zip文件，会得到一个cuda文件夹
   - 打开cuda文件夹，里面有bin、include、lib三个文件夹
   - 找到CUDA的安装目录（默认：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8）
   - 将cuda文件夹中的bin、include、lib三个文件夹的内容分别复制到CUDA安装目录对应的文件夹中：
     - cuda\bin\* 复制到 CUDA\v11.8\bin\
     - cuda\include\* 复制到 CUDA\v11.8\include\
     - cuda\lib\x64\* 复制到 CUDA\v11.8\lib\x64\
   - 如果提示文件已存在，选择替换

3. 验证cuDNN安装
   - 打开命令提示符
   - 输入：
     ```
     python
     ```
   - 然后输入：
     ```python
     import torch
     print(torch.cuda.is_available())
     ```
   - 如果输出True，说明CUDA和cuDNN配置成功（需要先安装PyTorch GPU版本）

#### 步骤4：安装PyTorch GPU版本
参考：https://blog.csdn.net/Little_Carter/article/details/135934842?ops_request_misc=%257B%2522request%255Fid%2522%253A%252294eafc058903182ddc0e633f93cdc6a2%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=94eafc058903182ddc0e633f93cdc6a2&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-135934842-null-null.142%5Ev102%5Econtrol&utm_term=pytorch%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8Bgpu&spm=1018.2226.3001.4187
1. 确定PyTorch版本
   - 访问：https://pytorch.org/get-started/locally/
   - 选择你的配置：
     - OS: Windows
     - Package: Pip
     - Language: Python
     - Compute Platform: CUDA 11.8或CUDA 12.1（根据你安装的CUDA版本）

2. 安装PyTorch
   - 打开Anaconda Prompt（确保已激活jsbsim环境）
   - 复制PyTorch官网提供的安装命令，例如：
     ```
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
     （cu118表示CUDA 11.8，如果是12.1则改为cu121）
   - 执行命令，等待安装完成（可能需要10-20分钟，取决于网络速度）

3. 验证PyTorch GPU支持
   - 在Python中执行：
     ```python
     import torch
     print(torch.__version__)
     print(torch.cuda.is_available())
     print(torch.cuda.get_device_name(0))
     ```
   - 应该显示：
     - PyTorch版本号
     - True（表示GPU可用）
     - 你的显卡名称（如NVIDIA GeForce RTX 3060）

如果torch.cuda.is_available()返回False，可能的原因：
- CUDA版本与PyTorch不匹配
- cuDNN未正确安装
- 显卡驱动版本过低

### 1.4 安装基础依赖包

在激活的jsbsim环境中，依次执行以下命令：

1. 安装PyTorch（如果1.3节已安装GPU版本，跳过此步）
   - 如果没有GPU或不需要GPU加速，安装CPU版本：
     ```
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     ```
   - 安装时间约5-10分钟，取决于网络速度

2. 安装其他基础依赖
   ```
   pip install pymap3d geographiclib gym==0.20.0 wandb icecream setproctitle numpy
   ```
   - 这些包安装时间约2-5分钟

3. 安装JSBSim
   ```
   pip install jsbsim==1.1.6
   ```
   - JSBSim是飞行动力学仿真库，安装时间约3-5分钟

如果安装过程中出现网络错误或超时，可以使用国内镜像源加速：

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pymap3d geographiclib gym==0.20.0 wandb icecream setproctitle numpy jsbsim==1.1.6
```

### 1.5 安装Shapely（Windows特殊处理）

Shapely是一个用于处理几何图形的Python库，在Windows上不能直接用pip安装，需要下载预编译的wheel文件。

#### 步骤1：确定Python版本和系统架构

1. 在Anaconda Prompt中（确保已激活jsbsim环境），输入：
   ```
   python --version
   ```
   - 应该显示Python 3.8.x

2. 确认系统是64位：
   - 右键点击此电脑，选择属性
   - 查看系统类型，应该显示64位操作系统

#### 步骤2：下载Shapely wheel文件

1. 访问下载页面
   - 打开浏览器，访问：https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely
   - 这个页面列出了所有Windows预编译的Python包

2. 找到正确的文件
   - 在页面中搜索Shapely
   - 找到Shapely‑1.7.1‑cp38‑cp38‑win_amd64.whl
   - 注意：cp38表示Python 3.8，win_amd64表示64位Windows
   - 如果Python版本不同，选择对应的版本（如cp39表示Python 3.9）

3. 下载文件
   - 点击文件名开始下载
   - 建议下载到容易找到的位置，如D:\Downloads\或桌面

#### 步骤3：安装wheel文件

1. 打开Anaconda Prompt（确保已激活jsbsim环境）

2. 使用pip安装下载的wheel文件
   ```
   pip install D:\Downloads\Shapely‑1.7.1‑cp38‑cp38‑win_amd64.whl
   ```
   - 将路径替换为你实际下载的路径
   - 如果路径中有空格，需要用引号括起来：
     ```
     pip install "D:\My Downloads\Shapely‑1.7.1‑cp38‑cp38‑win_amd64.whl"
     ```

3. 验证安装
   ```
   python -c "import shapely; print(shapely.__version__)"
   ```
   - 如果显示版本号（如1.7.1），说明安装成功
   - 如果报错，检查文件路径是否正确，以及Python版本是否匹配

### 1.6 配置JSBSim子模块

LAG项目使用Git子模块来管理JSBSim源代码，需要初始化子模块。

#### 步骤1：进入LAG项目目录

1. 打开Anaconda Prompt（确保已激活jsbsim环境）

2. 进入lqyLAG目录
   ```
   cd D:\Pycharm\LAG\lqyLAG
   ```
   - 将路径替换为你的实际路径
   - 如果路径中有空格，需要用引号：
     ```
     cd "D:\My Projects\LAG\lqyLAG"
     ```

#### 步骤2：初始化Git子模块

1. 检查是否已经是Git仓库
   ```
   git status
   ```
   - 如果显示fatal: not a git repository，说明不是Git仓库，需要先初始化：
     ```
     git init
     ```

2. 初始化子模块
   ```
   git submodule init
   ```
   - 这个命令会读取.gitmodules文件，准备初始化子模块

3. 更新子模块
   ```
   git submodule update
   ```
   - 这个命令会下载JSBSim源代码到envs/JSBSim/core目录
   - 下载时间取决于网络速度，可能需要几分钟

如果上述命令失败，可以尝试：

```
git submodule update --init --recursive
```

或者手动添加子模块：

```
git submodule add https://github.com/JSBSim-Team/jsbsim.git envs/JSBSim/core
```

### 1.7 验证环境配置

创建一个测试脚本来验证所有依赖是否正确安装：

1. 创建测试文件
   - 在lqyLAG目录下创建文件test_environment.py
   - 文件内容如下：
     ```python
     import sys
     import torch
     import numpy as np
     import jsbsim
     import gym
     import shapely
     
     print("Python版本:", sys.version)
     print("PyTorch版本:", torch.__version__)
     print("NumPy版本:", np.__version__)
     print("JSBSim版本:", jsbsim.__version__)
     print("Gym版本:", gym.__version__)
     print("Shapely版本:", shapely.__version__)
     print("\n所有依赖包安装成功！")
     ```

2. 运行测试
   - 在Anaconda Prompt中（确保已激活jsbsim环境），进入lqyLAG目录
   - 运行测试脚本：
     ```
     python test_environment.py
     ```

3. 预期输出
   - 应该显示各个包的版本号，没有报错
   - 如果出现ModuleNotFoundError，说明对应包未安装，需要重新安装

### 1.8 环境配置检查清单

在开始使用LAG之前，请确认以下项目：

- Anaconda已安装，conda --version可以运行
- 虚拟环境jsbsim已创建并可以激活
- Git已安装，git --version可以运行
- 所有Python依赖包已安装（PyTorch, JSBSim, Gym, Shapely等）
- JSBSim子模块已初始化
- 测试脚本运行无错误

## 第二部分：PyCharm IDE配置

PyCharm是JetBrains开发的Python IDE，提供了代码编辑、调试、运行等功能，非常适合开发LAG项目。

### 2.1 下载并安装PyCharm

1. 访问PyCharm官网
   - 打开浏览器，访问：https://www.jetbrains.com/pycharm/download/
   - 选择Windows版本
   - 有两个版本：Professional（专业版，收费）和Community（社区版，免费）
   - 对于学习使用，Community版本足够

2. 下载安装程序
   - 点击Download按钮下载Community版本
   - 文件大小约300MB

3. 安装PyCharm
   - 双击下载的安装程序
   - 安装过程中：
     - 选择安装路径（建议使用默认路径）
     - 勾选Create associations（关联.py文件）
     - 勾选Add launchers dir to the PATH（添加到PATH环境变量）
     - 其他选项保持默认即可
   - 点击Install开始安装，约2-5分钟

4. 启动PyCharm
   - 安装完成后，勾选Run PyCharm Community Edition
   - 首次启动会询问是否导入设置，选择Do not import settings
   - 接受用户协议
   - 可以选择是否发送使用统计（可选）

### 2.2 配置PyCharm使用Anaconda环境

1. 打开或创建项目
   - 启动PyCharm后，选择Open
   - 浏览到LAG项目目录：D:\Pycharm\LAG\lqyLAG
   - 点击OK打开项目

2. 配置Python解释器
   - 点击File -> Settings（或按Ctrl+Alt+S）
   - 在左侧菜单选择Project: lqyLAG -> Python Interpreter
   - 点击右上角的齿轮图标，选择Add
   - 在弹出的对话框中：
     - 左侧选择Conda Environment
     - 选择Existing environment
     - Interpreter路径选择：C:\Users\你的用户名\anaconda3\envs\jsbsim\python.exe
     - 如果找不到，点击右侧的文件夹图标浏览
     - 勾选Make available to all projects（可选）
   - 点击OK保存

3. 验证配置
   - 在Python Interpreter页面，应该能看到jsbsim环境
   - 下方会显示已安装的包列表，应该包括torch, jsbsim, gym等
   - 如果列表为空或缺少包，说明路径配置错误

### 2.3 PyCharm基本使用

1. 运行Python脚本
   - 右键点击.py文件
   - 选择Run '文件名'
   - 或者点击文件右上角的绿色运行按钮
   - 运行结果会显示在下方的Run窗口中

2. 调试代码
   - 在代码行号左侧点击，设置断点（红色圆点）
   - 右键点击文件，选择Debug '文件名'
   - 程序会在断点处暂停，可以查看变量值
   - 使用调试工具栏控制执行（继续、单步执行等）

3. 安装包
   - 在PyCharm中可以直接安装Python包
   - 点击File -> Settings -> Project -> Python Interpreter
   - 点击+号，搜索包名，点击Install Package
   - 或者使用Terminal标签页，在终端中执行pip install命令

4. 代码提示和自动补全
   - PyCharm会自动提供代码提示
   - 输入代码时按Ctrl+Space可以手动触发代码补全
   - 按Ctrl+Q可以查看函数或类的文档

### 2.4 配置Git集成

PyCharm内置了Git支持，可以方便地进行版本控制：

1. 启用Git
   - 如果项目已经是Git仓库，PyCharm会自动识别
   - 如果还没有初始化Git，可以：
     - 点击VCS -> Enable Version Control Integration
     - 选择Git，点击OK

2. 提交代码
   - 修改文件后，文件名会变成蓝色（已修改）
   - 右键点击文件，选择Git -> Commit File
   - 输入提交信息，点击Commit

3. 查看历史
   - 右键点击文件，选择Git -> Show History
   - 可以查看文件的修改历史

## 第三部分：C++环境配置（可选）

如果你不使用LAG平台，而是直接用C++来使用JSBSim，需要配置C++开发环境。这部分是可选的，只有在需要直接使用JSBSim C++库时才需要。

### 3.1 安装Visual Studio
参考：https://blog.csdn.net/Javachichi/article/details/131358012?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522cf54b12a572d231e36424fb139a232be%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=cf54b12a572d231e36424fb139a232be&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-131358012-null-null.142%5Ev102%5Econtrol&utm_term=visualstudio%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B&spm=1018.2226.3001.4187

Visual Studio是微软开发的集成开发环境，包含C++编译器和调试工具。

1. 下载Visual Studio
   - 访问：https://visualstudio.microsoft.com/downloads/
   - 点击下载Visual Studio Community（免费版本）
   - 文件大小约3-4GB

2. 运行安装程序
   - 双击下载的安装程序
   - 首次运行会提示安装Visual Studio Installer
   - 等待安装程序启动

3. 选择工作负载
   - 在安装程序的工作负载页面，找到使用C++的桌面开发
   - 勾选这个工作负载
   - 右侧会显示将要安装的组件，包括：
     - MSVC编译器
     - Windows SDK
     - CMake工具
     - Git for Windows（如果未安装）
   - 可以点击安装详细信息查看具体组件

4. 开始安装
   - 点击右下角的安装按钮
   - 等待下载和安装完成（可能需要30分钟到1小时，取决于网络速度）
   - 安装过程中可以继续使用电脑做其他事情

5. 重启电脑（如果需要）
   - 安装完成后，可能会提示重启电脑
   - 按照提示重启

6. 验证安装
   - 打开开始菜单，搜索Developer Command Prompt for VS
   - 打开后，输入：
     ```
     cl
     ```
   - 如果显示Microsoft C/C++编译器版本信息，说明安装成功
   - 如果显示不是内部或外部命令，可能需要重启电脑或检查安装

### 3.2 安装VSCode
参考：https://blog.csdn.net/weixin_60915103/article/details/131617196?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522ac03684c4e976246064dd6ec4621592a%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=ac03684c4e976246064dd6ec4621592a&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-131617196-null-null.142%5Ev102%5Econtrol&utm_term=vscode&spm=1018.2226.3001.4187

VSCode是微软开发的轻量级代码编辑器，可以配合Visual Studio的编译器使用。

1. 下载VSCode
   - 访问：https://code.visualstudio.com/
   - 点击Download for Windows
   - 下载Windows版本的安装程序

2. 安装VSCode
   - 双击下载的安装程序
   - 安装过程中：
     - 选择安装路径（建议使用默认路径）
     - 勾选添加到PATH（Add to PATH）
     - 勾选创建桌面快捷方式
     - 其他选项保持默认
   - 点击安装，等待完成

3. 安装C++扩展
   - 打开VSCode
   - 点击左侧的扩展图标（或按Ctrl+Shift+X）
   - 搜索C/C++
   - 找到Microsoft的C/C++扩展，点击安装
   - 还可以安装C/C++ Extension Pack，包含更多有用的扩展

4. 配置C++环境
   - 在VSCode中，按Ctrl+Shift+P打开命令面板
   - 输入C/C++: Edit Configurations (JSON)
   - 这会创建或打开c_cpp_properties.json文件
   - 配置编译器路径，例如：
     ```json
     {
         "configurations": [
             {
                 "name": "Win32",
                 "includePath": [
                     "${workspaceFolder}/**"
                 ],
                 "defines": [
                     "_DEBUG",
                     "UNICODE",
                     "_UNICODE"
                 ],
                 "compilerPath": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.xx.xxxxx/bin/Hostx64/x64/cl.exe",
                 "cStandard": "c17",
                 "cppStandard": "c++17",
                 "intelliSenseMode": "windows-msvc-x64"
             }
         ],
         "version": 4
     }
     ```
   - 注意：compilerPath需要根据你的Visual Studio安装路径调整
   - 可以在Developer Command Prompt中执行where cl来查找编译器路径

### 3.3 关于CMake的说明

CMake是跨平台的构建工具，JSBSim使用CMake来构建。关于CMake的安装，有两种情况：

#### 情况1：只使用Visual Studio（推荐，更简单）

如果你只使用Visual Studio来编译JSBSim，**不需要单独安装CMake**。

Visual Studio在安装"使用C++的桌面开发"工作负载时，已经自动包含了CMake工具。你可以直接在Visual Studio中打开CMake项目：

1. 打开Visual Studio
2. 选择文件 -> 打开 -> CMake
3. 选择JSBSim目录下的CMakeLists.txt文件
4. Visual Studio会自动配置和生成项目
5. 点击生成 -> 生成解决方案即可编译

这种方式不需要单独安装CMake，也不需要手动配置CMake命令。

#### 情况2：需要在命令行使用CMake（可选）

如果你需要在命令行中使用cmake命令（例如在VSCode终端中，或者使用脚本自动化构建），则需要单独安装CMake：

1. 下载CMake
   - 访问：https://cmake.org/download/
   - 选择Windows x64 Installer
   - 下载最新版本的安装程序

2. 安装CMake
   - 双击下载的安装程序
   - 安装过程中：
     - 勾选Add CMake to system PATH for all users（重要，这样可以在命令行直接使用cmake）
     - 或者选择Add CMake to system PATH for current user（如果只有当前用户使用）
     - 其他选项保持默认
   - 点击Next完成安装

3. 验证安装
   - 打开命令提示符（或VSCode的终端）
   - 输入：
     ```
     cmake --version
     ```
   - 如果显示版本号（如cmake version 3.xx.x），说明安装成功
   - 如果显示不是内部或外部命令，需要重启电脑或手动添加PATH

**建议**：对于初学者，如果只使用Visual Studio，可以跳过单独安装CMake的步骤，直接使用Visual Studio内置的CMake功能即可。

### 3.4 编译JSBSim C++库

有两种方式可以编译JSBSim，选择其中一种即可：

#### 方法1：使用Visual Studio直接打开CMake项目（推荐，更简单）

1. 获取JSBSim源代码
   - 如果使用LAG项目，JSBSim源代码在envs/JSBSim/core目录
   - 或者从GitHub克隆：
     ```
     git clone https://github.com/JSBSim-Team/jsbsim.git
     ```

2. 打开Visual Studio
   - 启动Visual Studio
   - 选择文件 -> 打开 -> CMake
   - 或者选择文件 -> 打开 -> 文件夹，然后选择JSBSim源代码目录

3. 选择CMakeLists.txt
   - 浏览到JSBSim源代码目录
   - 选择CMakeLists.txt文件（在源代码根目录）
   - 点击打开

4. Visual Studio自动配置
   - Visual Studio会自动检测CMakeLists.txt
   - 会在源代码目录下创建out/build目录用于构建
   - 等待CMake配置完成（底部状态栏会显示进度）

5. 选择配置
   - 在顶部工具栏，选择Release配置（默认可能是Debug）
   - 选择x64平台（64位）

6. 编译
   - 点击生成 -> 生成全部
   - 或者按F7快捷键
   - 等待编译完成（可能需要几分钟）
   - 编译进度会显示在输出窗口

7. 编译结果
   - 编译完成后，库文件在out/build/Release/src目录（或out/build/x64-Release/src）
   - Windows上会生成JSBSim.lib（静态库）和JSBSim.dll（动态库）
   - 头文件在源代码的src目录

#### 方法2：使用命令行和CMake（需要单独安装CMake）

如果你已经单独安装了CMake，可以使用命令行方式：

1. 获取JSBSim源代码
   - 如果使用LAG项目，JSBSim源代码在envs/JSBSim/core目录
   - 或者从GitHub克隆：
     ```
     git clone https://github.com/JSBSim-Team/jsbsim.git
     ```

2. 打开Developer Command Prompt
   - 在开始菜单搜索Developer Command Prompt for VS
   - 打开后会自动设置好编译环境

3. 进入JSBSim目录
   ```
   cd D:\Pycharm\LAG\lqyLAG\envs\JSBSim\core
   ```
   （将路径替换为你的实际路径）

4. 创建构建目录
   ```
   mkdir build
   cd build
   ```

5. 配置CMake
   ```
   cmake .. -G "Visual Studio 17 2022" -A x64
   ```
   - Visual Studio 17 2022对应Visual Studio 2022
   - 如果是Visual Studio 2019，使用Visual Studio 16 2019
   - -A x64表示生成64位版本

6. 编译
   - 方法1：使用CMake命令
     ```
     cmake --build . --config Release
     ```
   - 方法2：使用Visual Studio
     - 打开生成的JSBSim.sln文件
     - 在Visual Studio中选择Release配置
     - 右键解决方案，选择生成解决方案
     - 等待编译完成（可能需要几分钟）

7. 编译结果
   - 编译完成后，库文件在build/src/Release目录（或build/src目录）
   - Windows上会生成JSBSim.lib（静态库）和JSBSim.dll（动态库）
   - 头文件在src目录

**建议**：对于初学者，推荐使用方法1（Visual Studio直接打开CMake项目），更简单直观，不需要单独安装CMake，也不需要记住复杂的命令。

### 3.5 在VSCode中创建C++项目

1. 创建项目文件夹
   - 在合适的位置创建新文件夹，例如D:\MyJSBSimProject

2. 在VSCode中打开文件夹
   - 打开VSCode
   - 点击File -> Open Folder
   - 选择刚创建的文件夹

3. 创建CMakeLists.txt
   - 在项目根目录创建CMakeLists.txt文件
   - 内容示例：
     ```cmake
     cmake_minimum_required(VERSION 3.10)
     project(MyJSBSimProject)
     
     set(CMAKE_CXX_STANDARD 17)
     
     # 设置JSBSim路径（根据你的实际路径修改）
     set(JSBSIM_ROOT "D:/Pycharm/LAG/lqyLAG/envs/JSBSim/core")
     set(JSBSIM_INCLUDE_DIR "${JSBSIM_ROOT}/src")
     set(JSBSIM_LIB_DIR "${JSBSIM_ROOT}/build/src/Release")
     
     # 包含头文件
     include_directories(${JSBSIM_INCLUDE_DIR})
     
     # 添加可执行文件
     add_executable(main main.cpp)
     
     # 链接库
     target_link_libraries(main ${JSBSIM_LIB_DIR}/JSBSim.lib)
     ```

4. 创建main.cpp
   - 创建main.cpp文件，编写测试代码：
     ```cpp
     #include <iostream>
     #include "JSBSim/FGFDMExec.h"
     
     int main() {
         JSBSim::FGFDMExec* fdm = new JSBSim::FGFDMExec();
         std::cout << "JSBSim initialized successfully!" << std::endl;
         delete fdm;
         return 0;
     }
     ```

5. 配置CMake
   - 在VSCode中，按Ctrl+Shift+P
   - 输入CMake: Configure
   - 选择编译器（Visual Studio）
   - 等待配置完成

6. 编译和运行
   - 按F7或点击底部状态栏的Build按钮编译
   - 按F5运行（需要先配置launch.json）
   - 或者直接在终端运行生成的可执行文件

## 第四部分：算法集成指南

### 4.1 LAG算法架构概述

LAG的算法代码位于lqyLAG/algorithms目录，结构如下：

```
algorithms/
├── ppo/              # PPO算法（单智能体，用于1v1场景）
│   ├── ppo_actor.py      # Actor网络
│   ├── ppo_critic.py     # Critic网络
│   ├── ppo_policy.py     # 策略类（整合Actor和Critic）
│   └── ppo_trainer.py    # 训练器
├── mappo/            # MAPPO算法（多智能体，用于2v2场景）
│   ├── ppo_actor.py
│   ├── ppo_critic.py
│   ├── ppo_policy.py
│   └── ppo_trainer.py
└── utils/            # 工具函数
    ├── buffer.py         # 经验回放缓冲区
    ├── mlp.py            # 多层感知机
    ├── gru.py            # GRU循环网络
    └── ...
```

关键概念：
- Actor：策略网络，根据观测输出动作
- Critic：价值网络，评估状态价值
- Policy：整合Actor和Critic的策略类
- Trainer：执行算法训练逻辑

### 4.2 参考现有算法实现

在实现自己的算法之前，建议先查看和理解现有的算法实现：

1. 查看PPO实现（单智能体）
   - 文件位置：lqyLAG/algorithms/ppo/
   - 适合1v1场景
   - 注意：Critic和Actor使用相同的观测

2. 查看MAPPO实现（多智能体）
   - 文件位置：lqyLAG/algorithms/mappo/
   - 适合2v2场景
   - 注意：Critic使用中心化观测（包含所有智能体的信息）

3. 查看Runner实现
   - 单智能体：lqyLAG/runner/jsbsim_runner.py
   - 多智能体：lqyLAG/runner/share_jsbsim_runner.py

### 4.3 添加自己的算法

假设你要实现一个名为my_marl的MARL算法，需要完成以下步骤：

#### 步骤1：创建算法目录

在lqyLAG/algorithms目录下创建新文件夹：

```
cd lqyLAG/algorithms
mkdir my_marl
cd my_marl
```

创建必要的文件：
- my_marl_actor.py（Actor网络）
- my_marl_critic.py（Critic网络）
- my_marl_policy.py（策略类）
- my_marl_trainer.py（训练器）
- __init__.py（Python包初始化文件，可以为空）

#### 步骤2：实现算法文件

参考PPO或MAPPO的实现，创建对应的文件。关键是要实现以下接口：

1. Actor需要实现：
   - forward方法：根据观测输出动作
   - evaluate_actions方法：评估动作的对数概率

2. Critic需要实现：
   - forward方法：根据观测输出价值

3. Policy需要实现：
   - get_actions方法：获取动作（用于收集经验）
   - get_values方法：获取价值（用于计算优势）
   - evaluate_actions方法：评估动作（用于训练）
   - act方法：执行动作（用于评估）

4. Trainer需要实现：
   - train方法：执行训练逻辑

具体实现可以参考lqyLAG/algorithms/ppo/或lqyLAG/algorithms/mappo/中的代码。

#### 步骤3：在Runner中注册算法

修改Runner文件以支持新算法。

对于多智能体（2v2）场景，修改lqyLAG/runner/share_jsbsim_runner.py：

找到load方法中的算法加载部分，添加你的算法：

```python
if self.algorithm_name == "mappo":
    from algorithms.mappo.ppo_trainer import PPOTrainer as Trainer
    from algorithms.mappo.ppo_policy import PPOPolicy as Policy
elif self.algorithm_name == "my_marl":  # 添加你的算法
    from algorithms.my_marl.my_marl_trainer import MyMARLTrainer as Trainer
    from algorithms.my_marl.my_marl_policy import MyMARLPolicy as Policy
else:
    raise NotImplementedError
```

对于单智能体场景，修改lqyLAG/runner/jsbsim_runner.py中的相应部分。

#### 步骤4：在config.py中添加算法选项

修改lqyLAG/config.py，在算法选择参数中添加你的算法：

找到_get_prepare_config函数中的algorithm-name参数：

```python
group.add_argument("--algorithm-name", type=str, default='ppo', 
                   choices=["ppo", "mappo", "my_marl"],  # 添加你的算法
                   help="Specifiy the algorithm (default ppo)")
```

#### 步骤5：创建训练脚本

创建训练脚本lqyLAG/scripts/train_my_marl.bat（Windows）：

```batch
@echo off
set env=MultipleCombat
set scenario=2v2/NoWeapon/Selfplay
set algo=my_marl
set exp=my_experiment
set seed=1

python scripts/train/train_jsbsim.py ^
    --env-name %env% --algorithm-name %algo% --scenario-name %scenario% --experiment-name %exp% ^
    --seed %seed% --n-training-threads 1 --n-rollout-threads 4 --log-interval 1 --save-interval 1 ^
    --use-selfplay --selfplay-algorithm "fsp" --n-choose-opponents 1 ^
    --use-eval --n-eval-rollout-threads 1 --eval-interval 10 --eval-episodes 5 ^
    --num-mini-batch 1 --buffer-size 200 --num-env-steps 1e6 ^
    --lr 3e-4 --gamma 0.99 --ppo-epoch 10 --clip-param 0.2 --max-grad-norm 2 --entropy-coef 0.01 ^
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 10 ^
    --user-name "your_name"

pause
```

#### 步骤6：运行训练

1. 激活环境
   ```
   conda activate jsbsim
   ```

2. 进入项目目录
   ```
   cd D:\Pycharm\LAG\lqyLAG
   ```

3. 运行训练脚本
   ```
   scripts\train_my_marl.bat
   ```

### 4.4 算法集成检查清单

在开始训练前，请确认：

- 已创建算法目录（algorithms/my_marl/）
- 已实现所有必要的类（Actor, Critic, Policy, Trainer）
- 已在Runner中注册算法
- 已在config.py中添加算法选项
- 已创建训练脚本
- 测试运行无语法错误

### 4.5 评估和可视化

#### 使用TacView可视化

1. 安装TacView
   - 访问：https://www.tacview.net/
   - 下载并安装TacView（免费版即可）

2. 生成ACMI文件
   - 训练过程中会自动生成.acmi文件
   - 或使用渲染脚本：
     ```
     cd lqyLAG/renders
     python render_2v2.py --model-dir 你的模型路径
     ```

3. 在TacView中打开
   - 双击.acmi文件，TacView会自动打开
   - 可以回放整个空战过程

#### 分析训练指标

训练指标保存在scripts/results/环境名/场景名/算法名/实验名/runX/目录下：
- training_metrics.json：JSON格式的训练指标
- training_log.txt：文本格式的训练日志

## 第五部分：常见问题与解决方案

### 5.1 环境配置问题

#### 问题1：conda命令找不到

原因：Anaconda未添加到PATH环境变量

解决方案：
1. 重新安装Anaconda，确保勾选Add to PATH
2. 或手动添加：将C:\Users\你的用户名\anaconda3\Scripts和C:\Users\你的用户名\anaconda3添加到系统PATH

#### 问题2：pip install很慢或失败

原因：网络问题，访问PyPI速度慢

解决方案：使用国内镜像源
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 包名
```

#### 问题3：JSBSim安装失败

原因：可能需要C++编译器

解决方案：
1. 安装Visual Studio Build Tools：https://visualstudio.microsoft.com/downloads/
2. 选择C++ build tools工作负载
3. 重新安装JSBSim

#### 问题6：nvidia-smi命令找不到

原因：未安装NVIDIA显卡驱动

解决方案：
1. 访问NVIDIA官网：https://www.nvidia.com/Download/index.aspx
2. 输入你的显卡型号，下载并安装最新驱动
3. 安装完成后重启电脑
4. 再次运行nvidia-smi验证

#### 问题7：torch.cuda.is_available()返回False

原因：CUDA、cuDNN或PyTorch版本不匹配

解决方案：
1. 检查CUDA版本：nvcc --version
2. 检查PyTorch是否支持GPU：python -c "import torch; print(torch.version.cuda)"
3. 如果PyTorch的CUDA版本为None，说明安装的是CPU版本，需要重新安装GPU版本
4. 确认CUDA版本与PyTorch匹配：
   - CUDA 11.8对应PyTorch的cu118
   - CUDA 12.1对应PyTorch的cu121
5. 检查cuDNN是否正确安装：
   - 查看CUDA安装目录下的bin文件夹是否有cudnn64_8.dll
   - 如果没有，重新安装cuDNN

#### 问题8：CUDA安装后nvcc命令找不到

原因：CUDA的bin目录未添加到PATH

解决方案：
1. 找到CUDA安装目录（默认：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8）
2. 右键此电脑 -> 属性 -> 高级系统设置 -> 环境变量
3. 在系统变量中找到Path，点击编辑
4. 添加CUDA的bin目录：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
5. 添加CUDA的libnvvp目录：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
6. 点击确定，重启命令提示符或电脑

#### 问题4：Shapely安装后仍报错

原因：版本不匹配或安装路径问题

解决方案：
1. 确认Python版本：python --version（应该是3.8）
2. 确认下载的wheel文件对应Python 3.8和64位系统
3. 使用完整路径安装

#### 问题5：Git子模块初始化失败

原因：版本不匹配或安装路径问题

解决方案：
1. 确认Python版本：python --version（应该是3.8）
2. 确认下载的wheel文件对应Python 3.8和64位系统
3. 使用完整路径安装

#### 问题5：Git子模块初始化失败

原因：网络问题或Git配置问题

解决方案：
1. 检查网络连接
2. 尝试使用代理或VPN
3. 手动下载JSBSim源代码到envs/JSBSim/core目录

### 5.2 算法集成问题

#### 问题1：ModuleNotFoundError

原因：Python找不到你的算法模块

解决方案：
1. 确认算法目录下有__init__.py文件（可以为空）
2. 确认导入路径正确（从lqyLAG目录运行）
3. 检查sys.path是否包含项目根目录

#### 问题2：维度不匹配错误

原因：网络输入输出维度与环境不匹配

解决方案：
1. 打印观测空间和动作空间的维度
2. 根据实际维度调整网络结构

#### 问题3：训练时loss为NaN

原因：学习率过大、梯度爆炸等

解决方案：
1. 降低学习率（如从3e-4改为1e-4）
2. 减小max-grad-norm（如从2改为0.5）
3. 检查输入数据是否包含异常值

#### 问题4：内存不足

原因：缓冲区太大或并行环境太多

解决方案：
1. 减小--buffer-size（如从200改为100）
2. 减小--n-rollout-threads（如从32改为4）
3. 减小--num-env-steps

### 5.3 训练相关问题

#### 问题1：训练速度很慢

原因：使用CPU训练、并行度不够等

解决方案：
1. 如果有GPU，使用--cuda参数
2. 增加--n-rollout-threads（但注意内存限制）
3. 减小环境复杂度或减少训练步数

#### 问题2：模型不收敛

原因：超参数设置不当、算法实现有误

解决方案：
1. 参考PPO/MAPPO的默认超参数
2. 检查算法实现是否正确（特别是损失函数）
3. 使用wandb监控训练过程

#### 问题3：找不到保存的模型

原因：保存路径不正确

解决方案：
1. 模型保存在：scripts/results/环境名/场景名/算法名/实验名/runX/
2. 使用绝对路径或相对路径加载模型

### 5.4 PyCharm相关问题

#### 问题1：无法识别导入的模块

原因：Python解释器配置错误

解决方案：
1. 检查File -> Settings -> Project -> Python Interpreter
2. 确认选择了jsbsim环境
3. 如果还是不行，尝试Invalidate Caches：File -> Invalidate Caches

#### 问题2：代码提示不工作

原因：PyCharm索引未完成

解决方案：
1. 等待PyCharm完成索引（右下角有进度提示）
2. 如果长时间未完成，尝试File -> Invalidate Caches -> Invalidate and Restart

## 附录

### A. 快速参考命令

```
# 激活环境
conda activate jsbsim

# 进入项目目录
cd D:\Pycharm\LAG\lqyLAG

# 运行训练（Windows）
scripts\train_my_marl.bat

# 查看训练日志
# 日志会实时输出到控制台
```

### B. 重要文件路径

- 算法实现：lqyLAG/algorithms/
- 环境配置：lqyLAG/envs/JSBSim/configs/
- 训练脚本：lqyLAG/scripts/train/
- 运行器：lqyLAG/runner/
- 配置文件：lqyLAG/config.py
- 模型保存：lqyLAG/scripts/results/

### C. 推荐学习资源

1. 强化学习基础
   - 《强化学习：原理与Python实现》
   - OpenAI Spinning Up：https://spinningup.openai.com/

2. 多智能体强化学习
   - 《Multi-Agent Reinforcement Learning: A Selective Overview》
   - MAPPO论文：https://arxiv.org/abs/2103.01955

3. PyTorch教程
   - 官方教程：https://pytorch.org/tutorials/

4. LAG项目文档
   - README.md
   - docs/目录下的文档

### D. 获取帮助

如果遇到问题：

1. 检查日志：仔细阅读错误信息
2. 参考示例：查看PPO/MAPPO的实现
3. 调试技巧：
   - 添加print语句查看数据形状
   - 使用断点调试
   - 逐步测试各个组件

## 结语

本指南提供了从零开始配置LAG环境到集成自定义算法的完整流程。希望同学们能够：

1. 顺利完成环境配置
2. 理解LAG的算法架构
3. 成功集成自己的MARL算法
4. 专注于算法设计，而不是环境配置

祝大家毕设顺利！

文档版本：v1.0
最后更新：2024年
维护者：LAG项目组
