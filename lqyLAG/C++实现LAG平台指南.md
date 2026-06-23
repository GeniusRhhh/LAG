# 从零开始用C++实现LAG平台指南

本指南说明如何用C++实现类似于LAG平台的空战仿真和强化学习训练系统。

## 1. 系统架构

LAG平台主要包含以下层次：

```
强化学习训练框架 (PPO/MAPPO)
    ↓
环境封装层 (Gym-like接口)
    ↓
任务层 (单机控制/1v1/2v2)
    ↓
物理仿真层 (JSBSim)
```

## 2. 需要做什么

### 2.1 核心任务

1. **封装JSBSim**
   - 封装JSBSim C++接口，提供飞机控制接口
   - 实现飞机状态获取（位置、速度、姿态等）
   - 实现控制输入设置（油门、舵面等）

2. **实现环境接口**
   - 实现reset()函数：重置环境，返回初始观测
   - 实现step()函数：执行动作，返回(观测, 奖励, 是否结束)
   - 定义观测空间和动作空间

3. **实现强化学习算法**
   - 使用LibTorch实现神经网络（Actor/Critic）
   - 实现PPO算法（单智能体）或MAPPO算法（多智能体）
   - 实现经验回放和训练更新

4. **实现训练框架**
   - 实现训练循环
   - 实现模型保存/加载
   - 实现日志记录

## 3. 需要写什么

### 3.1 项目结构

```
cpp_lag/
├── include/
│   ├── core/
│   │   ├── environment.h          # 环境基类
│   │   └── task.h                 # 任务基类
│   ├── jsbsim/
│   │   └── aircraft_sim.h         # JSBSim封装
│   └── rl/
│       ├── ppo.h                  # PPO算法
│       ├── network.h              # 神经网络
│       └── buffer.h               # 经验缓冲区
├── src/
│   ├── core/environment.cpp
│   ├── jsbsim/aircraft_sim.cpp
│   └── rl/ppo.cpp
├── tasks/
│   ├── single_control.cpp         # 单机控制任务
│   ├── single_combat.cpp          # 1v1对抗任务
│   └── multi_combat.cpp           # 2v2对抗任务
├── CMakeLists.txt
└── main.cpp
```

### 3.2 关键文件说明

#### environment.h/cpp
- 定义环境基类接口
- 实现reset()和step()函数
- 定义观测和动作的数据结构

#### aircraft_sim.h/cpp
- 封装JSBSim的FGFDMExec类
- 提供飞机初始化、控制设置、状态获取接口
- 处理单位转换（英尺转米、弧度等）

#### ppo.h/cpp
- 使用LibTorch定义Actor和Critic网络
- 实现PPO算法的前向传播和更新逻辑
- 实现动作采样和概率计算

#### network.h/cpp
- 实现多层感知机（MLP）
- 实现Actor网络（输出动作均值和标准差）
- 实现Critic网络（输出状态价值）

#### buffer.h/cpp
- 实现经验回放缓冲区
- 存储观测、动作、奖励、回报等
- 实现批量采样功能

#### main.cpp
- 创建环境和算法实例
- 实现训练循环
- 处理模型保存和日志

## 4. 需要的库和工具

1. **JSBSim C++库**
   - 飞行动力学仿真（需要编译JSBSim C++版本）

2. **LibTorch**
   - PyTorch的C++版本，用于神经网络
   - 下载：https://pytorch.org/get-started/locally/（选择C++版本）

3. **YAML-CPP**
   - 解析YAML配置文件
   - GitHub: https://github.com/jbeder/yaml-cpp

4. **Eigen**（可选）
   - 矩阵运算库
   - 如果LibTorch已满足需求可不用

5. **spdlog**（可选）
   - 日志库
   - GitHub: https://github.com/gabime/spdlog

6. **CMake**
   - 构建工具（Visual Studio已包含）

## 5. 实现思路

### 5.1 JSBSim封装

```cpp
class AircraftSimulator {
    JSBSim::FGFDMExec* fdm_;
    
    // 初始化飞机模型
    bool initialize();
    
    // 设置控制输入（油门、舵面）
    void set_controls(float throttle, float elevator, float aileron, float rudder);
    
    // 步进仿真
    void step(double dt);
    
    // 获取飞机状态（位置、速度、姿态等）
    AircraftState get_state();
};
```

### 5.2 环境实现

```cpp
class Environment {
    // 重置环境
    Observation reset();
    
    // 执行动作
    std::tuple<Observation, float, bool> step(const Action& action);
    
    // 获取空间维度
    size_t get_observation_size();
    size_t get_action_size();
};
```

### 5.3 PPO算法

```cpp
class PPO {
    ActorNetwork actor_;      // 策略网络
    CriticNetwork critic_;    // 价值网络
    
    // 获取动作
    std::tuple<torch::Tensor, torch::Tensor> get_action(torch::Tensor obs);
    
    // 更新策略
    void update(经验批次);
};
```

### 5.4 训练循环

```cpp
for (int episode = 0; episode < num_episodes; ++episode) {
    obs = env.reset();
    while (!done) {
        action = ppo.get_action(obs);
        [obs, reward, done] = env.step(action);
        存储经验;
    }
    如果经验足够:
        ppo.update(经验批次);
}
```

## 6. CMakeLists.txt要点

```cmake
# 设置C++标准
set(CMAKE_CXX_STANDARD 17)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${JSBSIM_ROOT}/src
    ${LIBTORCH_ROOT}/include
)

# 链接库
target_link_libraries(cpp_lag
    JSBSim.lib
    torch
    torch_cpu
)
```

## 7. 实现步骤

1. **搭建项目框架**
   - 创建目录结构
   - 配置CMakeLists.txt
   - 配置依赖库路径

2. **实现JSBSim封装**
   - 封装FGFDMExec类
   - 实现控制接口
   - 实现状态获取

3. **实现环境接口**
   - 实现Environment基类
   - 实现具体任务环境（单机控制/1v1/2v2）

4. **实现神经网络**
   - 使用LibTorch实现MLP
   - 实现Actor和Critic网络

5. **实现PPO算法**
   - 实现前向传播
   - 实现PPO更新逻辑

6. **实现训练框架**
   - 实现经验缓冲区
   - 实现训练循环
   - 实现模型保存/加载

7. **测试和优化**
   - 单元测试
   - 集成测试
   - 性能优化

## 8. 关键技术点

1. **JSBSim使用**
   - 使用FGFDMExec类加载飞机模型
   - 通过PropertyManager设置和获取属性
   - 注意单位转换（英尺、弧度等）

2. **LibTorch使用**
   - 使用torch::nn::Module定义网络
   - 使用torch::optim::Adam作为优化器
   - 注意tensor的维度匹配

3. **多智能体实现**
   - 每个智能体独立的Actor网络
   - 共享的Critic网络（使用中心化观测）
   - 需要处理多智能体的经验收集

4. **性能优化**
   - 使用多线程并行环境
   - 批量处理tensor运算
   - 减少不必要的内存拷贝

## 9. 与Python版本对比

| 特性 | Python版本 | C++版本 |
|------|-----------|---------|
| 开发速度 | 快 | 慢 |
| 运行速度 | 较慢 | 快 |
| 内存占用 | 较高 | 较低 |
| 部署 | 需要Python环境 | 可独立运行 |

## 10. 参考资源

1. **JSBSim文档**
   - 官方文档：https://jsbsim.sourceforge.net/
   - C++ API：查看JSBSim源代码中的头文件

2. **LibTorch文档**
   - 官方文档：https://pytorch.org/cppdocs/
   - 示例：https://github.com/pytorch/examples/tree/master/cpp

3. **强化学习算法**
   - PPO论文：https://arxiv.org/abs/1707.06347
   - MAPPO论文：https://arxiv.org/abs/2103.01955

## 11. 总结

用C++实现LAG平台的核心工作：

1. **封装JSBSim**：提供简洁的飞机控制接口
2. **实现环境**：提供Gym-like的reset/step接口
3. **实现算法**：使用LibTorch实现PPO/MAPPO
4. **训练框架**：实现经验收集和模型更新

主要优势：性能好、可独立部署
主要挑战：开发周期长、调试相对困难

建议：先实现核心功能验证可行性，再逐步完善细节。
