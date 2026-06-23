import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置随机种子以确保可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class DoubleChannelAttention(nn.Module):
    """双通道注意力机制模块：时间域注意力 + 特征域注意力"""
    def __init__(self, hidden_size):
        super(DoubleChannelAttention, self).__init__()
        self.hidden_size = hidden_size
        
        # 时间域注意力机制
        self.time_attention = nn.Linear(hidden_size, 1)
        
        # 特征域注意力机制
        self.feature_attention = nn.Linear(hidden_size, hidden_size)
        self.feature_sigmoid = nn.Sigmoid()
        
        # 可训练的融合参数
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))  # 初始权重设为0.5
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, lstm_output):
        # lstm_output形状: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = lstm_output.size()
        
        # 1. 时间域注意力
        time_attention_weights = self.time_attention(lstm_output).squeeze(2)  # [batch_size, seq_len]
        time_attention_weights = self.softmax(time_attention_weights).unsqueeze(2)  # [batch_size, seq_len, 1]
        time_context_vector = torch.sum(time_attention_weights * lstm_output, dim=1)  # [batch_size, hidden_size]
        
        # 2. 特征域注意力
        # 对每个时间步应用特征注意力
        feature_attended = []
        for t in range(seq_len):
            time_step = lstm_output[:, t, :]  # [batch_size, hidden_size]
            feature_weights = self.feature_sigmoid(self.feature_attention(time_step))  # [batch_size, hidden_size]
            feature_attended_step = time_step * feature_weights  # [batch_size, hidden_size]
            feature_attended.append(feature_attended_step.unsqueeze(1))  # [batch_size, 1, hidden_size]
        
        feature_attended_output = torch.cat(feature_attended, dim=1)  # [batch_size, seq_len, hidden_size]
        # 对特征注意力后的结果应用平均池化
        feature_context_vector = torch.mean(feature_attended_output, dim=1)  # [batch_size, hidden_size]
        
        # 3. 加权融合两个通道的结果
        # 使用sigmoid确保权重在0-1之间
        fusion_alpha = torch.sigmoid(self.fusion_weight)
        fused_context_vector = fusion_alpha * time_context_vector + (1 - fusion_alpha) * feature_context_vector
        
        # 返回融合后的上下文向量和时间域注意力权重（用于可视化）
        return fused_context_vector, time_attention_weights

class LSTMWithAttention(nn.Module):
    """LSTM + 注意力机制模型"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0):
        super(LSTMWithAttention, self).__init__()
        # 添加批归一化层
        self.bn = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        # 使用双通道注意力机制
        self.attention = DoubleChannelAttention(hidden_dim * 2)  # *2 因为是双向LSTM
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        # 添加反归一化参数
        self.register_buffer('mean', torch.zeros(input_dim))
        self.register_buffer('std', torch.ones(input_dim))
    
    def set_normalization_params(self, mean, std):
        """设置反归一化参数"""
        self.mean.data = torch.tensor(mean, dtype=torch.float32)
        self.std.data = torch.tensor(std, dtype=torch.float32)
    
    def denormalize(self, x):
        """反归一化数据"""
        return x * self.std + self.mean
    
    def forward(self, x):
        # x形状: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.size()
        
        # 应用批归一化
        x = x.reshape(-1, input_dim)  # [batch_size*seq_len, input_dim]
        x = self.bn(x)
        x = x.reshape(batch_size, seq_len, input_dim)
        
        # LSTM层
        lstm_output, _ = self.lstm(x)
        
        # 注意力机制
        context_vector, attention_weights = self.attention(lstm_output)
        
        # 全连接层
        out = self.fc1(context_vector)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # softmax层获取6种意图的概率分布
        probabilities = self.softmax(out)
        
        return out, probabilities, attention_weights

def get_mode_with_last_occurrence(labels):
    """获取标签序列的众数，如果有多个众数，取最后出现的那个
    
    Args:
        labels: 形状为 (time_steps, num_classes) 的独热编码标签序列
        
    Returns:
        众数的类别索引
    """
    # 转换为类别索引
    class_indices = np.argmax(labels, axis=1)
    
    # 计算每个类别的出现次数
    unique_classes, counts = np.unique(class_indices, return_counts=True)
    
    # 找到最大出现次数
    max_count = np.max(counts)
    
    # 找出所有众数类别
    mode_classes = unique_classes[counts == max_count]
    
    # 如果只有一个众数，直接返回
    if len(mode_classes) == 1:
        return mode_classes[0]
    
    # 多个众数时，从后往前找第一个出现的众数
    for i in range(len(class_indices) - 1, -1, -1):
        if class_indices[i] in mode_classes:
            return class_indices[i]
    
    # 默认返回最后一个（理论上不会执行到这里）
    return class_indices[-1]

def load_and_preprocess_data(dataset_path):
    """加载并预处理数据集（使用已处理好众数标签的数据）"""
    print(f"加载数据集: {dataset_path}")
    
    # 加载数据集
    data = np.load(dataset_path)
    X_data = data['data']
    
    print(f"原始数据形状: {X_data.shape}")
    
    # 提取输入特征 (索引第8-27位和44-65位)
    # 注意：Python索引是从0开始的，所以第8位对应索引7，第27位对应索引26，以此类推
    feature_indices_1 = list(range(7, 27))  # 第8-27位
    feature_indices_2 = list(range(43, 65))  # 第44-65位
    feature_indices = feature_indices_1 + feature_indices_2
    
    X = X_data[:, :, feature_indices]
    print(f"提取特征后的数据形状: {X.shape}")
    
    # 直接从数据中提取标签值（使用已处理好的众数标签）
    # 标签位置：索引29是B0100标签值，索引36是B0200标签值
    n_samples, _, _ = X_data.shape
    y1_values = X_data[:, 0, 29].astype(int)  # 使用第一个时间步的标签值
    y2_values = X_data[:, 0, 36].astype(int)  # 使用第一个时间步的标签值
    
    # 将标签值转换为独热编码
    # 0-5六种类型，对应的独热编码分别是[1,0,0,0,0,0]到[0,0,0,0,0,1]
    # 使用float类型以匹配PyTorch交叉熵损失函数的期望
    num_classes = 6
    y1 = np.zeros((n_samples, num_classes), dtype=np.float32)
    y2 = np.zeros((n_samples, num_classes), dtype=np.float32)
    
    # 为每个样本设置对应类别的独热编码
    for i in range(n_samples):
        y1[i, y1_values[i]] = 1.0
        y2[i, y2_values[i]] = 1.0
    
    print(f"标签1独热编码形状: {y1.shape}")
    print(f"标签2独热编码形状: {y2.shape}")
    
    
    # 数据归一化
    print("进行数据归一化处理...")
    # 重塑数据以进行标准化 (samples*time_steps, features)
    n_samples, n_time_steps, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    
    # 使用StandardScaler进行标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    
    # 获取归一化参数
    mean = scaler.mean_
    std = scaler.scale_
    
    # 重塑回原始形状
    X_normalized = X_scaled.reshape(n_samples, n_time_steps, n_features)
    
    print(f"归一化后数据范围: 最小值={np.min(X_normalized):.4f}, 最大值={np.max(X_normalized):.4f}")
    print(f"归一化后数据均值: {np.mean(X_normalized):.4f}, 标准差: {np.std(X_normalized):.4f}")
    
    # 在读取chuli4数据后进行随机打乱
    print("随机打乱数据集...")
    # 生成随机索引
    random_indices = np.random.permutation(n_samples)
    # 使用随机索引打乱数据
    X_normalized = X_normalized[random_indices]
    y1 = y1[random_indices]
    y2 = y2[random_indices]
    print(f"数据集已打乱，样本数: {n_samples}")
    
    # 转换为PyTorch张量
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    y1_tensor = torch.tensor(y1, dtype=torch.long)
    y2_tensor = torch.tensor(y2, dtype=torch.long)
    
    return X_tensor, y1_tensor, y2_tensor, mean, std

def split_dataset(X_tensor, y1_tensor, y2_tensor, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """划分数据集为训练集、验证集和测试集"""
    # 验证比例
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("训练集、验证集和测试集的比例之和必须为1.0")
    
    # 先划分训练集和临时集
    X_train, X_temp, y1_train, y1_temp, y2_train, y2_temp = train_test_split(
        X_tensor, y1_tensor, y2_tensor, test_size=1-train_ratio, random_state=42
    )
    
    # 再从临时集中划分验证集和测试集
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y1_val, y1_test, y2_val, y2_test = train_test_split(
        X_temp, y1_temp, y2_temp, test_size=1-val_size, random_state=42
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")
    
    return X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test

def create_data_loaders(X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test, batch_size, num_workers=0):
    """创建数据加载器"""
    # 创建数据集
    train_dataset = TensorDataset(X_train, y1_train, y2_train)
    val_dataset = TensorDataset(X_val, y1_val, y2_val)
    test_dataset = TensorDataset(X_test, y1_test, y2_test)
    
    # 添加pin_memory=True来加速GPU训练
    pin_memory = torch.cuda.is_available()
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader

def check_v3_v4_status(inputs):
    """检查v3和v4特征的状态，判断是否从某一时刻后一直为0
    
    使用PyTorch向量化操作替代循环，显著提高计算效率
    """
    # 假设inputs的形状是 [batch_size, seq_len, feature_dim]
    # v3特征位于第20个位置（索引19），v4特征位于第39个位置（索引38）
    batch_size, seq_len, _ = inputs.shape
    
    # 提取v3和v4特征 [batch_size, seq_len]
    v3_features = inputs[:, :, 19]
    v4_features = inputs[:, :, 38]
    
    # 检查每个时间步特征是否接近零 [batch_size, seq_len]
    v3_is_zero = v3_features.abs() < 1e-6
    v4_is_zero = v4_features.abs() < 1e-6
    
    # 创建一个从后向前的累积和，用于检查是否存在连续的零
    # 对于每个时间步j，检查从j到seq_len-1是否全为零
    # 使用累积逻辑与操作来实现
    v3_cumulative_zero = torch.zeros_like(v3_is_zero, device=inputs.device)
    v4_cumulative_zero = torch.zeros_like(v4_is_zero, device=inputs.device)
    
    # 从后向前累积检查
    v3_cumulative_zero[:, -1] = v3_is_zero[:, -1]
    v4_cumulative_zero[:, -1] = v4_is_zero[:, -1]
    
    for j in range(seq_len-2, -1, -1):
        v3_cumulative_zero[:, j] = v3_is_zero[:, j] & v3_cumulative_zero[:, j+1]
        v4_cumulative_zero[:, j] = v4_is_zero[:, j] & v4_cumulative_zero[:, j+1]
    
    # 确保至少有5个连续的零值
    # 检查每个样本是否存在至少一个时间步j，使得从j到j+4都满足累积零条件
    # 注意：这里j+4必须小于seq_len
    valid_positions = seq_len - 4  # 确保有足够的时间步
    
    # 对每个样本，检查是否在有效位置范围内存在满足条件的时间步
    v3_zero_mask = v3_cumulative_zero[:, :valid_positions].any(dim=1)
    v4_zero_mask = v4_cumulative_zero[:, :valid_positions].any(dim=1)
    
    return v3_zero_mask, v4_zero_mask

def train_single_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30, 
                       model_save_path='best_model.pth', target_label='B0100'):
    """训练单个意图标签的模型
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 计算设备
        num_epochs: 训练轮次
        model_save_path: 模型保存路径
        target_label: 目标标签名称 ('B0100' 或 'B0200')
    """
    model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"开始训练{target_label}模型...")
    print(f"初始学习率: {scheduler.get_last_lr()[0]:.6f}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for inputs, targets1, targets2 in train_loader:
            inputs = inputs.to(device)
            # 根据目标标签选择对应的目标
            if target_label == 'B0100':
                targets = targets1.to(device).float()  # 确保是float类型
                v3_zero_mask, _ = check_v3_v4_status(inputs)
            else:  # B0200
                targets = targets2.to(device).float()  # 确保是float类型
                _, v4_zero_mask = check_v3_v4_status(inputs)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs, probs, attn = model(inputs)
            
            # 计算损失 - 使用已经过softmax处理的probs作为输入
            loss = criterion(probs, targets)
            
            # 添加L2正则化
            l2_lambda = 0.001  # L2正则化系数
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            loss += l2_lambda * l2_reg
            
            # 反向传播和优化
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计损失
            train_loss += loss.item() * inputs.size(0)
            
            # 计算准确率
            # 使用softmax概率分布找出最大概率对应的类别
            max_prob, predicted = torch.max(probs, 1)
            
            # 从独热编码的targets中提取类别索引
            _, target_classes = torch.max(targets, 1)
            
            # 根据目标标签应用零掩码
            if target_label == 'B0100':
                # 当v3从某一时刻后一直为0时，将B0100意图编号设置为5
                predicted[v3_zero_mask] = 5
            else:  # B0200
                # 当v4从某一时刻后一直为0时，将B0200意图编号设置为5
                predicted[v4_zero_mask] = 5
            
            correct_predictions += (predicted == target_classes).sum().item()
            total_samples += inputs.size(0)
        
        current_lr = scheduler.get_last_lr()[0]
        
        # 计算平均训练损失和准确率
        train_loss = train_loss / total_samples
        train_acc = correct_predictions / total_samples
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets1, targets2 in val_loader:
                inputs = inputs.to(device)
                # 根据目标标签选择对应的目标
                if target_label == 'B0100':
                    targets = targets1.to(device).float()  # 确保是float类型
                    v3_zero_mask, _ = check_v3_v4_status(inputs)
                else:  # B0200
                    targets = targets2.to(device).float()  # 确保是float类型
                    _, v4_zero_mask = check_v3_v4_status(inputs)
                
                # 前向传播
                outputs, probs, attn = model(inputs)
                
                # 计算损失 - 使用已经过softmax处理的probs作为输入
                loss = criterion(probs, targets)
                
                # 统计损失
                val_loss += loss.item() * inputs.size(0)
                
                # 计算准确率
                # 使用softmax概率分布找出最大概率对应的类别
                max_prob, predicted = torch.max(probs, 1)
                
                # 从独热编码的targets中提取类别索引
                _, target_classes = torch.max(targets, 1)
                
                # 根据目标标签应用零掩码
                if target_label == 'B0100':
                    predicted[v3_zero_mask] = 5
                else:  # B0200
                    predicted[v4_zero_mask] = 5
                
                correct_predictions += (predicted == target_classes).sum().item()
                total_samples += inputs.size(0)
        
        # 计算平均验证损失和准确率
        val_loss = val_loss / total_samples
        val_acc = correct_predictions / total_samples
        
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 在验证完成后更新学习率
        scheduler.step(val_loss)  # 传入验证损失作为指标
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"第 {epoch+1} 轮: 保存最佳{target_label}模型，验证损失: {best_val_loss:.6f}")
        
        print(f"第 {epoch+1}/{num_epochs} 轮:")
        print(f"  学习率: {current_lr:.6f}")
        print(f"  训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_curves(train_losses_b0100, val_losses_b0100, train_acc_b0100, val_acc_b0100,
                         train_losses_b0200, val_losses_b0200, train_acc_b0200, val_acc_b0200):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 6))
    
    # 绘制损失曲线对比
    plt.plot(train_losses_b0100, label='B0100训练损失')
    plt.plot(val_losses_b0100, label='B0100验证损失')
    plt.plot(train_losses_b0200, label='B0200训练损失')
    plt.plot(val_losses_b0200, label='B0200验证损失')
    plt.title('训练和验证损失对比')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('training_curves_separate.png')
    plt.show()
    print("训练损失曲线已保存为 training_curves_separate.png")

def evaluate_single_model(model, test_loader, device, target_label='B0100'):
    """评估单个模型并生成详细报告，输出6种意图的概率分布"""
    model.eval()
    all_targets = []
    all_predicted = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, targets1, targets2 in test_loader:
            inputs = inputs.to(device)
            # 根据目标标签选择对应的目标
            if target_label == 'B0100':
                targets = targets1.to(device)
                v3_zero_mask, _ = check_v3_v4_status(inputs)
            else:  # B0200
                targets = targets2.to(device)
                _, v4_zero_mask = check_v3_v4_status(inputs)
            
            # 从独热编码的targets中提取类别索引
            _, target_classes = torch.max(targets, 1)
            
            # 前向传播，获取原始输出和softmax概率分布
            outputs, probs, attn = model(inputs)
            
            # 使用softmax概率分布找出最大概率对应的类别
            max_prob, predicted = torch.max(probs, 1)
            
            # 根据目标标签应用零掩码
            if target_label == 'B0100':
                predicted[v3_zero_mask] = 5
                # 同时更新对应位置的概率分布，确保类别5的概率为1
                probs[v3_zero_mask] = torch.zeros_like(probs[v3_zero_mask])
                probs[v3_zero_mask, 5] = 1.0
            else:  # B0200
                predicted[v4_zero_mask] = 5
                # 同时更新对应位置的概率分布，确保类别5的概率为1
                probs[v4_zero_mask] = torch.zeros_like(probs[v4_zero_mask])
                probs[v4_zero_mask, 5] = 1.0
            
            all_targets.extend(target_classes.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    # 计算评估指标
    accuracy = np.mean(np.array(all_predicted) == np.array(all_targets))
    precision = precision_score(all_targets, all_predicted, average='weighted')
    recall = recall_score(all_targets, all_predicted, average='weighted')
    f1 = f1_score(all_targets, all_predicted, average='weighted')
    
    # 生成分类报告
    report = classification_report(all_targets, all_predicted, output_dict=True)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(all_targets, all_predicted, target_label, f'confusion_matrix_{target_label}_separate.png')
    
    # 打印评估结果
    print(f"\n{target_label}模型评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 保存评估结果
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'classification_report': report,
        'probabilities_sample': [probs.tolist() for probs in all_probabilities[:5]]  # 转换为Python列表
    }
    
    # 保存到JSON文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_{target_label}_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{title} 混淆矩阵')
    plt.colorbar()
    
    classes = ['攻击', '侦察', '协同', '防御', '逃逸', '未知']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # 在混淆矩阵上标注数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"{title} 混淆矩阵已保存为 {save_path}")

def main():
    """主函数"""
    # 数据集路径 - 使用已经处理过的带众数标签的数据集
    dataset_path = r'C:\Users\SONG\Desktop\yitushibie\chuli4\data_with_unified_labels.npz'
    
    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 找不到数据集文件 '{dataset_path}'")
        return
    
    # 设置设备 - 强制使用GPU
    print("强制使用GPU训练")
    device = torch.device('cuda')  # 直接指定使用GPU
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"使用设备: {device}")
    
    # 加载和预处理数据
    X_tensor, y1_tensor, y2_tensor, mean, std = load_and_preprocess_data(dataset_path)
    
    # 划分数据集
    X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test = split_dataset(
        X_tensor, y1_tensor, y2_tensor, 0.6, 0.2, 0.2
    )
    
    # 创建数据加载器
    # 为GPU训练设置更大的batch_size
    batch_size = 128  # GPU上使用更大的batch_size
    # 设置数据加载器的工作进程数
    num_workers = 4  # GPU上使用多进程加载数据
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test, 
        batch_size, num_workers=num_workers
    )
    
    # 创建模型参数
    input_dim = X_tensor.shape[2]  # 输入特征维度
    hidden_dim = 64
    num_layers = 1
    output_dim = 6  # 6种意图
    dropout = 0
    
    print("\n=== 训练B0100意图标签模型 ===")
    # 创建并训练B0100模型
    model_b0100 = LSTMWithAttention(input_dim, hidden_dim, num_layers, output_dim, dropout)
    model_b0100.set_normalization_params(mean, std)
    print("B0100模型结构:")
    print(model_b0100)
    
    # 定义损失函数和优化器
    # 使用BCELoss处理独热编码的概率分布标签
    criterion_b0100 = nn.BCELoss()
    optimizer_b0100 = optim.Adam(model_b0100.parameters(), lr=0.001, weight_decay=0.001)
    scheduler_b0100 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_b0100, mode='min', factor=0.5, patience=5)
    
    # 训练B0100模型
    num_epochs = 30
    trained_model_b0100, train_losses_b0100, val_losses_b0100, train_acc_b0100, val_acc_b0100 = train_single_model(
        model_b0100,
        train_loader,
        val_loader,
        criterion_b0100,
        optimizer_b0100,
        scheduler_b0100,
        device,
        num_epochs,
        model_save_path='best_intent_model_b0100_separate.pth',
        target_label='B0100'
    )
    
    print("\n=== 训练B0200意图标签模型 ===")
    # 创建并训练B0200模型
    model_b0200 = LSTMWithAttention(input_dim, hidden_dim, num_layers, output_dim, dropout)
    model_b0200.set_normalization_params(mean, std)
    print("B0200模型结构:")
    print(model_b0200)
    
    # 定义损失函数和优化器（与B0100保持相同参数，确保训练平等）
    # 使用BCELoss处理独热编码的概率分布标签
    criterion_b0200 = nn.BCELoss()
    optimizer_b0200 = optim.Adam(model_b0200.parameters(), lr=0.001, weight_decay=0.001)
    scheduler_b0200 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_b0200, mode='min', factor=0.5, patience=5)
    
    # 训练B0200模型
    trained_model_b0200, train_losses_b0200, val_losses_b0200, train_acc_b0200, val_acc_b0200 = train_single_model(
        model_b0200,
        train_loader,
        val_loader,
        criterion_b0200,
        optimizer_b0200,
        scheduler_b0200,
        device,
        num_epochs,
        model_save_path='best_intent_model_b0200_separate.pth',
        target_label='B0200'
    )
    
    # 绘制训练曲线
    plot_training_curves(
        train_losses_b0100, val_losses_b0100, train_acc_b0100, val_acc_b0100,
        train_losses_b0200, val_losses_b0200, train_acc_b0200, val_acc_b0200
    )
    
    # 评估模型
    print("\n=== 评估B0100模型 ===")
    results_b0100 = evaluate_single_model(trained_model_b0100, test_loader, device, target_label='B0100')
    
    print("\n=== 评估B0200模型 ===")
    results_b0200 = evaluate_single_model(trained_model_b0200, test_loader, device, target_label='B0200')
    
    # 保存评估结果到JSON文件
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'B0100': results_b0100,
        'B0200': results_b0200
    }
    
    with open('evaluation_results_separate.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print("\n评估结果已保存为 evaluation_results_separate.json")

if __name__ == "__main__":
    main()