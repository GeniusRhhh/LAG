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

class AttentionMechanism(nn.Module):
    """注意力机制模块"""
    def __init__(self, hidden_size):
        super(AttentionMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, lstm_output):
        # lstm_output形状: [batch_size, seq_len, hidden_size]
        attention_weights = self.attention(lstm_output).squeeze(2)  # [batch_size, seq_len]
        attention_weights = self.softmax(attention_weights).unsqueeze(2)  # [batch_size, seq_len, 1]
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # [batch_size, hidden_size]
        return context_vector, attention_weights

class LSTMWithAttention(nn.Module):
    """LSTM + 注意力机制模型"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMWithAttention, self).__init__()
        # 添加批归一化层
        self.bn = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.attention = AttentionMechanism(hidden_dim * 2)  # *2 因为是双向LSTM
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
        probabilities = self.softmax(out)
        
        return out, probabilities, attention_weights

class MultiAircraftLSTM(nn.Module):
    """多飞机独立模型架构"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(MultiAircraftLSTM, self).__init__()
        # 为两架飞机创建独立的模型实例
        self.aircraft1_model = LSTMWithAttention(input_dim, hidden_dim, num_layers, output_dim, dropout)
        self.aircraft2_model = LSTMWithAttention(input_dim, hidden_dim, num_layers, output_dim, dropout)
    
    def set_normalization_params(self, mean, std):
        """设置两架飞机模型的反归一化参数"""
        self.aircraft1_model.set_normalization_params(mean, std)
        self.aircraft2_model.set_normalization_params(mean, std)
    
    def forward(self, x):
        # 分别通过两架飞机的模型
        output1, prob1, attn1 = self.aircraft1_model(x)
        output2, prob2, attn2 = self.aircraft2_model(x)
        
        return (output1, prob1, attn1), (output2, prob2, attn2)

def load_and_preprocess_data(dataset_path):
    """加载并预处理数据集"""
    print(f"加载数据集: {dataset_path}")
    
    # 加载数据集
    data = np.load(dataset_path)
    X_data = data['data']
    
    print(f"原始数据形状: {X_data.shape}")
    
    # 提取输入特征 (索引第8-27位和44-65位)
    # 注意：Python索引是从0开始的，所以第8位对应索引7，第27位对应索引26，以此类推
    feature_indices_1 = list(range(7, 26))  # 第8-27位
    feature_indices_2 = list(range(43, 64))  # 第44-65位
    feature_indices = feature_indices_1 + feature_indices_2
    
    X = X_data[:, :, feature_indices]
    print(f"提取特征后的数据形状: {X.shape}")
    
    # 提取标签 (索引第31-36位和38-43位)
    # 第31-36位对应飞机1的意图独热编码，第38-43位对应飞机2的意图独热编码
    label_indices_1 = list(range(30, 35))  # 第31-36位
    label_indices_2 = list(range(37, 42))  # 第38-43位
    
    # 对于每个样本，取最后一个时间步的标签
    y1 = X_data[:, -1, label_indices_1]
    y2 = X_data[:, -1, label_indices_2]
    
    # 将独热编码转换为类别索引
    y1 = np.argmax(y1, axis=1)
    y2 = np.argmax(y2, axis=1)
    
    print(f"标签形状 (飞机1): {y1.shape}")
    print(f"标签形状 (飞机2): {y2.shape}")
    print(f"标签1唯一值: {np.unique(y1)}")
    print(f"标签2唯一值: {np.unique(y2)}")
    
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
    
    # 转换为PyTorch张量
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    y1_tensor = torch.tensor(y1, dtype=torch.long)
    y2_tensor = torch.tensor(y2, dtype=torch.long)
    
    return X_tensor, y1_tensor, y2_tensor, mean, std

def split_dataset(X_tensor, y1_tensor, y2_tensor, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """划分数据集为训练集、验证集和测试集"""
    # 验证比例
    assert train_ratio + val_ratio + test_ratio == 1.0, "划分比例必须总和为1"
    
    # 先将数据集分为训练集和剩余集
    X_train, X_temp, y1_train, y1_temp, y2_train, y2_temp = train_test_split(
        X_tensor, y1_tensor, y2_tensor, train_size=train_ratio, random_state=42
    )
    
    # 再将剩余集分为验证集和测试集
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y1_val, y1_test, y2_val, y2_test = train_test_split(
        X_temp, y1_temp, y2_temp, train_size=val_ratio_adjusted, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test

def create_data_loaders(X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test, batch_size=32):
    """创建数据加载器"""
    # 创建数据集
    train_dataset = TensorDataset(X_train, y1_train, y2_train)
    val_dataset = TensorDataset(X_val, y1_val, y2_val)
    test_dataset = TensorDataset(X_test, y1_test, y2_test)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30, model_save_path='best_intent_model.pth', loss_weights=(1.0, 1.0)):
    """训练模型
    
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
        loss_weights: 两架飞机的损失权重 (weight1, weight2)
    """
    model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies_1 = []
    train_accuracies_2 = []
    val_accuracies_1 = []
    val_accuracies_2 = []
    
    print("开始训练模型...")
    print(f"初始学习率: {scheduler.get_last_lr()[0]:.6f}")
    print(f"损失权重 - B0100: {loss_weights[0]}, B0200: {loss_weights[1]}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct_predictions_1 = 0
        correct_predictions_2 = 0
        total_samples = 0
        
        for inputs, targets1, targets2 in train_loader:
            inputs, targets1, targets2 = inputs.to(device), targets1.to(device), targets2.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播 - 根据模型类型处理输出
            if isinstance(model, MultiAircraftLSTM):
                # 使用多飞机独立模型
                (outputs1, probs1, attn1), (outputs2, probs2, attn2) = model(inputs)
            else:
                # 使用原始单一模型
                outputs1, probs1, _ = model(inputs)
                outputs2, probs2, _ = model(inputs)
            
            # 计算损失，应用权重
            loss1 = criterion(outputs1, targets1) * loss_weights[0]
            loss2 = criterion(outputs2, targets2) * loss_weights[1]
            loss = loss1 + loss2
            
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
            
            # 检查v3和v4的状态
            v3_zero_mask, v4_zero_mask = check_v3_v4_status(inputs)
            
            # 计算准确率
            _, predicted1 = torch.max(probs1, 1)
            _, predicted2 = torch.max(probs2, 1)
            
            # 当v3从某一时刻后一直为0时，将B0100意图编号设置为5
            predicted1[v3_zero_mask] = 5
            # 当v4从某一时刻后一直为0时，将B0200意图编号设置为5
            predicted2[v4_zero_mask] = 5
            
            correct_predictions_1 += (predicted1 == targets1).sum().item()
            correct_predictions_2 += (predicted2 == targets2).sum().item()
            total_samples += inputs.size(0)
        
        current_lr = scheduler.get_last_lr()[0]
        
        # 计算平均训练损失和准确率
        train_loss = train_loss / total_samples
        train_acc1 = correct_predictions_1 / total_samples
        train_acc2 = correct_predictions_2 / total_samples
        
        train_losses.append(train_loss)
        train_accuracies_1.append(train_acc1)
        train_accuracies_2.append(train_acc2)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_predictions_1 = 0
        correct_predictions_2 = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets1, targets2 in val_loader:
                inputs, targets1, targets2 = inputs.to(device), targets1.to(device), targets2.to(device)
                
                # 前向传播 - 根据模型类型处理输出
                if isinstance(model, MultiAircraftLSTM):
                    (outputs1, probs1, attn1), (outputs2, probs2, attn2) = model(inputs)
                else:
                    outputs1, probs1, _ = model(inputs)
                    outputs2, probs2, _ = model(inputs)
                
                # 计算损失，应用权重
                loss1 = criterion(outputs1, targets1) * loss_weights[0]
                loss2 = criterion(outputs2, targets2) * loss_weights[1]
                loss = loss1 + loss2
                
                # 统计损失
                val_loss += loss.item() * inputs.size(0)
                
                # 检查v3和v4的状态
                v3_zero_mask, v4_zero_mask = check_v3_v4_status(inputs)
                
                # 计算准确率
                _, predicted1 = torch.max(probs1, 1)
                _, predicted2 = torch.max(probs2, 1)
                
                # 当v3从某一时刻后一直为0时，将B0100意图编号设置为5
                predicted1[v3_zero_mask] = 5
                # 当v4从某一时刻后一直为0时，将B0200意图编号设置为5
                predicted2[v4_zero_mask] = 5
                
                correct_predictions_1 += (predicted1 == targets1).sum().item()
                correct_predictions_2 += (predicted2 == targets2).sum().item()
                total_samples += inputs.size(0)
        
        # 计算平均验证损失和准确率
        val_loss = val_loss / total_samples
        val_acc1 = correct_predictions_1 / total_samples
        val_acc2 = correct_predictions_2 / total_samples
        
        val_losses.append(val_loss)
        val_accuracies_1.append(val_acc1)
        val_accuracies_2.append(val_acc2)
        
        # 在验证完成后更新学习率
        scheduler.step(val_loss)  # 传入验证损失作为指标
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"第 {epoch+1} 轮: 保存最佳模型，验证损失: {best_val_loss:.6f}")
        
        print(f"第 {epoch+1}/{num_epochs} 轮:")
        print(f"  学习率: {current_lr:.6f}")
        print(f"  训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
        print(f"  B0100训练准确率: {train_acc1:.4f}, B0100验证准确率: {val_acc1:.4f}")
        print(f"  B0200训练准确率: {train_acc2:.4f}, B0200验证准确率: {val_acc2:.4f}")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accuracies_1, val_accuracies_1, train_accuracies_2, val_accuracies_2)
    
    return model

def plot_training_curves(train_losses, val_losses, train_acc1, val_acc1, train_acc2, val_acc2):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 6))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='训练损失')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_acc1) + 1), train_acc1, 'b-', label='B0100训练准确率')
    plt.plot(range(1, len(val_acc1) + 1), val_acc1, 'b--', label='B0100验证准确率')
    plt.plot(range(1, len(train_acc2) + 1), train_acc2, 'r-', label='B0200训练准确率')
    plt.plot(range(1, len(val_acc2) + 1), val_acc2, 'r--', label='B0200验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    print("训练曲线已保存为 training_curves.png")

def check_v3_v4_status(inputs):
    """检查v3和v4是否从某一时刻后一直为0
    
    Args:
        inputs: 输入数据，形状为[batch_size, seq_len, features]
    
    Returns:
        v3_zero_mask: v3从某一时刻后一直为0的样本掩码
        v4_zero_mask: v4从某一时刻后一直为0的样本掩码
    """
    # 注意：这里假设v3和v4的索引在原始数据中是17和18
    # 但在预处理后的数据中，特征可能已经被重新排列
    # 根据load_and_preprocess_data函数，我们使用的是索引7-26和43-65
    # 因此需要确定v3和v4在预处理后数据中的位置
    
    # 假设在原始数据中，v3和v4的索引是17和18，它们属于前20个特征（索引7-26）
    # 所以在预处理后的数据中，它们的索引应该是17-7=10和18-7=11
    v3_idx = 10  # 预处理后数据中v3的索引
    v4_idx = 11  # 预处理后数据中v4的索引
    
    batch_size = inputs.size(0)
    v3_zero_mask = torch.zeros(batch_size, dtype=torch.bool)
    v4_zero_mask = torch.zeros(batch_size, dtype=torch.bool)
    
    for i in range(batch_size):
        # 获取单个样本的v3和v4序列
        v3_seq = inputs[i, :, v3_idx]
        v4_seq = inputs[i, :, v4_idx]
        
        # 检查v3是否从某一时刻后一直为0
        zero_indices_v3 = (v3_seq == 0).nonzero()
        if len(zero_indices_v3) > 0:
            # 找到第一个0的位置
            first_zero_idx = zero_indices_v3[0].item()
            # 检查从第一个0之后的所有值是否都是0
            if torch.all(v3_seq[first_zero_idx:] == 0):
                v3_zero_mask[i] = True
        
        # 检查v4是否从某一时刻后一直为0
        zero_indices_v4 = (v4_seq == 0).nonzero()
        if len(zero_indices_v4) > 0:
            # 找到第一个0的位置
            first_zero_idx = zero_indices_v4[0].item()
            # 检查从第一个0之后的所有值是否都是0
            if torch.all(v4_seq[first_zero_idx:] == 0):
                v4_zero_mask[i] = True
    
    return v3_zero_mask, v4_zero_mask

def evaluate_model(model, test_loader, device):
    """评估模型并生成详细报告"""
    model.eval()
    all_targets1 = []
    all_predicted1 = []
    all_probabilities1 = []
    all_targets2 = []
    all_predicted2 = []
    all_probabilities2 = []
    
    with torch.no_grad():
        for inputs, targets1, targets2 in test_loader:
            inputs, targets1, targets2 = inputs.to(device), targets1.to(device), targets2.to(device)
            
            # 检查v3和v4的状态
            v3_zero_mask, v4_zero_mask = check_v3_v4_status(inputs)
            
            # 根据模型类型处理输出
            if isinstance(model, MultiAircraftLSTM):
                # 使用多飞机独立模型
                (outputs1, probs1, attn1), (outputs2, probs2, attn2) = model(inputs)
            else:
                # 使用原始单一模型
                outputs1, probs1, _ = model(inputs)
                outputs2, probs2, _ = model(inputs)
            
            _, predicted1 = torch.max(probs1, 1)
            _, predicted2 = torch.max(probs2, 1)
            
            # 当v3从某一时刻后一直为0时，将B0100意图编号设置为5
            predicted1[v3_zero_mask] = 5
            # 当v4从某一时刻后一直为0时，将B0200意图编号设置为5
            predicted2[v4_zero_mask] = 5
            
            all_targets1.extend(targets1.cpu().numpy())
            all_predicted1.extend(predicted1.cpu().numpy())
            all_probabilities1.extend(probs1.cpu().numpy())
            all_targets2.extend(targets2.cpu().numpy())
            all_predicted2.extend(predicted2.cpu().numpy())
            all_probabilities2.extend(probs2.cpu().numpy())
    
    # 计算评估指标 - 飞机1
    accuracy1 = np.mean(np.array(all_predicted1) == np.array(all_targets1))
    precision1 = precision_score(all_targets1, all_predicted1, average='weighted')
    recall1 = recall_score(all_targets1, all_predicted1, average='weighted')
    f11 = f1_score(all_targets1, all_predicted1, average='weighted')
    
    # 计算评估指标 - 飞机2
    accuracy2 = np.mean(np.array(all_predicted2) == np.array(all_targets2))
    precision2 = precision_score(all_targets2, all_predicted2, average='weighted')
    recall2 = recall_score(all_targets2, all_predicted2, average='weighted')
    f12 = f1_score(all_targets2, all_predicted2, average='weighted')
    
    # 打印评估结果
    print("\n=== B0100 评估结果 ===")
    print(f"准确率: {accuracy1:.4f}")
    print(f"精确率: {precision1:.4f}")
    print(f"召回率: {recall1:.4f}")
    print(f"F1分数: {f11:.4f}")
    
    print("\n=== B0200 评估结果 ===")
    print(f"准确率: {accuracy2:.4f}")
    print(f"精确率: {precision2:.4f}")
    print(f"召回率: {recall2:.4f}")
    print(f"F1分数: {f12:.4f}")
    
    # 生成分类报告
    print("\n=== B0100 分类报告 ===")
    print(classification_report(all_targets1, all_predicted1, target_names=['攻击', '侦察', '协同', '防御', '逃逸', '未知']))
    
    print("\n=== B0200 分类报告 ===")
    print(classification_report(all_targets2, all_predicted2, target_names=['攻击', '侦察', '协同', '防御', '逃逸', '未知']))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(all_targets1, all_predicted1, 'B0100', 'confusion_matrix_B0100.png')
    plot_confusion_matrix(all_targets2, all_predicted2, 'B0200', 'confusion_matrix_B0200.png')
    
    # 保存评估结果
    results = {
        'B0100': {
            'accuracy': accuracy1,
            'precision': precision1,
            'recall': recall1,
            'f1_score': f11
        },
        'B0200': {
            'accuracy': accuracy2,
            'precision': precision2,
            'recall': recall2,
            'f1_score': f12
        }
    }
    
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n评估结果已保存为 evaluation_results.json")
    
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
    # 数据集路径
    dataset_path = r'c:\Users\86188\Desktop\yitushibie\chuli3\带动作独热编码的数据集\enhanced_time_series_data_with_actions.npz'
    
    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 找不到数据集文件 '{dataset_path}'")
        return
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载和预处理数据
    X_tensor, y1_tensor, y2_tensor, mean, std = load_and_preprocess_data(dataset_path)
    
    # 划分数据集
    X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test = split_dataset(
        X_tensor, y1_tensor, y2_tensor, 0.6, 0.2, 0.2
    )
    
    # 创建数据加载器
    batch_size = 32
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test, batch_size
    )
    
    # 创建模型
    input_dim = X_tensor.shape[2]  # 输入特征维度
    hidden_dim = 256
    num_layers = 2
    output_dim = 6  # 6种意图
    dropout = 0.2
    
    # 使用多飞机独立模型架构
    use_multi_aircraft_model = True  # 设置为True使用独立模型，False使用共享模型
    
    if use_multi_aircraft_model:
        print("使用多飞机独立模型架构")
        model = MultiAircraftLSTM(input_dim, hidden_dim, num_layers, output_dim, dropout)
    else:
        print("使用共享模型架构")
        model = LSTMWithAttention(input_dim, hidden_dim, num_layers, output_dim, dropout)
    
    # 设置反归一化参数
    model.set_normalization_params(mean, std)
    print("模型结构:")
    print(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 设置损失权重 - 增加B0200的权重以提高其准确率
    loss_weights = (1.0, 2.0)  # (B0100权重, B0200权重)
    
    # 训练模型
    num_epochs = 30
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        device, 
        num_epochs,
        loss_weights=loss_weights
    )
    
    # 评估模型
    evaluation_results = evaluate_model(trained_model, test_loader, device)
    
    print("\n意图识别模型训练和评估完成!")

if __name__ == "__main__":
    main()