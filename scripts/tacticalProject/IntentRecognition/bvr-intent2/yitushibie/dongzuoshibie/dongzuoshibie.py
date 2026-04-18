import os
import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 定义LSTM模型
class LSTMActionClassifier(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, num_classes=11, dropout=0.2):
        super(LSTMActionClassifier, self).__init__()
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        # 全连接层用于分类
        self.fc = nn.Linear(hidden_size, num_classes)
        # Dropout层用于防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 激活函数
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x形状: (batch_size, seq_length, input_size)
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_time_step = lstm_out[:, -1, :]
        # Dropout
        out = self.dropout(last_time_step)
        # 全连接层
        out = self.fc(out)
        # 返回原始输出和softmax概率分布
        return out, self.softmax(out)


# 加载数据
def load_data():
    try:
        # 加载训练集
        train_data = np.load('training_data.npz')
        X_train_np = train_data['X']
        y_train = torch.tensor(train_data['y'], dtype=torch.float32)

        # 加载验证集
        val_data = np.load('validation_data.npz')
        X_val_np = val_data['X']
        y_val = torch.tensor(val_data['y'], dtype=torch.float32)

        # 加载测试集
        test_data = np.load('test_data.npz')
        X_test_np = test_data['X']
        y_test = torch.tensor(test_data['y'], dtype=torch.float32)

        # 创建并训练数据归一化器
        print("对数据进行归一化处理...")
        scaler = StandardScaler()

        # 对每个特征维度分别进行归一化
        # 重塑数据以适应StandardScaler的输入格式
        n_features = X_train_np.shape[2]
        train_shape = X_train_np.shape
        val_shape = X_val_np.shape
        test_shape = X_test_np.shape

        # 对训练数据进行归一化
        X_train_flat = X_train_np.reshape(-1, n_features)
        scaler.fit(X_train_flat)
        X_train_norm = scaler.transform(X_train_flat).reshape(train_shape)

        # 使用与训练数据相同的参数对验证和测试数据进行归一化
        X_val_flat = X_val_np.reshape(-1, n_features)
        X_val_norm = scaler.transform(X_val_flat).reshape(val_shape)

        X_test_flat = X_test_np.reshape(-1, n_features)
        X_test_norm = scaler.transform(X_test_flat).reshape(test_shape)

        # 保存归一化参数以便后续反归一化
        np.savez('normalization_params.npz', mean=scaler.mean_, scale=scaler.scale_)
        print("归一化参数已保存，均值:", scaler.mean_, " 标准差:", scaler.scale_)

        # 转换为张量
        X_train = torch.tensor(X_train_norm, dtype=torch.float32)
        X_val = torch.tensor(X_val_norm, dtype=torch.float32)
        X_test = torch.tensor(X_test_norm, dtype=torch.float32)

        # 创建数据集
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        # 加载动作类型映射
        with open('action_types.json', 'r') as f:
            action_types = json.load(f)
            # 反转映射，用于从索引获取动作名称
            idx_to_action = {v: k for k, v in action_types.items()}

        return train_dataset, val_dataset, test_dataset, idx_to_action, scaler
    except Exception as e:
        print(f"加载数据时出错: {e}")
        # 尝试运行数据预处理
        try:
            import shujuji
            print("尝试运行数据预处理...")
            samples, labels = shujuji.process_data(os.path.dirname(os.path.abspath(__file__)), window_size=50)
            shujuji.split_data(samples, labels)
            # 重新加载数据
            return load_data()
        except Exception as e2:
            print(f"数据预处理也失败了: {e2}")
            raise


# 训练模型
def train_model(model, train_loader, val_loader, device, learning_rate=0.001, num_epochs=20, patience=10):
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 记录训练和验证的损失和准确率
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 最佳验证准确率
    best_val_accuracy = 0.0
    # 早停计数器
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算训练损失和准确率
            train_loss += loss.item() * inputs.size(0)
            # 找到预测的类别
            _, predicted = torch.max(outputs.data, 1)
            _, true_labels = torch.max(labels.data, 1)
            total_train += true_labels.size(0)
            correct_train += (predicted == true_labels).sum().item()

        # 计算平均训练损失和准确率
        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 在验证集上评估
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)

                # 计算验证损失和准确率
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                _, true_labels = torch.max(labels.data, 1)
                total_val += true_labels.size(0)
                correct_val += (predicted == true_labels).sum().item()

        # 计算平均验证损失和准确率
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 更新学习率
        scheduler.step(val_loss)

        # 打印当前epoch的结果
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_lstm_model.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # 早停机制
        if early_stop_counter >= patience:
            print(f"早停: {patience}个epoch没有提高验证准确率")
            break

    # 绘制训练和验证的损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()

    # 绘制训练和验证的准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='训练准确率')
    plt.plot(val_accuracies, label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

    return model


# 测试模型
def test_model(model, test_loader, device, idx_to_action, scaler=None):
    model.eval()
    all_preds = []
    all_true = []
    all_probs = []

    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs, probs = model(inputs)

            # 记录概率分布
            all_probs.extend(probs.cpu().numpy())

            # 找到预测的类别
            _, predicted = torch.max(outputs.data, 1)
            _, true_labels = torch.max(labels.data, 1)

            # 记录预测和真实标签
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(true_labels.cpu().numpy())

            total += true_labels.size(0)
            correct += (predicted == true_labels).sum().item()

    # 计算测试准确率
    test_accuracy = 100 * correct / total
    print(f'测试准确率: {test_accuracy:.2f}%')

    # 随机选择500个样本用于混淆矩阵
    if len(all_preds) > 500:
        indices = np.random.choice(len(all_preds), 500, replace=False)
        sample_preds = [all_preds[i] for i in indices]
        sample_true = [all_true[i] for i in indices]
        print(f'随机选择了500个测试样本生成混淆矩阵')
    else:
        sample_preds = all_preds
        sample_true = all_true
        print(f'测试样本不足500个，使用全部{len(all_preds)}个样本生成混淆矩阵')

    # 按照用户指定的顺序定义动作类型
    action_order = ['level_flight', 'accelerate', 'decelerate', 'turn_left', 'turn_right',
                    'climb', 'climb_left', 'climb_right', 'dive', 'dive_left', 'dive_right']

    # 生成完整的11x11混淆矩阵
    # 创建一个空的11x11矩阵
    cm_11x11 = np.zeros((11, 11), dtype=int)

    # 遍历所有样本，填充混淆矩阵
    for true_idx, pred_idx in zip(sample_true, sample_preds):
        true_action = idx_to_action[true_idx]
        pred_action = idx_to_action[pred_idx]

        # 找到在action_order中的索引
        true_pos = action_order.index(true_action)
        pred_pos = action_order.index(pred_action)

        # 填充矩阵：行=预测动作，列=真实动作
        cm_11x11[pred_pos, true_pos] += 1

    # 创建混淆矩阵图表
    plt.figure(figsize=(14, 12))

    # 使用完整的11x11矩阵，行和列按照用户指定的顺序排列
    sns.heatmap(cm_11x11, annot=True, fmt='d', cmap='Blues',
                xticklabels=[action.replace('_', ' ') for action in action_order],  # 将下划线替换为空格以提高可读性
                yticklabels=[action.replace('_', ' ') for action in action_order])

    # 设置标题和标签以反映用户要求
    plt.title('混淆矩阵（行=预测动作，列=真实动作）', fontsize=16)
    plt.xlabel('真实动作', fontsize=14)
    plt.ylabel('预测动作', fontsize=14)

    # 调整标签字体大小
    plt.xticks(fontsize=11, rotation=45, ha='right')
    plt.yticks(fontsize=11)

    # 计算正确识别的样本数量（对角线元素之和）
    correct_predictions = np.trace(cm_11x11)
    total_predictions = len(sample_preds)
    accuracy = (correct_predictions / total_predictions) * 100

    # 打印混淆矩阵统计信息
    print("\n混淆矩阵统计详情（行=预测动作，列=真实动作）：")
    print("动作顺序：平飞、加速、减速、左转、右转、爬升、左爬升、右爬升、俯冲、左俯冲、右俯冲")
    for i, action_row in enumerate(action_order):
        row_str = f"{action_row.replace('_', ' ')}: "
        for j in range(11):
            row_str += f"{cm_11x11[i, j]:3d} "
        print(row_str)

    # 打印正确识别的样本数量和正确率
    print(f"\n500个随机测试样本的识别结果：")
    print(f"正确识别的样本数量: {correct_predictions}")
    print(f"总样本数量: {total_predictions}")
    print(f"识别正确率: {accuracy:.2f}%")

    # 保存图表
    plt.savefig('confusion_matrix.png')
    plt.close()

    print("混淆矩阵已生成并保存为'confusion_matrix.png'。")
    print("矩阵说明：行代表模型预测的动作，列代表样本实际的真实动作。")
    print("例如：如果某个单元格的值为20，表示有20个样本实际属于该列的动作类别，但被模型预测为该行的动作类别。")

    # 获取实际存在的类别索引
    unique_classes = np.unique(all_true)
    print(f"实际存在的类别索引: {unique_classes}")
    print(f"实际存在的类别数量: {len(unique_classes)}")

    # 根据实际存在的类别索引筛选类别名称
    actual_target_names = [idx_to_action[i] for i in unique_classes]

    # 生成分类报告，指定labels参数以匹配实际类别
    report = classification_report(all_true, all_preds, labels=unique_classes, target_names=actual_target_names)
    print("分类报告:\n", report)

    # 保存分类报告到文件
    with open('classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
        # 同时保存类别信息
        f.write("\n\n实际使用的类别索引和名称:\n")
        for idx, name in zip(unique_classes, actual_target_names):
            f.write(f"{idx}: {name}\n")

    return test_accuracy


# 预测单个样本
def predict_sample(model, sample, device, idx_to_action, scaler=None):
    # 反归一化数据
    if scaler is not None:
        # 重塑样本以适应反归一化
        sample_shape = sample.shape
        n_features = sample_shape[-1] if len(sample_shape) > 1 else 1
        sample_flat = sample.reshape(-1, n_features)

        # 反归一化
        sample_denorm_flat = sample_flat * scaler.scale_ + scaler.mean_
        sample_denorm = sample_denorm_flat.reshape(sample_shape)

        print(f"\n原始样本数据（反归一化后）:")
        print(f"样本形状: {sample_denorm.shape}")
        # 仅显示前3个时间步的特征（如果有时间维度）
        if len(sample_denorm.shape) > 2:
            print(f"样本部分数据（前3个时间步）: {sample_denorm[0, :3, :]}")
        else:
            print(f"样本部分数据: {sample_denorm[:3]}")
    elif os.path.exists('normalization_params.npz'):
        # 如果没有传入scaler，尝试从文件加载
        params = np.load('normalization_params.npz')
        mean = params['mean']
        scale = params['scale']

        # 重塑样本以适应反归一化
        sample_shape = sample.shape
        n_features = sample_shape[-1] if len(sample_shape) > 1 else 1
        sample_flat = sample.reshape(-1, n_features)

        # 反归一化
        sample_denorm_flat = sample_flat * scale + mean
        sample_denorm = sample_denorm_flat.reshape(sample_shape)

        print(f"\n原始样本数据（反归一化后）:")
        print(f"样本形状: {sample_denorm.shape}")
        # 仅显示前3个时间步的特征
        if len(sample_denorm.shape) > 2:
            print(f"样本部分数据（前3个时间步）: {sample_denorm[0, :3, :]}")
        else:
            print(f"样本部分数据: {sample_denorm[:3]}")

    # 确保样本形状正确
    if isinstance(sample, np.ndarray):
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        sample_tensor = sample.to(device)

    model.eval()
    with torch.no_grad():
        # 模型返回两个值：原始输出和softmax概率分布
        _, probabilities = model(sample_tensor)  # 只关注概率分布
        probabilities = probabilities.squeeze().cpu().numpy()

        # 找到概率最大的类别
        max_prob_index = np.argmax(probabilities)
        max_prob = probabilities[max_prob_index]

        # 获取对应的动作名称
        predicted_action = idx_to_action[max_prob_index]

        # 打印11种动作的概率分布
        print(f"\n动作概率分布:")
        for action_idx, prob in enumerate(probabilities):
            action_name = idx_to_action[action_idx]
            print(f"{action_name}: {prob:.8f}")

        print(f"\n最终预测结果: 使用概率分布中置信度最高的动作 '{predicted_action}' (置信度: {max_prob:.6f})")

    return predicted_action, max_prob, probabilities


def main():
    # 设置随机种子以确保结果可复现
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    train_dataset, val_dataset, test_dataset, idx_to_action, scaler = load_data()

    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型 - 使用窗口长度50
    input_size = 4  # X, Y, Z, Velocity
    hidden_size = 128
    num_layers = 2
    num_classes = 11  # 11种动作类型
    model = LSTMActionClassifier(input_size, hidden_size, num_layers, num_classes).to(device)

    print(f"模型配置: 输入大小={input_size}, 隐藏层大小={hidden_size}, 层数={num_layers}, 类别数={num_classes}")
    print(f"训练数据样本数: {len(train_dataset)}, 验证数据样本数: {len(val_dataset)}, 测试数据样本数: {len(test_dataset)}")

    # 训练模型
    trained_model = train_model(model, train_loader, val_loader, device, learning_rate=0.001, num_epochs=20, patience=15)

    # 测试模型
    test_model(trained_model, test_loader, device, idx_to_action)

    print("动作识别模型训练和测试完成！")
    print("请运行 shujuji.py 以处理chuli2数据并将动作识别结果保存到chuli3目录")

if __name__ == "__main__":
    main()