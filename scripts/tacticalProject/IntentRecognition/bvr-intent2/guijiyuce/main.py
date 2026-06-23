import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import time
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)


class LSTMTrajectoryPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, output_steps, dropout=0.2):
        super(LSTMTrajectoryPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.output_steps = output_steps

        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # 简化的全连接层
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # 预测未来轨迹
        outputs = []

        # 初始化用于预测的隐藏状态和细胞状态
        current_h = hn
        current_c = cn

        # 从最后一个时间步的输出开始
        current_input = x[:, -1:, :]

        # 预测output_steps个时间步
        for _ in range(self.output_steps):
            # 使用LSTM层进行预测
            lstm_out, (current_h, current_c) = self.lstm(current_input, (current_h, current_c))

            # 应用全连接层和激活函数
            out = self.fc1(lstm_out.squeeze(1))
            out = self.relu(out)

            current_pred = self.fc2(out)
            outputs.append(current_pred)

            # 将预测结果重塑为LSTM的输入格式
            current_input = current_pred.unsqueeze(1)

        # 将输出列表转换为张量
        outputs = torch.stack(outputs, dim=1)
        return outputs


def load_and_preprocess_data(data_path):
    # 加载数据
    data = np.load(data_path, allow_pickle=True)

    # 提取历史数据和标签数据（仅使用前3个特征，即x、y、z坐标）
    history_data = data['history_data'][:, :, :3]  # 形状: (num_windows, 50, 3)
    label_data = data['label_data'][:, :, :3]  # 形状: (num_windows, 25, 3)

    # 数据归一化处理
    num_windows, history_steps, num_features = history_data.shape
    history_reshaped = history_data.reshape(-1, num_features)

    # 对历史数据进行归一化
    scaler = StandardScaler()
    scaler.fit(history_reshaped)

    # 应用归一化到历史数据和标签数据
    history_normalized = scaler.transform(history_data.reshape(-1, num_features)).reshape(num_windows, history_steps,
                                                                                          num_features)
    label_reshaped = label_data.reshape(-1, num_features)
    label_normalized = scaler.transform(label_reshaped).reshape(num_windows, 25, num_features)

    # 转换为PyTorch张量
    history_tensor = torch.from_numpy(history_normalized).float()
    label_tensor = torch.from_numpy(label_normalized).float()

    return history_tensor, label_tensor, scaler


def split_data(history_tensor, label_tensor):
    # 创建数据集
    dataset = TensorDataset(history_tensor, label_tensor)

    # 计算数据集分割大小
    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    # 分割数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 创建数据加载器
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, test_dataset


def train_model(model, train_loader, val_loader, device, epochs=30):
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_model_path = "best_lstm_trajectory_model.pth"
    patience_counter = 0
    max_patience = 8  # 早停机制的耐心值

    # 用于记录损失曲线
    train_losses = []
    val_losses = []

    print("开始训练简化版LSTM轨迹预测模型...")
    train_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # 计算平均训练损失
        train_mse = train_loss / len(train_loader.dataset)
        train_losses.append(train_mse)

        # 在验证集上评估模型
        model.eval()
        val_mse = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_mse += loss.item() * inputs.size(0)

        # 计算平均验证损失
        val_mse /= len(val_loader.dataset)
        val_losses.append(val_mse)

        # 调整学习率
        scheduler.step(val_mse)

        # 保存最佳模型和早停检查
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0  # 重置耐心计数器
        else:
            patience_counter += 1

        print(f'Epoch {epoch + 1}/{epochs}, 训练损失: {train_mse:.6f}, 验证损失: {val_mse:.6f}')

        # 早停机制
        if patience_counter >= max_patience:
            print(f"验证损失连续{max_patience}轮没有改善，提前停止训练。")
            break

    train_end_time = time.time()
    total_train_time = train_end_time - train_start_time
    print(f"训练完成！总耗时: {total_train_time:.2f} 秒")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, 'b-', label='训练损失')
    plt.plot(range(1, epochs + 1), val_losses, 'r-', label='验证损失')
    plt.title('LSTM模型训练与验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MSE损失')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_model_path


def evaluate_model(model, test_loader, scaler, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # 保存预测和目标值
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # 将预测结果转换回原始数据范围
    all_predictions_np = np.concatenate(all_predictions)
    all_targets_np = np.concatenate(all_targets)

    # 重塑数据以应用反归一化
    predictions_reshaped = all_predictions_np.reshape(-1, 3)
    targets_reshaped = all_targets_np.reshape(-1, 3)

    # 应用反归一化（单位：米）
    predictions_denormalized = scaler.inverse_transform(predictions_reshaped).reshape(all_predictions_np.shape)
    targets_denormalized = scaler.inverse_transform(targets_reshaped).reshape(all_targets_np.shape)

    # 计算每个样本每个时间步的误差（单位：米）
    sample_rmse_list = []
    sample_mae_list = []

    for i in range(len(predictions_denormalized)):
        # 获取第i个样本的预测轨迹和真实轨迹
        pred_trajectory = predictions_denormalized[i]
        true_trajectory = targets_denormalized[i]

        # 计算每个时间步的误差
        step_errors = pred_trajectory - true_trajectory

        # 计算每个时间步的RMSE
        step_mse = np.mean(step_errors ** 2, axis=1)
        step_rmse = np.sqrt(step_mse)

        # 计算每个时间步的MAE
        step_mae = np.mean(np.abs(step_errors), axis=1)

        # 计算该样本的平均RMSE和MAE
        sample_rmse = np.mean(step_rmse)
        sample_mae = np.mean(step_mae)

        sample_rmse_list.append(sample_rmse)
        sample_mae_list.append(sample_mae)

    # 计算测试集所有样本的平均RMSE和MAE
    test_rmse = np.mean(sample_rmse_list)
    test_mae = np.mean(sample_mae_list)

    # 打印评估结果
    print(f"测试集平均RMSE: {test_rmse:.6f} (米)")
    print(f"测试集平均MAE: {test_mae:.6f} (米)")

    return predictions_denormalized, targets_denormalized


def visualize_trajectory(history_data, predictions, targets, scaler):
    # 选择测试集中的一个样本进行可视化
    sample_index = 0

    # 获取该样本的历史数据（需要反归一化）
    history_normalized = history_data[sample_index].numpy()
    history_reshaped = history_normalized.reshape(-1, 3)
    history_denormalized = scaler.inverse_transform(history_reshaped).reshape(history_normalized.shape)

    # 创建3D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制真实历史轨迹（绿色实线）
    ax.plot(
        history_denormalized[:, 0],
        history_denormalized[:, 1],
        history_denormalized[:, 2],
        'g-', linewidth=2, label='真实历史轨迹'
    )

    # 获取历史轨迹的最后一个点
    last_history_point = history_denormalized[-1]

    # 绘制预测轨迹（蓝色虚线）
    prediction_x = np.insert(predictions[sample_index][:, 0], 0, last_history_point[0])
    prediction_y = np.insert(predictions[sample_index][:, 1], 0, last_history_point[1])
    prediction_z = np.insert(predictions[sample_index][:, 2], 0, last_history_point[2])

    ax.plot(
        prediction_x,
        prediction_y,
        prediction_z,
        'b--', linewidth=2, label='预测轨迹'
    )

    # 绘制真实未来轨迹（绿色实线带标记）
    real_x = np.insert(targets[sample_index][:, 0], 0, last_history_point[0])
    real_y = np.insert(targets[sample_index][:, 1], 0, last_history_point[1])
    real_z = np.insert(targets[sample_index][:, 2], 0, last_history_point[2])

    ax.plot(
        real_x,
        real_y,
        real_z,
        'g-', linewidth=2, marker='o', markersize=4, label='真实未来轨迹'
    )

    # 添加标题和标签
    ax.set_title('LSTM空战目标轨迹预测结果', fontsize=15)
    ax.set_xlabel('X坐标 (米)', fontsize=12)
    ax.set_ylabel('Y坐标 (米)', fontsize=12)
    ax.set_zlabel('Z坐标 (米)', fontsize=12)

    # 添加图例
    ax.legend(fontsize=12, loc='upper right')

    # 设置坐标轴范围，使图形更美观
    all_x = np.concatenate([history_denormalized[:, 0], predictions[sample_index][:, 0], targets[sample_index][:, 0]])
    all_y = np.concatenate([history_denormalized[:, 1], predictions[sample_index][:, 1], targets[sample_index][:, 1]])
    all_z = np.concatenate([history_denormalized[:, 2], predictions[sample_index][:, 2], targets[sample_index][:, 2]])

    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    z_range = all_z.max() - all_z.min()
    max_range = max(x_range, y_range, z_range) * 0.5

    mean_x = all_x.mean()
    mean_y = all_y.mean()
    mean_z = all_z.mean()

    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)

    # 设置视角
    ax.view_init(elev=30, azim=45)

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()


def main():
    # 设置数据路径
    data_path = r"c:\Users\86188\Desktop\try\windowed_data\windowed_trajectory_data_75_50_25.npz"

    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载和预处理数据
    history_tensor, label_tensor, scaler = load_and_preprocess_data(data_path)

    # 分割数据
    train_loader, val_loader, test_loader, test_dataset = split_data(history_tensor, label_tensor)

    # 初始化模型（简化版本）
    input_size = 3  # x, y, z坐标
    hidden_size = 64  # 减小隐藏层大小
    num_layers = 1  # 减少LSTM层数为1层
    output_size = 3
    output_steps = 25  # 预测25个时间步

    model = LSTMTrajectoryPredictor(input_size, hidden_size, num_layers, output_size, output_steps)
    model.to(device)

    # 训练模型
    best_model_path = train_model(model, train_loader, val_loader, device, epochs=50)

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))

    # 评估模型
    print("\n在测试集上评估模型...")
    predictions_denormalized, targets_denormalized = evaluate_model(model, test_loader, scaler, device)

    # 可视化结果
    print("\n生成轨迹可视化...")
    # 获取测试集的原始历史数据
    test_history_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    visualize_trajectory(test_history_data, predictions_denormalized, targets_denormalized, scaler)

    print("可视化完成！")


if __name__ == "__main__":
    main()