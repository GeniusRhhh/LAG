\# BVR 意图识别数据处理与 BiLSTM-Self-Attention 训练脚本



本项目包含两个核心功能：



1\. \*\*构建滑动窗口数据集（prepare\_intent\_dataset.py）\*\*  

&nbsp;  从 NEU 特征 CSV 构建滑窗样本，标准化处理，并划分 train/val/test（npz）。

2\. \*\*训练意图识别模型（train\_intent\_bilstm\_from\_npz.py）\*\*  

&nbsp;  使用 BiLSTM + Self-Attention 模型进行训练与评估，输出最佳模型、日志和图表。



项目适用于超视距空战（BVR）目标的战术意图识别任务。



---



\## 📌 1. 环境依赖



建议 Python 3.8+。



安装依赖：



```bash

pip install -r requirements.txt



\## 📌 2. 使用示例

（1）python prepare\_intent\_dataset.py \\

&nbsp; --input "D:\\datas\\output\\features\_NEU\_from\_radar\_pairs.csv" \\

&nbsp; --out\_dir "D:\\datas\\output\\intent\_npz\_T32\_S1" \\

&nbsp; --T 32 --S 1 \\

&nbsp; --use-status-embed \\

&nbsp; --perclass-step "防御=4,攻击=2,逃逸=1,协同=1,探测=1,中立=1"

跑完以后，out\_dir 里会有：

train.npz

val.npz

test.npz

meta.npy

（2)

