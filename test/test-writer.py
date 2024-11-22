from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 初始化 TensorBoard
writer = SummaryWriter(log_dir="runs/multi_classifier_comparison")

num_classifiers = 3  # 假设有 3 个分类器
epochs = 50

for epoch in range(epochs):
    # 创建一个字典，记录每个分类器的准确率
    accuracy_dict = {}
    for classifier_id in range(num_classifiers):
        accuracy = np.random.uniform(0.7, 0.9) + classifier_id * 0.01
        accuracy_dict[f"Classifier_{classifier_id}"] = accuracy

    # 使用 add_scalars 将所有分类器的结果写入同一图表
    writer.add_scalars("Accuracy", accuracy_dict, epoch)
writer.close()

# tensorboard --logdir=runs