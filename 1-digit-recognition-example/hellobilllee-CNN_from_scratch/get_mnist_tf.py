import matplotlib.pyplot as plt
import tensorflow as tf

(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data()

print(train_X.shape, train_Y.shape)  # 输出训练集样本和标签的大小

# 查看数据，例如训练集中第一个样本的内容和标签
# print(train_X[0])       #是一个包含784个元素且值在[0,1]之间的向量
# print(train_Y[0])

# 可视化样本，下面是输出了训练集中前20个样本
figure, axes = plt.subplots(2, 3, figsize=[40, 20])
axes = axes.flatten()
for i in range(6):
    img = train_X[i].reshape(28, 28)
    axes[i].imshow(img, cmap='Greys')
axes[0].set_xticks([])
axes[0].set_yticks([])
plt.tight_layout()
plt.show()
