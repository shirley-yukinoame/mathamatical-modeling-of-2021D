import pandas as pd
import numpy as np

# 读取输入数据和输出数据
X = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\Molecular_Descriptor_Training_1.xlsx')  # 输入数据
y = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\ADMET_training.xlsx')  # 输出数据
X.columns = X.columns.astype(str)
y.columns = y.columns.astype(str)
# 确保输出数据是二分类格式（0和1）
y = y.astype(int).values  # 转换为numpy数组

from sklearn.model_selection import train_test_split

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 构建模型
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(5, activation='sigmoid')  # 输出层，5个输出，每个输出是0或1
])

# 编译模型
model.compile(optimizer='adam', 
              loss='binary_crossentropy',  # 使用binary_crossentropy作为多标签分类的损失函数
              metrics=['accuracy'])

# 打印模型总结
model.summary()


# 训练模型
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')



from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

# 将数据打乱
X_shuffled, y_shuffled = shuffle(X, y, random_state=42)

# 使用KFold进行交叉验证
cv_scores = cross_val_score(model, X_shuffled, y_shuffled, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", np.mean(cv_scores))

from sklearn.metrics import multilabel_confusion_matrix

# 预测
y_pred = (model.predict(X_test) > 0.5).astype(int)  # 将概率转换为0或1

# 计算混淆矩阵
conf_matrices = multilabel_confusion_matrix(y_test, y_pred)
for i, cm in enumerate(conf_matrices):
    print(f"Confusion Matrix for Output {i}:")
    print(cm)
