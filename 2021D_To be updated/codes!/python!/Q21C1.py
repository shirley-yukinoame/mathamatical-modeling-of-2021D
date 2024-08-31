import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd

# 读取Excel文件
file_path = 'C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\E_activity.xlsx'  
df = pd.read_excel(file_path, header=1)  # 跳过第一行描述

# 提取输入和输出数据
X = df.iloc[:, 1].values.reshape(-1, 1)  # 第二列为输入特征
X = np.log(X)
y = df.iloc[:, 2].values  # 第三列为输出标签


# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建和训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.ravel())

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方误差 (MSE): {mse}")
print(f"R^2 得分: {r2}")

# 绘制结果
X_curve = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_curve = model.predict(X_curve)

plt.scatter(X_test, y_test, color='blue', label='actual value')
plt.plot(X_curve, y_curve, color='red', linewidth=2, label='prediction value')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Random Forest Regression Output')
plt.legend()
plt.show()
