import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import  mean_absolute_error, r2_score

# 读取数据
# 读取输入数据和输出数据
X = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\Molecular_Descriptor_Training_1.xlsx')  # 输入数据
# 找出数值完全相同的列
equal_columns = X.apply(lambda x: x.nunique() == 1, axis=0)
# 删除数值完全相同的列
X = X.loc[:, ~equal_columns]

y = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\ADMET_training.xlsx')  # 输出数据

## 计算相关系数矩阵
#correlation_matrix = X.corr()

# 使用 seaborn 可视化相关系数矩阵
#plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
#plt.title('Correlation Matrix')
#plt.show()

# 合并因变量和自变量
data_df = pd.concat([X, y], axis=1)
data_df = data_df.astype(float)
data_df.columns = data_df.columns.astype(str)

# 确保 y 是一维数组
y = y.values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
