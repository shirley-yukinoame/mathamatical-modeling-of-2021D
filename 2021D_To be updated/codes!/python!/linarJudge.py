import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
# 读取 Excel 文件，跳过第一行和第一列的描述信息
target_df = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\R_activity.xlsx', sheet_name='Sheet1')
features_df = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\Molecular_Descriptor_Training.xlsx', sheet_name='Sheet1')

# 确保因变量和自变量具有相同的样本数量
assert len(target_df) == len(features_df), "因变量和自变量样本数量不匹配"

# 合并因变量和自变量
data_df = pd.concat([features_df, target_df], axis=1)
data_df = data_df.astype(float)
data_df.columns = data_df.columns.astype(str)

# 分离自变量和因变量
X = data_df.iloc[:, :-1]  # 除了最后一列的所有列为自变量
y = data_df.iloc[:, -1]   # 最后一列为因变量

# 初始化一个字典存储相关系数
correlation_dict = {}

# 计算每个自变量与因变量之间的皮尔逊相关系数
for column in X.columns:
    corr, _ = pearsonr(X[column], y)
    correlation_dict[column] = corr

# 将相关系数转换为DataFrame以便查看
correlation_df = pd.DataFrame(list(correlation_dict.items()), columns=['Variable', 'Pearson Correlation'])

correlation_df.to_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\Rho.xlsx', index=False)
import seaborn as sns

# 将因变量添加到自变量数据集中
data_with_y = X.copy()
data_with_y['Y'] = y

# 绘制散点图矩阵
sns.pairplot(data_with_y, y_vars='Y', x_vars=X.columns[:10])  # 这里只显示前10个自变量，可以调整显示更多
plt.show()
