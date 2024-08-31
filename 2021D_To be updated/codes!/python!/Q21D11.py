import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# 读取 Excel 文件，跳过第一行和第一列的描述信息
target_df = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\R_activity.xlsx', sheet_name='Sheet1')
features_df = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\Molecular_Descriptor_Training.xlsx', sheet_name='Sheet1')
# 找出数值完全相同的列
equal_columns = features_df.apply(lambda x: x.nunique() == 1, axis=0)

# 删除数值完全相同的列
features_df = features_df.loc[:, ~equal_columns]

## 计算相关系数矩阵
#correlation_matrix = features_df.corr()

# 使用 seaborn 可视化相关系数矩阵
#plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
#plt.title('Correlation Matrix')
#plt.show()



# 确保因变量和自变量具有相同的样本数量
assert len(target_df) == len(features_df), "因变量和自变量样本数量不匹配"

# 合并因变量和自变量
data_df = pd.concat([features_df, target_df], axis=1)
data_df = data_df.astype(float)
data_df.columns = data_df.columns.astype(str)

# 分离自变量和因变量
X = data_df.iloc[:, :-1]  # 除了最后一列的所有列为自变量
y = data_df.iloc[:, -1]   # 最后一列为因变量

# 标准化自变量
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 初始化线性回归模型
#model = LinearRegression()

# 初始化 RFE，选择前 20 个特征
#rfe = RFE(model, n_features_to_select=20)
#rfe = rfe.fit(X_scaled, y)

# 获取支持的特征（即被选中的特征）
#selected_features = X.columns[rfe.support_]

# 输出选择的特征
#print(f"选择的特征:\n{selected_features}")


# 标准化自变量
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用岭回归代替线性回归
model = Ridge(alpha=1.0)

# 初始化 RFE，选择前 20 个特征
rfe = RFE(model, n_features_to_select=20)
rfe = rfe.fit(X_scaled, y)

# 获取支持的特征
selected_features = X.columns[rfe.support_]
print(f"选择的特征:\n{selected_features}")

# 可视化选择的特征与因变量的关系
for feature in selected_features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_df[feature], y=data_df[target_df.columns[0]], alpha=0.7)
    plt.title(f'{feature} vs {target_df.columns[0]}')
    plt.xlabel(feature)
    plt.ylabel(target_df.columns[0])
    plt.grid(True)
    plt.show()

# 可视化选择的特征的分布
for feature in selected_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data_df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


