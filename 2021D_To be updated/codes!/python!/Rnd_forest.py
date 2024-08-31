import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # 使用回归随机森林
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


# 标准化自变量
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 获取特征重要性
importances = model.feature_importances_

# 将特征重要性与特征名称配对
features_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# 输出前 20 个最重要的特征
print("前 20 个最重要的特征:")
print(features_importance.head(20))

# 可视化特征重要性
plt.figure(figsize=(10, 8))
features_importance.head(20).plot(kind='barh')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()



