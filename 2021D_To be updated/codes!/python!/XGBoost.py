import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

# 对目标变量进行编码（如果目标变量是分类的）
# le = LabelEncoder()
# y = le.fit_transform(y)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化XGBoost模型
model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')

# 训练模型
model.fit(X_train, y_train)

# 获取特征重要性
feature_importances = model.feature_importances_

# 创建特征重要性数据框
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# 显示前20个最重要的特征
top_20_features = importance_df.head(20)
print(top_20_features)

# 绘制特征重要性图
plt.figure(figsize=(10, 8))
plt.barh(top_20_features['Feature'], top_20_features['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 20 Features by Importance')
plt.gca().invert_yaxis()
plt.show()
