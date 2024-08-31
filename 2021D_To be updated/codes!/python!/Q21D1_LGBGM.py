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
X = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\Molecular_Descriptor_Training.xlsx')  # 输入数据
# 找出数值完全相同的列
equal_columns = X.apply(lambda x: x.nunique() == 1, axis=0)
# 删除数值完全相同的列
X = X.loc[:, ~equal_columns]

y = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\R_activity.xlsx')  # 输出数据

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
y = y.values.ravel()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 LightGBM 数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 设置 LightGBM 参数
params = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.15,
    'feature_fraction': 0.8
}

# 训练模型
model = lgb.train(params,
                   train_data,
                   valid_sets=[test_data],
                   num_boost_round=100)

# 进行预测
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# 计算和打印均方误差
mse = mean_squared_error(y_test, y_pred)
print(f"均方误差 (MSE): {mse:.2f}")

# 获取特征重要性
feature_importance = model.feature_importance()

# 创建 DataFrame 存储特征重要性
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})

# 按重要性排序，并选择前 20 个特征
top_features = importance_df.sort_values(by='Importance', ascending=False).head(20)

# 可视化特征重要性
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title('Top 20 Features Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance
}).sort_values(by='importance', ascending=False)

top_20_features = importance_df.head(20)['feature'].tolist()

# 打印前20个重要变量的名称
print("前20个重要变量的名称：")
for i, feature in enumerate(top_20_features, 1):
    print(f"{i}. {feature}")

# 绘制前20个重要变量的散点图与分布特征
plt.figure(figsize=(15, 10))

for i, feature in enumerate(top_20_features):
    plt.subplot(4, 5, i+1)
    sns.scatterplot(x = X[feature], y = y, alpha=0.7)
    plt.title(f'{feature}')
    plt.xlabel('')
    plt.ylabel('Target')

plt.tight_layout()
plt.show()

# 绘制前20个重要变量的分布特征
plt.figure(figsize=(15, 10))

for i, feature in enumerate(top_20_features):
    plt.subplot(4, 5, i+1)
    sns.histplot(X[feature], kde=True, color='green')
    plt.title(f'{feature}')
    plt.xlabel('')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 删除输入数据中其他变量
X = X[top_20_features]  # 保留前20个重要变量和目标变量

#使用随机森林进行
# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#feature_names = X_train.columns

# 实例化随机森林分类器
clf = RandomForestRegressor(n_estimators=100, random_state=42)
# 训练模型
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出评价指标
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")

#可视化误差
plt.figure(figsize=(15, 10))
sns.histplot(y_test-y_pred, kde=True, color='green')
#plt.title(f'{feature}')
plt.xlabel('')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

# 交叉验证
cv_scores = cross_val_score(clf, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", np.mean(cv_scores))


#生成验证数据集
X_t = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\Molecular_Descriptor_testing.xlsx')
#y_t = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\ADMET_testing.xlsx')

X_t = X_t[top_20_features]

#预测验证结果
y_t = clf.predict(X_t)

y_pred_df = pd.DataFrame(y_t, columns=['pIC50'])

y_pred_df.to_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\B___.xlsx', index=False)


#使用lightgbm方法求解

# 创建 LightGBM 数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 设置 LightGBM 参数
params = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.15,
    'feature_fraction': 0.8
}

# 训练模型
model = lgb.train(params,
                   train_data,
                   valid_sets=[test_data],
                   num_boost_round=100)

# 进行预测
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# 计算和打印均方误差等指标
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出评价指标
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")

#可视化误差
plt.figure(figsize=(15, 10))
sns.histplot(y_test-y_pred, kde=True, color='green')
#plt.title(f'{feature}')
plt.xlabel('')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()