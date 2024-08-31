import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# 加载数据
# 读取输入数据和输出数据
X = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\Molecular_Descriptor_Training_1.xlsx')  # 输入数据
y_total = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\ADMET_training.xlsx')  # 输出数据



# 如果 y 不是一列，确保将其转换为一列的二分类标签
if y.shape[1] > 1:
    y = y.values.argmax(axis=1)  # 假设 y 是多列的二分类编码，转换为单列的类别标签
else:
    y = y.values.ravel()  # 确保 y 是一维数组

# 可能需要对特征进行标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建XGBoost分类器
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss'
)

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估模型准确性
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确性: {accuracy * 100:.2f}%")

# 打印分类报告，包含精确度、召回率、F1-score
print("\n分类报告:\n", classification_report(y_test, y_pred))

# 计算并打印AUC-ROC分数
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"\nAUC-ROC分数: {roc_auc:.2f}")

# 鲁棒性测试：添加随机噪声
import numpy as np

# 添加噪声
noise_factor = 0.1
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

# 对带噪声的数据进行预测
y_pred_noisy = model.predict(X_test_noisy)

# 评估带噪声数据的模型准确性
accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
print(f"\n带噪声数据的模型准确性: {accuracy_noisy * 100:.2f}%")

# 打印带噪声数据的分类报告
print("\n带噪声数据的分类报告:\n", classification_report(y_test, y_pred_noisy))
