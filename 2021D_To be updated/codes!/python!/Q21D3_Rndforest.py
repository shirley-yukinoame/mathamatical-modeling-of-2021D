import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt

# 读取输入数据和输出数据
X = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\Molecular_Descriptor_Training_1.xlsx')  # 输入数据
y = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\ADMET_training.xlsx')  # 输出数据

X.columns = X.columns.astype(str)
y.columns = y.columns.astype(str)
# 将输出数据从五列拆分为五个二分类任务（每列为一个输出）
# 假设每列为一个独立的标签，这里将其转化为适合模型的格式

y = y.values  # 转换为numpy数组

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#feature_names = X_train.columns

# 实例化随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 评估模型
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))



from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

# 交叉验证
cv_scores = cross_val_score(clf, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", np.mean(cv_scores))

# 混淆矩阵
y_pred_train = clf.predict(X_train)
cm = confusion_matrix(np.argmax(y_train, axis=1), np.argmax(y_pred_train, axis=1))
print("Confusion Matrix:\n", cm)


#生成验证数据集
X_t = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\Molecular_Descriptor_testing.xlsx')
#y_t = pd.read_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\ADMET_testing.xlsx')

X_t.columns = X_t.columns.astype(str)
#y_t.columns = y_t.columns.astype(str)
#预测验证结果
#X_t = X_t[feature_names]
y_t = clf.predict(X_t)

y_pred_df = pd.DataFrame(y_t, columns=['Caco-2','CYP3A4','hERG','HOB','MN'])

y_pred_df.to_excel('C:\\Users\\吴钰贤\\Desktop\\2024第四次模拟题\\A___.xlsx', index=False)