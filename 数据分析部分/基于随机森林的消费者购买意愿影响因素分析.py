from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置微软雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 加载数据集
data_path = '问卷调查结果1.xlsx'
df = pd.read_excel(data_path)
df.replace(-3, np.nan, inplace=True)

# 选择特征
selected_columns = [
    '您的性别', '您的年龄', '您的学历', '您的家庭月收入', '您的工作类型是',
    '您是否经历过以下情况？(腰酸背痛)', '6(颈部、腰部感到疲惫)', '6(脖子痛)',
    '您是否通过体检、拍X光等方式检查自己的脊柱健康状况',
    '您感觉您的脊柱是健康的吗',
    '您了解过脊柱矫形器吗',
    '您对脊柱矫形器发展前景持何种态度',
    '您是否购买过脊柱矫形器'
]

# 选取作为变量的列
df_selected = df[selected_columns]

# 去除缺失值
df_selected.dropna(inplace=True)

# 定义目标变量 (Y) 和自变量 (X)
X = df_selected.drop('您是否购买过脊柱矫形器', axis=1)
y = df_selected['您是否购买过脊柱矫形器']

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化随机森林分类器
rf = RandomForestClassifier(n_estimators=150, random_state=42)

# 训练模型
rf.fit(X_train_scaled, y_train)

# 预测
y_pred = rf.predict(X_test_scaled)

# 模型评估
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# 计算特征重要性
feature_importances = rf.feature_importances_

# 构建模型特征和重要性的哈希表
feature_importance_dict = {feature: importance for feature, importance in zip(X.columns, feature_importances)}

old_key = feature_importance_dict.pop('您是否通过体检、拍X光等方式检查自己的脊柱健康状况')
new_key = '自身脊柱健康了解程度'
# 更新dict,使用新的key
feature_importance_dict.update({new_key: old_key})
old_key1 = feature_importance_dict.pop('您是否经历过以下情况？(腰酸背痛)')
new_key1 = '是否在经历腰酸背痛'
# 更新dict,使用新的key
feature_importance_dict.update({new_key1: old_key1})
old_key2 = feature_importance_dict.pop('您的家庭月收入')
new_key2 = '家庭月收入情况'
# 更新dict,使用新的key
feature_importance_dict.update({new_key2: old_key2})
#old_key3 = feature_importance_dict.pop('您对脊柱健康的重视程度是')
#new_key3 = '脊柱健康的重视程度'
# 更新dict,使用新的key
#feature_importance_dict.update({new_key3: old_key3})

old_key4 = feature_importance_dict.pop('您的工作类型是')
new_key4 = '工作类型情况'
# 更新dict,使用新的key
feature_importance_dict.update({new_key4: old_key4})
old_key5 = feature_importance_dict.pop('您了解过脊柱矫形器吗')
new_key5 = '脊柱矫形器了解程度'
feature_importance_dict.update({new_key5: old_key5})
old_key6 = feature_importance_dict.pop('您感觉您的脊柱是健康的吗')
new_key6 = '脊柱健康个人感受'
# 更新dict,使用新的key
feature_importance_dict.update({new_key6: old_key6})
old_key7 = feature_importance_dict.pop('您的年龄')
new_key7 = '年龄'
# 更新dict,使用新的key
feature_importance_dict.update({new_key7: old_key7})
# Sorting the feature importances in descending order
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)

print(sorted_feature_importance, accuracy)

sorted_feature_importance = sorted_feature_importance[:8]
print()
print(accuracy_score(y_test, y_pred))
print(classification_rep)
# 创建一个 DataFrame 来存储特征名称和它们的重要性得分
feature_importance_df = pd.DataFrame(sorted_feature_importance)
feature_importance_df.to_csv('feature_importance.csv', index=False)
# 生成混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Extract feature names and their importance scores
features, importances = zip(*sorted_feature_importance)
print(features)
# Creating the bar plot
plt.figure(figsize=(10, 8))
plt.barh(features, importances, color='skyblue')
plt.xlabel('重要性')
plt.title('预测脊柱矫形器购买意向的重要特征')
plt.gca().invert_yaxis()  # Invert y axis to have the most important feature at the top
plt.show()
# 保存为CSV文件