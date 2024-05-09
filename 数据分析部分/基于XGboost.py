import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load the dataset
data_path = '问卷调查结果1.xlsx'
df = pd.read_excel(data_path)

# Preprocessing steps as before
df.replace(-3, np.nan, inplace=True)

selected_columns = [
    '您的性别', '您的年龄', '您的学历', '您的家庭月收入', '您的工作类型是',  # Demographics
    '您是否经历过以下情况？(腰酸背痛)', '6(颈部、腰部感到疲惫)', '6(脖子痛)',  # Health status
    '您是否通过体检、拍X光等方式检查自己的脊柱健康状况',  # Health actions
    '您感觉您的脊柱是健康的吗', '您对脊柱健康的重视程度是',  # Attitudes towards spinal health
    '您了解过脊柱矫形器吗',  # Awareness of spinal orthosis
    '您对脊柱矫形器发展前景持何种态度',
    '您是否购买过脊柱矫形器'  # Target variable
]

df_selected = df[selected_columns].dropna()

X = df_selected.drop('您是否购买过脊柱矫形器', axis=1)
y = df_selected['您是否购买过脊柱矫形器']-1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the XGBoost Classifier
xgb = XGBClassifier(eval_metric='logloss', random_state=41)

# Train the XGBoost Classifier
xgb.fit(X_train_scaled, y_train)

# Predictions
y_pred = xgb.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Feature importance
feature_importances = xgb.feature_importances_

# Mapping feature names with their importance scores and updating names
feature_importance_dict = {feature: importance for feature, importance in zip(X.columns, feature_importances)}
updated_keys = {
    '您是否通过体检、拍X光等方式检查自己的脊柱健康状况': '自身脊柱健康了解程度',
    '您是否经历过以下情况？(腰酸背痛)': '是否在经历腰酸背痛',
    '您的家庭月收入': '家庭月收入情况',
    '您对脊柱健康的重视程度是': '脊柱健康的重视程度',
    '您的工作类型是': '工作类型情况',
    '您了解过脊柱矫形器吗': '脊柱矫形器了解程度',
    '您感觉您的脊柱是健康的吗': '脊柱健康个人感受',
    '您的年龄': '年龄'
}
feature_importance_dict = {updated_keys.get(k, k): v for k, v in feature_importance_dict.items()}

sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)[:8]

# Generating confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Preparing data for the bar plot
features, importances = zip(*sorted_feature_importance)

print(sorted_feature_importance, accuracy, classification_rep, cm, features)