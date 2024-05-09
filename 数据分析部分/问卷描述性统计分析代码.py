import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Setting the aesthetic style of the plots
sns.set_style("whitegrid")
# Load the survey results
survey_results_path = '问卷调查结果1.xlsx'
survey_results = pd.read_excel(survey_results_path)

# Basic demographic information analysis
demographics = survey_results[['您的性别', '您的年龄', '您的学历', '您的家庭月收入']]

# Rename columns for clarity
demographics.columns = ['Gender', 'Age', 'Education', 'MonthlyIncome']


# Convert categorical data to more meaningful strings based on the questionnaire structure
survey_results_renamed = survey_results.copy()

# Map the responses based on the questionnaire content
gender_map = {1: '女', 2: '男'}
age_map = {1: '16岁及以下', 2: '17-22岁', 3: '23-28岁', 4: '29-34岁', 5: '35岁及以上'}
education_map = {1: '初中及以下', 2: '高中/中专', 3: '本科/大专', 4: '硕士研究生', 5: '博士研究生'}
monthly_income_map = {1: '3000元及以下', 2: '3001-8000元', 3: '8001-13000元', 4: '13001-18000元', 5: '18001元以上'}

survey_results_renamed['您的性别'] = survey_results['您的性别'].map(gender_map)
survey_results_renamed['您的年龄'] = survey_results['您的年龄'].map(age_map)
survey_results_renamed['您的学历'] = survey_results['您的学历'].map(education_map)
survey_results_renamed['您的家庭月收入'] = survey_results['您的家庭月收入'].map(monthly_income_map)

# Now, provide summary statistics for these demographics
demographics_summary = survey_results_renamed[['您的性别', '您的年龄', '您的学历', '您的家庭月收入']].describe()

# Overview of awareness and attitude towards spine health
awareness_columns = ['您知道脊柱侧弯吗', '您对脊柱健康的重视程度是', '您对脊柱侧弯的矫正态度']
awareness_summary = survey_results_renamed[awareness_columns].describe()


# Preparing data for plotting
gender_distribution = survey_results_renamed['您的性别'].value_counts(normalize=True) * 100
age_distribution = survey_results_renamed['您的年龄'].value_counts(normalize=True) * 100
education_distribution = survey_results_renamed['您的学历'].value_counts(normalize=True) * 100
monthly_income_distribution = survey_results_renamed['您的家庭月收入'].value_counts(normalize=True) * 100
print(age_distribution)
print(education_distribution)
print(monthly_income_distribution)
# Creating subplots for each demographic variable
fig, ax = plt.subplots(2, 2, figsize=(16, 12))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 设置负号显示为减号

# Reorder the index based on the survey option order for accurate plotting
gender_order = ['男', '女']
age_order = ['16岁及以下', '17-22岁', '23-28岁', '29-34岁', '35岁及以上']
education_order = ['初中及以下', '高中/中专', '本科/大专', '硕士研究生', '博士研究生']
monthly_income_order = ['3000元及以下', '3001-8000元', '8001-13000元', '13001-18000元', '18001元以上']



# Plotting with the correct order
sns.barplot(x=gender_order, y=gender_distribution.reindex(gender_order).values, ax=ax[0, 0], palette="coolwarm")
ax[0, 0].set_title("Gender Distribution")
ax[0, 0].set_ylabel("Percentage")

sns.barplot(x=age_order, y=age_distribution.reindex(age_order).values, ax=ax[0, 1], palette="coolwarm")
ax[0, 1].set_title("Age Distribution")
ax[0, 1].set_ylabel("Percentage")

sns.barplot(x=education_order, y=education_distribution.reindex(education_order).values, ax=ax[1, 0], palette="coolwarm")
ax[1, 0].set_title("Education Distribution")
ax[1, 0].set_ylabel("Percentage")
ax[1, 0].tick_params(axis='x', rotation=45)

sns.barplot(x=monthly_income_order, y=monthly_income_distribution.reindex(monthly_income_order).values, ax=ax[1, 1], palette="coolwarm")
ax[1, 1].set_title("Monthly Income Distribution")
ax[1, 1].set_ylabel("Percentage")
ax[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()