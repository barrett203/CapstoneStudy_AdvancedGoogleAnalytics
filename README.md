# CapstoneStudy_AdvancedGoogleAnalytics
Why is company demonstrating a high employee turnover rate?- Following example project provided by the course to expand my Python coding skills. 
## Saliford Motors data analysis case study 
<img align="right" width="450" height="425" src="https://www.4cornerresources.com/wp-content/uploads/2021/11/Measuring-Employee-Satisfaction-scaled.jpeg">
Salifort Motors is a fictional French-based alternative energy vehicle manufacturer. Its global workforce of over 100,000 employees research, design, construct, validate, and distribute electric, solar, algae, and hydrogen-based vehicles. Salifort’s end-to-end vertical integration model has made it a global leader at the intersection of alternative energy and automobiles.

**For the purpose of this case study, a scenario has been constructed:**                                                                                            
As a data specialist working for Salifort Motors, you have received the results of a recent employee survey. The HR department has tasked you with analyzing the data to come up with ideas for how to increase employee retention. To help with this, they would like you to design a model that predicts whether an employee will leave the company based on their  department, number of projects, average monthly hours, and any other data points you deem helpful. For this deliverable, you are asked to choose a method to approach this data challenge based on your prior course work. Select either a regression model or a tree-based machine learning model to predict whether an employee will leave the company.

## Ask
### Key stakeholders
* The HR department at Saliford Motors- wants to take some initiatives to improve employee satisfaction levels at the company

### Business task
* Analyze the data collected by the HR department and build a model that predicts whether or not an employee will leave the company.

### Key Question
* What are the factors that contribute to employees wanting to leave the company?

## Prepare 

### Data source: 

Hr Analytics Job Prediction Data on [Kaggle](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?select=HR_comma_sep.csv). The data contains survey responses from 14,999 employees at Saliford Motors. This dataset contains input related to satisfaction level, performance review, number of projects, average monthly hours, time spent at the company, number of work accidents, promotions over the last 5 years, work department and salary. The data dictionary can be found [here](https://github.com/barrett203/CapstoneStudy_AdvancedGoogleAnalytics/blob/main/Data%20dictionary%20.png).

### Limitations: 
* Since the data was collected through a survey, the results may not be accurate as such participants may not provide honest and accurate answers.

# Process

## Applications
Excel will be used to load the data and initially look for any issues. Python will then be used to transform and explore the data, as well as to construct a predictive machine learning model.  

## Transform and Explore 
All Python code can be found [here](https://github.com/barrett203/CapstoneStudy_AdvancedGoogleAnalytics/blob/main/PythonScript).

1) Load the necessary packages
2) Check to see if the data has been loaded correctly

```
# For data manipulation
!pip install numpy
import numpy as np
!pip install pandas
import pandas as pd


# For data visualization
!pip install matplotlib
import matplotlib.pyplot as plt
!pip install seaborn
import seaborn as sns

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

# For data modeling
!pip install xgboost
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

!pip install scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

#Load dataset into a dataframe
df0 = pd.read_csv("C:\\Users\\ocbar\\Documents\\Advanced google data analytics\\Capstone study\\archive\\HR_capstone.csv")
```

3)Rename the columns as needed 
```
df0 = df0.rename(columns={'Work_accident': 'work_accident',
                          'average_montly_hours': 'average_monthly_hours',
                          'time_spend_company': 'tenure',
                          'Department': 'department'})
```
4)Check for missing values and duplicates. 
```
df0.isna().sum()
df0.duplicated().sum()
```
* There are no missing values in the dataset but there are 3,008 rows contain duplicates. That is apprximately 20% of the data.

5)Drop duplicates and save resulting dataframe in a new variable as needed
```
df1 = df0.drop_duplicates(keep='first')
```
6)Check for outliers by visualising distribution of time spent at the company 
```
plt.figure(figsize=(6,6))
plt.title('Boxplot to detect outliers for tenure', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df1['tenure'])
plt.show()
```
![Boxplot](https://github.com/barrett203/CapstoneStudy_AdvancedGoogleAnalytics/blob/main/Boxplot_OutliersForTenure.png "Boxplot")

7)Determine the number of rows containing outliers 
```
# Compute the 25th percentile value in `tenure`
percentile25 = df1['tenure'].quantile(0.25)

# Compute the 75th percentile value in `tenure`
percentile75 = df1['tenure'].quantile(0.75)

# Compute the interquartile range in `tenure`
iqr = percentile75 - percentile25

# Define the upper limit and lower limit for non-outlier values in `tenure`
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

# Identify subset of data containing outliers in `tenure`
outliers = df1[(df1['tenure'] > upper_limit) | (df1['tenure'] < lower_limit)]

# Count how many rows in the data contain outliers in `tenure`
print("Number of rows in the data containing outliers in `tenure`:", len(outliers))
```
* 824 rows in the time spent at company column contain outliers. As certain types of models are more sensitive to outliers than others, these will be considered at the model building stage of analysis.

# Analyze 

## Select summary statistics and visualizations 
1)Showcase descriptive statistics 
```
df0.describe()
```
![Summary](https://github.com/barrett203/CapstoneStudy_AdvancedGoogleAnalytics/blob/main/Descriptive%20statistics%20.png "Summary")
* The average employee rated their satisfaction as 6.13/10, was scored at 7.16/10 on their last evaluation and has completed 3.80 projects. On average, they have also spent 3.50 years working at the company, have experienced 1.45 work accidents and have had 0.21 promotions over the last 5 years. 2.38/10 employees have also left the company.

2)Create a stacked boxplot showing average monthly hours distributions for number of project, comparing the distributions of employees who stayed versus those who left. Also plot a stacked histogram to visualize the distribution of number_project for those who stayed and those who left.
```
# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Create boxplot showing `average_monthly_hours` distributions for `number_project`, comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='average_monthly_hours', y='number_project', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Monthly hours by number of projects', fontsize='14')

# Create histogram showing distribution of `number_project`, comparing employees who stayed versus those who left
tenure_stay = df1[df1['left']==0]['number_project']
tenure_left = df1[df1['left']==1]['number_project']
sns.histplot(data=df1, x='number_project', hue='left', multiple='dodge', shrink=2, ax=ax[1])
ax[1].set_title('Number of projects histogram', fontsize='14')

# Display the plots
plt.show()
```
![Projects](https://github.com/barrett203/CapstoneStudy_AdvancedGoogleAnalytics/blob/main/MonthlyHours_ByNumberOfProjects.png "Projects")
* The mean hours of each group (stayed and left) increases with number of projects worked.
* Everyone with seven projects left the company, and the interquartile ranges of this group and those who left with six projects was ~255–295 hours/month—much more than any other group. It seems that employees here are overworked.
3)Examine the average monthly hours versus the satisfaction levels.
```
# Create scatterplot of `average_monthly_hours` versus `satisfaction_level`, comparing employees who stayed versus those who left
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');
```
![Scatterplot](https://github.com/barrett203/CapstoneStudy_AdvancedGoogleAnalytics/blob/main/MonthlyHours_ByEvaluationScore.png "Scatterplot")
* The scatterplot above shows that there was a sizeable group of employees who worked ~240–315 hours per month. 315 hours per month is over 75 hours per week for a whole year. It's likely this is related to their satisfaction levels being close to zero.
4)Visualize satisfaction levels by time spent at the company.
```
# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Create boxplot showing distributions of `satisfaction_level` by tenure, comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='satisfaction_level', y='tenure', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Satisfaction by tenure', fontsize='14')

# Create histogram showing distribution of `tenure`, comparing employees who stayed versus those who left
tenure_stay = df1[df1['left']==0]['tenure']
tenure_left = df1[df1['left']==1]['tenure']
sns.histplot(data=df1, x='tenure', hue='left', multiple='dodge', shrink=5, ax=ax[1])
ax[1].set_title('Tenure histogram', fontsize='14')

plt.show();
```
![Tenure](https://github.com/barrett203/CapstoneStudy_AdvancedGoogleAnalytics/blob/main/Satisfaction_ByTenure.png "Tenure")
* Employees who left fall into two general categories: dissatisfied employees with shorter tenures and very satisfied employees with medium-length tenures.
* Four-year employees who left seem to have an unusually low satisfaction level. It's worth investigating changes to company policy that might have affected people specifically at the four-year mark, if possible.
5)Examine salary levels for different times spent at the company. 
```
# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Define short-tenured employees
tenure_short = df1[df1['tenure'] < 7]

# Define long-tenured employees
tenure_long = df1[df1['tenure'] > 6]

# Plot short-tenured histogram
sns.histplot(data=tenure_short, x='tenure', hue='salary', discrete=1, 
             hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.5, ax=ax[0])
ax[0].set_title('Salary histogram by tenure: short-tenured people', fontsize='14')

# Plot long-tenured histogram
sns.histplot(data=tenure_long, x='tenure', hue='salary', discrete=1, 
             hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.4, ax=ax[1])
ax[1].set_title('Salary histogram by tenure: long-tenured people', fontsize='14');
```
![Salary](https://github.com/barrett203/CapstoneStudy_AdvancedGoogleAnalytics/blob/main/SalaryHistogram_ByTenure.png "Salary")
* The plots above show that long-tenured employees were not disproportionately comprised of higher-paid employees.
6)Explore whether there's a correlation between working long hours and receiving high evaluation scores.
```
# Create scatterplot of `average_monthly_hours` versus `last_evaluation`
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='last_evaluation', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');
```
![Evaluation](https://github.com/barrett203/CapstoneStudy_AdvancedGoogleAnalytics/blob/main/MonthlyHoursBy_LastEvaluationScore.png "Evaluation")
* The scatterplot indicates two groups of employees who left: overworked employees who performed very well and employees who worked slightly under the nominal monthly average of 166.67 hours with lower evaluation scores.
* There seems to be a correlation between hours worked and evaluation score.
* There isn't a high percentage of employees in the upper left quadrant of this plot; but working long hours doesn't guarantee a good evaluation score.
* Most of the employees in this company work well over 167 hours per month.
7)Examine whether employees who worked very long hours were promoted in the last five years.
```
# Create plot to examine relationship between `average_monthly_hours` and `promotion_last_5years`
plt.figure(figsize=(16, 3))
sns.scatterplot(data=df1, x='average_monthly_hours', y='promotion_last_5years', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by promotion last 5 years', fontsize='14');
```
![Promotion](https://github.com/barrett203/CapstoneStudy_AdvancedGoogleAnalytics/blob/main/MonthlyHoursByPromotion_LastFiveYears.png "Promotion")
* Very few employees who were promoted in the last five years left
* Very few employees who worked the most hours were promoted
* All of the employees who left were working the longest hours
8)Inspect how the employees who left are distributed across departments.
```
# Create stacked histogram to compare department distribution of employees who left to that of employees who didn't
plt.figure(figsize=(11,8))
sns.histplot(data=df1, x='department', hue='left', discrete=1, 
             hue_order=[0, 1], multiple='dodge', shrink=.5)
plt.xticks(rotation='45')
plt.title('Counts of stayed/left by department', fontsize=14);
```
![Left](https://github.com/barrett203/CapstoneStudy_AdvancedGoogleAnalytics/blob/main/CountOfLeft_ByDepartment.png "Left")
* There doesn't seem to be any department that differs significantly in its proportion of employees who left to those who stayed.
9)Check for strong correlations between variables in the data.
```
# Plot a correlation heatmap
plt.figure(figsize=(16, 9))
heatmap = sns.heatmap(df0.corr(), vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12);
```
![Correlation](https://github.com/barrett203/CapstoneStudy_AdvancedGoogleAnalytics/blob/main/CorrelationHeatmap.png "Correlation")
* The correlation heatmap confirms that the number of projects, monthly hours, and evaluation scores all have some positive correlation with each other, and whether an employee leaves is negatively correlated with their satisfaction level.
# Construct

## Build a logistic regression machine learning model 

1)Before splitting the data, encode the non-numeric variables. There are two: department and salary.
```
# Copy the dataframe
df_enc = df1.copy()

# Encode the `salary` column as an ordinal numeric category
df_enc['salary'] = (
    df_enc['salary'].astype('category')
    .cat.set_categories(['low', 'medium', 'high'])
    .cat.codes
)

# Dummy encode the `department` column
df_enc = pd.get_dummies(df_enc, drop_first=False)

# Display the new dataframe
df_enc.head()
```
2)Create a heatmap to visualize how correlated variables are. Consider which variables you're interested in examining correlations between.
```
plt.figure(figsize=(8, 6))
sns.heatmap(df_enc[['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'tenure']]
            .corr(), annot=True, cmap="crest")
plt.title('Heatmap of the dataset')
plt.show()
```
![Heatmap](https://github.com/barrett203/CapstoneStudy_AdvancedGoogleAnalytics/blob/main/HeatMapOfDataset.png "Heatmap")

3)Create a stacked bar plot to visualize number of employees across department, comparing those who left with those who didn't.
```
# In the legend, 0 (purple color) represents employees who did not leave, 1 (red color) represents employees who left
pd.crosstab(df1['department'], df1['left']).plot(kind ='bar',color='mr')
plt.title('Counts of employees who left versus stayed across department')
plt.ylabel('Employee count')
plt.xlabel('Department')
plt.show()
```
![Barplot](https://github.com/barrett203/CapstoneStudy_AdvancedGoogleAnalytics/blob/main/StayedVsLeft_ByDepartment.png "Barplot")

4)Since logistic regression is quite sensitive to outliers, it would be a good idea at this stage to remove the outliers in the tenure column that were identified earlier.
```
# Select rows without outliers in `tenure` and save resulting dataframe in a new variable
df_logreg = df_enc[(df_enc['tenure'] >= lower_limit) & (df_enc['tenure'] <= upper_limit)]

# Display first few rows of new dataframe
df_logreg.head()
```
5)Isolate the outcome variable, which is the variable you want your model to predict.
```
y = df_logreg['left']

# Display first few rows of the outcome variable
y.head()
```
6)Select the features you want to use in your model. Consider which variables will help you predict the outcome variable
```
X = df_logreg.drop('left', axis=1)

# Display the first few rows of the selected features 
X.head()
```
7)Split the data into training set and testing set. Don't forget to stratify based on the values in y, since the classes are unbalanced.
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
```
8)Construct a logistic regression model and fit it to the training dataset.
```
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)
```
9)Test the logistic regression model: use the model to make predictions on the test set.
```
y_pred = log_clf.predict(X_test)
```
10)Create a confusion matrix to visualize the results of the logistic regression model.
```
# Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, 
                                  display_labels=log_clf.classes_)

# Plot confusion matrix
log_disp.plot(values_format='')

# Display plot
plt.show()
```
![ConfusionMatrix](https://github.com/barrett203/CapstoneStudy_AdvancedGoogleAnalytics/blob/main/ConfusionMatrix.png "ConfusionMatrix")
* The upper-left quadrant displays the number of true negatives. The upper-right quadrant displays the number of false positives. The bottom-left quadrant displays the number of false negatives. The bottom-right quadrant displays the number of true positives.
* True negatives: The number of people who did not leave that the model accurately predicted did not leave.
* False positives: The number of people who did not leave the model inaccurately predicted as leaving.
* False negatives: The number of people who left that the model inaccurately predicted did not leave
* True positives: The number of people who left the model accurately predicted as leaving
* A perfect model would yield all true negatives and true positives, and no false negatives or false positives.

11)Check the class balance in the data. In other words, check the value counts in the left column. Since this is a binary classification task, the class balance informs the way you interpret accuracy metrics.
```
df_logreg['left'].value_counts(normalize=True)
```
* There is an approximately 83%-17% split. So the data is not perfectly balanced, but it is not too imbalanced. If it was more severely imbalanced, you might want to resample the data to make it more balanced. In this case, you can use this data without modifying the class balance and continue evaluating the model.

12)Create classification report for logistic regression model
```
target_names = ['Predicted would not leave', 'Predicted would leave']
print(classification_report(y_test, y_pred, target_names=target_names))
```
* The classification report above shows that the logistic regression model achieved a precision of 79%, recall of 82%, f1-score of 80% (all weighted averages), and accuracy of 82%. However, if it's most important to predict employees who leave, then the scores are significantly lower.

## Act
The model and the feature importances extracted from the model confirms that employees at the company are overworked.

To retain employees, the following recommendations could be presented to the stakeholders:

* Cap the number of projects that employees can work on.
* Consider promoting employees who have been with the company for atleast four years, or conduct further investigation about why four-year tenured employees are so dissatisfied.
* Either reward employees for working longer hours, or don't require them to do so.
* If employees aren't familiar with the company's overtime pay policies, inform them about this. If the expectations around workload and time off aren't explicit, make them clear.
*  Hold company-wide and within-team discussions to understand and address the company work culture, across the board and in specific contexts.
* High evaluation scores should not be reserved for employees who work 200+ hours per month. Consider a proportionate scale for rewarding employees who contribute more/put in more effort.

















