# CapstoneStudy_AdvancedGoogleAnalytics
Advanced google analytics capstone study. Following example project provided by the course to expand my python coding skills. 
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
Excel will be used to load the data and initially look for any issues. Python will then be used to transform and explore the data. 

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
There are no missing values in the dataset but there are 3,008 rows contain duplicates. That is apprximately 20% of the data.

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




