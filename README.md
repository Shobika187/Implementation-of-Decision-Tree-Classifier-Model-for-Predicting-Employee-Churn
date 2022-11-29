# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required packages.
2.Read the data set.
3.Apply label encoder to the non-numerical column inoreder to convert into numerical values.
4.Determine training and test data set.
5.Apply decision tree Classifier and get the values of accuracy and data prediction.
```
## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Shobbika P
RegisterNumber:  212221230096

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])


```

## Output:
![image](https://user-images.githubusercontent.com/94508142/204559766-dd3e4d9c-a654-4c55-aff5-c02082b3ce2a.png)

![image](https://user-images.githubusercontent.com/94508142/204559919-2a32e3d7-131e-4b96-8569-5fbea865b643.png)

![image](https://user-images.githubusercontent.com/94508142/204560025-54e4ae01-7552-4423-944d-3c26134347f3.png)

![image](https://user-images.githubusercontent.com/94508142/204560174-82f21e26-e8f1-4f4c-85dc-089b16036da9.png)

![image](https://user-images.githubusercontent.com/94508142/204560409-3d7e6be9-7890-414b-b635-7c19e0ea6150.png)

![image](https://user-images.githubusercontent.com/94508142/204560558-5118188a-dbc4-427f-88bd-de10140b96d3.png)

![image](https://user-images.githubusercontent.com/94508142/204560671-54c3e8b2-5565-4d79-9a77-2010c0ee00d6.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
