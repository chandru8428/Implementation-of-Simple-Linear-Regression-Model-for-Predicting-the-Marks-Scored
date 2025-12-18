# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph. 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: chandru.k
RegisterNumber:  212224220017

*/
```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('data.csv')
print(dataset.head())
dataset=pd.read_csv('data.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```


## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="792" height="722" alt="image" src="https://github.com/user-attachments/assets/1f271b98-ef9d-44d6-8a47-13c07809510a" />
<img width="1274" height="603" alt="image" src="https://github.com/user-attachments/assets/9bcc0e7f-b27f-4490-94c5-bb3b59c8077b" />
<img width="1081" height="664" alt="Screenshot 2025-09-19 104424" src="https://github.com/user-attachments/assets/28a2a29a-5eac-4806-acd1-3077b68ae863" />
<img width="979" height="717" alt="image" src="https://github.com/user-attachments/assets/9363ef8a-9aaf-4072-b18e-728974070629" />






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
