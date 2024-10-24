import pandas as pd         
import numpy as np           
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier   
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

milk=pd.read_csv("C:/Users/akshi/Downloads/milknew.csv")
print(milk.head())
print(milk.info())
print(milk.isnull().sum())

#scaling the numerical values
scaler=StandardScaler()
milk['Temprature']=scaler.fit_transform(milk[['Temprature']])
milk['pH']=scaler.fit_transform(milk[['pH']])

#target variable
X=milk.drop('Grade',axis=1)
y=milk['Grade']

#seperating the train and test variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.31, random_state=42)

#trining the model
model = RandomForestClassifier(n_estimators=150,random_state=42,max_depth=20,bootstrap=True)
model.fit(X_train, y_train)

#predictions
y_pred=model.predict(X_test)

#calculating accuracy
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy*100)

print()
print("TESTING FOR OVERFITTING")
print()
train_accuracy=model.score(X_train,y_train)
print("train accuracy: ",train_accuracy)

test_accuracy=model.score(X_test,y_test)
print("test accuracy: ",test_accuracy)