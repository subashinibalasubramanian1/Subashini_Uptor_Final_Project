import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report


import seaborn as sns
#Reading Dataset using pandas

df = pd.read_csv("framingham.csv")
print(df.head())

#drop the column education
df.drop(['education'], inplace = True, axis = 1)
df.rename(columns ={'male':'Sex_male'}, inplace = True)

df.dropna(axis=0,inplace = True)
print(df.head())
print(df.info())

print(df['TenYearCHD'].value_counts())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#splitting the  dataset  into train and test sets

X = df[["age","Sex_male","cigsPerDay","totChol","sysBP","glucose"]]
y = df["TenYearCHD"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=4)

print("Train set:",X_train.shape,y_train.shape)
print("Test set,",X_test.shape,y_test.shape)

#TenYearCHD records of all the patients using seaborn
plt.figure(figsize=(7,5))
sns.countplot(x='TenYearCHD',data = df,palette="BuGn_r")
plt.show()

#counting no of patients affected by CHD

laste = df['TenYearCHD'].plot()
plt.show()

#Fitting Logistic regression model for Heart Disease Prediction

model = LogisticRegression()
model.fit(X_train,y_train)
y_prediction = model.predict(X_test)
print(y_prediction)
#Evaluatic Logistic regression model
#Accuracy Score
Accuracy_Score = accuracy_score(y_test,y_prediction)
print("Accuracy of the model is ",Accuracy_Score)

#Confusion Matrix
Cl_report = classification_report(y_test,y_prediction)
print("Classification Report is = \n",Cl_report)

cm = confusion_matrix(y_test,y_prediction)

print("Conf matrix\n",cm)

conf_matrix= pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix,annot = True, fmt = 'd',cmap = 'Greens')
plt.show()

"""
#Unsupervised machine Learning KMeans algorithm

from sklearn.cluster import KMeans
print(X)
kmeans_obj = KMeans(n_clusters=2, random_state=42)
fitted_model = kmeans_obj.fit_predict(X)

plt.scatter(X.iloc[:,0],X.iloc[:,1],c = fitted_model,marker="*",cmap="rainbow",label="kmeans")

plt.show()

"""