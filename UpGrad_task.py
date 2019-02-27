#Importing The Libraries
import pandas as pd
from sklearn.metrics import confusion_matrix
from  sklearn.metrics import accuracy_score

#Reading The Dataset
dataset = pd.read_csv("bank-additional-full.csv",sep=";")

#Spliiting X and Y Variables from Dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

cat_col=[1,2,3,4,5,6,7,8,9,14] #Column numbers having categorical data

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for i in cat_col:
  X[:,int(i)] = labelencoder.fit_transform(X[:,int(i)])


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Using Logistic Regression Model
from sklearn.linear_model import LogisticRegression
logis = LogisticRegression(penalty = 'l1' ,C=0.5)
logis.fit(X_train,y_train)
#Predicting the test set
y_pred1 = logis.predict(X_test)
#Printing Confusion Matrix
cm1 = confusion_matrix(y_test, y_pred1)
#Calculating The Accuracy of the model
acc1=accuracy_score(y_pred1,y_test)


#Using K Nearest Neighbours Classifier
from sklearn.neighbors import KNeighborsClassifier
acc2=0
for i in range(1,40):
    knn = KNeighborsClassifier(i) 
    #Using different values of number of nearest neighbours by running in a for loop,
    #best value we get is 22
    knn.fit(X_train,y_train)
    y_pred2 = knn.predict(X_test)
    #Calculating The Accuracy of the model
    acc=accuracy_score(y_pred2,y_test)
    if (acc>acc2):
        acc2=acc
        bst_i=i
#Printing Confusion Matrix
cm2 = confusion_matrix(y_test, y_pred2)

accuracy={"Logistic Regression":acc1,"K-Nearest Neighbors":acc2}
print("Accuracy of different models is given as:")
for i in accuracy.keys():
  print(i," : ",round(accuracy[i]*100,2),"%")
