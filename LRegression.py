import numpy as np
import pandas as pd
dataset = pd.read_csv("C:/Users/Tathagata Sengupta/OneDrive/Desktop/ATS Project/Breast Cancer/Data.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
#pred1=classifier.predict(sc.transform([[1,2,3,4,5,7,5,2,1]]))
def pred(a,b,c,d,e,f,g,h,i):
    pred1=int(classifier.predict(sc.transform([[a,b,c,d,e,f,g,h,i]])))
    return pred1
