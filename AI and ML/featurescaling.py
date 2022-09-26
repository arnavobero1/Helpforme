import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Data.csv')
df.head()

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan,strategy = 'mean')
imputer.fit(x.iloc[:,1:])
x.iloc[:,1:] = imputer.transform(x.iloc[:,1:])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x.iloc[:, 0] = labelencoder_X.fit_transform(x.iloc[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.transform(X_test[:,3:])


