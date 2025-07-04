#IMPORT LIBRARY AND DEPENDENCIES
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

#LOAD THE DATA
Cancer = load_breast_cancer()
X = Cancer.data
Y = Cancer.target

#GET THE TARGET NAMES 
class_names = Cancer.target_names

#DATA SPLIT 
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42)

#SCALING
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale  = scaler.transform(X_test)

#MODEL
model = LogisticRegression()

#FIT
model.fit(X_train_scale , Y_train)

#ACCURACY 
accuracy = model.score(X_test_scale , Y_test)

#OUTPUT
print(f"ACCURACY {accuracy:.2f}")