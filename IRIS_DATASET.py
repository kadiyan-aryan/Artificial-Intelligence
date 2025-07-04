# IMPORT LIBRARY AND DEPENDENCIES
from sklearn.datasets import load_iris  # load_*(): Dataset drawer
from sklearn.model_selection import train_test_split  # model_selection: Split data
from sklearn.preprocessing import StandardScaler  # Preprocessing: Scale features
from sklearn.linear_model import LogisticRegression  # Model: Multinomial regression
from sklearn.pipeline import Pipeline  # Pipeline: Combine preprocessing and model
from sklearn.metrics import accuracy_score  # Metrics: Compute accuracy

# LOAD THE DATA
iris = load_iris()  # Load Iris dataset (like getting ingredients)
X = iris.data  # Features: Sepal length, sepal width, petal length, petal width
y = iris.target  # Labels: 0=Setosa, 1=Versicolor, 2=Virginica

# DATA SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(solver='saga', penalty="none", random_state=42))])

pipeline.fit(X_train, y_train)  # Fit: Train the model with scaled data

# ACCURACY
accuracy = pipeline.score(X_test, y_test)  # Score: Compute accuracy on test data
print(f"ACCURACY: {accuracy:.2f}")  # Print accuracy


# FOR NEW DATA
New_Flower = [[6.0, 3.0, 4.0, 1.5], [5.0, 3.4, 1.5, 0.2], [7.0, 3.2, 4.7, 1.4]]


# APPLY PREDICTION AND PROBABILITY
prediction = pipeline.predict(New_Flower)  # Predict: Guess class based on weights       
probabilities = pipeline.predict_proba(New_Flower)  # predict_proba: Get class probabilities from softmax
# OUTPUT
for i , p in enumerate(prediction):
    print(f"FLOWER {i+1}: {iris.target_names[p]}")  # Print predicted class

for i, p in enumerate(probabilities):
    print(f"PROBABILITIES: Flower {i+1} : Setosa={p[0]:.2f}, Versicolor={p[1]:.2f}, Virginica={p[2]:.2f}")  # Print probabilities (low loss = high probability for correct class

