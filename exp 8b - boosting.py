import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
iris = load_iris()
features = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20)
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
pred_final = model.predict(X_test)
print(accuracy_score(y_test, pred_final))
