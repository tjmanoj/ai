import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# importing machine learning models for prediction
import xgboost as xgb
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import load_iris
iris = load_iris()
target = iris.target
train = pd.DataFrame(iris.data, columns=iris.feature_names)
X_train, X_test, y_train, y_test = train_test_split(
train, target, test_size=0.20)
model = BaggingRegressor(estimator=xgb.XGBRegressor())
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(mean_squared_error(y_test, pred))
