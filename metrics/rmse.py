from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# データの読み込み
boston = load_boston()
X = boston['data']
y = boston['target']
X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.3, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('RMSE is', np.sqrt(mean_squared_error(y_true, y_pred)))
