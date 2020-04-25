from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np

# データの読み込み
wine = load_boston()
X = wine['data']
y = wine['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test)

params = {
    'objective': 'regression',
    'metric': 'mse',
}
num_round = 100

model = lgb.train(
    params,
    lgb_train,
    valid_sets=lgb_eval,
    num_boost_round=num_round,
)

y_pred = model.predict(X_test)

score = mean_squared_error(y_test, y_pred)

print('score is', score)
