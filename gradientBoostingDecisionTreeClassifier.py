from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import numpy as np

# データの読み込み
wine = load_wine()
X = wine['data']
y = wine['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test)

params = {
    'objective': 'multiclass',
    'num_class': 3,
}
num_round = 100

model = lgb.train(
    params,
    lgb_train,
    valid_sets=lgb_eval,
    num_boost_round=num_round,
)

pred = model.predict(X_test)
y_pred = []
for p in pred:
    y_pred.append(np.argmax(p))

score = accuracy_score(y_test, y_pred)

print('score is', score)
