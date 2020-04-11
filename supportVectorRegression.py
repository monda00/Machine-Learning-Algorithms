from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# データの読み込み
boston = load_boston()
X = boston['data']
y = boston['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = SVR(kernel='linear', gamma='auto')
model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print('score is', score)
