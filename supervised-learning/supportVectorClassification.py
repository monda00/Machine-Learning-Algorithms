from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# データの読み込み
wine = load_wine()
X = wine['data']
y = wine['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = SVC(gamma='auto')
model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print('score is', score)
