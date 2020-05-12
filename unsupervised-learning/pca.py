from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

# データの読み込み
wine = load_wine()
X = wine['data']

model = PCA(n_components=4)
model.fit(X)

print('Before Transform:', X.shape[1])
print('After Transform:', model.transform(X).shape[1])
