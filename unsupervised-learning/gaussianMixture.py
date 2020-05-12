from sklearn.datasets import load_wine
from sklearn.mixture import GaussianMixture

# データの読み込み
wine = load_wine()
X = wine['data']

model = GaussianMixture(n_components=4)
model.fit(X)

print("means: \n", model.means_)
print("predict result: \n", model.predict(X))
