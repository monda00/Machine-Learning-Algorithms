from sklearn.datasets import load_wine
from sklearn.cluster import KMeans

# データの読み込み
wine = load_wine()
X = wine['data']

model = KMeans(n_clusters=3, random_state=0)
model.fit(X)

print("labels: \n", model.labels_)
print("cluster centers: \n", model.cluster_centers_)
print("predict result: \n", model.predict(X))
