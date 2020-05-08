from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers

# データの読み込み
boston = load_boston()
X = boston['data']
y = boston['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

model.fit(X_train, y_train)

mse, mae = model.evaluate(X_test, y_test)

print('MSE is', mse)
print('MAE is', mae)
