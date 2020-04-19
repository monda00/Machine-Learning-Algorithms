from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras.utils import np_utils

# データの読み込み
boston = load_wine()
X = boston['data']
y = np_utils.to_categorical(boston['target'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train)

crossentropy, acc = model.evaluate(X_test, y_test)

print('Categorical Crossentropy is', crossentropy)
print('Accuracy is', acc)
