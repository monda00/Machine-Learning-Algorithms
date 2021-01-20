from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import numpy as np
import matplotlib.pyplot as plt


def build_model(n_features):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(n_features,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# データの読み込み
boston = load_boston()
X, X_test, y, y_test = train_test_split(
    boston['data'], boston['target'], test_size=0.3, random_state=0)

# k-fold cross validation
FOLD = 5
EPOCH = 10
BATCH_SIZE = 32

valid_scores = []
histories = []
models = []
kf = KFold(n_splits=FOLD, shuffle=True, random_state=42)

for fold, (train_indices, valid_indices) in enumerate(kf.split(X)):
    X_train, X_valid = X[train_indices], X[valid_indices]
    y_train, y_valid = y[train_indices], y[valid_indices]

    model = build_model(X_train.shape[1])
    rlr = ReduceLROnPlateau(monitor='val_loss',
                            factor=0.1,
                            patience=3,
                            verbose=0,
                            min_delta=1e-4,
                            mode='max')
    ckp = ModelCheckpoint(f'model_{fold}.hdf5',
                          monitor='val_loss',
                          verbose=0,
                          save_best_only=True,
                          save_weights_only=True,
                          mode='max')
    es = EarlyStopping(monitor='val_loss',
                       min_delta=1e-4,
                       patience=7,
                       mode='max',
                       baseline=None,
                       restore_best_weights=True,
                       verbose=0)

    hostory = model.fit(X_train, y_train,
                        validation_data=(X_valid, y_valid),
                        epochs=EPOCH,
                        batch_size=BATCH_SIZE,
                        callbacks=[rlr, ckp, es],
                        verbose=0)
    histories.append(hostory)

    y_valid_pred = model.predict(X_valid)
    score = mean_absolute_error(y_valid, y_valid_pred)
    print(f'fold {fold} MAE: {score}')
    valid_scores.append(score)

    models.append(model)


cv_score = np.mean(valid_scores)
print(f'CV score: {cv_score}')

plt.plot(histories[0].history['loss'])
plt.plot(histories[0].history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
