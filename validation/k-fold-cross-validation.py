from sklearn.datasets import load_wine
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

import numpy as np


def build_model(n_features):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(n_features,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# データの読み込み
boston = load_wine()
X, X_test, y, y_test = train_test_split(
    boston['data'], utils.to_categorical(boston['target']), test_size=0.3, random_state=0)


# k-fold cross validation
FOLD = 5
EPOCH = 10
BATCH_SIZE = 32

oof = np.zeros(y.shape)
scores = []
y_preds = []
histories = []
kf = KFold(n_splits=FOLD, shuffle=True, random_state=42)

for fold, (train_indices, valid_indices) in enumerate(kf.split(X)):
    print(f'fold {fold}')

    X_train, X_valid = X[train_indices], X[valid_indices]
    y_train, y_valid = y[train_indices], y[valid_indices]

    model = build_model(X_train[1])
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

    oof[valid_indices] += model.predict(X_valid,
                                        num_iteration=model.best_iteration)
    score = accuracy_score(y_valid, oof[valid_indices])
    print(f'fold {fold} ACCURACY: {score}')
    scores.append(score)

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_preds.append(y_pred)

cv_score = sum(scores) / FOLD
print(f'CV score: {cv_score}')
