from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

num_tags = 12
num_words = 10000
num_departments = 4

# ダミーデータの作成
title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tags_data = np.random.randint(2, size=(1280, num_tags)).astype("float32")
priority_targets = np.random.random(size=(1280, 1))
dept_targets = np.random.randint(2, size=(1280, num_departments))

# タイトルの層
title_input = keras.Input(
    shape=(None,), name="title"
)
title_features = layers.Embedding(num_words, 64)(title_input)
title_features = layers.LSTM(128)(title_features)

# 本文の層
body_input = keras.Input(shape=(None,), name="body")
body_features = layers.Embedding(num_words, 64)(body_input)
body_features = layers.LSTM(32)(body_features)

# タグの層
tags_input = keras.Input(
    shape=(num_tags,), name="tags"
)
tags_features = layers.Dense(36, activation='relu')(tags_input)

# 層を結合
x = layers.concatenate([title_features, body_features, tags_features])

# 出力層
priority_output = layers.Dense(1, name="priority")(x)
department_output = layers.Dense(num_departments, name="department")(x)

model = keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs=[priority_output, department_output],
)

keras.utils.plot_model(model, 'multi_inputoutput_model.png', show_shapes=True)

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={
        "priority": keras.losses.BinaryCrossentropy(from_logits=True),
        "department": keras.losses.CategoricalCrossentropy(from_logits=True),
    },
    loss_weights=[1.0, 0.5],
)

model.fit(
    {"title": title_data, "body": body_data, "tags": tags_data},
    {"priority": priority_targets, "department": dept_targets},
    epochs=2,
    batch_size=32,
)
