import lib
from tensorflow.keras import (
    Sequential,
    layers,
    optimizers,
    losses,
    preprocessing,
    applications,
    activations,
    callbacks,
    metrics)
import pandas as pd
import cv2 as cv
import numpy as np
from pathlib import Path
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


save_dir = lib.get_save_dir("runs/train")

model = Sequential([
    applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        pooling=None),
    layers.Flatten(),
    layers.Dense(16),
    layers.Dense(3, activation=activations.relu)
])
# model = Sequential([
#     layers.Input(shape=(100, 100, 3)),
#     layers.Flatten(),
#     layers.Dense(1)
# ])
model.summary()
model.compile(
    optimizer=optimizers.Adam(1e-4),
    loss=losses.MSE,
    metrics=['acc', 'mse'])

data_root = Path("/dataset/data01")
label_path = data_root / "train_data01.csv"
images_dir = data_root / "train_data01"
df = pd.read_csv(str(label_path))
df.iloc[:, 0] = df.iloc[:, 0].astype(str) + ".jpg"
print(df.iloc[:, 0])
print(df.head(10))
datagen = preprocessing.image.ImageDataGenerator(validation_split=0.2)
train_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory=images_dir,
    x_col="id",
    y_col=['circle', 'square', 'triangle'],
    class_mode='raw',
    target_size=(224, 224),
    batch_size=4,
    subset='training',
    interpolation='bicubic',
    seed=0
)
val_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory=images_dir,
    x_col="id",
    y_col=['circle', 'square', 'triangle'],
    class_mode='raw',
    target_size=(224, 224),
    batch_size=1,
    subset='validation',
    interpolation='bicubic',
    seed=0
)
# for x, y in train_gen:
#     print(x.shape, y.shape)
#     out = model(x)
#     print(out.shape)
history = model.fit(train_gen,
                    epochs=50,
                    callbacks=[
                        callbacks.EarlyStopping(patience=5),
                        callbacks.ModelCheckpoint(
                            filepath=str(
                                save_dir / '{epoch:02d}_{val_acc:.4f}_{val_loss:.4f}.hdf5'),
                            monitor='val_loss',
                        ),
                        callbacks.TensorBoard(
                            log_dir=str(save_dir / 'logs')
                        ),
                        callbacks.History()
                    ],
                    validation_data=val_gen)
print(history)
