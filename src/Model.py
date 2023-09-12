import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, model, lr: float, model_checkpoint_loc: str, loss_function: str):
        self.best_model = model_checkpoint_loc+'/weights.{:02d}-{:.2f}-{:.2f}.hdf5'
        self.headless_model = None

        tf.random.set_seed(
            42
        )

        optim = tf.optimizers.Adam(learning_rate=lr)

        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_checkpoint_loc+'/weights.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.hdf5',
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )

        # Compile
        self.model = model.compile(optimizer=optim, loss=loss_function, metrics=['accuracy'])

        self.callbacks = [
            lr_reducer,
            checkpoint
        ]

    def train(self, images: np.array, labels: np.array, batch_size: int, epoch: int, validation_split: int = 0.1):
        # Training
        history = self.model.fit(
            images,
            labels,
            batch_size=batch_size,
            epoch=epoch,
            shuffle=True,
            validation_split=validation_split,
            callbacks=self.callbacks
        )

        idx = np.argmax(history.history['val_accuracy'])
        print("Epoch:", idx+1)
        for key in history.history.keys():
            print(f'{key}:{round(history.history[key][idx], 2)}')

        self.best_model = self.best_model.format(idx+1, history.history["val_loss"][idx], history.history["val_accuracy"][idx])
        self.model.load_weights(self.best_model)

    def remove_layer(self, layers_to_remove: int):
        self.headless_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-(layers_to_remove+1)].output
        )



