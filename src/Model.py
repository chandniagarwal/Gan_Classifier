import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self, model, lr: float, model_checkpoint_loc: str, loss_function: str):
        self.best_model = model_checkpoint_loc+'/weights.{:02d}-{:.2f}-{:.2f}.hdf5'
        self.headless_model = None
        self.model = model

        tf.random.set_seed(
            42
        )

        optim = tf.optimizers.Adam(learning_rate=lr)

        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=0.00001
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_checkpoint_loc+'/weights.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.hdf5',
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )

        # Compile
        self.model.compile(optimizer=optim, loss=loss_function, metrics=['accuracy'])

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
            epochs=epoch,
            shuffle=True,
            validation_split=validation_split,
            callbacks=self.callbacks
        )
        self.history = history

        idx = np.argmin(history.history['val_loss'])

        # Get training and test loss histories
        training_loss = history.history['loss']
        test_loss = history.history['val_loss']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)

        # Visualize loss history
        plt.plot(epoch_count, training_loss, 'r--')
        plt.plot(epoch_count, test_loss, 'b-')
        plt.legend(['Training Loss', 'Test Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.show()

        # Get training and test accuracy histories
        training_accuracy = history.history['accuracy']
        test_accuracy = history.history['val_accuracy']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_accuracy) + 1)

        # Visualize loss history
        plt.plot(epoch_count, training_accuracy, 'r--')
        plt.plot(epoch_count, test_accuracy, 'b-')
        plt.legend(['Training Accuracy', 'Accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.show()

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



