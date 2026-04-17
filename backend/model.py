import tensorflow as tf
from tensorflow.keras import layers

def build_model(filters=32, lr=0.001):
    model = tf.keras.Sequential([
        layers.Conv2D(filters, (2,2), activation='relu', input_shape=(5,30,1)),
        layers.MaxPooling2D((1,2)),

        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
