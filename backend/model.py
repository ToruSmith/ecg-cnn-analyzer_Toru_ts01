def build_model(input_shape, filters=32, lr=0.001):
    import tensorflow as tf
    from tensorflow.keras import layers

    model = tf.keras.Sequential([
        layers.Conv1D(filters, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),

        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
