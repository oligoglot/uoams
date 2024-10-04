import tensorflow as tf

def create_auto_mpg_deep_and_wide_networks(n_inputs: int, n_outputs: int):
    # Deep network
    deep_model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(n_inputs,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(n_outputs)
    ])

    # Wide network
    wide_model = tf.keras.Sequential([
        tf.keras.layers.Dense(56, activation='relu', input_shape=(n_inputs,)),
        tf.keras.layers.Dense(n_outputs)
    ])

    return deep_model, wide_model

def create_activity_dropout_and_nodropout_networks(n_inputs: int, n_outputs: int):
    def create_base_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(n_inputs,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(n_outputs, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # Model with dropout
    dropout_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(n_inputs,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_outputs, activation='softmax')
    ])
    dropout_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Model without dropout
    no_dropout_model = create_base_model()

    return dropout_model, no_dropout_model

def create_income_earlystopping_and_noearlystopping_networks(n_inputs: int, n_outputs: int):
    def create_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(n_inputs,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(n_outputs, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    early_model = create_model()
    late_model = create_model()

    early_fit_kwargs = {
        'callbacks': [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        'validation_split': 0.2
    }
    late_fit_kwargs = {}

    return early_model, early_fit_kwargs, late_model, late_fit_kwargs
