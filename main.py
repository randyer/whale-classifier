# TensorFlow and tf.keras
import tensorflow as tf
import keras_tuner as kt

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


def model_builder(hp):
    hp_units = hp.Int('units', min_value=32, max_value=1028, step=32)
    deep_model_hype = tf.keras.Sequential([
        tf.keras.layers.Dense(units=hp_units, activation='relu'),
        tf.keras.layers.Dense(units=hp_units, activation='relu'),
        tf.keras.layers.Dense(units=hp_units, activation='relu'),
        tf.keras.layers.Dense(30)
    ])

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    deep_model_hype.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

    return deep_model_hype


if __name__ == '__main__':
    traindataX = np.load("traindataX.npy")
    traindataY = np.load("traindataY.npy")
    testdataX = np.load("testdataX.npy")
    testdataY = np.load("testdataY.npy")

    traindataX = traindataX / 255.0
    testdataX = testdataX / 255.0

    train_dataset = tf.data.Dataset.from_tensor_slices((testdataX, testdataY))

    species = ['melon_headed_whale',
               'humpback_whale',
               'false_killer_whale',
               'bottlenose_dolphin',
               'beluga',
               'minke_whale',
               'fin_whale',
               'blue_whale',
               'gray_whale',
               'southern_right_whale',
               'common_dolphin',
               'kiler_whale',
               'pilot_whale',
               'dusky_dolphin',
               'killer_whale',
               'long_finned_pilot_whale',
               'sei_whale',
               'spinner_dolphin',
               'bottlenose_dolpin',
               'cuviers_beaked_whale',
               'spotted_dolphin',
               'globis',
               'brydes_whale',
               'commersons_dolphin',
               'white_sided_dolphin',
               'short_finned_pilot_whale',
               'rough_toothed_dolphin',
               'pantropic_spotted_dolphin',
               'pygmy_killer_whale',
               'frasiers_dolphin']

    three_layer_model1 = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(30)
    ])

    three_layer_model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(30)
    ])

    deep_model1 = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='linear'),
        tf.keras.layers.Dense(30)
    ])

    deep_model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(1028, activation='relu'),
        tf.keras.layers.Dense(512, activation='sigmoid'),
        tf.keras.layers.Dense(128, activation='linear'),
        tf.keras.layers.Dense(30)
    ])

    deep_model3 = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1000, activation='sigmoid'),
        tf.keras.layers.Dense(30)
    ])

    deep_model4 = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='swish'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1000, activation='swish'),
        tf.keras.layers.Dense(30)
    ])

    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='dolphin')
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(traindataX, traindataY, epochs=10, validation_split=0.2, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(traindataX, traindataY, epochs=10, validation_split=0.2)

    _, acce = model.evaluate(testdataX, testdataY)
    print(f"Final accuracy on testing data with optimized hyperparameter: {acce}")

    # val_acc_per_epoch = history.history['val_accuracy']
    # best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    # print('Best epoch: %d' % (best_epoch,))

    # three_layer_model1.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    #
    # three_layer_model1.fit(testdataX, testdataY, epochs=10)
    #
    # print("Three layer model 2")
    #
    # three_layer_model2.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    #
    # three_layer_model2.fit(testdataX, testdataY, epochs=10)

    # print("Deep model 1")
    #
    # deep_model1.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    #
    # deep_model1.fit(testdataX, testdataY, epochs=10)
    #
    # print("Deep model 2")
    #
    # deep_model2.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    #
    # deep_model2.fit(testdataX, testdataY, epochs=10)
    #
    # print("Deep model 3")
    #
    # deep_model3.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    #
    # deep_model3.fit(testdataX, testdataY, epochs=10)

# print("Deep model 4")
#
# deep_model4.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# deep_model4.fit(traindataX, traindataY, epochs=10)

test_loss, test_acc = three_layer_model1.evaluate(testdataX,  testdataY, verbose=2)

print('\nTest accuracy:', test_acc)