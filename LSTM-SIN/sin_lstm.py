# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Avoid 'Blas GEMM launch failed'
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

HIDDEN_SIZE = 64
NUM_LAYERS = 2
TIMESTEPS = 10
BATCH_SIZE = 128
TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01
NUM_EPOC = 50

test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
train_data = np.sin(np.linspace(
    0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32))
train_data = train_data.reshape(-1, 1)
gen_train_data = keras.preprocessing.sequence.TimeseriesGenerator(
    train_data[:TRAINING_EXAMPLES], train_data[TIMESTEPS:], length=TIMESTEPS,
    batch_size=BATCH_SIZE, shuffle=True
)

model = keras.Sequential()
model.add(keras.layers.CuDNNLSTM(units=HIDDEN_SIZE, input_shape=(TIMESTEPS, 1), return_sequences=True))
model.add(keras.layers.CuDNNLSTM(units=HIDDEN_SIZE, return_sequences=False))
model.add(keras.layers.Dense(1, activation=None))
model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.1), loss='mse')

early_stopping = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                                               patience=3, verbose=0, mode='auto',
                                               baseline=None, restore_best_weights=False)

model.fit_generator(generator=gen_train_data, epochs=NUM_EPOC, verbose=2,
                    callbacks=[early_stopping])

val_data = np.sin(np.sin(np.linspace(
    test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
val_data = val_data.reshape(-1, 1)
predict_result = []
label_result = []

for i in range(TESTING_EXAMPLES):
    predict_input = val_data[i:i+TIMESTEPS]
    predict_input = predict_input.reshape(-1, TIMESTEPS, 1)
    predict_val = model.predict(predict_input)
    predict_result.append(predict_val)
    label_result.append(val_data[i+TIMESTEPS])

# Computing rmse
predict_result = np.array(predict_result).squeeze()
label_result = np.array(label_result).squeeze()
rmse = np.sqrt(((predict_result - label_result) ** 2).mean(axis=0))
print('Mean Square Error is: %f' % rmse)

plt.figure()
plt.plot(predict_result, label='predictions')
plt.plot(label_result, label='real_sin')
plt.legend()
plt.show()
