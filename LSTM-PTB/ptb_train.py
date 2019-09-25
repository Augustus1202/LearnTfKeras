import numpy as np
import tensorflow as tf
from tensorflow import keras

TRAIN_DATA = './ptb.train'
EVAL_DATA = './ptb.valid'
VOCAB = './ptb.vocab'          # Vocabulary file
HIDDEN_SIZE = 500
NUM_LAYERS = 2
VOCAB_SIZE = 10000
TRAIN_BATCH_SIZE = 128
TRAIN_NUM_STEP = 30

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 50
LSTM_KEEP_PROB = 0.5
EMBEDDING_KEEP_PROB = 0.5
MAX_GRAD_NORM = 5
SHARE_EMB_AND_SOFTMAX = True

# Avoid 'Blas GEMM launch failed'
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))


def load_data(data_file):
    with open(data_file, 'r') as fin:
        # read full file as a long string
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]  # Convert word id to integer
    return id_list


# Load data from file
data_train = np.array(load_data(TRAIN_DATA))
data_val = np.array(load_data(EVAL_DATA))

len_train = len(data_train)
len_val = len(data_val)


class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx +
                                   1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :] = temp_y
                self.current_idx += self.skip_step
            # x = x.reshape(self.batch_size, self.num_steps, 1)
            py = y.reshape(self.batch_size, self.num_steps, 1)
            yield x, py


gen_train_data = KerasBatchGenerator(
    data_train, TRAIN_NUM_STEP, TRAIN_BATCH_SIZE, VOCAB_SIZE,
    skip_step=TRAIN_NUM_STEP
)

gen_val_data = KerasBatchGenerator(
    data_val, TRAIN_NUM_STEP, TRAIN_BATCH_SIZE, VOCAB_SIZE,
    skip_step=TRAIN_NUM_STEP
)

# items = next(gen_train_data.generate())
# print(items[0].shape)
# print(items[1].shape)
# print(items)
# exit()

# gen_train_data = keras.preprocessing.sequence.TimeseriesGenerator(
#     data_train[:len_train-1], data_train[1:], length=TRAIN_NUM_STEP,
#     batch_size=TRAIN_BATCH_SIZE
# )
#
# gen_val_data = keras.preprocessing.sequence.TimeseriesGenerator(
#     data_val[:len_val-1], data_val[1:], length=TRAIN_NUM_STEP,
#     batch_size=TRAIN_BATCH_SIZE
# )

model = keras.Sequential()
model.add(keras.layers.Embedding(VOCAB_SIZE, HIDDEN_SIZE, input_length=TRAIN_NUM_STEP))
# model.add(keras.layers.Dropout(1 - EMBEDDING_KEEP_PROB))
for _ in range(NUM_LAYERS):
    model.add(keras.layers.CuDNNLSTM(units=HIDDEN_SIZE, return_sequences=True))
model.add(keras.layers.Dropout(1 - LSTM_KEEP_PROB))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(VOCAB_SIZE)))
model.add(keras.layers.Activation('softmax'))
model.summary()
keras.utils.plot_model(model, 'model.png')

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['sparse_categorical_accuracy'])
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath='./models/model-{epoch:02d}.hdf5', verbose=1)
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1, batch_size=TRAIN_BATCH_SIZE,
    write_graph=True, write_grads=False, write_images=True,
    embeddings_freq=0, embeddings_layer_names=None,
    embeddings_metadata=None, embeddings_data=None, update_freq=500
    )

model.fit_generator(generator=gen_train_data.generate(),
                    steps_per_epoch=len_train // (TRAIN_BATCH_SIZE * TRAIN_NUM_STEP),
                    epochs=NUM_EPOCH, callbacks=[cp_callback, tb_callback],
                    validation_data=gen_val_data.generate(),
                    validation_steps=len_val // (TRAIN_BATCH_SIZE * TRAIN_NUM_STEP))

