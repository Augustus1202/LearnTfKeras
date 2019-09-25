import numpy as np
import tensorflow as tf
from tensorflow import keras
import codecs

TRAIN_DATA = './ptb.train'
TEST_DATA = './ptb.test'
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

VOCAB = './ptb.vocab'          # Vocabulary file
# Project word to id
with codecs.open(VOCAB, 'r', 'utf-8') as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}
reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))


def load_data(data_file):
    with open(data_file, 'r') as fin:
        # read full file as a long string
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]  # Convert word id to integer
    return id_list


# Load data from file
data_train = np.array(load_data(TRAIN_DATA))
data_test = np.array(load_data(TEST_DATA))

len_train = len(data_train)
len_test = len(data_test)


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


model = keras.models.load_model('./models/model-50.hdf5')
dummy_iters = 40
example_training_generator = KerasBatchGenerator(data_train, TRAIN_NUM_STEP, 1, VOCAB_SIZE,
                                                 skip_step=1)
print("Training data:")
for i in range(dummy_iters):
    dummy = next(example_training_generator.generate())

num_predict = 100
true_print_out = "Actual words: "
pred_print_out = "Predicted words: "
for i in range(num_predict):
    data = next(example_training_generator.generate())
    prediction = model.predict(data[0])
    predict_word = np.argmax(prediction[:, TRAIN_NUM_STEP - 1, :])
    true_print_out += reversed_dictionary[data_train[TRAIN_NUM_STEP + dummy_iters + i]] + " "
    pred_print_out += reversed_dictionary[predict_word] + " "
print(true_print_out)
print(pred_print_out)

example_testing_generator = KerasBatchGenerator(data_test, TRAIN_NUM_STEP, 1, VOCAB_SIZE,
                                                skip_step=1)
print("Testing data:")
for i in range(dummy_iters):
    dummy = next(example_testing_generator.generate())

num_predict = 100
true_print_out = "Actual words: "
pred_print_out = "Predicted words: "
for i in range(num_predict):
    data = next(example_testing_generator.generate())
    prediction = model.predict(data[0])
    predict_word = np.argmax(prediction[:, TRAIN_NUM_STEP - 1, :])
    true_print_out += reversed_dictionary[data_test[TRAIN_NUM_STEP + dummy_iters + i]] + " "
    pred_print_out += reversed_dictionary[predict_word] + " "
print(true_print_out)
print(pred_print_out)
