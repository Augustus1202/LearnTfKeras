# Convert the words to numbers according to the vocabulary

import codecs
import sys

RAW_DATA_PATH = '../../../Projects/Data/PTB-simple/data/'  # data path
RAW_DATA_FILE_NAME = ['ptb.train.txt', 'ptb.test.txt', 'ptb.valid.txt']  # files to convert
OUTPUT_FILE_NAME = ['ptb.train', 'ptb.test', 'ptb.valid']  # files to convert
VOCAB = './ptb.vocab'          # Vocabulary file

# Project word to id
with codecs.open(VOCAB, 'r', 'utf-8') as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}


# Convert missing word to '<unk>'
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id['<unk>']


# Convert words in file to numbers
def word_to_number(input_file, output_file):
    fin = codecs.open(input_file, 'r', 'utf-8')
    fout = codecs.open(output_file, 'w', 'utf-8')
    for line in fin:
        words = line.strip().split() + ['<eos>']  # Read the words and add <eos> to the end of lines
        # covert word to number
        out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
        fout.write(out_line)
    fin.close()
    fout.close()


for i in range(len(RAW_DATA_FILE_NAME)):
    input_name = RAW_DATA_PATH + RAW_DATA_FILE_NAME[i]
    output_name = OUTPUT_FILE_NAME[i]
    word_to_number(input_name, output_name)

