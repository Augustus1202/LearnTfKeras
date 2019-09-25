# Create vocabulary file from training data

import codecs
import collections

RAW_DATA = '../../../Projects/Data/PTB-simple/data/ptb.train.txt'  # training data
VACAB_OUTPUT = 'ptb.vocab'  # vocabulary file

counter = collections.Counter()     # Word frequency statistics
with codecs.open(RAW_DATA, 'r', 'utf-8') as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

# Sort the words in frequency order
sorted_word_to_cnt = sorted(counter.items(), key=lambda item: item[1], reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]

# Add '<eos>' (End of the string)
sorted_words = ['<eos>'] + sorted_words

# sorted_words = ['<unk>', '<sos>', '<eos>'] + sorted_words
# if len(sorted_words) > 10000:
#     sorted_words = sorted_words[:10000]

with codecs.open(VACAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + '\n')

