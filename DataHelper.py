import torch
import numpy as np
from torch.utils import data
from torch.autograd import Variable
from itertools import combinations


class Vocabulary(object):

    def __init__(self):
        self.char2idx = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
        self.idx2char = {0: 'SOS', 1: 'EOS', 2: 'PAD', 3: 'UNK'}
        self.num_chars = 4
        self.max_length = 0
        self.word_list = [[], [], [], []]
        self.num_words = 0

    def build_vocab(self, data_path):
        """Construct the relation between words and indices"""
        with open(data_path, 'r', encoding='utf-8') as dataset:
            for word in dataset:
                word = word.strip('\n')
                words = word.split()
                self.num_words += 1

                for idx, word in enumerate(words):
                    self.word_list[idx].append(word)
                    if self.max_length < len(word):
                        self.max_length = len(word)

                    chars = self.split_sequence(word)
                    for char in chars:
                        if char not in self.char2idx:
                            self.char2idx[char] = self.num_chars
                            self.idx2char[self.num_chars] = char
                            self.num_chars += 1

    def sequence_to_indices(self, sequence, add_eos=False, add_sos=False):
        """Transform a char sequence to index sequence
            :param sequence: a string composed with chars
            :param add_eos: if true, add the <EOS> tag at the end of given sentence
            :param add_sos: if true, add the <SOS> tag at the beginning of given sentence
        """
        index_sequence = [self.char2idx['SOS']] if add_sos else []

        for char in self.split_sequence(sequence):
            if char not in self.char2idx:
                index_sequence.append((self.char2idx['UNK']))
            else:
                index_sequence.append(self.char2idx[char])

        if add_eos:
            index_sequence.append(self.char2idx['EOS'])

        return index_sequence

    def indices_to_sequence(self, indices):
        """Transform a list of indices
            :param indices: a list
        """
        sequence = ""
        for idx in indices:
            char = self.idx2char[idx]
            if char == "EOS":
                break
            else:
                sequence += char
        return sequence

    def split_sequence(self, sequence):
        """Vary from languages and tasks. In our task, we simply return chars in given sentence
        For example:
            Input : alphabet
            Return: [a, l, p, h, a, b, e, t]
        """
        return [char for char in sequence]
    


    def __str__(self):
        str = "Vocab information:\n"
        for idx, char in self.idx2char.items():
            str += "Char: %s Index: %d\n" % (char, idx)
        return str


class DataTransformer(object):

    def __init__(self, path):
        self.indices_sequences = []

        # Load and build the vocab
        self.vocab = Vocabulary()
        self.vocab.build_vocab(path)
        self.PAD_ID = self.vocab.char2idx["PAD"]
        self.SOS_ID = self.vocab.char2idx["SOS"]
        self.vocab_size = self.vocab.num_chars
        self.max_length = self.vocab.max_length

        self._build_training_set(path)

    def _build_training_set(self, path):
        # Change sentences to indices, and append <EOS> at the end of all pairs
        for word_idx in range(self.vocab.num_words):
            same_words = [sub_list[word_idx] for sub_list in self.vocab.word_list]

            indices_seq = [self.pad_sequence(self.vocab.sequence_to_indices(word, add_eos=True)) for word in same_words]
            # print(indices_seq)
            for idx1, word1 in enumerate(indices_seq):
                for idx2, word2 in enumerate(indices_seq, start= idx1):
                    self.indices_sequences.append([[idx1]+word1, [idx2-idx1]+ word2])

    def pad_sequence(self, sequence):
        sequence += [self.PAD_ID for i in range(self.max_length - len(sequence)+1)]
        return sequence

    def mini_batches(self, batch_size):
        input_batches = []
        target_batches = []

        np.random.shuffle(self.indices_sequences)
        mini_batches = [
            self.indices_sequences[k: k + batch_size]
            for k in range(0, len(self.indices_sequences), batch_size)
        ]

        for batch in mini_batches:
            seq_pairs = sorted(batch, key=lambda seqs: len(seqs[0]), reverse=True)  # sorted by input_lengths
            input_seqs = [pair[0] for pair in seq_pairs]
            target_seqs = [pair[1] for pair in seq_pairs]

            input_lengths = [len(s) for s in input_seqs]
            in_max = input_lengths[0]
            input_padded = [self.pad_sequence(s, in_max) for s in input_seqs]

            target_lengths = [len(s) for s in target_seqs]
            out_max = target_lengths[0]
            target_padded = [self.pad_sequence(s, out_max) for s in target_seqs]

            input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)  # time * batch
            target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)  # time * batch

            if self.use_cuda:
                input_var = input_var.cuda()
                target_var = target_var.cuda()

            yield (input_var, input_lengths), (target_var, target_lengths)

class EncoDataSet(data.Dataset):
    def __init__(self, DataTransformer, mode):
        super(data.Dataset, self).__init__()
        self.vocab = DataTransformer.vocab
        self.data = DataTransformer.indices_sequences
        self.mode = mode
        self.spltpoint = len(self.data)*7//10
        if mode == 'train':
            self.data = self.data[:self.spltpoint]
        if mode == 'test':
            self.data = self.data[self.spltpoint:]
            


    def __len__(self):
        """'return the size of dataset"""
        return len(self.data)

    def __getitem__(self, index):

        
        x = self.data[index][0]
        x = torch.LongTensor(x)
        target = self.data[index][1]
        target = torch.LongTensor(target)
        return x, target
if __name__ == '__main__':
    vocab = Vocabulary()
    vocab.build_vocab('./lab4_dataset/train.txt')
    # print(vocab)

    # test = "helloworld"
    # print("Sequence before transformed:", test)
    # ids = vocab.sequence_to_indices(test)
    # print("Indices sequence:", ids)
    # sent = vocab.indices_to_sequence(ids)
    # print("Sequence after transformed:",sent)

    data_transformer = DataTransformer('./lab4_dataset/train.txt')
    # print(data_transformer.indices_sequences)
    train_set = EncoDataSet(data_transformer, 'train')
    test_set = EncoDataSet(data_transformer, 'test')

    train_loader = data.DataLoader(train_set, batch_size=8)

    for x, target in train_loader:
        pass