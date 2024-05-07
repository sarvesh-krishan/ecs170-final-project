import os
import torch
from torchtext.vocab import Vectors
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


current_dir = os.getcwd()
glove_file = os.path.join(current_dir, 'data', '.vector_cache', 'glove.6B.100d.txt')
glove = Vectors(glove_file)

def collate_fn(batch):
    max_padding_length = 200  # Define the maximum padding length here
    sequences, labels = zip(*batch)

    # Pad the sequences to the maximum padding length
    sequences_padded = pad_sequence([seq[:max_padding_length] for seq in sequences], batch_first=True)

    # Stack the labels into a single tensor
    labels = torch.stack(labels)

    return sequences_padded, labels


def load_data(directory, label):
    examples = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            unique_id, rating = filename.split("_")
            if label == 'positive':
                examples.append((text, 1))  # Assign label 1 for positive
            elif label == 'negative':
                examples.append((text, 0))  # Assign label 0 for negative
    return examples


class CustomDataset(Dataset):
    def __init__(self, examples, vocab):
        self.examples = examples
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        tokenized_text = self.tokenizer(text)
        indexed_text = [self.vocab.stoi[word] for word in tokenized_text if word in self.vocab.stoi]
        tensor_text = torch.tensor(indexed_text)
        label_tensor = torch.tensor(label)
        return tensor_text, label_tensor
