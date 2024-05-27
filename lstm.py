import os
from torch.utils.data import DataLoader
import torch
from torch import nn
from processing_data import load_data, CustomDataset, glove, collate_fn
from torch.nn.utils.rnn import pad_sequence
from lstmmodel import LSTMClassifier, train, evaluate
from torchtext.data.utils import get_tokenizer
from collections import Counter
from histogram import generate_histogram


def collate_fn(batch):
    # Separate the inputs and labels
    texts, labels = zip(*batch)
    
    # Convert the list of sequences into a list of tensors (word embeddings)
    embedded_texts = [torch.tensor(glove.get_vecs_by_tokens(text.split()), dtype=torch.float) for text in texts]
    
    # Pad the sequences so that they all have the same length
    padded_texts = pad_sequence(embedded_texts, batch_first=True)
    
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_texts, labels

# LOADING DATA

batch_size = 512

# Get the current working directory
current_dir = os.getcwd()
train_pos_examples = load_data(os.path.join(current_dir, 'data', 'train', 'pos'), 'positive')
train_neg_examples = load_data(os.path.join(current_dir, 'data', 'train', 'neg'), 'negative')
test_pos_examples = load_data(os.path.join(current_dir, 'data', 'test', 'pos'), 'positive')
test_neg_examples = load_data(os.path.join(current_dir, 'data', 'test', 'neg'), 'negative')

len_train_pos = int(len(train_pos_examples) * 0.8)
len_train_neg = int(len(train_neg_examples) * 0.8)

train_pos_subset = train_pos_examples[:len_train_pos]
train_neg_subset = train_neg_examples[:len_train_neg]

val_pos_subset = train_pos_examples[len_train_pos:]
val_neg_subset = train_neg_examples[len_train_neg:]

train_examples = train_pos_subset + train_neg_subset
val_examples = val_pos_subset + val_neg_subset
test_examples = test_pos_examples + test_neg_examples

train_dataset = CustomDataset(train_examples, glove)
val_dataset = CustomDataset(val_examples, glove)
test_dataset = CustomDataset(test_examples, glove)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# EXPLORATORY DATA ANALYSIS

# Most frequent words per class
if 1:
    tokenizer = get_tokenizer('basic_english')
    ignore_words = {'.',',','and','to','a','of','is','in','the','it','\'','i','that','this','s','as','with','for','was','but','film','movie','(',')','his','on','you','he','are','not','t','one','have','be','she','they','by','!','all','!','?','so','like','just','by','an','has','from','her','him','them','who','at','about','very','there','out','what','or','more','when','some','if','can','my','time','their','see','up','had','really','would','which','we','me','will','story','do','than','even','most','only','also','other','-','its','were','been','much','get','because','people','into','first','how','way','life','films','many','made','think','two','too','movies','character','characters','don','any','make','made','first','too','plot','movies','way','after','think','watch','something','ve','scene','scenes','these','go','re','few','want','before','minutes','then','could'}

    positive_counter = Counter()
    negative_counter = Counter()

    for text, label in train_examples:
        tokens = [token for token in tokenizer(text) if token not in ignore_words]
        if label == 1:
            positive_counter.update(tokens)
        else:
            negative_counter.update(tokens)

    top_positive_words = positive_counter.most_common(10)
    top_negative_words = negative_counter.most_common(10)

    print("Top 10 most frequent words in positive reviews:")
    for word, count in top_positive_words:
        print(f"{word}: {count}")

    print("\nTop 10 most frequent words in negative reviews:")
    for word, count in top_negative_words:
        print(f"{word}: {count}")

# Distribution of Movie Ratings
if 1:
    directory_path1 = os.path.join('data', 'test', 'neg')
    directory_path2 = os.path.join('data', 'test', 'pos')
    generate_histogram(directory_path1, directory_path2)

# DEPLOYING THE MODELS
if 1:
    input_size = 100
    hidden_size = 128
    num_classes = 2
    num_layers = 1
    bidirectional = False
    num_epochs = 10

    lstm_classifier = LSTMClassifier(input_size, hidden_size, num_classes, num_layers, bidirectional)

    criterion = nn.CrossEntropyLoss()
    optimizer_lstm = torch.optim.Adam(lstm_classifier.parameters(), lr=0.001)

    # Verify the shape of input data
    for batch in train_loader:
        sequences, labels = batch
        print(f"Input shape: {sequences.shape}")  # Should be (batch_size, sequence_length, input_size)
        break

    m1 = train(lstm_classifier, num_epochs, train_loader, val_loader, optimizer_lstm, criterion)
    evaluate(lstm_classifier, test_loader, test_dataset, criterion)