import os
from torch.utils.data import DataLoader
import torch
from torch import nn
from processing_data import load_data, CustomDataset, glove, collate_fn
from model import RNNClassifier, train, evaluate
from model import train, evaluate

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

# Use the collate_fn in your DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

class BiRNNClassifier(nn.Module):
    def __init__(self, input_size=100, hidden_size=128, num_layers=1, num_classes=2):
        super(BiRNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(glove.vectors)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        last_hidden_state = output[:, -1, :]
        logits = self.fc(last_hidden_state)
        return logits

if 1:
    # Define the hyperparameters

    input_size = 100  # Size of the input vectors (e.g., GloVe word embeddings)
    hidden_size = 128  # Size of the hidden state in the RNN
    num_layers = 1
    num_classes = 2  # Number of output classes (positive and negative)
    num_epochs = 10

    # Create an instance of the classifier
    bi_rnn_classifier = BiRNNClassifier(input_size, hidden_size, num_classes=num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn = torch.optim.Adam(bi_rnn_classifier.parameters(), lr=0.001)

    # Train the RNN model
    m1 = train(bi_rnn_classifier, num_epochs, train_loader, val_loader, optimizer_rnn, criterion)

    # Evaluate the RNN model
    evaluate(bi_rnn_classifier, test_loader, test_dataset, criterion)