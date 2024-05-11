import os
from torch.utils.data import DataLoader
import torch
from torch import nn
from processing_data import load_data, CustomDataset, glove, collate_fn
from model import RNNClassifier, train
import matplotlib.pyplot as plt

def plot_convergence(model_losses, model_labels):
    if len(model_losses) != len(model_labels):
        return

    for i, losses in enumerate(model_losses):
        plt.plot(losses, label = model_labels[i])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Convergence')
    plt.legend()
    plt.show()


# LOADING DATA

batch_size = 512

# Get the current working directory
current_dir = os.getcwd()
train_pos_examples = load_data(os.path.join(current_dir, 'data', 'train', 'pos'), 'positive')
train_neg_examples = load_data(os.path.join(current_dir, 'data', 'train', 'neg'), 'negative')
test_pos_examples = load_data(os.path.join(current_dir, 'data', 'test', 'pos'), 'positive')
test_neg_examples = load_data(os.path.join(current_dir, 'data', 'test', 'neg'), 'negative')

train_examples = train_pos_examples + train_neg_examples
test_examples = test_pos_examples + test_neg_examples

train_dataset = CustomDataset(train_examples, glove)
test_dataset = CustomDataset(test_examples, glove)

# Use the collate_fn in your DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

input_size = 100  # Size of the input vectors (e.g., GloVe word embeddings)
hidden_size = 128  # Size of the hidden state in the RNN
num_classes = 2  # Number of output classes (positive and negative)
num_epochs = 10

# Create an instance of the classifier
rnn_classifier = RNNClassifier(input_size, hidden_size, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_rnn = torch.optim.Adam(rnn_classifier.parameters(), lr=0.001)

# Train the model
train_losses = train(rnn_classifier, num_epochs, train_loader, optimizer_rnn, criterion)

# Plot the training convergence
model_losses = [train_losses]
model_label = ['Model 1']
plot_convergence(model_losses, model_label)