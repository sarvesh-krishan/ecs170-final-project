import os
from torch.utils.data import DataLoader
import torch
import pandas as pd
import seaborn as sns
from torch import nn
from processing_data import load_data, CustomDataset, glove, collate_fn
from model import RNNClassifier, train
import matplotlib.pyplot as plt

def plot_convergence(model_losses, model_labels):
    if len(model_losses) != len(model_labels):
        return

    data = pd.DataFrame({'Training': train_losses, 'Validation': val_losses})

    # Plot using Seaborn
    sns.lineplot(data=data)

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Convergence Plot')

    # Display the plot
    plt.show()


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
train_losses, val_losses = train(rnn_classifier, num_epochs, train_loader, val_loader, optimizer_rnn, criterion)

# Plot the training convergence
model_losses = [train_losses]
model_losses_2 = [val_losses]
model_label = ['Model 1']
model_label_2 = ['Model 2']
plot_convergence(model_losses, model_label)
plot_convergence(model_losses_2, model_label_2)