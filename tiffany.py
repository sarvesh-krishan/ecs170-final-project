import torchtext; torchtext.disable_torchtext_deprecation_warning()

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os
from torch.utils.data import DataLoader
import torch
from torch import nn
from processing_data import load_data, CustomDataset, glove, collate_fn
from model import RNNClassifier, train, evaluate
from torchtext.data.utils import get_tokenizer
from collections import Counter
from histogram import generate_histogram

def find_ratings_skew(datasets, class_type):
    ratings = []
    # get ratings from file names
    for dataset in datasets:
        files = os.listdir(dataset)
        file_names = [file.split('_', 1)[1].split('.')[0] for file in files if '_' in file and file.endswith('.txt')]
        ratings.extend(file_names)
    # sort increasing order
    ratings = sorted(ratings, key=lambda x: int(x))
    ratings_bins = np.unique(ratings)

    # create empty list for frequency of each rating
    frequency = [0] * len(ratings_bins)
    for i in range(len(ratings_bins)):
        frequency[i] = ratings.count(ratings_bins[i])

    # create distribution plot
    df = pd.DataFrame(ratings, columns=['Ratings'])
    sns.displot(data=df, kde=True, x="Ratings")
    plt.show()


if 0:
    # Get the current working directory
    current_dir = os.getcwd()
    train_pos_examples = os.path.join(current_dir, 'data', 'train', 'pos')
    train_neg_examples = os.path.join(current_dir, 'data', 'train', 'neg')
    test_pos_examples = os.path.join(current_dir, 'data', 'test', 'pos')
    test_neg_examples = os.path.join(current_dir, 'data', 'test', 'neg')

    find_ratings_skew((train_pos_examples, test_pos_examples), "Positive")
    find_ratings_skew((train_neg_examples, test_neg_examples), "Negative")

def getData():
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

    return test_dataset, train_loader, val_loader, test_loader

def tuneLayers(test_dataset, train_loader, val_loader, test_loader, hidden_size=128, num_layers=1):
    # Define the hyperparameters
    input_size = 100  # Size of the input vectors (e.g., GloVe word embeddings)
    num_classes = 2  # Number of output classes (positive and negative)
    num_epochs = 10 

    # Create an instance of the classifier
    rnn_classifier = RNNClassifier(input_size, hidden_size, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn = torch.optim.Adam(rnn_classifier.parameters(), lr=0.001)

    # Train the RNN model
    m1 = train(rnn_classifier, num_epochs, train_loader, val_loader, optimizer_rnn, criterion)

    # Evaluate the RNN model
    evaluate(rnn_classifier, test_loader, test_dataset, criterion)


if 1:
    hidden_size_arr = [156, 256, 512]
    num_layers_arr = [1, 2, 3, 4]

    test_dataset, train_loader, val_loader, test_loader = getData()

    for hidden_size in hidden_size_arr:
        for num_layers in num_layers_arr:
            tuneLayers(test_dataset, train_loader, val_loader, test_loader, hidden_size, num_layers)
            print()