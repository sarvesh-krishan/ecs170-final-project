import os
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from processing_data import load_data, CustomDataset, glove, collate_fn, generate_histogram
from model import RNNClassifier, LSTMClassifier, GRUClassifier, train, evaluate
from torchtext.data.utils import get_tokenizer
from collections import Counter


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


# EXPLORATORY DATA ANALYSIS

# Five Number Summary Statistics

if 0:
    positive_reviews = train_pos_examples + test_pos_examples
    negative_reviews = train_neg_examples + test_neg_examples

    # Print first review from positive and negative lists
    print("Positive Review Sample:", positive_reviews[0][0])
    print("Negative Review Sample:", negative_reviews[0][0])

    # Create the pandas DataFrame for positive reviews
    df_positive = pd.DataFrame(positive_reviews, columns=['Review', 'Label'])
    df_positive['Words'] = df_positive['Review'].apply(lambda x: len(x.split()))
    df_positive['Characters'] = df_positive['Review'].apply(len)

    # Create the pandas DataFrame for negative reviews
    df_negative = pd.DataFrame(negative_reviews, columns=['Review', 'Label'])
    df_negative['Words'] = df_negative['Review'].apply(lambda x: len(x.split()))
    df_negative['Characters'] = df_negative['Review'].apply(len)

    # Print dataframes and their statistical descriptions for positive reviews
    print("\nPositive Reviews DataFrame:")
    print(df_positive)
    print("\nStatistics for Words in Positive Reviews:")
    print(df_positive['Words'].describe())
    print("\nStatistics for Characters in Positive Reviews:")
    print(df_positive['Characters'].describe())

    # Print dataframes and their statistical descriptions for negative reviews
    print("\nNegative Reviews DataFrame:")
    print(df_negative)
    print("\nStatistics for Words in Negative Reviews:")
    print(df_negative['Words'].describe())
    print("\nStatistics for Characters in Negative Reviews:")
    print(df_negative['Characters'].describe())


# Most frequent words per class

if 0:
    # Get the basic English tokenizer
    tokenizer = get_tokenizer('basic_english')

    # Define the words to ignore
    ignore_words = {'.',',','and','to','a','of','is','in','the','it','\'','i','that','this','s','as','with','for','was','but','film','movie','(',')','his','on','you','he','are','not','t','one','have','be','she','they','by','!','all','!','?','so','like','just','by','an','has','from','her','him','them','who','at','about','very','there','out','what','or','more','when','some','if','can','my','time','their','see','up','had','really','would','which','we','me','will','story','do','than','even','most','only','also','other','-','its','were','been','much','get','because','people','into','first','how','way','life','films','many','made','think','two','too','movies','character','characters','don','any','make','made','first','too','plot','movies','way','after','think','watch','something','ve','scene','scenes','these','go','re','few','want','before','minutes','then','could'}

    # Counters for positive and negative words
    positive_counter = Counter()
    negative_counter = Counter()

    # Update counters based on label
    for text, label in train_examples:
        # Tokenize text and filter out ignored words
        tokens = [token for token in tokenizer(text) if token not in ignore_words]
        if label == 1:
            positive_counter.update(tokens)
        else:
            negative_counter.update(tokens)

    # Get top 10 most frequent words for positive and negative classes
    top_positive_words = positive_counter.most_common(10)
    top_negative_words = negative_counter.most_common(10)

    print("Top 10 most frequent words in positive reviews:")
    for word, count in top_positive_words:
        print(f"{word}: {count}")

    print("\nTop 10 most frequent words in negative reviews:")
    for word, count in top_negative_words:
        print(f"{word}: {count}")


# Distribution of Movie Ratings

if 0:
    directory_path1 = os.path.join('data', 'test', 'neg')
    directory_path2 = os.path.join('data', 'test', 'pos')
    generate_histogram(directory_path1, directory_path2)


# DEPLOYING THE MODELS
# RNN Deployment
if 0:
    print("RNN Model Deployment RNN 1 LR:0.0001")
    # Define the hyperparameters
    input_size = 100  # Size of the input vectors (e.g., GloVe word embeddings)
    num_classes = 2  # Number of output classes (positive and negative)
    num_epochs = 15

    # Create an instance of the classifier
    rnn_1 = RNNClassifier(input_size, 128, num_classes)
    rnn_2 = RNNClassifier(input_size, 128, num_classes)
    rnn_3 = RNNClassifier(input_size, 256, num_classes)
    rnn_4 = RNNClassifier(input_size, 256, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn_1 = torch.optim.Adam(rnn_1.parameters(), lr=0.0001)
    optimizer_rnn_2 = torch.optim.SGD(rnn_2.parameters(), lr=0.0001)
    optimizer_rnn_3 = torch.optim.Adam(rnn_3.parameters(), lr=0.0001)
    optimizer_rnn_4 = torch.optim.SGD(rnn_4.parameters(), lr=0.0001)

    # Train the models
    rnn_1_train_losses, rnn_1_val_losses = train(rnn_1, num_epochs, train_loader, val_loader, optimizer_rnn_1, criterion)
    rnn_2_train_losses, rnn_2_val_losses = train(rnn_2, num_epochs, train_loader, val_loader, optimizer_rnn_2, criterion)
    rnn_3_train_losses, rnn_3_val_losses = train(rnn_3, num_epochs, train_loader, val_loader, optimizer_rnn_3, criterion)
    rnn_4_train_losses, rnn_4_val_losses = train(rnn_4, num_epochs, train_loader, val_loader, optimizer_rnn_4, criterion)

    # Pad the list of losses with 0s for early stopping
    rnn_1_train_losses = np.pad(rnn_1_train_losses, (0, num_epochs-len(rnn_1_train_losses)), mode='constant', constant_values=np.nan)
    rnn_2_train_losses = np.pad(rnn_2_train_losses, (0, num_epochs-len(rnn_2_train_losses)), mode='constant', constant_values=np.nan)
    rnn_3_train_losses = np.pad(rnn_3_train_losses, (0, num_epochs-len(rnn_3_train_losses)), mode='constant', constant_values=np.nan)
    rnn_4_train_losses = np.pad(rnn_4_train_losses, (0, num_epochs-len(rnn_4_train_losses)), mode='constant', constant_values=np.nan)
    rnn_1_val_losses = np.pad(rnn_1_val_losses, (0, num_epochs-len(rnn_1_val_losses)), mode='constant', constant_values=np.nan)
    rnn_2_val_losses = np.pad(rnn_2_val_losses, (0, num_epochs-len(rnn_2_val_losses)), mode='constant', constant_values=np.nan)
    rnn_3_val_losses = np.pad(rnn_3_val_losses, (0, num_epochs-len(rnn_3_val_losses)), mode='constant', constant_values=np.nan)
    rnn_4_val_losses = np.pad(rnn_4_val_losses, (0, num_epochs-len(rnn_4_val_losses)), mode='constant', constant_values=np.nan)

    # Evaluate the models
    rnn_1_roc_data = evaluate(rnn_1, test_loader, test_dataset, criterion)
    rnn_2_roc_data = evaluate(rnn_2, test_loader, test_dataset, criterion)
    rnn_3_roc_data = evaluate(rnn_3, test_loader, test_dataset, criterion)
    rnn_4_roc_data = evaluate(rnn_4, test_loader, test_dataset, criterion)

    # Plot the training convergence
    df = pd.DataFrame({'Num Epochs':np.arange(1,num_epochs+1),
                       'Model 1 Train Loss':rnn_1_train_losses, 'Model 1 Val Loss':rnn_1_val_losses,
                       'Model 2 Train Loss':rnn_2_train_losses, 'Model 2 Val Loss':rnn_2_val_losses,
                       'Model 3 Train Loss':rnn_3_train_losses, 'Model 3 Val Loss':rnn_3_val_losses,
                       'Model 4 Train Loss':rnn_4_train_losses, 'Model 4 Val Loss':rnn_4_val_losses})
    for col in df.columns[1:]:
        sns.lineplot(x='Num Epochs', y=col, data=df, label=col)
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.title('Training Convergence Plot')
    plt.legend()
    plt.show()

    # Plot the ROC curve
    plt.figure()  
    plt.plot(rnn_1_roc_data[0], rnn_1_roc_data[1], label='Model 1 (area = %0.2f)' % rnn_1_roc_data[2])
    plt.plot(rnn_2_roc_data[0], rnn_2_roc_data[1], label='Model 2 (area = %0.2f)' % rnn_2_roc_data[2])
    plt.plot(rnn_3_roc_data[0], rnn_3_roc_data[1], label='Model 3 (area = %0.2f)' % rnn_3_roc_data[2])
    plt.plot(rnn_4_roc_data[0], rnn_4_roc_data[1], label='Model 4 (area = %0.2f)' % rnn_4_roc_data[2])
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


if 0:
    print("RNN Model Deployment RNN 2 LR:0.001")
    # Define the hyperparameters
    input_size = 100  # Size of the input vectors (e.g., GloVe word embeddings)
    num_classes = 2  # Number of output classes (positive and negative)
    num_epochs = 15

    # Create an instance of the classifier
    rnn_1 = RNNClassifier(input_size, 128, num_classes)
    rnn_2 = RNNClassifier(input_size, 128, num_classes)
    rnn_3 = RNNClassifier(input_size, 256, num_classes)
    rnn_4 = RNNClassifier(input_size, 256, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn_1 = torch.optim.Adam(rnn_1.parameters(), lr=0.001)
    optimizer_rnn_2 = torch.optim.SGD(rnn_2.parameters(), lr=0.001)
    optimizer_rnn_3 = torch.optim.Adam(rnn_3.parameters(), lr=0.001)
    optimizer_rnn_4 = torch.optim.SGD(rnn_4.parameters(), lr=0.001)

    # Train the models
    rnn_1_train_losses, rnn_1_val_losses = train(rnn_1, num_epochs, train_loader, val_loader, optimizer_rnn_1, criterion)
    rnn_2_train_losses, rnn_2_val_losses = train(rnn_2, num_epochs, train_loader, val_loader, optimizer_rnn_2, criterion)
    rnn_3_train_losses, rnn_3_val_losses = train(rnn_3, num_epochs, train_loader, val_loader, optimizer_rnn_3, criterion)
    rnn_4_train_losses, rnn_4_val_losses = train(rnn_4, num_epochs, train_loader, val_loader, optimizer_rnn_4, criterion)

    # Pad the list of losses with 0s for early stopping
    rnn_1_train_losses = np.pad(rnn_1_train_losses, (0, num_epochs-len(rnn_1_train_losses)), mode='constant', constant_values=np.nan)
    rnn_2_train_losses = np.pad(rnn_2_train_losses, (0, num_epochs-len(rnn_2_train_losses)), mode='constant', constant_values=np.nan)
    rnn_3_train_losses = np.pad(rnn_3_train_losses, (0, num_epochs-len(rnn_3_train_losses)), mode='constant', constant_values=np.nan)
    rnn_4_train_losses = np.pad(rnn_4_train_losses, (0, num_epochs-len(rnn_4_train_losses)), mode='constant', constant_values=np.nan)
    rnn_1_val_losses = np.pad(rnn_1_val_losses, (0, num_epochs-len(rnn_1_val_losses)), mode='constant', constant_values=np.nan)
    rnn_2_val_losses = np.pad(rnn_2_val_losses, (0, num_epochs-len(rnn_2_val_losses)), mode='constant', constant_values=np.nan)
    rnn_3_val_losses = np.pad(rnn_3_val_losses, (0, num_epochs-len(rnn_3_val_losses)), mode='constant', constant_values=np.nan)
    rnn_4_val_losses = np.pad(rnn_4_val_losses, (0, num_epochs-len(rnn_4_val_losses)), mode='constant', constant_values=np.nan)

    # Evaluate the models
    rnn_1_roc_data = evaluate(rnn_1, test_loader, test_dataset, criterion)
    rnn_2_roc_data = evaluate(rnn_2, test_loader, test_dataset, criterion)
    rnn_3_roc_data = evaluate(rnn_3, test_loader, test_dataset, criterion)
    rnn_4_roc_data = evaluate(rnn_4, test_loader, test_dataset, criterion)

    # Plot the training convergence
    df = pd.DataFrame({'Num Epochs':np.arange(1,num_epochs+1),
                       'Model 5 Train Loss':rnn_1_train_losses, 'Model 5 Val Loss':rnn_1_val_losses,
                       'Model 6 Train Loss':rnn_2_train_losses, 'Model 6 Val Loss':rnn_2_val_losses,
                       'Model 7 Train Loss':rnn_3_train_losses, 'Model 7 Val Loss':rnn_3_val_losses,
                       'Model 8 Train Loss':rnn_4_train_losses, 'Model 8 Val Loss':rnn_4_val_losses})
    for col in df.columns[1:]:
        sns.lineplot(x='Num Epochs', y=col, data=df, label=col)
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.title('Training Convergence Plot')
    plt.legend()
    plt.show()

    # Plot the ROC curve
    plt.figure()  
    plt.plot(rnn_1_roc_data[0], rnn_1_roc_data[1], label='Model 5 (area = %0.2f)' % rnn_1_roc_data[2])
    plt.plot(rnn_2_roc_data[0], rnn_2_roc_data[1], label='Model 6 (area = %0.2f)' % rnn_2_roc_data[2])
    plt.plot(rnn_3_roc_data[0], rnn_3_roc_data[1], label='Model 7 (area = %0.2f)' % rnn_3_roc_data[2])
    plt.plot(rnn_4_roc_data[0], rnn_4_roc_data[1], label='Model 8 (area = %0.2f)' % rnn_4_roc_data[2])
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

if 0:
    print("RNN Model Deployment RNN 3 LR:0.01")
    # Define the hyperparameters
    input_size = 100  # Size of the input vectors (e.g., GloVe word embeddings)
    num_classes = 2  # Number of output classes (positive and negative)
    num_epochs = 15

    # Create an instance of the classifier
    rnn_1 = RNNClassifier(input_size, 128, num_classes)
    rnn_2 = RNNClassifier(input_size, 128, num_classes)
    rnn_3 = RNNClassifier(input_size, 256, num_classes)
    rnn_4 = RNNClassifier(input_size, 256, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn_1 = torch.optim.Adam(rnn_1.parameters(), lr=0.01)
    optimizer_rnn_2 = torch.optim.SGD(rnn_2.parameters(), lr=0.01)
    optimizer_rnn_3 = torch.optim.Adam(rnn_3.parameters(), lr=0.01)
    optimizer_rnn_4 = torch.optim.SGD(rnn_4.parameters(), lr=0.01)

    # Train the models
    rnn_1_train_losses, rnn_1_val_losses = train(rnn_1, num_epochs, train_loader, val_loader, optimizer_rnn_1, criterion)
    rnn_2_train_losses, rnn_2_val_losses = train(rnn_2, num_epochs, train_loader, val_loader, optimizer_rnn_2, criterion)
    rnn_3_train_losses, rnn_3_val_losses = train(rnn_3, num_epochs, train_loader, val_loader, optimizer_rnn_3, criterion)
    rnn_4_train_losses, rnn_4_val_losses = train(rnn_4, num_epochs, train_loader, val_loader, optimizer_rnn_4, criterion)

    # Pad the list of losses with 0s for early stopping
    rnn_1_train_losses = np.pad(rnn_1_train_losses, (0, num_epochs-len(rnn_1_train_losses)), mode='constant', constant_values=np.nan)
    rnn_2_train_losses = np.pad(rnn_2_train_losses, (0, num_epochs-len(rnn_2_train_losses)), mode='constant', constant_values=np.nan)
    rnn_3_train_losses = np.pad(rnn_3_train_losses, (0, num_epochs-len(rnn_3_train_losses)), mode='constant', constant_values=np.nan)
    rnn_4_train_losses = np.pad(rnn_4_train_losses, (0, num_epochs-len(rnn_4_train_losses)), mode='constant', constant_values=np.nan)
    rnn_1_val_losses = np.pad(rnn_1_val_losses, (0, num_epochs-len(rnn_1_val_losses)), mode='constant', constant_values=np.nan)
    rnn_2_val_losses = np.pad(rnn_2_val_losses, (0, num_epochs-len(rnn_2_val_losses)), mode='constant', constant_values=np.nan)
    rnn_3_val_losses = np.pad(rnn_3_val_losses, (0, num_epochs-len(rnn_3_val_losses)), mode='constant', constant_values=np.nan)
    rnn_4_val_losses = np.pad(rnn_4_val_losses, (0, num_epochs-len(rnn_4_val_losses)), mode='constant', constant_values=np.nan)

    # Evaluate the models
    rnn_1_roc_data = evaluate(rnn_1, test_loader, test_dataset, criterion)
    rnn_2_roc_data = evaluate(rnn_2, test_loader, test_dataset, criterion)
    rnn_3_roc_data = evaluate(rnn_3, test_loader, test_dataset, criterion)
    rnn_4_roc_data = evaluate(rnn_4, test_loader, test_dataset, criterion)

    # Plot the training convergence
    df = pd.DataFrame({'Num Epochs':np.arange(1,num_epochs+1),
                       'Model 9 Train Loss':rnn_1_train_losses, 'Model 9 Val Loss':rnn_1_val_losses,
                       'Model 10 Train Loss':rnn_2_train_losses, 'Model 10 Val Loss':rnn_2_val_losses,
                       'Model 11 Train Loss':rnn_3_train_losses, 'Model 11 Val Loss':rnn_3_val_losses,
                       'Model 12 Train Loss':rnn_4_train_losses, 'Model 12 Val Loss':rnn_4_val_losses})
    for col in df.columns[1:]:
        sns.lineplot(x='Num Epochs', y=col, data=df, label=col)
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.title('Training Convergence Plot')
    plt.legend()
    plt.show()

    # Plot the ROC curve
    plt.figure()  
    plt.plot(rnn_1_roc_data[0], rnn_1_roc_data[1], label='Model 9 (area = %0.2f)' % rnn_1_roc_data[2])
    plt.plot(rnn_2_roc_data[0], rnn_2_roc_data[1], label='Model 10 (area = %0.2f)' % rnn_2_roc_data[2])
    plt.plot(rnn_3_roc_data[0], rnn_3_roc_data[1], label='Model 11 (area = %0.2f)' % rnn_3_roc_data[2])
    plt.plot(rnn_4_roc_data[0], rnn_4_roc_data[1], label='Model 12 (area = %0.2f)' % rnn_4_roc_data[2])
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
# LSTM Deployment
if 0:
    print("LSTM Model Deployment LSTM 1 LR:0.01")
    # Define the hyperparameters
    input_size = 100  # Size of the input vectors (e.g., GloVe word embeddings)
    num_classes = 2  # Number of output classes (positive and negative)
    num_epochs = 15

    # Create an instance of the classifier
    rnn_1 = LSTMClassifier(input_size, 128, num_classes)
    rnn_2 = LSTMClassifier(input_size, 128, num_classes)
    rnn_3 = LSTMClassifier(input_size, 256, num_classes)
    rnn_4 = LSTMClassifier(input_size, 256, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn_1 = torch.optim.Adam(rnn_1.parameters(), lr=0.0001)
    optimizer_rnn_2 = torch.optim.SGD(rnn_2.parameters(), lr=0.0001)
    optimizer_rnn_3 = torch.optim.Adam(rnn_3.parameters(), lr=0.0001)
    optimizer_rnn_4 = torch.optim.SGD(rnn_4.parameters(), lr=0.0001)

    # Train the models
    rnn_1_train_losses, rnn_1_val_losses = train(rnn_1, num_epochs, train_loader, val_loader, optimizer_rnn_1, criterion)
    rnn_2_train_losses, rnn_2_val_losses = train(rnn_2, num_epochs, train_loader, val_loader, optimizer_rnn_2, criterion)
    rnn_3_train_losses, rnn_3_val_losses = train(rnn_3, num_epochs, train_loader, val_loader, optimizer_rnn_3, criterion)
    rnn_4_train_losses, rnn_4_val_losses = train(rnn_4, num_epochs, train_loader, val_loader, optimizer_rnn_4, criterion)

    # Pad the list of losses with 0s for early stopping
    rnn_1_train_losses = np.pad(rnn_1_train_losses, (0, num_epochs-len(rnn_1_train_losses)), mode='constant', constant_values=np.nan)
    rnn_2_train_losses = np.pad(rnn_2_train_losses, (0, num_epochs-len(rnn_2_train_losses)), mode='constant', constant_values=np.nan)
    rnn_3_train_losses = np.pad(rnn_3_train_losses, (0, num_epochs-len(rnn_3_train_losses)), mode='constant', constant_values=np.nan)
    rnn_4_train_losses = np.pad(rnn_4_train_losses, (0, num_epochs-len(rnn_4_train_losses)), mode='constant', constant_values=np.nan)
    rnn_1_val_losses = np.pad(rnn_1_val_losses, (0, num_epochs-len(rnn_1_val_losses)), mode='constant', constant_values=np.nan)
    rnn_2_val_losses = np.pad(rnn_2_val_losses, (0, num_epochs-len(rnn_2_val_losses)), mode='constant', constant_values=np.nan)
    rnn_3_val_losses = np.pad(rnn_3_val_losses, (0, num_epochs-len(rnn_3_val_losses)), mode='constant', constant_values=np.nan)
    rnn_4_val_losses = np.pad(rnn_4_val_losses, (0, num_epochs-len(rnn_4_val_losses)), mode='constant', constant_values=np.nan)

    # Evaluate the models
    rnn_1_roc_data = evaluate(rnn_1, test_loader, test_dataset, criterion)
    rnn_2_roc_data = evaluate(rnn_2, test_loader, test_dataset, criterion)
    rnn_3_roc_data = evaluate(rnn_3, test_loader, test_dataset, criterion)
    rnn_4_roc_data = evaluate(rnn_4, test_loader, test_dataset, criterion)

    # Plot the training convergence
    df = pd.DataFrame({'Num Epochs':np.arange(1,num_epochs+1),
                       'Model 13 Train Loss':rnn_1_train_losses, 'Model 13 Val Loss':rnn_1_val_losses,
                       'Model 14 Train Loss':rnn_2_train_losses, 'Model 14 Val Loss':rnn_2_val_losses,
                       'Model 16 Train Loss':rnn_3_train_losses, 'Model 15 Val Loss':rnn_3_val_losses,
                       'Model 16 Train Loss':rnn_4_train_losses, 'Model 16 Val Loss':rnn_4_val_losses})
    for col in df.columns[1:]:
        sns.lineplot(x='Num Epochs', y=col, data=df, label=col)
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.title('Training Convergence Plot')
    plt.legend()
    plt.show()

    # Plot the ROC curve
    plt.figure()  
    plt.plot(rnn_1_roc_data[0], rnn_1_roc_data[1], label='Model 13 (area = %0.2f)' % rnn_1_roc_data[2])
    plt.plot(rnn_2_roc_data[0], rnn_2_roc_data[1], label='Model 14 (area = %0.2f)' % rnn_2_roc_data[2])
    plt.plot(rnn_3_roc_data[0], rnn_3_roc_data[1], label='Model 15 (area = %0.2f)' % rnn_3_roc_data[2])
    plt.plot(rnn_4_roc_data[0], rnn_4_roc_data[1], label='Model 16 (area = %0.2f)' % rnn_4_roc_data[2])
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()    

if 0:
    print("LSTM Model Deployment LSTM 2 LR:0.001")
    # Define the hyperparameters
    input_size = 100  # Size of the input vectors (e.g., GloVe word embeddings)
    num_classes = 2  # Number of output classes (positive and negative)
    num_epochs = 15

    # Create an instance of the classifier
    rnn_1 = LSTMClassifier(input_size, 128, num_classes)
    rnn_2 = LSTMClassifier(input_size, 128, num_classes)
    rnn_3 = LSTMClassifier(input_size, 256, num_classes)
    rnn_4 = LSTMClassifier(input_size, 256, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn_1 = torch.optim.Adam(rnn_1.parameters(), lr=0.001)
    optimizer_rnn_2 = torch.optim.SGD(rnn_2.parameters(), lr=0.001)
    optimizer_rnn_3 = torch.optim.Adam(rnn_3.parameters(), lr=0.001)
    optimizer_rnn_4 = torch.optim.SGD(rnn_4.parameters(), lr=0.001)

    # Train the models
    rnn_1_train_losses, rnn_1_val_losses = train(rnn_1, num_epochs, train_loader, val_loader, optimizer_rnn_1, criterion)
    rnn_2_train_losses, rnn_2_val_losses = train(rnn_2, num_epochs, train_loader, val_loader, optimizer_rnn_2, criterion)
    rnn_3_train_losses, rnn_3_val_losses = train(rnn_3, num_epochs, train_loader, val_loader, optimizer_rnn_3, criterion)
    rnn_4_train_losses, rnn_4_val_losses = train(rnn_4, num_epochs, train_loader, val_loader, optimizer_rnn_4, criterion)

    # Pad the list of losses with 0s for early stopping
    rnn_1_train_losses = np.pad(rnn_1_train_losses, (0, num_epochs-len(rnn_1_train_losses)), mode='constant', constant_values=np.nan)
    rnn_2_train_losses = np.pad(rnn_2_train_losses, (0, num_epochs-len(rnn_2_train_losses)), mode='constant', constant_values=np.nan)
    rnn_3_train_losses = np.pad(rnn_3_train_losses, (0, num_epochs-len(rnn_3_train_losses)), mode='constant', constant_values=np.nan)
    rnn_4_train_losses = np.pad(rnn_4_train_losses, (0, num_epochs-len(rnn_4_train_losses)), mode='constant', constant_values=np.nan)
    rnn_1_val_losses = np.pad(rnn_1_val_losses, (0, num_epochs-len(rnn_1_val_losses)), mode='constant', constant_values=np.nan)
    rnn_2_val_losses = np.pad(rnn_2_val_losses, (0, num_epochs-len(rnn_2_val_losses)), mode='constant', constant_values=np.nan)
    rnn_3_val_losses = np.pad(rnn_3_val_losses, (0, num_epochs-len(rnn_3_val_losses)), mode='constant', constant_values=np.nan)
    rnn_4_val_losses = np.pad(rnn_4_val_losses, (0, num_epochs-len(rnn_4_val_losses)), mode='constant', constant_values=np.nan)

    # Evaluate the models
    rnn_1_roc_data = evaluate(rnn_1, test_loader, test_dataset, criterion)
    rnn_2_roc_data = evaluate(rnn_2, test_loader, test_dataset, criterion)
    rnn_3_roc_data = evaluate(rnn_3, test_loader, test_dataset, criterion)
    rnn_4_roc_data = evaluate(rnn_4, test_loader, test_dataset, criterion)

    # Plot the training convergence
    df = pd.DataFrame({'Num Epochs':np.arange(1,num_epochs+1),
                       'Model 17 Train Loss':rnn_1_train_losses, 'Model 17 Val Loss':rnn_1_val_losses,
                       'Model 18 Train Loss':rnn_2_train_losses, 'Model 18 Val Loss':rnn_2_val_losses,
                       'Model 19 Train Loss':rnn_3_train_losses, 'Model 19 Val Loss':rnn_3_val_losses,
                       'Model 20 Train Loss':rnn_4_train_losses, 'Model 20 Val Loss':rnn_4_val_losses})
    for col in df.columns[1:]:
        sns.lineplot(x='Num Epochs', y=col, data=df, label=col)
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.title('Training Convergence Plot')
    plt.legend()
    plt.show()

    # Plot the ROC curve
    plt.figure()  
    plt.plot(rnn_1_roc_data[0], rnn_1_roc_data[1], label='Model 17 (area = %0.2f)' % rnn_1_roc_data[2])
    plt.plot(rnn_2_roc_data[0], rnn_2_roc_data[1], label='Model 18 (area = %0.2f)' % rnn_2_roc_data[2])
    plt.plot(rnn_3_roc_data[0], rnn_3_roc_data[1], label='Model 19 (area = %0.2f)' % rnn_3_roc_data[2])
    plt.plot(rnn_4_roc_data[0], rnn_4_roc_data[1], label='Model 20 (area = %0.2f)' % rnn_4_roc_data[2])
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()    

if 0:
    print("LSTM Model Deployment LSTM 3 LR:0.01")
    # Define the hyperparameters
    input_size = 100  # Size of the input vectors (e.g., GloVe word embeddings)
    num_classes = 2  # Number of output classes (positive and negative)
    num_epochs = 15

    # Create an instance of the classifier
    rnn_1 = LSTMClassifier(input_size, 128, num_classes)
    rnn_2 = LSTMClassifier(input_size, 128, num_classes)
    rnn_3 = LSTMClassifier(input_size, 256, num_classes)
    rnn_4 = LSTMClassifier(input_size, 256, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn_1 = torch.optim.Adam(rnn_1.parameters(), lr=0.01)
    optimizer_rnn_2 = torch.optim.SGD(rnn_2.parameters(), lr=0.01)
    optimizer_rnn_3 = torch.optim.Adam(rnn_3.parameters(), lr=0.01)
    optimizer_rnn_4 = torch.optim.SGD(rnn_4.parameters(), lr=0.01)

    # Train the models
    rnn_1_train_losses, rnn_1_val_losses = train(rnn_1, num_epochs, train_loader, val_loader, optimizer_rnn_1, criterion)
    rnn_2_train_losses, rnn_2_val_losses = train(rnn_2, num_epochs, train_loader, val_loader, optimizer_rnn_2, criterion)
    rnn_3_train_losses, rnn_3_val_losses = train(rnn_3, num_epochs, train_loader, val_loader, optimizer_rnn_3, criterion)
    rnn_4_train_losses, rnn_4_val_losses = train(rnn_4, num_epochs, train_loader, val_loader, optimizer_rnn_4, criterion)

    # Pad the list of losses with 0s for early stopping
    rnn_1_train_losses = np.pad(rnn_1_train_losses, (0, num_epochs-len(rnn_1_train_losses)), mode='constant', constant_values=np.nan)
    rnn_2_train_losses = np.pad(rnn_2_train_losses, (0, num_epochs-len(rnn_2_train_losses)), mode='constant', constant_values=np.nan)
    rnn_3_train_losses = np.pad(rnn_3_train_losses, (0, num_epochs-len(rnn_3_train_losses)), mode='constant', constant_values=np.nan)
    rnn_4_train_losses = np.pad(rnn_4_train_losses, (0, num_epochs-len(rnn_4_train_losses)), mode='constant', constant_values=np.nan)
    rnn_1_val_losses = np.pad(rnn_1_val_losses, (0, num_epochs-len(rnn_1_val_losses)), mode='constant', constant_values=np.nan)
    rnn_2_val_losses = np.pad(rnn_2_val_losses, (0, num_epochs-len(rnn_2_val_losses)), mode='constant', constant_values=np.nan)
    rnn_3_val_losses = np.pad(rnn_3_val_losses, (0, num_epochs-len(rnn_3_val_losses)), mode='constant', constant_values=np.nan)
    rnn_4_val_losses = np.pad(rnn_4_val_losses, (0, num_epochs-len(rnn_4_val_losses)), mode='constant', constant_values=np.nan)

    # Evaluate the models
    rnn_1_roc_data = evaluate(rnn_1, test_loader, test_dataset, criterion)
    rnn_2_roc_data = evaluate(rnn_2, test_loader, test_dataset, criterion)
    rnn_3_roc_data = evaluate(rnn_3, test_loader, test_dataset, criterion)
    rnn_4_roc_data = evaluate(rnn_4, test_loader, test_dataset, criterion)

    # Plot the training convergence
    df = pd.DataFrame({'Num Epochs':np.arange(1,num_epochs+1),
                       'Model 21 Train Loss':rnn_1_train_losses, 'Model 21 Val Loss':rnn_1_val_losses,
                       'Model 22 Train Loss':rnn_2_train_losses, 'Model 22 Val Loss':rnn_2_val_losses,
                       'Model 23 Train Loss':rnn_3_train_losses, 'Model 23 Val Loss':rnn_3_val_losses,
                       'Model 24 Train Loss':rnn_4_train_losses, 'Model 24 Val Loss':rnn_4_val_losses})
    for col in df.columns[1:]:
        sns.lineplot(x='Num Epochs', y=col, data=df, label=col)
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.title('Training Convergence Plot')
    plt.legend()
    plt.show()

    # Plot the ROC curve
    plt.figure()  
    plt.plot(rnn_1_roc_data[0], rnn_1_roc_data[1], label='Model 21 (area = %0.2f)' % rnn_1_roc_data[2])
    plt.plot(rnn_2_roc_data[0], rnn_2_roc_data[1], label='Model 22 (area = %0.2f)' % rnn_2_roc_data[2])
    plt.plot(rnn_3_roc_data[0], rnn_3_roc_data[1], label='Model 23 (area = %0.2f)' % rnn_3_roc_data[2])
    plt.plot(rnn_4_roc_data[0], rnn_4_roc_data[1], label='Model 24 (area = %0.2f)' % rnn_4_roc_data[2])
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
# GRU Deployment
if 0:
    print("GRU Model Deployment GRU 1 LR=0.0001 ")
    # Define the hyperparameters
    input_size = 100  # Size of the input vectors (e.g., GloVe word embeddings)
    num_classes = 2  # Number of output classes (positive and negative)
    num_epochs = 15

    # Create an instance of the classifier
    rnn_1 = GRUClassifier(input_size, 128, num_classes)
    rnn_2 = GRUClassifier(input_size, 128, num_classes)
    rnn_3 = GRUClassifier(input_size, 256, num_classes)
    rnn_4 = GRUClassifier(input_size, 256, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn_1 = torch.optim.Adam(rnn_1.parameters(), lr=0.0001)
    optimizer_rnn_2 = torch.optim.SGD(rnn_2.parameters(), lr=0.0001)
    optimizer_rnn_3 = torch.optim.Adam(rnn_3.parameters(), lr=0.0001)
    optimizer_rnn_4 = torch.optim.SGD(rnn_4.parameters(), lr=0.0001)

    # Train the models
    rnn_1_train_losses, rnn_1_val_losses = train(rnn_1, num_epochs, train_loader, val_loader, optimizer_rnn_1, criterion)
    rnn_2_train_losses, rnn_2_val_losses = train(rnn_2, num_epochs, train_loader, val_loader, optimizer_rnn_2, criterion)
    rnn_3_train_losses, rnn_3_val_losses = train(rnn_3, num_epochs, train_loader, val_loader, optimizer_rnn_3, criterion)
    rnn_4_train_losses, rnn_4_val_losses = train(rnn_4, num_epochs, train_loader, val_loader, optimizer_rnn_4, criterion)

    # Pad the list of losses with 0s for early stopping
    rnn_1_train_losses = np.pad(rnn_1_train_losses, (0, num_epochs-len(rnn_1_train_losses)), mode='constant', constant_values=np.nan)
    rnn_2_train_losses = np.pad(rnn_2_train_losses, (0, num_epochs-len(rnn_2_train_losses)), mode='constant', constant_values=np.nan)
    rnn_3_train_losses = np.pad(rnn_3_train_losses, (0, num_epochs-len(rnn_3_train_losses)), mode='constant', constant_values=np.nan)
    rnn_4_train_losses = np.pad(rnn_4_train_losses, (0, num_epochs-len(rnn_4_train_losses)), mode='constant', constant_values=np.nan)
    rnn_1_val_losses = np.pad(rnn_1_val_losses, (0, num_epochs-len(rnn_1_val_losses)), mode='constant', constant_values=np.nan)
    rnn_2_val_losses = np.pad(rnn_2_val_losses, (0, num_epochs-len(rnn_2_val_losses)), mode='constant', constant_values=np.nan)
    rnn_3_val_losses = np.pad(rnn_3_val_losses, (0, num_epochs-len(rnn_3_val_losses)), mode='constant', constant_values=np.nan)
    rnn_4_val_losses = np.pad(rnn_4_val_losses, (0, num_epochs-len(rnn_4_val_losses)), mode='constant', constant_values=np.nan)

    # Evaluate the models
    rnn_1_roc_data = evaluate(rnn_1, test_loader, test_dataset, criterion)
    rnn_2_roc_data = evaluate(rnn_2, test_loader, test_dataset, criterion)
    rnn_3_roc_data = evaluate(rnn_3, test_loader, test_dataset, criterion)
    rnn_4_roc_data = evaluate(rnn_4, test_loader, test_dataset, criterion)

    # Plot the training convergence
    df = pd.DataFrame({'Num Epochs':np.arange(1,num_epochs+1),
                       'Model 25 Train Loss':rnn_1_train_losses, 'Model 25 Val Loss':rnn_1_val_losses,
                       'Model 26 Train Loss':rnn_2_train_losses, 'Model 26 Val Loss':rnn_2_val_losses,
                       'Model 27 Train Loss':rnn_3_train_losses, 'Model 27 Val Loss':rnn_3_val_losses,
                       'Model 28 Train Loss':rnn_4_train_losses, 'Model 28 Val Loss':rnn_4_val_losses})
    for col in df.columns[1:]:
        sns.lineplot(x='Num Epochs', y=col, data=df, label=col)
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.title('Training Convergence Plot')
    plt.legend()
    plt.show()

    # Plot the ROC curve
    plt.figure()  
    plt.plot(rnn_1_roc_data[0], rnn_1_roc_data[1], label='Model 25 (area = %0.2f)' % rnn_1_roc_data[2])
    plt.plot(rnn_2_roc_data[0], rnn_2_roc_data[1], label='Model 26 (area = %0.2f)' % rnn_2_roc_data[2])
    plt.plot(rnn_3_roc_data[0], rnn_3_roc_data[1], label='Model 27 (area = %0.2f)' % rnn_3_roc_data[2])
    plt.plot(rnn_4_roc_data[0], rnn_4_roc_data[1], label='Model 28 (area = %0.2f)' % rnn_4_roc_data[2])
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
if 0:
    print("GRU Model Deployment GRU 2 LR:0.001")
    # Define the hyperparameters
    input_size = 100  # Size of the input vectors (e.g., GloVe word embeddings)
    num_classes = 2  # Number of output classes (positive and negative)
    num_epochs = 15

    # Create an instance of the classifier
    rnn_1 = GRUClassifier(input_size, 128, num_classes)
    rnn_2 = GRUClassifier(input_size, 128, num_classes)
    rnn_3 = GRUClassifier(input_size, 256, num_classes)
    rnn_4 = GRUClassifier(input_size, 256, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn_1 = torch.optim.Adam(rnn_1.parameters(), lr=0.001)
    optimizer_rnn_2 = torch.optim.SGD(rnn_2.parameters(), lr=0.001)
    optimizer_rnn_3 = torch.optim.Adam(rnn_3.parameters(), lr=0.001)
    optimizer_rnn_4 = torch.optim.SGD(rnn_4.parameters(), lr=0.001)

    # Train the models
    rnn_1_train_losses, rnn_1_val_losses = train(rnn_1, num_epochs, train_loader, val_loader, optimizer_rnn_1, criterion)
    rnn_2_train_losses, rnn_2_val_losses = train(rnn_2, num_epochs, train_loader, val_loader, optimizer_rnn_2, criterion)
    rnn_3_train_losses, rnn_3_val_losses = train(rnn_3, num_epochs, train_loader, val_loader, optimizer_rnn_3, criterion)
    rnn_4_train_losses, rnn_4_val_losses = train(rnn_4, num_epochs, train_loader, val_loader, optimizer_rnn_4, criterion)

    # Pad the list of losses with 0s for early stopping
    rnn_1_train_losses = np.pad(rnn_1_train_losses, (0, num_epochs-len(rnn_1_train_losses)), mode='constant', constant_values=np.nan)
    rnn_2_train_losses = np.pad(rnn_2_train_losses, (0, num_epochs-len(rnn_2_train_losses)), mode='constant', constant_values=np.nan)
    rnn_3_train_losses = np.pad(rnn_3_train_losses, (0, num_epochs-len(rnn_3_train_losses)), mode='constant', constant_values=np.nan)
    rnn_4_train_losses = np.pad(rnn_4_train_losses, (0, num_epochs-len(rnn_4_train_losses)), mode='constant', constant_values=np.nan)
    rnn_1_val_losses = np.pad(rnn_1_val_losses, (0, num_epochs-len(rnn_1_val_losses)), mode='constant', constant_values=np.nan)
    rnn_2_val_losses = np.pad(rnn_2_val_losses, (0, num_epochs-len(rnn_2_val_losses)), mode='constant', constant_values=np.nan)
    rnn_3_val_losses = np.pad(rnn_3_val_losses, (0, num_epochs-len(rnn_3_val_losses)), mode='constant', constant_values=np.nan)
    rnn_4_val_losses = np.pad(rnn_4_val_losses, (0, num_epochs-len(rnn_4_val_losses)), mode='constant', constant_values=np.nan)

    # Evaluate the models
    rnn_1_roc_data = evaluate(rnn_1, test_loader, test_dataset, criterion)
    rnn_2_roc_data = evaluate(rnn_2, test_loader, test_dataset, criterion)
    rnn_3_roc_data = evaluate(rnn_3, test_loader, test_dataset, criterion)
    rnn_4_roc_data = evaluate(rnn_4, test_loader, test_dataset, criterion)

    # Plot the training convergence
    df = pd.DataFrame({'Num Epochs':np.arange(1,num_epochs+1),
                       'Model 29 Train Loss':rnn_1_train_losses, 'Model 29 Val Loss':rnn_1_val_losses,
                       'Model 30 Train Loss':rnn_2_train_losses, 'Model 30 Val Loss':rnn_2_val_losses,
                       'Model 31 Train Loss':rnn_3_train_losses, 'Model 31 Val Loss':rnn_3_val_losses,
                       'Model 32 Train Loss':rnn_4_train_losses, 'Model 32 Val Loss':rnn_4_val_losses})
    for col in df.columns[1:]:
        sns.lineplot(x='Num Epochs', y=col, data=df, label=col)
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.title('Training Convergence Plot')
    plt.legend()
    plt.show()

    # Plot the ROC curve
    plt.figure()  
    plt.plot(rnn_1_roc_data[0], rnn_1_roc_data[1], label='Model 29 (area = %0.2f)' % rnn_1_roc_data[2])
    plt.plot(rnn_2_roc_data[0], rnn_2_roc_data[1], label='Model 30 (area = %0.2f)' % rnn_2_roc_data[2])
    plt.plot(rnn_3_roc_data[0], rnn_3_roc_data[1], label='Model 31 (area = %0.2f)' % rnn_3_roc_data[2])
    plt.plot(rnn_4_roc_data[0], rnn_4_roc_data[1], label='Model 32 (area = %0.2f)' % rnn_4_roc_data[2])
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()    

if 0:
    print("GRU Model Deployment GRU 3 LR:0.01")
    # Define the hyperparameters
    input_size = 100  # Size of the input vectors (e.g., GloVe word embeddings)
    num_classes = 2  # Number of output classes (positive and negative)
    num_epochs = 15

    # Create an instance of the classifier
    rnn_1 = GRUClassifier(input_size, 128, num_classes)
    rnn_2 = GRUClassifier(input_size, 128, num_classes)
    rnn_3 = GRUClassifier(input_size, 256, num_classes)
    rnn_4 = GRUClassifier(input_size, 256, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn_1 = torch.optim.Adam(rnn_1.parameters(), lr=0.01)
    optimizer_rnn_2 = torch.optim.SGD(rnn_2.parameters(), lr=0.01)
    optimizer_rnn_3 = torch.optim.Adam(rnn_3.parameters(), lr=0.01)
    optimizer_rnn_4 = torch.optim.SGD(rnn_4.parameters(), lr=0.01)

    # Train the models
    rnn_1_train_losses, rnn_1_val_losses = train(rnn_1, num_epochs, train_loader, val_loader, optimizer_rnn_1, criterion)
    rnn_2_train_losses, rnn_2_val_losses = train(rnn_2, num_epochs, train_loader, val_loader, optimizer_rnn_2, criterion)
    rnn_3_train_losses, rnn_3_val_losses = train(rnn_3, num_epochs, train_loader, val_loader, optimizer_rnn_3, criterion)
    rnn_4_train_losses, rnn_4_val_losses = train(rnn_4, num_epochs, train_loader, val_loader, optimizer_rnn_4, criterion)

    # Pad the list of losses with 0s for early stopping
    rnn_1_train_losses = np.pad(rnn_1_train_losses, (0, num_epochs-len(rnn_1_train_losses)), mode='constant', constant_values=np.nan)
    rnn_2_train_losses = np.pad(rnn_2_train_losses, (0, num_epochs-len(rnn_2_train_losses)), mode='constant', constant_values=np.nan)
    rnn_3_train_losses = np.pad(rnn_3_train_losses, (0, num_epochs-len(rnn_3_train_losses)), mode='constant', constant_values=np.nan)
    rnn_4_train_losses = np.pad(rnn_4_train_losses, (0, num_epochs-len(rnn_4_train_losses)), mode='constant', constant_values=np.nan)
    rnn_1_val_losses = np.pad(rnn_1_val_losses, (0, num_epochs-len(rnn_1_val_losses)), mode='constant', constant_values=np.nan)
    rnn_2_val_losses = np.pad(rnn_2_val_losses, (0, num_epochs-len(rnn_2_val_losses)), mode='constant', constant_values=np.nan)
    rnn_3_val_losses = np.pad(rnn_3_val_losses, (0, num_epochs-len(rnn_3_val_losses)), mode='constant', constant_values=np.nan)
    rnn_4_val_losses = np.pad(rnn_4_val_losses, (0, num_epochs-len(rnn_4_val_losses)), mode='constant', constant_values=np.nan)

    # Evaluate the models
    rnn_1_roc_data = evaluate(rnn_1, test_loader, test_dataset, criterion)
    rnn_2_roc_data = evaluate(rnn_2, test_loader, test_dataset, criterion)
    rnn_3_roc_data = evaluate(rnn_3, test_loader, test_dataset, criterion)
    rnn_4_roc_data = evaluate(rnn_4, test_loader, test_dataset, criterion)

    # Plot the training convergence
    df = pd.DataFrame({'Num Epochs':np.arange(1,num_epochs+1),
                       'Model 33 Train Loss':rnn_1_train_losses, 'Model 33 Val Loss':rnn_1_val_losses,
                       'Model 34 Train Loss':rnn_2_train_losses, 'Model 34 Val Loss':rnn_2_val_losses,
                       'Model 35 Train Loss':rnn_3_train_losses, 'Model 35 Val Loss':rnn_3_val_losses,
                       'Model 36 Train Loss':rnn_4_train_losses, 'Model 36 Val Loss':rnn_4_val_losses})
    for col in df.columns[1:]:
        sns.lineplot(x='Num Epochs', y=col, data=df, label=col)
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.title('Training Convergence Plot')
    plt.legend()
    plt.show()

    # Plot the ROC curve
    plt.figure()  
    plt.plot(rnn_1_roc_data[0], rnn_1_roc_data[1], label='Model 33 (area = %0.2f)' % rnn_1_roc_data[2])
    plt.plot(rnn_2_roc_data[0], rnn_2_roc_data[1], label='Model 34 (area = %0.2f)' % rnn_2_roc_data[2])
    plt.plot(rnn_3_roc_data[0], rnn_3_roc_data[1], label='Model 35 (area = %0.2f)' % rnn_3_roc_data[2])
    plt.plot(rnn_4_roc_data[0], rnn_4_roc_data[1], label='Model 36 (area = %0.2f)' % rnn_4_roc_data[2])
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show() 