import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import pandas as pd
from processing_data import load_data, CustomDataset, glove, collate_fn
from model import RNNClassifier,BiRNNClassifier, train, evaluate
from torchtext.data.utils import get_tokenizer
from collections import Counter
from histogram import generate_histogram
from convergence import plot_convergence


# LOADING DATA
batch_size = 512

# Get the current working directory
current_dir = os.getcwd()
train_pos_examples = load_data(os.path.join(current_dir, 'data', 'train', 'pos'), 'positive')
train_neg_examples = load_data(os.path.join(current_dir, 'data', 'train', 'neg'), 'negative')
test_pos_examples = load_data(os.path.join(current_dir, 'data', 'test', 'pos'), 'positive')
test_neg_examples = load_data(os.path.join(current_dir, 'data', 'test', 'neg'), 'negative')

positive_reviews = train_pos_examples + test_pos_examples
negative_reviews = train_neg_examples + test_neg_examples

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

# Most frequent words per class

if 1:
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

if 1:
    directory_path1 = os.path.join('data', 'test', 'neg')
    directory_path2 = os.path.join('data', 'test', 'pos')
    generate_histogram(directory_path1, directory_path2)

#five number summary 

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

# DEPLOYING THE MODELS

if 1:
    # Define the hyperparameters

    input_size = 100  # Size of the input vectors (e.g., GloVe word embeddings)
    hidden_size = 128  # Size of the hidden state in the RNN
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
    optimizer_rnnA1 = torch.optim.Adam(rnn_classifier.parameters(), lr=0.01)
    optimizer_rnnA2 = torch.optim.Adam(rnn_classifier.parameters(), lr=0.001)
    optimizer_rnnA3 = torch.optim.Adam(rnn_classifier.parameters(), lr=0.0001)
    optimizer_rnnS1 = torch.optim.SGD(rnn_classifier.parameters(), lr=0.01)
    optimizer_rnnS2 = torch.optim.SGD(rnn_classifier.parameters(), lr=0.001)
    optimizer_rnnS3 = torch.optim.SGD(rnn_classifier.parameters(), lr=0.0001)

    # Train the RNN model
    m1 = train(rnn_classifier, num_epochs, train_loader, val_loader, optimizer_rnnA1, criterion)
    m2 = train(rnn_classifier, num_epochs, train_loader, val_loader, optimizer_rnnA2, criterion)
    m3 = train(rnn_classifier, num_epochs, train_loader, val_loader, optimizer_rnnA3, criterion)
    m4 = train(rnn_classifier, num_epochs, train_loader, val_loader, optimizer_rnnS1, criterion)
    m5 = train(rnn_classifier, num_epochs, train_loader, val_loader, optimizer_rnnS2, criterion)
    m6 = train(rnn_classifier, num_epochs, train_loader, val_loader, optimizer_rnnS3, criterion)

    # Evaluate the RNN model
    evaluate(rnn_classifier, test_loader, test_dataset, criterion)

    # Plot Convergence
    model_losses = {
    'Model 1': m1,
    'Model 2': m2,
    'Model 3': m3,
    'Model 4': m4,
    'Model 5': m5,
    'Model 6': m6
}
    plot_convergence(list(model_losses.values()), list(model_losses.keys()))