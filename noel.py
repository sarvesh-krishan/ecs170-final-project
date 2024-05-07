import os
from torch.utils.data import DataLoader
import torch
from torch import nn
from processing_data import load_data, CustomDataset, glove, collate_fn
from model import RNNClassifier, train, evaluate

from torchtext.data.utils import get_tokenizer
from collections import Counter

if 1:
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


    # Define the hyperparameters

    input_size = 100  # Size of the input vectors (e.g., GloVe word embeddings)
    hidden_size = 128  # Size of the hidden state in the RNN
    num_classes = 2  # Number of output classes (positive and negative)
    num_epochs = 1

    # Create an instance of the classifier
    rnn_classifier = RNNClassifier(input_size, hidden_size, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn = torch.optim.Adam(rnn_classifier.parameters(), lr=0.001)

    # Train the RNN model
    m1 = train(rnn_classifier, num_epochs, train_loader, optimizer_rnn, criterion)
    
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

    # Evaluate the RNN model
    evaluate(rnn_classifier, test_loader, test_dataset, criterion)

    