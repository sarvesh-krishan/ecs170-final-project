from torch.utils.data import DataLoader
import torch
from torch import nn
from processing_data import load_data, CustomDataset, glove, collate_fn
# from model import RNNClassifier, train, evaluate


if 1:
    batch_size = 512

    train_pos_examples = load_data('../data/train/pos', 'positive')
    train_neg_examples = load_data('../data/train/neg', 'negative')
    test_pos_examples = load_data('../data/test/pos', 'positive')
    test_neg_examples = load_data('../data/test/neg', 'negative')

    train_examples = train_pos_examples + train_neg_examples
    test_examples = test_pos_examples + test_neg_examples

    train_dataset = CustomDataset(train_examples, glove)
    test_dataset = CustomDataset(test_examples, glove)

    # Use the collate_fn in your DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(train_examples[0])

    # # Define the hyperparameters

    # input_size = 100  # Size of the input vectors (e.g., GloVe word embeddings)
    # hidden_size = 128  # Size of the hidden state in the RNN
    # num_classes = 2  # Number of output classes (positive and negative)
    # num_epochs = 15

    # # Create an instance of the classifier
    # rnn_classifier = RNNClassifier(input_size, hidden_size, num_classes)

    # # Define the loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer_rnn = torch.optim.Adam(rnn_classifier.parameters(), lr=0.001)

    # # Train the RNN model
    # m1 = train(rnn_classifier, num_epochs, train_loader, optimizer_rnn, criterion)

    # # Evaluate the RNN model
    # evaluate(rnn_classifier, test_loader, test_dataset, criterion)

