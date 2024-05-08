import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    df = pd.DataFrame(ratings_bins, columns=['Rating'])
    df.insert(1, "Frequency", frequency)
    df.sort_values('Rating')
    print(df)
    sns.displot(data=df, kde=True, x="Rating")


# Get the current working directory
current_dir = os.getcwd()
train_pos_examples = os.path.join(current_dir, 'data', 'train', 'pos')
train_neg_examples = os.path.join(current_dir, 'data', 'train', 'neg')
test_pos_examples = os.path.join(current_dir, 'data', 'test', 'pos')
test_neg_examples = os.path.join(current_dir, 'data', 'test', 'neg')

find_ratings_skew((train_pos_examples, test_pos_examples), "Positive")
find_ratings_skew((train_neg_examples, test_neg_examples), "Negative")