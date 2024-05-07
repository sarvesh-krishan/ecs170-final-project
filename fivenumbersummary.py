import os
import numpy as np
from processing_data import load_data

def lengths(reviews):
    characters = [len(review) for review in reviews]
    words = [len(review.split()) for review in reviews]
    return characters, words

def five_number_summary(data):
    q1 =np.percentile(data,25) 
    median = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    mean = np.mean(data)
    return min(data), q1, median, q3, max(data), mean

if 1:
    batch_size = 512

    # Get the current working directory
    current_dir = os.getcwd()
    train_pos_examples = load_data(os.path.join(current_dir, 'data', 'train', 'pos'), 'positive')
    train_neg_examples = load_data(os.path.join(current_dir, 'data', 'train', 'neg'), 'negative')
    test_pos_examples = load_data(os.path.join(current_dir, 'data', 'test', 'pos'), 'positive')
    test_neg_examples = load_data(os.path.join(current_dir, 'data', 'test', 'neg'), 'negative')

positive_reviews = train_pos_examples + test_pos_examples
negative_reviews = train_neg_examples + test_neg_examples

# Calculate lengths
characters_pos, words_pos = lengths(positive_reviews)
characters_neg, words_neg = lengths(negative_reviews)

# Compute five-number summaries
characters_pos_summary = five_number_summary(characters_pos)
words_pos_summary = five_number_summary(words_pos)
characters_neg_summary = five_number_summary(characters_neg)
words_neg_summary = five_number_summary(words_neg)

# Print results
print("Five Number Summary(min, q1, median, q3, max, mean)")
print("Positive Reviews - Characters",characters_pos_summary)
print("Positive Reviews - Words",words_pos_summary)
print("Negative Reviews - Characters",characters_neg_summary)
print("Negative Reviews - Words",words_neg_summary)
