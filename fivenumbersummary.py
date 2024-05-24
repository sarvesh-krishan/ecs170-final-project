import os
import numpy as np
import pandas as pd
from processing_data import load_data

current_dir = os.getcwd()
train_pos_examples = load_data(os.path.join(current_dir, 'data', 'train', 'pos'), 'positive')
train_neg_examples = load_data(os.path.join(current_dir, 'data', 'train', 'neg'), 'negative')
test_pos_examples = load_data(os.path.join(current_dir, 'data', 'test', 'pos'), 'positive')
test_neg_examples = load_data(os.path.join(current_dir, 'data', 'test', 'neg'), 'negative')

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




