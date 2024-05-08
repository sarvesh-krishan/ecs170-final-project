import os
import numpy as np

def load_data(directory, label):
    examples = []
    char_lengths = []
    word_lengths = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()

                examples.append(content)
                
                char_length = len(content)
                char_lengths.append(char_length)
                
                word_length = len(content.split())
                word_lengths.append(word_length)

    return examples, char_lengths, word_lengths

batch_size = 512
current_dir = os.getcwd()

train_pos_examples, train_char_lengths_pos, train_word_lengths_pos = load_data(os.path.join(current_dir, 'data', 'train', 'pos'), 'positive')
train_neg_examples, train_char_lengths_neg, train_word_lengths_neg = load_data(os.path.join(current_dir, 'data', 'train', 'neg'), 'negative')
test_pos_examples, test_char_lengths_pos, test_word_lengths_pos = load_data(os.path.join(current_dir, 'data', 'test', 'pos'), 'positive')
test_neg_examples, test_char_lengths_neg, test_word_lengths_neg = load_data(os.path.join(current_dir, 'data', 'test', 'neg'), 'negative')

positive_reviews_characters = train_char_lengths_pos + test_char_lengths_pos
positive_reviews_words = train_word_lengths_pos + test_word_lengths_pos
negative_reviews_characters = train_char_lengths_neg + test_char_lengths_neg
negative_reviews_words = train_word_lengths_neg + test_word_lengths_neg


# Calculate the 5-number summary for character lengths
char_summary = np.percentile(positive_reviews_characters, [0, 25, 50, 75, 100])
print("5-Number Summary for Characters (Positive Reviews)")
print(f"Min: {char_summary[0]}")
print(f"Q1: {char_summary[1]}")
print(f"Median: {char_summary[2]}")
print(f"Q3: {char_summary[3]}")
print(f"Max: {char_summary[4]}")

# Calculate the 5-number summary for words-positive reviews
word_summary = np.percentile(positive_reviews_words, [0, 25, 50, 75, 100])
print("\n5-Number Summary for Words (Positive Reviews):")
print(f"Min: {word_summary[0]}")
print(f"Q1: {word_summary[1]}")
print(f"Median: {word_summary[2]}")
print(f"Q3: {word_summary[3]}")
print(f"Max: {word_summary[4]}")

# Calculate the 5-number summary for characters-positive reviews
char_summary = np.percentile(positive_reviews_characters, [0, 25, 50, 75, 100])
print("5-Number Summary for Characters (Positive Reviews)")
print(f"Min: {char_summary[0]}")
print(f"Q1: {char_summary[1]}")
print(f"Median: {char_summary[2]}")
print(f"Q3: {char_summary[3]}")
print(f"Max: {char_summary[4]}")

# Calculate the 5-number summary for words-negative reviews
word_summary = np.percentile(negative_reviews_words, [0, 25, 50, 75, 100])
print("\n5-Number Summary for Words (Negative Reviews):")
print(f"Min: {word_summary[0]}")
print(f"Q1: {word_summary[1]}")
print(f"Median: {word_summary[2]}")
print(f"Q3: {word_summary[3]}")
print(f"Max: {word_summary[4]}")

# Calculate the 5-number summary for characters-negative reviews
char_summary = np.percentile(negative_reviews_characters, [0, 25, 50, 75, 100])
print("5-Number Summary for Characters (Negative Reviews)")
print(f"Min: {char_summary[0]}")
print(f"Q1: {char_summary[1]}")
print(f"Median: {char_summary[2]}")
print(f"Q3: {char_summary[3]}")
print(f"Max: {char_summary[4]}")
