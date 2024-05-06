import os
import matplotlib.pyplot as plt

def generate_histogram(*directories):
    ratings = []
    for directory in directories:
        files = os.listdir(directory)
        file_names = [file.split('_', 1)[1].split('.')[0] for file in files if '_' in file and file.endswith('.txt')]
        ratings.extend(file_names)
    sorted_file_names = sorted(ratings, key=lambda x: int(x))

    plt.hist(sorted_file_names, bins=len(set(ratings)),color='blue', edgecolor='black')
    plt.xlabel('Movie Rating')
    plt.ylabel('Frequency')
    plt.title('Movie Rating Frequency')
    plt.show()


directory_path1 = os.path.join('data', 'test', 'neg')
directory_path2 = os.path.join('data', 'test', 'pos')
generate_histogram(directory_path1, directory_path2)
