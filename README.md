# ECS 170 Group 18 Final Project

By Tobias Cheung, Simona Kamat, Sarvesh Krishan, Noel Lee, Megan Phan, Tiffany Su, Heidi Trinh

The objective of this project was to build a machine learning model that would classify movie reviews from IMDB as positive or negative sentiment. We sought out to learn how to build a Recurrent Neural Network (RNN) that would provide us with a basic understanding of the applications of artificial intelligence in natural language processing and sentiment analysis.

Throughout the process of developing this project, we learned about working with data and how to design and optimize a Recurrent Neural Network.

Please run `script.py` to train and evaluate four (4) Recurrent Neural Network models. These include a base RNN model that performs poorly on the data set (approximately 54% testing accuracy), one LSTM model and two GRU models that perform exceptionally well on the data (approximately 80%+ testing accuracy).

### Dependencies

Please ensure the following libraries are installed on your local machine:
1. `numpy`: Imported as `np` to do numerical and array operations
2. `torch`: Imported to build and train our neural network
3. `seaborn`: Imported to supplement `matplotlib` to visualize data and create plots
4. `pandas`: Imported to do data manipulation (creating a dataframe of the model losses)
5. `matplotlib`: Imported as a dependency of `seaborn`
6. `os`: Imported to assist with file paths
7. `time`: Imported to compute run-time of training and evaluating models
8. `torchtext`: Imported to embed words as vectors to feed into the RNN
9. `sklearn`: Imported to utilize performance metrics functions, such as accuracy, precision, etc.

You will also need GloVe pre-trained word vectors that can be downloaded from [The Stanford Natural Language Processing Group](https://nlp.stanford.edu/projects/glove/) and placed into `./ecs170-final-project/data/.vector_cache`. The file is `glove.6B.100d.txt`.