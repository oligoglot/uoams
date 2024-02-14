# Objectives

The learning objectives of this assignment are to:

1. obtain the numeric representation of neural network input and output
2. create, train and run a recurrent neural network 
3. initialize an embedding layer with pre-trained word embeddings

# Setup your environment and prepare data

First, carefully follow the *General Instructions for Programming Assignments*.

To install the libraries required for this assignment run:

    pip install -r requirements.txt

or in case you work with macOS:

    pip install -r requirements-macos.txt

Download and unzip GloVe embeddings:

    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip -d glove glove.6B.zip

# Grading

The grading distribution for this assignment is listed below:
- test_get_vocabulary = 16%
- test_format_examples = 16%
- test_create_model = 16%
- test_train_model = 16%
- test_make_predictions = 16%
- test_create_embedding_matrix = 10%
- test_create_model_with_embeddings = 10%