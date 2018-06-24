#If you want to create the word embeddings yourself, follow the tutorial on https://github.com/dalab/lecture_cil_public/tree/master/exercises/ex6
#Keep compute_embedding, if you want to use mine

#TODO
#Check which embedding to use xs or ys
#Check if logistic regression has a regularizer
#Train the embeddings with the full data set (so far only done with the small one)
#Maybe try a higher embedding dimension
#What to do if we don't have an embedding for all the words in the tweet


#!/usr/bin/env python3
from scipy.sparse import * 
import numpy as np
import pickle
import random
import sys
from sklearn.linear_model import LogisticRegression

#Parameters
compute_embedding = False
validation_data_size = 10000
embedding_dim = 20
epochs = 20
full_data_suffix = '' #To decide if we use the full data for training, for full data specify: '_full'
selected_embedding = 'arr_0' # Choose 'arr_1' for ys
output_file_name = 'results.csv'

#Code taken from the tutorial
def compute_glove_embeddings():
    print("loading cooccurrence matrix")
    with open('cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings");
    print("cooc shape 0: ", cooc.shape[0], "cooc shape 1: ", cooc.shape[1])
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    np.savez('embeddings', xs, ys)

def read_input_data(filename, matrix_list, embeddings, vocab, is_test_data, label):
    with open(filename, 'r+', encoding="utf-8") as f: #todo do the same for the covariance matrix
        for line in f:
            tokens = [vocab.get(t, -1) for t in line.strip().split()]
            tokens = [t for t in tokens if t >= 0]
            #at the moment: if we don't have an embedding for any of the words in the tweet -> don't use them for training (are not that many so far)
            if(len(tokens) > 0):
                result = np.zeros(embedding_dim)
                for t in tokens:
                    result += embeddings[t]
                result = result / len(tokens)
                if not is_test_data:
                    result = np.append(result, label)
                matrix_list.append(result)
            elif is_test_data:
                matrix_list.append(np.zeros(embedding_dim))
                print('Did not find this an embedding for any of the words')

def main():
    if compute_embedding:
        print('Compute embeddings')
        compute_glove_embeddings()
    #compute a mapping from the word to the index (to be able to get the right embedding for a given word)
    vocab = dict()
    with open('vocab_cut.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx
    npzfile = np.load('embeddings.npz')

    xs = npzfile[selected_embedding]
    matrix_list = []
    print('Reading input')
    read_input_data('train_neg' + full_data_suffix + '.txt', matrix_list, xs, vocab, False, 0)
    read_input_data('train_pos' + full_data_suffix + '.txt', matrix_list, xs, vocab, False, 1)
    input_data = np.array(matrix_list)
    print(input_data.shape)
    np.random.seed(42)
    np.random.shuffle(input_data)
    logisticRegr = LogisticRegression()
    validation_data = input_data[0:validation_data_size,0:embedding_dim]
    validation_labels = input_data[0:validation_data_size,embedding_dim:]
    training_data = input_data[validation_data_size:,0:embedding_dim]
    training_labels = input_data[validation_data_size:,embedding_dim:]
    logisticRegr.fit(training_data,training_labels)

    print(logisticRegr.score(validation_data, validation_labels))
    test_data = []
    read_input_data('test_data.txt', test_data, xs, vocab, True, 0)
    to_predict_data = np.array(test_data)  
    print(to_predict_data.shape)    
    predicted_results = logisticRegr.predict(to_predict_data)
    predicted_results[predicted_results == 0] = -1
    #Concat IDs with result
    output = np.concatenate((np.expand_dims(np.arange(1,len(predicted_results)+1),0),np.expand_dims(predicted_results,0)), axis=0).T
    #Save to file
    np.savetxt(output_file_name, output, delimiter=',', fmt = "%d,%d", header="Id,Prediction", comments='')

##TODO Read in the data by words
##For each word check the word embedding
##Average the embeddings
##Do logistic regression with sklearn based on the input

if __name__ == '__main__':
    main()
