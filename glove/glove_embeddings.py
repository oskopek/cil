#This code is used to run logistic regression or the random forest classifier on the embedding of a tweet.
#To get an embeddings for the whole tweet we are averaging glove word embeddings.

#The Stanford University already has precomputed glove word embeddings for twitter (https://nlp.stanford.edu/projects/glove/)
#We also have code to compute word embeddings ourselves and basically followed the tutorial from the CIL tutoril
#on https://github.com/dalab/lecture_cil_public/tree/master/exercises/ex6

import numpy as np
import pickle
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from numpy import genfromtxt

#Parameters
#use the standford word embedding
use_stanford = False

#if we want to compute the word embeddings ourselves, is only considered if use_stanford is false
compute_embedding = True
#how many dimensions the embedding has which should be computed
embedding_dim = 200
#how many epochs are used to compute the word embedding
epochs = 20
#which embedding from glove is chosen
selected_embedding = 'arr_0' #Choose 'arr_1' for ys
output_file_name = 'results.csv'
#We observed that the length of negative tweets is in general higher, if useLen is true we are using the "number of words"-information in ML models
useLen = False
#which classifier to use: True means to use logistic regression, False leads to using random forests
use_logistic_regression = True

#Method taken from the CIL tutorial https://github.com/dalab/lecture_cil_public/tree/master/exercises/ex6
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
        print("epoch " + str(epoch), flush=True)
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    np.savez('embeddings', xs, ys)


def read_input_data(filename, matrix_list, embeddings, vocab, is_test_data, labels = None):
    lineID = 0
    with open(filename, 'r+', encoding="utf-8") as f:
        for line in f:
            tokens = [vocab.get(t, -1) for t in line.strip().split()]
            if useLen:
                sentLen = len(tokens)
            tokens = [t for t in tokens if t >= 0]
            #At the moment: if we do not have an embedding for any of the words in the tweet -> do not use them for training (are not that many)
            #Compute the avrerage
            if(len(tokens) > 0):
                result = np.zeros(embedding_dim)
                for t in tokens:
                    result += embeddings[t]
                result = result / len(tokens)
                if useLen:
                    result = np.append(result, sentLen)
                if not is_test_data:
                   result = np.append(result, labels[lineID])
                matrix_list.append(result)
            elif is_test_data:
                zero_embedding = np.zeros(embedding_dim)
                if useLen:
                    zero_embedding = np.append(zero_embedding, 0)
                matrix_list.append(zero_embedding)
                print('Did not find this an embedding for any of the words')
            lineID = lineID + 1

def str_to_bool(string):
  return string.lower() in ("yes", "true", "t", "1")

def main():
    #Check if any command line parameter was added. If so, use stanford embeddings.
    global use_stanford
    global output_file_name
    global use_logistic_regression

    if (len(sys.argv) <= 1):
        use_stanford = True
        use_logistic_regression = False
        output_file_name = 'results_stanford_random_forests.csv'
    elif(len(sys.argv) == 3):
        use_logistic_regression = str_to_bool(sys.argv[1])
        use_stanford = str_to_bool(sys.argv[2])
        output_file_name = 'results_'
        if(use_stanford):
            output_file_name += 'stanford_'
        if(use_logistic_regression):
            output_file_name += 'logistic_regression.csv'
        elif:
            output_file_name += 'random_forests.csv'


    if compute_embedding and not use_stanford:
        print('Compute and save embeddings')
        compute_glove_embeddings()
        #Compute a mapping from the word to the index (to be able to get the right embedding for a given word)

    if use_stanford:
        print("Loading Stanford 200D Vocab...")
        stanford_label = genfromtxt('glove.twitter.27B.200d.txt', delimiter=' ', encoding="utf8", usecols=(0),dtype=None)
        print("Loading Stanford 200D Embeddings...")
        vocab = dict()
        for ii in range(stanford_label.shape[0]):
            vocab[stanford_label[ii]] = ii

        stanford_data = genfromtxt('glove.twitter.27B.200d.txt', delimiter=' ', encoding="utf8")
        stanford_data = stanford_data[:,1:]
        print(stanford_data.shape)

        glove_words_embeddings = stanford_data
        print("Embeddings Loaded")

    else:
        vocab = dict()
        with open('vocab_cut.txt') as f:
            for idx, line in enumerate(f):
                vocab[line.strip()] = idx
        npzfile = np.load('embeddings.npz')
        glove_words_embeddings = npzfile[selected_embedding]


    matrix_list = []
    print('Reading input')
    train_labels = genfromtxt('train_label.txt')
    read_input_data('train_sent.txt', matrix_list, glove_words_embeddings, vocab, False, train_labels) #label deactivated
    train_sentences = np.array(matrix_list)
    train_labels = train_sentences[:,-1]
    train_sentences = train_sentences[:, :-1]

    matrix_list = []
    eval_labels = genfromtxt('eval_label.txt')
    read_input_data('eval_sent.txt', matrix_list, glove_words_embeddings, vocab, False, eval_labels)  # label deactivated
    eval_sentences = np.array(matrix_list)
    eval_labels = eval_sentences[:, -1]
    eval_sentences = eval_sentences[:, :-1]

    if useLen:
        global embedding_dim
        embedding_dim = embedding_dim + 1
    print(train_sentences.shape)

    if use_logistic_regression:
        classifier = LogisticRegression()
    else:
        classifier = RandomForestClassifier(128, n_jobs = -1)

    classifier.fit(train_sentences, train_labels)
    print(classifier.score(eval_sentences, eval_labels))

    if useLen:
        embedding_dim = embedding_dim - 1
    test_data = []
    read_input_data('test_data.txt', test_data, glove_words_embeddings, vocab, True, None)
    to_predict_data = np.array(test_data)
    print(to_predict_data.shape)
    predicted_results = classifier.predict(to_predict_data)
    predicted_results[predicted_results == 0] = -1
    #Concat IDs with result
    output = np.concatenate((np.expand_dims(np.arange(1,len(predicted_results)+1),0),np.expand_dims(predicted_results,0)), axis=0).T
    #Save to file
    np.savetxt(output_file_name, output, delimiter=',', fmt = "%d,%d", header="Id,Prediction", comments='')

if __name__ == '__main__':
    main()