import numpy as np
import re
from gensim.models import Word2Vec
import numpy as np
import sys


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def buildAndGetModel(vocabulary, vector_size):
# define training data
    x_text, y = load_data_and_labels('./data/rt-polaritydata/rt-polarity.pos', './data/rt-polaritydata/rt-polarity.neg')

# Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    x_words=[]
    for ll in range( len(x_text) ):
      x_words.append( x_text[ll].split(' ') )
   
    print ( 'x_testSize= {0} Y Size= {1} MaxDocLength {2} length x_words {3}'.format( len(x_text), y.shape, max_document_length, len(x_words) ) )
    for ll in range(10):
       print( 'item {0} <{1}>'.format( ll, x_text[ll] ))
    # train model
    #model = Word2Vec(sentences, min_count=1)
    model = Word2Vec(x_words, size= vector_size, min_count=1, window=10, batch_words=128)
    
    model.train( x_words, total_examples=len(x_words), epochs=100, compute_loss=True)
    # summarize the loaded model
    print(model)
    # summarize vocabulary
    words = list(model.wv.vocab)
    #print(words)
    # access vector for one word
    #print(model['sentence'])
    print('Corresponding to word beauty\n{0}\n'.format(model['beauty']) )
    # save model
    model.save('w2v_rtpolmodel.bin')
    # load model
    #new_model = Word2Vec.load('model.bin')
    #print(new_model)
    #print(model['beauty'])
    print ('Vocab Size {0} Dim {1} Loss {2}'.format( len(model.wv.vocab), vector_size, model.get_latest_training_loss()))

    #embedding_vectors = np.random.uniform(-1.0, 1.0, (len(vocabulary), vector_size))
    embedding_vectors = np.zeros( (len(vocabulary), vector_size))
    mcount= 0
    for mword in words:
        idx = vocabulary.get(mword)
        if idx != 0:
           embedding_vectors[idx]= np.asarray( model[ mword], dtype='float32' )
           mcount +=1 
        else :
           print( 'Non existant word in Dict <{0}>'.format( mword) )

    print ('Successfully loaded models corresponding to {0} words out of {1} words, w2vLen {2} Shape {3} Max {4} Min {5}'.format(mcount, len(vocabulary), len(words), embedding_vectors.shape, np.amax(embedding_vectors), np.amin(embedding_vectors) ) )
    return embedding_vectors

def buildAndGetGloVeModel(vocabulary, vector_size):
    from gensim.scripts.glove2word2vec import glove2word2vec
    from gensim.models import KeyedVectors

    glove_input_file = 'glove.6B.100d.txt'
    word2vec_output_file = 'glove.6B.100d.txt.word2vec'
    glove2word2vec(glove_input_file, word2vec_output_file)


    # load the Stanford GloVe model
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    words = list(model.wv.vocab)
    print ('Number of words= {0} GMODEL {1}'.format( len(words), model) )

    #Now construct word embedding matrix
    embedding_vectors = np.zeros( (len(vocabulary), vector_size))
    missingW=0
    mcount= 0
    for mword in words:
        idx = vocabulary.get(mword)
        if idx != 0:
           wemb= model[ mword ]
           embedding_vectors[idx]= np.asarray( wemb, dtype='float32' )
           mcount +=1
        else :
           if( missingW < 20):
              print( 'Non existant word in Dict <{0}>'.format(mword)  )
           missingW += 1
    setMissing= (len(vocabulary)- mcount)
    print ('Successfully loaded GloVE corresponding to {0} words out of {1} words Missing {2}/{3} Shape {4} Max {5} Min {6}'.format(mcount, len(vocabulary), setMissing, missingW, embedding_vectors.shape, np.amax(embedding_vectors), np.amin(embedding_vectors) ) )
    #sys.exit()
    return embedding_vectors

         
