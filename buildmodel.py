from gensim.models import Word2Vec
import sys
import data_helpers
import numpy as np

# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
x_text, y = data_helpers.load_data_and_labels('./data/rt-polaritydata/rt-polarity.pos', './data/rt-polaritydata/rt-polarity.neg')

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
x_words=[]
for ll in range( len(x_text) ):
  x_words.append( x_text[ll].split(' ') )
   
print ( 'x_testSize= {0} Y Size= {1} MaxDocLength {2} length x_words {3}'.format( len(x_text), y.shape, max_document_length, len(x_words) ) )
for ll in range(10):
   print( 'item {0} <{1}>'.format( ll, x_text[ll] ))
#sentences=x_text
# train model
#model = Word2Vec(sentences, size= xdim, min_count=1)
xdim= 100
model = Word2Vec(x_words, size= xdim, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
#print(words)
# access vector for one word
#print(model['sentence'])
print('Corresponding to word beauty\n{0}\n'.format(model['beauty']) )
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)
print ('Vocab Size {0} Dim {1} '.format( len(model.wv.vocab), xdim))
#print(model['beauty'])
