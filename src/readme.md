# results

## first
- single bi-lstm, one linear layer, .5 dropout
	- 60% accuracy score
- single bi-lstm, two linear, .5 dropout
- use pretrained glove/word2vec word vectors
	- load vectors 
	```python
	def load_glove_vectors(glove_file="./data/glove.6B/glove.6B.50d.txt"):
	    """Load the glove word vectors"""
	    word_vectors = {}
	    with open(glove_file) as f:
		for line in f:
		    split = line.split()
		    word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
	    return word_vectors
	```
	- get vector rep from text
	```python 
		def get_emb_matrix(pretrained, word_counts, emb_size = 50):
	    """ Creates embedding matrix from word vectors"""
	    vocab_size = len(word_counts) + 2
	    vocab_to_idx = {}
	    vocab = ["", "UNK"]
	    W = np.zeros((vocab_size, emb_size), dtype="float32")
	    W[0] = np.zeros(emb_size, dtype='float32') # adding a vector for padding
	    W[1] = np.random.uniform(-0.25, 0.25, emb_size) # adding a vector for unknown words 
	    vocab_to_idx["UNK"] = 1
	    i = 2
	    for word in word_counts:
		if word in word_vecs:
		    W[i] = word_vecs[word]
		else:
		    W[i] = np.random.uniform(-0.25,0.25, emb_size)
		vocab_to_idx[word] = i
		vocab.append(word)
		i += 1   
	    return W, np.array(vocab), vocab_to_idx
	```
