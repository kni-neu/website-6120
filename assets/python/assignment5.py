#@title Question 1

def process_data(filename, min_cnt, max_cnt, min_win = 5, min_letters = 3):
    '''
    Preprocesses and builds the distribution of words in sorted order 
    (from maximum occurrence to minimum occurrence) after reading the 
    file. Preprocessing will include filtering out:
    * words that have non-letters in them,
    * words that are too short (under minletters)
    
    Arguments:
    * filename: name of file
    * min_cnt: min occurrence of words to include
    * max_cnt: max occurrence of words to include
    * min_win: minimum number of words in a title after word filtering
    * min_letters: min length of words to include (3)
    
    Returns:
    * word_freqs: A sorted (max to min) list of tuples of form -
       [(word1, count1), (wordN, countN), ... (wordN, countN)]
    * dataset: A list of strings with OOV words removed -
       ["this is title 1", "this is title 2", ...]
    '''
    # <YOUR-CODE-HERE>
    return None, None

#@title Question 2.1

def create_adjacency(dataset, word2index, win = 10):
    '''
    Builds an adjacency matrix based on word co-occurrence within a window.

    Args:
        dataset: List of processed titles
        word2index: Dictionary mapping word to index
        win: The window size for co-occurrence.

    Returns:
        adjacency_matrix: A NumPy array representing the adjacency matrix.
    '''
    <YOUR-CODE-HERE>
    return None

#@title Question 2.2

def train_svd(adjacency_matrix, min_index = 3, max_index = 103):
    """
    Creates an embedding space using SVD on the adjacency matrix.

    Args:
        adjacency_matrix: The adjacency matrix.
        embedding_dim: The desired dimensionality of the embedding space.

    Returns:
        A NumPy array representing the embedding space (num_words x embedding_dim)
    """
    return None

#@title Question 3.1

def sample_bow(dataset, word2index, win=5):
    '''
    Randomly samples a title and a window within that title, returning
    one-hot and multi-hot vectors.

    Args:
        dataset: A list of preprocessed titles.
        word2index: A dictionary of words and their indices
        win: The size of the context window.

    Returns:
        content_vector: the one-hot vector for the center word
        context_vector: multi-hot vector for window around center
    '''
    return None, None

#@title Question 3.2

def train_bow(dataset, word2index, iters,
              win = 5, embedding_dim = 100, learning_rate=0.01):
    """
    Creates an embedding space using SVD on the adjacency matrix.

    Args:
        dataset: A list of preprocessed titles.
        word2index: Dictinoary assigning word to index
        win: The size of the context window for sampling.
        iters: Number of iterations to run for
        embedding_dim: The desired dimensionality of the embedding space.
        learning_rate: Learning rate or any other DNN params with defaults.
                       The autograder won't touch this.

    Returns:
        V_bow: an array representing the embedding space (num_words x
               embedding_dim)
        List of losses (to print out)
    """
    return None, None

#@title Question 4.1

def sample_w2v(data, word2index, neg_samples=5, win=10):
    '''
    Randomly samples a title and a window within that title, returning
    one-hot and multi-hot vectors.

    Args:
        dataset: A list of preprocessed titles.
        word2index: A dictionary of words and their indices
        neg_samples: Number of negative samples
        win: The size of the context window.

    Returns:
        wi - target vector index
        wo - context vector index
        Wn - negative vectors index
    '''
    return None, None, None

#@title Question 4.3

def w2vgrads( vi, vo, Vns ):
  """
  This function implements the gradient for all vectors in
  input matrix Vi and output matrix Vo.

  Args:
    vi:  Vector of shape (d,), a sample in the input word
         vector matrix
    vo:  Vector of shape (d,), a positive sample in the output
         word vector matrix
    vns: Vector of shape (d, k), k negative samples in the
         output word vector matrix

  Returns:
    dvi, dvo, dVns: the gradients of J with respect to vi
                    and vo.
  """
  # dvi, dvo, dVns = <YOUR-CODE-HERE>
  return None, None, None

#@title Question 4.4

def train_w2v(dataset, word2index, iters, negsamps = 5,
              win = 5, embedding_dim = 100, learning_rate=0.01):
    """
    Creates an embedding space using SVD on the adjacency matrix.

    Args:
        dataset: A list of preprocessed titles.
        word2index: Dictinoary assigning word to index
        iters: Number of iterations to run for
        negsamps: Number of negative samples
        win: The size of the context window for sampling.
        embedding_dim: The desired dimensionality of the embedding space.
        learning_rate: Learning rate or any other DNN params with defaults.
                       The autograder won't touch this.

    Returns:
        V_w2v: an array representing the embedding space (num_words x
               embedding_dim)
        List of losses (to print out)
    """
    return None, None

#@title Save Models / Word Vectors

def save_models(word_freqs, V_svd, V_bow, V_w2v):

    '''
    Save all the appropriate data. We are expecting the variables
    shown below.
    '''
    data = {
        'word_freqs': word_freqs,
        'V_svd': V_svd,
        'V_bow': V_bow,
        'V_w2v': V_w2v
    }
    
    with open('assignment5.pkl', 'wb') as f:
        pickle.dump(data, f)

