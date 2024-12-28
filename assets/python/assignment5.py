
def build_word_distribution(filename, min_cnt, max_cnt, min_letters = 3):
  '''
  Preprocesses and builds the distribution of words in sorted order 
  (from maximum occurrence to minimum occurrence) after reading the 
  file. Preprocessing will include filtering out:
    * words that have non-letters in them,
    * words that are too short (under minletters)

  Arguments:
    * filename: name of file
    * min_cnt: min occurrence of words to include
    * cut_out: max occurrence of words to include
    * min_letters: min length of words to include

  Returns:
    * A list of tuples of form: 
       [(word1, count1), (wordN, countN), ... (wordN, countN)]
  '''
  #<YOUR-CODE-HERE>#
  return [(None, None), (None, None), (None, None)]


def build_adjacency_matrix(filename, vocabulary, win = 10):
    '''
    Builds an adjacency matrix based on word co-occurrence within a window.
    
    Args:
        filename: Path to the text file.
        vocabulary: List or set of valid words
        win: The window size for co-occurrence.
    
    Returns:
        
        adjacency_matrix: A NumPy array representing the adjacency matrix.
        word_to_index: A dictionary mapping words to their indices in the
                       matrix.
    '''
    #<YOUR-CODE-HERE>
    return None, None


def embeddings_svd(adjacency_matrix, min_index = 3, max_index = 103):
    """
    Creates an embedding space using SVD on the adjacency matrix.

    Args:
        adjacency_matrix: The adjacency matrix.
        embedding_dim: The desired dimensionality of the embedding space.

    Returns:
        A NumPy array representing the embedding space.
    """
    # <YOUR-CODE-HERE>
    return None
