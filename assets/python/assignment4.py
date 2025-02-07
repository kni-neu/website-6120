import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

class SpecialTokens:
    """ 
    Class of special tokens
    """
    def __init__(self, start_token = "<s>", end_token = "<e>", unknown_token = "<unk>"):
        self.start_token = start_token
        self.end_token = end_token
        self.unknown_token = unknown_token

#@title Question 1

def preprocess_data(filename, count_threshold, special_tokens,
                    sample_delimiter='\n', split_ratio=0.8):
    """
    Preprocess data, i.e.,
        - Find tokens that appear at least N times in the training data.
        - Replace tokens that appear less than N times by "<unk>" .
    Args:
        count_threshold: Words whose count is less than this are
                      treated as unknown.

    Returns:
        training_data = list of lists denoting tokenized sentence. This looks like
                        the following:
 
                        [ ["this", "<unk>", "example"], 
                          ["another", "sentence", "<unk>", "right"],
                         ...
                        ] 
        test_data = Same format as above.
        vocabulary = list of vocabulary words. This looks like the following:

                        ["vocab-word-1", "vocab-word-2", etc.]
    """

    # Create sentences and tokenize the data to create a list of strings. 
    tokenized_data = read_and_tokenize_sentences(filename, sample_delimiter)

    # Create the training / test splits
    train_size = int(len(tokenized_data) * split_ratio)
    train_data = tokenized_data[0:train_size]
    test_data = tokenized_data[train_size:]

    # Get the closed vocabulary using the train data
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)

    # For the train data, replace less common words with "<unk>"
    train_data_replaced = replace_oov_words_by_unk(
        train_data, vocabulary, unknown_token)

    # For the test data, replace less common words with "<unk>"
    test_data_replaced = replace_oov_words_by_unk(
        test_data, vocabulary, unknown_token)

    return train_data_replaced, test_data_replaced, vocabulary

#@title Q1.1 Read / Tokenize Data from Sentences

def read_and_tokenize_sentences(filename, sample_delimiter="\n"):
    '''
    Input is a filename (e.g., "en_US.twitter.txt"

    Example usage: 
       $ read_and_tokenize_sentences(sentences) 

       [['sky', 'is', 'blue', '.'],
        ['leaves', 'are', 'green'],
        ['roses', 'are', 'red', '.']]A

    You can use nltk's tokenize function here.

       nltk.word_tokenize(sentence)
    '''
    return None

def get_words_with_nplus_frequency(train_data, count_threshold):
    return None

#@title Q1.2 Replace OOV Words with Special Token

def replace_oov_words_by_unk(data, vocabulary, unknown_token="<unk>"):
    return None

#@title Q2 Count N-Grams

def count_n_grams(data, n, special_tokens):
    """
    Count all n-grams in the data

    Args:
        data: List of lists of words
        n: Number of words in a sequence
        special_tokens: A structure that contains:
          - start_token = "<s>"
          - end_token = "<e>"
          - unknown_token = "unk"

    Returns:
        A dictionary that maps a tuple of n-words to its frequency
    """

    # Initialize dictionary of n-grams and their counts
    n_grams = {}
    # <YOUR-CODE-HERE>
    return n_grams

#@title Q3 Estimate the Probabilities

def estimate_probabilities(context_tokens, ngram_model):
    """
    Estimate the probabilities of a next word using the n-gram counts
    with k-smoothing

    Args:
        word: next word
        previous_n_gram: A sequence of words of length n
        ngram_model: a structure that contains:
            - n_gram_counts: Dictionary of counts of n-grams
            - n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
            - vocabulary_size: number of words
            - k: positive constant, smoothing parameter

    Returns:
        A dictionary mapping from next words to probability
    """
    probabilities = {}
    # <YOUR-CODE-HERE>
    return probabilities

#@title Q4 Inference

def predict_next_word(sentence_beginning, model):
  '''
  Args:
    sentence_beginning: a string
    model: an NGramModel object

  Returns:
    a string with the next word that his most likely to appear after the 
    sentence_beginning input using the define model
  '''
  # <YOUR-CODE-HERE>
  return None

#@title Q5 Extra Credit

class StyleGram:

    def __init__(self, style_files):
        """
        We will only be passing style_files in. All your processing and 
        training should be done by the time this function retunrs.
        """
        self.style_files = style_files
        # <YOUR-CODE-HERE>
        return

    def write_in_style_ngram(self, passage):
        """
        Takes a passage in, matches it with a style, given a list of
        filenames, and predicts the next word that will appear
        using a bigram model. 
            
        Args:
            passage: A string that contains a passage
            style_file: a list of filenames to be used to determine the style
            
        Returns:
             single word <string>
             probability associated with the word <float>
             index of "style" it originated from (e.g., 0 for 1st file) <int8>
             probability associated with the style <float>
        """
      # <YOUR-CODE_HERE>
      return word, probability_word, style_file, probability_style
