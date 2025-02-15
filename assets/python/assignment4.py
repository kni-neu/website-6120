import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.data.path.append('.')

class SpecialTokens:
    """ 
    Class of special tokens
    """
    def __init__(self, start_token = "<s>", end_token = "<e>", unknown_token = "<unk>"):
        self.start_token = start_token
        self.end_token = end_token
        self.unknown_token = unknown_token

class NGramModel():
    """
    This class holds your n-gram model and all its parameters, which include:

      - previous_n_gram: the last n-gram that appeared prior to the current one
      - n-gram_counts: a dictionary specifying the context words
    """
    def __init__(self, n_gram_counts,
               n_plus1_gram_counts, vocabulary, special_tokens, k=1):
        self.n_gram_counts = n_gram_counts
        self.n_plus1_gram_counts = n_plus1_gram_counts
        self.vocabulary = vocabulary
        self.special_tokens = special_tokens
        self.k = k

#@title Question 1

def preprocess_data(filename, count_threshold, special_tokens,
                    sample_delimiter='\n', split_ratio=0.8):
    """
    Ungraded: You do not need to change this function.

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

    # For the train data, replace less common words with unknown token
    train_data_replaced = replace_oov_words_by_unk(
        train_data, vocabulary, unknown_token = special_tokens.unknown_token)

    # For the test data, replace less common words with "<unk>"
    test_data_replaced = replace_oov_words_by_unk(
        test_data, vocabulary, unknown_token = special_tokens.unknown_token)

    return train_data_replaced, test_data_replaced, vocabulary

def preprocess_data_test():
    """
    Ungraded: You can use this function to test out preprocess_data. 
    """
    tmp_train = "the sky is blue.\nleaves are green.\nsmell all the roses."
    tmp_test = "roses are red."

    with open('tmp_data.txt', 'w') as f:
      f.write(str(tmp_train) + '\n')
      f.write(str(tmp_test) + '\n')

    special_tokens = SpecialTokens()
    count_threshold = 1

    tmp_train_repl, tmp_test_repl, tmp_vocab = preprocess_data(
        "tmp_data.txt", count_threshold, special_tokens, split_ratio = 0.75)

    assert tmp_test_repl == [['roses', 'are', '<unk>', '.']] or \
      tmp_test_repl == [[special_tokens.start_token, 
                         'roses', 'are', '<unk>', 
                         special_tokens.end_token]] or \
      tmp_test_repl == [[special_tokens.start_token, 
                         'roses', 'are', '<unk>', '.',
                         special_tokens.end_token]], \
      print("tmp_test_repl is not correct")

    assert tmp_train_repl == [['the', 'sky', 'is', 'blue', '.'],
                              ['leaves', 'are', 'green', '.'],
                              ['smell', 'all', 'the', 'roses', '.']] or \
           tmp_train_repl == [[special_tokens.start_token, 
                               'the', 'sky', 'is', 'blue', 
                               special_tokens.end_token],
                              [special_tokens.start_token, 
                               'leaves', 'are', 'green', 
                               special_tokens.end_token],
                              [special_tokens.start_token, 
                               'smell', 'all', 'the', 'roses', 
                               special_tokens.end_token]] or \
           tmp_train_repl == [[special_tokens.start_token, 
                               'the', 'sky', 'is', 'blue', '.',
                               special_tokens.end_token],
                              [special_tokens.start_token, 
                               'leaves', 'are', 'green', '.',
                               special_tokens.end_token],
                              [special_tokens.start_token, 
                               'smell', 'all', 'the', 'roses', '.',
                               special_tokens.end_token]], \
      print("tmp_train_repl is not correct")

    print("\033[92m Successful test")

    return 

#@title Q1.1 Read / Tokenize Data from Sentences

def read_and_tokenize_sentences(filename, sample_delimiter="\n"):
    """
    Args:
        - filename = (e.g., "en_US.twitter.txt")
        - sample_delimiter = delimits each sample (i.e., each tweet)

    Example usage: 
       $ read_and_tokenize_sentences(sentences) 

       [['sky', 'is', 'blue', '.'],
        ['leaves', 'are', 'green'],
        ['roses', 'are', 'red', '.']]A

    You can use nltk's tokenize function here.

       nltk.word_tokenize(sentence)
    """
    # <YOUR-CODE-HERE>
    return None

def get_words_with_nplus_frequency(train_data, count_threshold):
    # <YOUR-CODE-HERE>
    return None

#@title Q1.2 Replace OOV Words with Special Token

def replace_oov_words_by_unk(data, vocabulary, unknown_token="<unk>"):
    # <YOUR-CODE-HERE>
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

def count_n_grams_test():
    """
    Ungraded: You can use this function to test out count_n_grams. 
    """
    tmp_data = "i like a cat\nthis dog is like a cat"
    with open('tmp_data.txt', 'w') as f:
      f.write(tmp_data + '\n')

    sentences, _, _ = preprocess_data(
        "tmp_data.txt", 0, SpecialTokens(), split_ratio = 1.0)

    received = count_n_grams(sentences, 2, SpecialTokens())
    expected = { ('<s>', 'i'): 1,
      ('i', 'like'): 1, ('like', 'a'): 2, ('a', 'cat'): 2, ('cat', '<e>'): 2,
      ('<s>', 'this'): 1, ('this', 'dog'): 1, ('dog', 'is'): 1, ('is', 'like'): 1}

    assert received == expected, print("Received: \n", received, 
                                       "\n\nExpected: \n", expected)

    print("\033[92m Successful test")

    return

count_n_grams_test()

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


def estimate_probabilities_test():
    """
    Ungraded: You can use this function to test out estimate_probabilities. 
    """
    tmp_data = "i like a cat\nthis dog is like a cat"
    with open('tmp_data.txt', 'w') as f:
      f.write(tmp_data + '\n')

    sentences, _, vocabulary = preprocess_data(
        "tmp_data.txt", 0, SpecialTokens(), split_ratio = 1.0)

    # unique_words = list(set(sentences[0] + sentences[1]))
    unigram_counts = count_n_grams(sentences, 1, SpecialTokens())
    bigram_counts = count_n_grams(sentences, 2, SpecialTokens())

    ngram_model = NGramModel(unigram_counts, bigram_counts, vocabulary,
                             SpecialTokens(), k=1)
    
    expected = {'i': 0.09090909090909091, 'like': 0.09090909090909091, 
                'a': 0.09090909090909091, 'cat': 0.2727272727272727, 
                'this': 0.09090909090909091, 'dog': 0.09090909090909091, 
                'is': 0.09090909090909091, '<e>': 0.09090909090909091, 
                '<unk>': 0.09090909090909091}

    assert estimate_probabilities(["a"], ngram_model) == expected, \
      print("estimate_probabilities is not correct")
    
    print("\033[92m Successful test")

#@title Q4 Inference

def predict_next_word(sentence_beginning, model):
    """
    Args:
        sentence_beginning: a string
        model: an NGramModel object

    Returns:
        next_word = a string with the next word that his most likely to appear 
        after the sentence_beginning input based ont he model. (You do not need to 
        add in any top K or random sampling.)
        probability = corresponding probability of that word
    """
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


# Example Usage
if __name__ == "__main__":
    # Create an instance of your NGramModel here, using your training data
    special_tokens = SpecialTokens()
    count_threshold = 10
    train_data_replaced, test_data_replaced, vocabulary = preprocess_data(
        "data/en_US.twitter.txt", count_threshold, special_tokens
    )

    # n=2
    unigram_counts = count_n_grams(train_data_replaced, 1, special_tokens)
    bigram_counts = count_n_grams(train_data_replaced, 2, special_tokens)

    ngram_model = NGramModel(unigram_counts, bigram_counts, vocabulary,
                            special_tokens, k=1)

    partial_sentence = "i love"  # Example partial sentence
    predicted_word = predict_next_word(partial_sentence, ngram_model)
    print(f"The predicted next word for '{partial_sentence}' is: {predicted_word}")
