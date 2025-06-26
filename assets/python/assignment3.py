import re
from collections import Counter
import numpy as np
import pandas as pd
from utils import extract_featuers

nltk.download('twitter_samples')
nltk.download('stopwords')

#@title Q1 Process Data

def process_data(file_name):
    """
    Input:
        A file_name which is found in your current directory. You just have to read it in.
    Output:
        wordprobs: a dictionary where keys are all the processed lowercase words and the 
             values are the frequency that it occurs in the corpus (text file you read).
    """
    words = [] # return this variable correctly

    ### START CODE HERE ###

    #Open the file, read its contents into a string variable
    data = None

    # convert all letters to lower case
    lower_case_data = None

    # with words as keys, ensure that we have a count of their occurrences
    words = dict()

    ### END CODE HERE ###

    return words

def process_data_test():

    probs = process_data('./shakespeare-7k.txt')
    assert np.allclose(probs['thee'], 0.004476442720185026), \
                      "expected \"thee\" frequency of 0.004476442720185026"
    assert len(probs) == 6116, "vocabulary should have total of 6116 words.\n\t" + \
        f"length of vocabulary in your implementation is {len(probs)}"

    # Check maximum frequency word
    max_word = ""; max_freq = 0.0
    for word in probs:
      assert word == word.lower(), "all words must be lowercase.\n\t" + \
          f"word \"{word}\" is not lowercase"
      if probs[word] > max_freq:
        max_freq = probs[word]
        max_word = word
    assert max_word == 'the', "\"the\" should be the most frequent word\n\t" + \
        f"your maximum word is \"{max_word}\""
    assert np.allclose(max_freq, 0.028444063117842356)

    print("process_data: \033[1;32mtests OK.\033[0m")

#@title Q2: N-gram most likely next words

def probable_substitutes(word, probs, maxret = 10):
    """
    Determine the most probable words for a misspelled string that are TWO edits
    away. The edits that are possible are:

        * delete_letter: removing a character
        * switch_letters: switching two adjacent characters
        * replace_letter: replacing one character by another different one
        * insert_letter: inserting a character

    There may be fewer but no more than maxret words.

    Input:
        word - The misspelled word
        probs - A dictionary of word --> prob
        maxret - Maximum number of words to return
    Returns:
        Tuples of the words and their probabilities, ordered by highest frequency.
        [(word1, prob1), ... ]
    """
    # <YOUR-CODE-HERE>
    def example_function(word):
        """
        To make your code modular, feel free to add sub-functions (e.g., like the 
        one we've templated below).
        """
        return None

    return [("hello", 0.5), ("world", 0.3)]

def probable_substitutes_test():
    probs = dict()
    probs['raid'] = 0.25
    probs['rain'] = 0.1
    probs['raise'] = 0.1
    probs['void'] = 0.15
    probs['roid'] = 0.20
    probs['rein'] = 0.2

    # Check delete one add one
    probable_substitutes_result = probable_substitutes('braid', probs, maxret = 5)   
    probable_substitutes_expected = [('raid', 0.25), ('roid', 0.2), ('rain', 0.1)]
    assert probable_substitutes_result == probable_substitutes_expected, \
      str(probable_substitutes_expected) + "\n\tReceived" + str(probable_substitutes_result)
    
    # Check insert two
    probable_substitutes_result = probable_substitutes('id', probs, maxret = 5)   
    probable_substitutes_expected = [('raid', 0.25), ('roid', 0.2), ('void', 0.15)]
    assert probable_substitutes_result == probable_substitutes_expected, \
      str(probable_substitutes_expected) + "\n\tReceived" + str(probable_substitutes_result)

    print("probable_substitutes: \033[1;32mtests OK.\033[0m")

#@title Q3: The Minimum Edit Distnance

def min_edit_distance(source, target, ins_cost = 1, 
                      del_cost = 1, rep_cost = 2):
    '''
    Input:
        source: starting string
        target: ending string
        ins_cost: integer representing insert cost
        del_cost: integer representing delete cost
        rep_cost: integer representing replace cost
    Output:
        D: matrix of size (len(source)+1 , len(target)+1) 
           with minimum edit distances
        med: the minimum edit distance required to convert
             source to target
    '''
    # <YOUR-CODE-HERE>
    return D, med

def min_edit_distance_test():

    matrix, mindist = min_edit_distance("hello", "help", ins_cost = 1, del_cost = 1, rep_cost = 2)
    assert mindist == 3, "Expected distance of 3, but received: " + str(mindist)   

    matrix, mindist = min_edit_distance("hello", "jello", ins_cost = 1, del_cost = 1, rep_cost = 2)
    assert mindist == 2, "Expected distance of 2, but received: " + str(mindist)

    matrix, mindist = min_edit_distance("ello", "jello", ins_cost = 1, del_cost = 1, rep_cost = 2)
    assert mindist == 1, "Expected distance of 1, but received: " + str(mindist)

    print("min_edit_distance: \033[1;32mtests OK.\033[0m")

if __name__ == '__main__':

    print("Running unit tests ... ")
    process_data_test()
    probable_substitutes_test()
    min_edit_distance_test()

    print("Processing data, assuming that shakespeare-7k.txt is in working directory...")
    probs = process_data('./shakespeare-7k.txt')

    print("\n\nLet's try autocorrect. \"(Q)/(q)uit\" to end.")
    entry = input("Enter any word:")
    entry = entry.lower()
    while not entry == "q" and not entry == 'quit':
        print(probable_substitutes(entry, probs, maxret = 5))
        entry = input("Enter any word:")
        entry = entry.lower()

    print("\n\nLet's try min edit distance.")
    while not entry == 'n' and not entry == 'no':
        source = input("Enter any source word:")
        source = source.lower()
        target = input("Enter any target word:")
        target = target.lower()
        D, d = min_edit_distance(source, target)
        print(source, " and ", target, " are ", d, " apart")
        entry = input("Continue? (Y/y)es / (N/n)o:")
