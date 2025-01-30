# In a shell, download the data and extract to the base folder:
# !wget -nc https://course.ccs.neu.edu/cs6120s25/data/named-entities/ner-data.zip
# !unzip -n ner-data.zip
# Copy data/load_data into your source directory from ner-data.zip. If you do not, load_data will not import.

# Imports. You will not need any libraries beyond these.
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from load_data import load_data

#@title Question 1

def get_sentence_vectorizer(sentences):
    tf.keras.utils.set_random_seed(33) ## Do not change this line.
    """
    Create a TextVectorization layer for sentence tokenization and adapt it to the provided sentences.

    Parameters:
    sentences (list of str): Sentences for vocabulary adaptation.

    Returns:
    sentence_vectorizer (tf.keras.layers.TextVectorization): TextVectorization layer for sentence tokenization.
    vocab (list of str): Extracted vocabulary.
    """
    ### START CODE HERE ###

    # Define TextVectorization object with the appropriate standardize parameter
    sentence_vectorizer = None
    # Adapt the sentence vectorization object to the given sentences
    None.adapt(None)
    # Get the vocabulary
    vocab = None

    ### END CODE HERE ###

    return sentence_vectorizer, vocab

#@title Question 2

def label_vectorizer(labels, tag_map):
    """
    Convert list of label strings to padded label IDs using a tag mapping.

    Parameters:
    labels (list of str): List of label strings.
    tag_map (dict): Dictionary mapping tags to IDs.
    Returns:
    label_ids (numpy.ndarray): Padded array of label IDs.
    """
    label_ids = [] # It can't be a numpy array yet, since each sentence has a different size

    ### START CODE HERE ###

    # Each element in labels is a string of tags so for each of them:
    for element in None:
        # Split it into single tokens. You may use .split function for strings. Be aware to split it by a blank space!
        tokens = element.split(None)

        # Use the dictionaty tag_map passed as an argument to the label_vectorizer function
        # to make the correspondence between tags and numbers.
        element_ids = None

        for token in tokens:
            element_ids.append(None)

        # Append the found ids to corresponding to the current element to label_ids list
        label_ids.append(None)

    # Pad the elements
    label_ids = None

    ### END CODE HERE ###

    return label_ids

#@title Question 3.1

def NER(len_tags, vocab_size, embedding_dim = 50):
    """
    Create a Named Entity Recognition (NER) model.

    Parameters:
    len_tags (int): Number of NER tags (output classes).
    vocab_size (int): Vocabulary size.
    embedding_dim (int, optional): Dimension of embedding and LSTM layers (default is 50).

    Returns:
    model (Sequential): NER model.
    """

    ### START CODE HERE ###

    model = tf.keras.Sequential(name = 'sequential')
    # Add the tf.keras.layers.Embedding layer. Do not forget to mask out the zeros!
    model.add(None)
    # Add the LSTM layer. Make sure you are passing the right dimension (defined in the docstring above)
    # and returning every output for the tf.keras.layers.LSTM layer and not the very last one.
    model.add(None)
    # Add the final tf.keras.layers.Dense with the appropriate activation function. Remember you must pass the activation function itself ant not its call!
    # You must use tf.nn.log_softmax instead of tf.nn.log_softmax().
    model.add(None)

    ### END CODE HERE ###

    return model

#@title Question 3.2

def masked_loss(y_true, y_pred):
    """
    Calculate the masked sparse categorical cross-entropy loss.

    Parameters:
    y_true (tensor): True labels.
    y_pred (tensor): Predicted logits.

    Returns:
    loss (tensor): Calculated loss.
    """

    ### START CODE HERE ###

    # Calculate the loss for each item in the batch. Remember to pass the right arguments, as discussed above!
    loss = tf.keras.losses.SparseCategoricalCrossentropy 

    ### END CODE HERE ###

    return  loss

#@title Question 3.3

def masked_accuracy(y_true, y_pred):
    """
    Calculate masked accuracy for predicted labels.

    Parameters:
    y_true (tensor): True labels.
    y_pred (tensor): Predicted logits.

    Returns:
    accuracy (tensor): Masked accuracy.

    """
    
    ### START CODE HERE ### 
    
    # Calculate the loss for each item in the batch.
    # You must always cast the tensors to the same type in order to use them in training. Since you will make divisions, it is safe to use tf.float32 data type.
    y_true = tf.cast(y_true, tf.float32) 
    # Create the mask, i.e., the values that will be ignored
    mask = None
    mask = tf.cast(mask, tf.float32) 
    # Perform argmax to get the predicted values
    y_pred_class = None
    y_pred_class = tf.cast(y_pred_class, tf.float32) 
    # Compare the true values with the predicted ones
    matches_true_pred  = tf.equal(None, None)
    matches_true_pred = tf.cast(matches_true_pred , tf.float32) 
    # Multiply the acc tensor with the masks
    matches_true_pred *= None
    # Compute masked accuracy (quotient between the total matches and the total valid values, i.e., the amount of non-masked values)
    masked_acc = None/None
    
    ### END CODE HERE ### 

    return masked_acc

#@title Question 4

def predict(sentence, model, sentence_vectorizer, tag_map):
    """
    Predict NER labels for a given sentence using a trained model.

    Parameters:
    sentence (str): Input sentence.
    model (tf.keras.Model): Trained NER model.
    sentence_vectorizer (tf.keras.layers.TextVectorization): Sentence vectorization layer.
    tag_map (dict): Dictionary mapping tag IDs to labels.

    Returns:
    predictions (list): Predicted NER labels for the sentence.

    """

    ### START CODE HERE ### 

    # Convert the sentence into ids
    sentence_vectorized = None
    # Expand its dimension to make it appropriate to pass to the model
    sentence_vectorized = tf.expand_dims(None, None)
    # Get the model output
    output = None
    # Get the predicted labels for each token, using argmax function and specifying the correct axis to perform the argmax
    outputs = np.argmax(None, axis = None)
    # Next line is just to adjust outputs dimension. Since this function expects only one input to get a prediction, outputs will be something like [[1,2,3]]
    # so to avoid heavy notation below, let's transform it into [1,2,3]
    outputs = outputs[0] 
    # Get a list of all keys, remember that the tag_map was built in a way that each label id matches its index in a list
    labels = list(tag_map.keys()) 
    pred = [] 
    # Iterating over every predicted token in outputs list
    for tag_idx in None:
        pred_label = None
        pred.append(None)
    
    ### END CODE HERE ### 
    
    return pred

######################################################################################
##  Provided Functions
######################################################################################

def generate_dataset(sentences, labels, sentence_vectorizer, tag_map, tfdata=True):
    sentences_ids = sentence_vectorizer(sentences)
    labels_ids = label_vectorizer(labels, tag_map = tag_map)
    if tfdata:
      dataset = tf.data.Dataset.from_tensor_slices((sentences_ids, labels_ids))
      return dataset
    else:
      return sentences_ids, labels_ids

def get_tags(labels):
    tag_set = set() # Define an empty set
    for el in labels:
        for tag in el.split(" "):
            tag_set.add(tag)
    tag_list = list(tag_set)
    tag_list.sort()
    return tag_list

def make_tag_map(tags):
    tag_map = {}
    for i,tag in enumerate(tags):
        tag_map[tag] = i
    return tag_map

def main(train_sentences, val_sentences, test_sentences,
         train_labels, val_labels, test_labels):

    SEED = 33
    BATCH_SIZE = 64
    tf.keras.utils.set_random_seed(33) ## Setting again a random seed to ensure reproducibility

    # Read Data In
    train_sentences = load_data(train_sentences)
    train_labels = load_data(train_labels)

    val_sentences = load_data(val_sentences)
    val_labels = load_data(val_labels)

    test_sentences = load_data(test_sentences)
    test_labels = load_data(test_labels)

    # Get the tag and tag map
    tags = get_tags(train_labels)
    tag_map = make_tag_map(tags)
    print(tag_map)
    sentence_vectorizer, vocab = get_sentence_vectorizer(train_sentences)

    # Generate tf.Dataset training sets (provided function)
    train_dataset = generate_dataset(train_sentences,train_labels, sentence_vectorizer, tag_map)
    val_dataset = generate_dataset(val_sentences,val_labels,  sentence_vectorizer, tag_map)
    test_sentences_id, test_labels_id = generate_dataset(test_sentences, test_labels, sentence_vectorizer, tag_map, tfdata = False)

    model = NER(len(tag_map), len(vocab))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss = masked_loss,
                   metrics = [masked_accuracy])

    model.summary()

    model.fit(train_dataset.batch(BATCH_SIZE),
              validation_data = val_dataset.batch(BATCH_SIZE),
              shuffle=True,
              epochs = 1, steps_per_epoch=100)

    model.save('saved_model.keras')  # Saves the model in Keras v3 format


    # Convert the sentences into ids
    test_sentences_id, test_labels_id = generate_dataset(test_sentences, test_labels, sentence_vectorizer, tag_map, tfdata = False)
    test_predictions = model.predict(test_sentences_id)
    print(f"The model's accuracy in test set is: ",
          masked_accuracy(test_labels_id,test_predictions).numpy())

    return

def print_usage():
    print("Exected six argmuents. Got ", len(sys.argv))
    print("Usage: ")
    print("$> python3 assignment6.py <train-sentences-path> <val-sentences-path> <test-sentences-path> \\")
    print("                          <train-labels-path> <val-labels-path> <test-label-path")
    print("")
    print("")
    print("Example: ")
    print("$> python3 assignment6.py data/large/train/sentences.txt data/large/val/sentences.txt data/large/test/sentences.txt \\")
    print("                          data/large/train/labels.txt data/large/val/labels.txt data/large/test/labels.txt")
    return

if __name__ == '__main__':

    if len(sys.argv) != (6 + 1):
        print_usage()
    else:
        print("Please provide two arguments.")
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])


