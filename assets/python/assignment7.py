import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import utils
import w2_unittest

tf.keras.utils.set_random_seed(10)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

################################################################################
#@title Provided Functions: Part I
################################################################################

# Preprocess Data

def preprocess_data(data_dir, encoder_maxlen = 150, decoder_maxlen = 50):
    '''
    Creates a matrix mask for the padding cells

    Arguments:
        data_dir: folder where your data is (e.g., corpus)
        encoder_maxlen: maximum number of tokens (default = 150)
        decoder_maxlen: maximum number of decoding tokens (default = 50)
    Returns:
        dataset: tf.Dataset iterator
    '''

    train_data, test_data = utils.get_train_test_data(data_dir)

    document, summary = utils.preprocess(train_data)
    document_test, summary_test = utils.preprocess(test_data)

    # The [ and ] from default tokens cannot be removed, because they mark the SOS and EOS token.
    filters = '!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n'
    oov_token = '[UNK]'

    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token, lower=False)

    documents_and_summary = pd.concat([document, summary], ignore_index=True)

    tokenizer.fit_on_texts(documents_and_summary)

    inputs = tokenizer.texts_to_sequences(document)
    targets = tokenizer.texts_to_sequences(summary)

    vocab_size = len(tokenizer.word_index) + 1

    # Pad the sequences.
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=encoder_maxlen, padding='post', truncating='post')
    targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=decoder_maxlen, padding='post', truncating='post')

    inputs = tf.cast(inputs, dtype=tf.int32)
    targets = tf.cast(targets, dtype=tf.int32)

    # Create the final training dataset.
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    return tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE), tokenizer

# Positional Encoding

def positional_encoding(positions, d_model):
    """
    Precomputes a matrix with all the positional encodings

    Arguments:
        positions (int): Maximum number of positions to be encoded
        d_model (int): Encoding size

    Returns:
        pos_encoding (tf.Tensor): A matrix of shape (1, position, d_model) with the positional encodings
    """

    position = np.arange(positions)[:, np.newaxis]
    k = np.arange(d_model)[np.newaxis, :]
    i = k // 2

    # initialize a matrix angle_rads of all the angles
    angle_rates = 1 / np.power(10000, (2 * i) / np.float32(d_model))
    angle_rads = position * angle_rates

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

# Padding mask

def create_padding_mask(decoder_token_ids):
    """
    Creates a matrix mask for the padding cells

    Arguments:
        decoder_token_ids (matrix like): matrix of size (n, m)

    Returns:
        mask (tf.Tensor): binary tensor of size (n, 1, m)
    """
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)

    # add extra dimensions to add the padding to the attention logits.
    # this will allow for broadcasting later when comparing sequences
    return seq[:, tf.newaxis, :]

# Mask

def create_look_ahead_mask(sequence_length):
    """
    Returns a lower triangular matrix filled with ones

    Arguments:
        sequence_length (int): matrix size

    Returns:
        mask (tf.Tensor): binary tensor of size (sequence_length, sequence_length)
    """
    mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
    return mask

################################################################################
#@title Question 1: Scaled Dot Product Attention
################################################################################

def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead)
      but it must be broadcastable for addition.

    Arguments:
        q (tf.Tensor): query of shape (..., seq_len_q, depth)
        k (tf.Tensor): key of shape (..., seq_len_k, depth)
        v (tf.Tensor): value of shape (..., seq_len_v, depth_v)
        mask (tf.Tensor): mask with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output -- attention_weights
    """
    ### START CODE HERE ###

    # Multiply q and k transposed.
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk with the square root of dk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:  # Don't replace this None
        scaled_attention_logits += (1 - mask) * (-1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiply the attention weights by v
    output = tf.matmul(attention_weights, v)

    ### END CODE HERE ###

    return output, attention_weights

def scaled_dot_product_attention_test():

    # Test your function!
    q = np.array([[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1]]).astype(np.float32)
    k = np.array([[1, 1, 0, 1], [1, 0, 1, 1 ], [1, 1, 1, 0], [0, 0, 0, 1], [0, 1, 0, 1]]).astype(np.float32)
    v = np.array([[0, 0], [1, 0], [1, 0], [1, 1], [1, 1]]).astype(np.float32)
    mask = np.array([[[0, 1, 0, 1, 1], [1, 0, 0, 1, 1], [1, 1, 0, 1, 1]]])

    ou, atw = scaled_dot_product_attention(q, k, v, mask)
    ou = np.around(ou, decimals=2)
    atw = np.around(atw, decimals=2)

    print(f"Output:\n {ou}")
    print(f"\nAttention weigths:\n {atw}")

    print("Expected Output: ")
    print("[[[1.   0.62]\n",
          "[0.62 0.62]\n",
          "[0.74 0.31]]]\n")

    print("Attention weigths:")
    print("[[[0.   0.38 0.   0.23 0.38]\n",
          "[0.38 0.   0.   0.23 0.38]\n",
          "[0.26 0.43 0.   0.16 0.16]]]")

################################################################################
#@title Provided Functions: Part II
################################################################################

# Fully Connected Layer

def FullyConnected(embedding_dim, fully_connected_dim):
    """
    Returns a sequential model consisting of two dense layers. The first dense layer has
    fully_connected_dim neurons and is activated by relu. The second dense layer has
    embedding_dim and no activation.

    Arguments:
        embedding_dim (int): output dimension
        fully_connected_dim (int): dimension of the hidden layer

    Returns:
        _ (tf.keras.Model): sequential model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim, activation='relu'),  # (batch_size, seq_len, d_model)
        tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, d_model)
    ])

# Encoder Layer

class EncoderLayer(tf.keras.layers.Layer):
    """
    The encoder layer is composed by a multi-head self-attention mechanism,
    followed by a simple, positionwise fully connected feed-forward network.
    This architecture includes a residual connection around each of the two
    sub-layers, followed by layer normalization.
    """
    def __init__(self, embedding_dim, num_heads, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):

        super(EncoderLayer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim,
            dropout=dropout_rate
        )

        self.ffn = FullyConnected(
            embedding_dim=embedding_dim,
            fully_connected_dim=fully_connected_dim
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
        Forward pass for the Encoder Layer

        Arguments:
            x (tf.Tensor): Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
            training (bool): Boolean, set to true to activate
                        the training mode for dropout layers
            mask (tf.Tensor): Boolean mask to ensure that the padding is not
                    treated as part of the input
        Returns:
            encoder_layer_out (tf.Tensor): Tensor of shape (batch_size, input_seq_len, embedding_dim)
        """
        # calculate self-attention using mha(~1 line).
        # Dropout is added by Keras automatically if the dropout parameter is non-zero during training
        self_mha_output = self.mha(x, x, x, mask)  # Self attention (batch_size, input_seq_len, fully_connected_dim)

        # skip connection
        # apply layer normalization on sum of the input and the attention output to get the
        # output of the multi-head attention layer
        skip_x_attention = self.layernorm1(x + self_mha_output)  # (batch_size, input_seq_len, fully_connected_dim)

        # pass the output of the multi-head attention layer through a ffn
        ffn_output = self.ffn(skip_x_attention)  # (batch_size, input_seq_len, fully_connected_dim)

        # apply dropout layer to ffn output during training
        # use `training=training`
        ffn_output = self.dropout_ffn(ffn_output, training=training)

        # apply layer normalization on sum of the output from multi-head attention (skip connection) and ffn output
        # to get the output of the encoder layer
        encoder_layer_out = self.layernorm2(skip_x_attention + ffn_output)  # (batch_size, input_seq_len, embedding_dim)

        return encoder_layer_out

class Encoder(tf.keras.layers.Layer):
    """
    # Full Encoder

    # The full encoder will take an embedded input and positional encoding that you have
    # calculate. Your encoded embeddings will be fed to a stack of Encoder layers.

    # The Encoder class is implemented for you. It performs the following steps:
    # 1. Pass the input through the Embedding layer.
    # 2. Scale the embedding by multiplying it by the square root of the embedding dimension.
    # 3. Add the position encoding: self.pos_encoding `[:, :seq_len, :]` to the embedding.
    # 4. Pass the encoded embedding through a dropout layer
    # 5. Pass the output of the dropout layer through the stack of encoding layers using a for loop.

    The entire Encoder starts by passing the input to an embedding layer
    and using positional encoding to then pass the output through a stack of
    encoder Layers

    """
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
               maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.embedding_dim)


        self.enc_layers = [EncoderLayer(embedding_dim=self.embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps)
                           for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
        Forward pass for the Encoder

        Arguments:
            x (tf.Tensor): Tensor of shape (batch_size, seq_len)
            training (bool): Boolean, set to true to activate
                        the training mode for dropout layers
            mask (tf.Tensor): Boolean mask to ensure that the padding is not
                    treated as part of the input

        Returns:
            x (tf.Tensor): Tensor of shape (batch_size, seq_len, embedding dim)
        """
        seq_len = tf.shape(x)[1]

        # Pass input through the Embedding layer
        x = self.embedding(x)  # (batch_size, input_seq_len, embedding_dim)
        # Scale embedding by multiplying it by the square root of the embedding dimension
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        # Add the position encoding to embedding
        x += self.pos_encoding[:, :seq_len, :]
        # Pass the encoded embedding through a dropout layer
        # use `training=training`
        x = self.dropout(x, training=training)
        # Pass the output through the stack of encoding layers
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training = training, mask = mask)

        return x  # (batch_size, input_seq_len, embedding_dim)

################################################################################
#@title Question 2: The Decoder Layer
################################################################################

class DecoderLayer(tf.keras.layers.Layer):
    """
    The decoder layer is composed by two multi-head attention blocks,
    one that takes the new input and uses self-attention, and the other
    one that combines it with the output of the encoder, followed by a
    fully connected block.
    """
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()

        self.mha1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim,
            dropout=dropout_rate
        )

        self.mha2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim,
            dropout=dropout_rate
        )

        self.ffn = FullyConnected(
            embedding_dim=embedding_dim,
            fully_connected_dim=fully_connected_dim
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass for the Decoder Layer

        Arguments:
            x (tf.Tensor): Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            enc_output (tf.Tensor): Tensor of shape(batch_size, input_seq_len, fully_connected_dim)
            training (bool): Boolean, set to true to activate
                        the training mode for dropout layers
            look_ahead_mask (tf.Tensor): Boolean mask for the target_input
            padding_mask (tf.Tensor): Boolean mask for the second multihead attention layer
        Returns:
            out3 (tf.Tensor): Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            attn_weights_block1 (tf.Tensor): Tensor of shape (batch_size, num_heads, target_seq_len, target_seq_len)
            attn_weights_block2 (tf.Tensor): Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """

        ### START CODE HERE ###
        # enc_output.shape == (batch_size, input_seq_len, fully_connected_dim)

        # BLOCK 1
        # calculate self-attention and return attention scores as attn_weights_block1.
        # Dropout will be applied during training (~1 line).
        mult_attn_out1, attn_weights_block1 = self.mha1(
            x, x, x, attention_mask=look_ahead_mask, return_attention_scores=True)

        # apply layer normalization (layernorm1) to the sum of the attention output and the input (~1 line)
        Q1 = self.layernorm1(mult_attn_out1 + x)

        # BLOCK 2
        # calculate self-attention using the Q from the first block and K and V from the encoder output.
        # Dropout will be applied during training
        # Return attention scores as attn_weights_block2 (~1 line)
        if padding_mask is None:
            mult_attn_out2, attn_weights_block2 = self.mha2(
                Q1, enc_output, enc_output, return_attention_scores=True)
        else:
            mult_attn_out2, attn_weights_block2 = self.mha2(
                Q1, enc_output, enc_output, attention_mask=padding_mask, return_attention_scores=True)

        # apply layer normalization (layernorm2) to the sum of the attention output and the Q from the first block (~1 line)
        mult_attn_out2 = self.layernorm2(Q1 + mult_attn_out2)

        #BLOCK 3
        # pass the output of the second block through a ffn
        ffn_output = self.ffn(mult_attn_out2)  # (batch_size, target_seq_len, fully_connected_dim)

        # apply a dropout layer to the ffn output
        # use `training=training`
        ffn_output = self.dropout_ffn(ffn_output, training=training)

        # apply layer normalization (layernorm3) to the sum of the ffn output and the output of the second block
        out3 = self.layernorm3(mult_attn_out2 + ffn_output)  # (batch_size, target_seq_len, fully_connected_dim)

        ### END CODE HERE ###

        return out3, attn_weights_block1, attn_weights_block2

# Test your function!

def decoder_layer_test():

    key_dim = 12
    n_heads = 16

    decoderLayer_test = DecoderLayer(embedding_dim=key_dim, num_heads=n_heads, fully_connected_dim=32)

    q = np.ones((1, 15, key_dim))
    encoder_test_output = tf.convert_to_tensor(np.random.rand(1, 7, 8))
    look_ahead_mask = create_look_ahead_mask(q.shape[1])

    out, attn_w_b1, attn_w_b2 = decoderLayer_test(
        q, encoder_test_output, training = False, look_ahead_mask = look_ahead_mask, padding_mask = None)

    print(f"Using embedding_dim={key_dim} and num_heads={n_heads}:\n")
    print(f"q has shape:{q.shape}")
    print(f"Output of encoder has shape:{encoder_test_output.shape}\n")

    print(f"Output of decoder layer has shape:{out.shape}")
    print(f"Att Weights Block 1 has shape:{attn_w_b1.shape}")
    print(f"Att Weights Block 2 has shape:{attn_w_b2.shape}")


################################################################################
#@title Question 3: Full Decoder
################################################################################

class Decoder(tf.keras.layers.Layer):
    """
    The entire Encoder starts by passing the target input to an embedding layer
    and using positional encoding to then pass the output through a stack of
    decoder Layers

    """
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, target_vocab_size,
               maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim)

        self.dec_layers = [DecoderLayer(embedding_dim=self.embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps)
                           for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):
        """
        Forward  pass for the Decoder

        Arguments:
            x (tf.Tensor): Tensor of shape (batch_size, target_seq_len)
            enc_output (tf.Tensor):  Tensor of shape(batch_size, input_seq_len, fully_connected_dim)
            training (bool): Boolean, set to true to activate
                        the training mode for dropout layers
            look_ahead_mask (tf.Tensor): Boolean mask for the target_input
            padding_mask (tf.Tensor): Boolean mask for the second multihead attention layer
        Returns:
            x (tf.Tensor): Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            attention_weights (dict[str: tf.Tensor]): Dictionary of tensors containing all the attention weights
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        ### START CODE HERE ###
        # create word embeddings
        x = self.embedding(x)  # (batch_size, target_seq_len, embedding_dim)

        # scale embeddings by multiplying by the square root of their dimension
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))

        # add positional encodings to word embedding
        x += self.pos_encoding[:, :seq_len, :]

        # apply a dropout layer to x
        # use `training=training`
        x = self.dropout(x, training=training)

        # use a for loop to pass x through a stack of decoder layers and update attention_weights (~4 lines total)
        for i in range(self.num_layers):
            # pass x and the encoder output through a stack of decoder layers and save the attention weights
            # of block 1 and 2 (~1 line)
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, training = training,
                look_ahead_mask = look_ahead_mask, padding_mask = padding_mask)

            #update attention_weights dictionary with the attention weights of block 1 and block 2
            attention_weights['decoder_layer{}_block1_self_att'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2_decenc_att'.format(i+1)] = block2
        ### END CODE HERE ###

        # x.shape == (batch_size, target_seq_len, fully_connected_dim)
        return x, attention_weights

# Test your function!

def decoder_test():
    n_layers = 5
    emb_d = 13
    n_heads = 17
    fully_connected_dim = 16
    target_vocab_size = 300
    maximum_position_encoding = 6

    x = np.array([[3, 2, 1, 1], [2, 1, 1, 0], [2, 1, 1, 0]])

    encoder_test_output = tf.convert_to_tensor(np.random.rand(3, 7, 9))

    look_ahead_mask = create_look_ahead_mask(x.shape[1])

    decoder_test = Decoder(n_layers, emb_d, n_heads, fully_connected_dim, target_vocab_size,maximum_position_encoding)

    outd, att_weights = decoder_test(
        x, encoder_test_output, training = False, look_ahead_mask = look_ahead_mask, padding_mask = None)


    print("-----------------------------------------------")
    print("Generated Output:")
    print("")
    print(f"Using num_layers={n_layers}, embedding_dim={emb_d} and num_heads={n_heads}:\n")
    print(f"x has shape:{x.shape}")
    print(f"Output of encoder has shape:{encoder_test_output.shape}\n")

    print(f"Output of decoder has shape:{outd.shape}\n")
    print("Attention weights:")
    for name, tensor in att_weights.items():
        print(f"{name} has shape:{tensor.shape}")

    print("")
    print("-----------------------------------------------")
    print("Expected Output:")
    print("")
    expected_output="""
    Using num_layers=5, embedding_dim=13 and num_heads=17:

    x has shape:(3, 4)
    Output of encoder has shape:(3, 7, 9)

    Output of decoder has shape:(3, 4, 13)

    Attention weights:
    decoder_layer1_block1_self_att has shape:(3, 17, 4, 4)
    decoder_layer1_block2_decenc_att has shape:(3, 17, 4, 7)
    decoder_layer2_block1_self_att has shape:(3, 17, 4, 4)
    decoder_layer2_block2_decenc_att has shape:(3, 17, 4, 7)
    decoder_layer3_block1_self_att has shape:(3, 17, 4, 4)
    decoder_layer3_block2_decenc_att has shape:(3, 17, 4, 7)
    decoder_layer4_block1_self_att has shape:(3, 17, 4, 4)
    decoder_layer4_block2_decenc_att has shape:(3, 17, 4, 7)
    decoder_layer5_block1_self_att has shape:(3, 17, 4, 4)
    decoder_layer5_block2_decenc_att has shape:(3, 17, 4, 7)
    """
    print(expected_output)

################################################################################
#@title Question 4: Transformer
################################################################################

class Transformer(tf.keras.Model):
    """
    Complete transformer with an Encoder and a Decoder
    """
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
               target_vocab_size, max_positional_encoding_input,
               max_positional_encoding_target, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers=num_layers,
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               fully_connected_dim=fully_connected_dim,
                               input_vocab_size=input_vocab_size,
                               maximum_position_encoding=max_positional_encoding_input,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.decoder = Decoder(num_layers=num_layers,
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               fully_connected_dim=fully_connected_dim,
                               target_vocab_size=target_vocab_size,
                               maximum_position_encoding=max_positional_encoding_target,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size, activation='softmax')

    def call(self, input_sentence, output_sentence, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """
        Forward pass for the entire Transformer
        Arguments:
            input_sentence (tf.Tensor): Tensor of shape (batch_size, input_seq_len)
                              An array of the indexes of the words in the input sentence
            output_sentence (tf.Tensor): Tensor of shape (batch_size, target_seq_len)
                              An array of the indexes of the words in the output sentence
            training (bool): Boolean, set to true to activate
                        the training mode for dropout layers
            enc_padding_mask (tf.Tensor): Boolean mask to ensure that the padding is not
                    treated as part of the input
            look_ahead_mask (tf.Tensor): Boolean mask for the target_input
            dec_padding_mask (tf.Tensor): Boolean mask for the second multihead attention layer
        Returns:
            final_output (tf.Tensor): The final output of the model
            attention_weights (dict[str: tf.Tensor]): Dictionary of tensors containing all the attention weights for the decoder
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)

        """
        ### START CODE HERE ###
        # call self.encoder with the appropriate arguments to get the encoder output
        enc_output = self.encoder(input_sentence, training = training,
                                  mask = enc_padding_mask)

        # call self.decoder with the appropriate arguments to get the decoder output
        # dec_output.shape == (batch_size, tar_seq_len, fully_connected_dim)
        dec_output, attention_weights = self.decoder(
            output_sentence, enc_output, training = training,
            look_ahead_mask = look_ahead_mask, padding_mask = dec_padding_mask)

        # pass decoder output through a linear layer and softmax (~1 line)
        final_output = self.final_layer(dec_output)
        ### END CODE HERE ###

        return final_output, attention_weights

# Test your function!
def transformer_test():
    n_layers = 3
    emb_d = 13
    n_heads = 17
    fully_connected_dim = 8
    input_vocab_size = 300
    target_vocab_size = 350
    max_positional_encoding_input = 12
    max_positional_encoding_target = 12

    transformer = Transformer(n_layers,
        emb_d,
        n_heads,
        fully_connected_dim,
        input_vocab_size,
        target_vocab_size,
        max_positional_encoding_input,
        max_positional_encoding_target)

    # 0 is the padding value
    sentence_a = np.array([[2, 3, 1, 3, 0, 0, 0]])
    sentence_b = np.array([[1, 3, 4, 0, 0, 0, 0]])

    enc_padding_mask = create_padding_mask(sentence_a)
    dec_padding_mask = create_padding_mask(sentence_a)

    look_ahead_mask = create_look_ahead_mask(sentence_a.shape[1])

    test_summary, att_weights = transformer(
        sentence_a,
        sentence_b,
        training = False,
        enc_padding_mask = enc_padding_mask,
        look_ahead_mask = look_ahead_mask,
        dec_padding_mask = dec_padding_mask
    )

    print("-----------------------------------------------")
    print("Generated Output:")
    print("")
    print(f"Using num_layers={n_layers}, target_vocab_size={target_vocab_size} and num_heads={n_heads}:\n")
    print(f"sentence_a has shape:{sentence_a.shape}")
    print(f"sentence_b has shape:{sentence_b.shape}")

    print(f"\nOutput of transformer (summary) has shape:{test_summary.shape}\n")
    print("Attention weights:")
    for name, tensor in att_weights.items():
        print(f"{name} has shape:{tensor.shape}")

    print("")
    print("-----------------------------------------------")
    print("Expected Output:")
    print("")

    expected_output = """
    Using num_layers=3, target_vocab_size=350 and num_heads=17:

    sentence_a has shape:(1, 7)
    sentence_b has shape:(1, 7)

    Output of transformer (summary) has shape:(1, 7, 350)

    Attention weights:
    decoder_layer1_block1_self_att has shape:(1, 17, 7, 7)
    decoder_layer1_block2_decenc_att has shape:(1, 17, 7, 7)
    decoder_layer2_block1_self_att has shape:(1, 17, 7, 7)
    decoder_layer2_block2_decenc_att has shape:(1, 17, 7, 7)
    decoder_layer3_block1_self_att has shape:(1, 17, 7, 7)
    decoder_layer3_block2_decenc_att has shape:(1, 17, 7, 7)
    """
    print(expected_output)


################################################################################
#@title Provided Functions: Part III
################################################################################

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, dtype=tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def masked_loss(real, pred):

    # Next, you set up the loss. Since the target sequences are padded, it is important to 
    # apply a padding mask when calculating the loss.
    # 
    # You will use the sparse categorical cross-entropy loss function 
    # (`tf.keras.losses.SparseCategoricalCrossentropy`) and set the parameter `from_logits` to 
    # False since the Transformer does not output raw logits since the last layer has a 
    # softmax activation:

    # Set loss to categorical crossentropy
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

# Custom training function. If you are not very advanced with tensorflow, you can understand 
# this function as an alternative to using `model.compile()` and `model.fit()`, but with 
# added extra flexibility."""

@tf.function
def train_step(model, inp, tar, train_loss, optimizer):
    """
    One training step for the transformer
    Arguments:
        inp (tf.Tensor): Input data to summarize
        tar (tf.Tensor): Target (summary)
        optimizer (tf.keras.optimizers): Optimizatizer
    Returns:
        None
    """
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    # Create masks
    enc_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
    dec_padding_mask = create_padding_mask(inp) # Notice that both encoder and decoder padding masks are equal

    with tf.GradientTape() as tape:
        predictions, _ = model(
            inp,
            tar_inp,
            training = True,
            enc_padding_mask = enc_padding_mask,
            look_ahead_mask = look_ahead_mask,
            dec_padding_mask = dec_padding_mask
        )
        loss = masked_loss(tar_real, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)

"""Now you are ready for training the model. But before starting the training, you can also define one more set of functions to perform the inference. Because you are using a custom training loop, you can do whatever you want between the training steps. And wouldnt't it be fun to see after each epoch some examples of how the model performs?

<a name='11'></a>
## 11 - Summarization

The last thing you will implement is inference. With this, you will be able to produce actual summaries of the documents. You will use a simple method called greedy decoding, which means you will predict one word at a time and append it to the output. You will start with an `[SOS]` token and repeat the word by word inference until the model returns you the `[EOS]` token or until you reach the maximum length of the sentence (you need to add this limit, otherwise a poorly trained model could give you infinite sentences without ever producing the `[EOS]` token.

<a name='ex-5'></a>
### Exercise 5 - next_word
Write a helper function that predicts the next word, so you can use it to write the whole sentences. Hint: this is very similar to what happens in the train_step, but you have to set the training of the model to False.
"""

################################################################################
#@title Question 5: Next Word Inference
################################################################################

def next_word(model, encoder_input, output):
    """
    Helper function for summarization that uses the model to predict just the next word.
    Arguments:
        encoder_input (tf.Tensor): Input data to summarize
        output (tf.Tensor): (incomplete) target (summary)
    Returns:
        predicted_id (tf.Tensor): The id of the predicted word
    """
    ### START CODE HERE ###
    # Create a padding mask for the input (encoder)
    enc_padding_mask = create_padding_mask(encoder_input)
    # Create a look-ahead mask for the output
    look_ahead_mask = create_look_ahead_mask(tf.shape(output)[1])
    # Create a padding mask for the input (decoder)
    dec_padding_mask = create_padding_mask(encoder_input)

    # Run the prediction of the next word with the transformer model
    predictions, attention_weights = model(
        encoder_input,
        output,
        training = False,
        enc_padding_mask = enc_padding_mask,
        look_ahead_mask = look_ahead_mask,
        dec_padding_mask = dec_padding_mask
    )
    # Select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]
    # Get the predicted id: you can choose the maximum score prediction
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    ### END CODE HERE ###

    predictions = predictions[: ,-1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    return predicted_id

"""Check if your function works."""

def next_word_test(tokenizer):
    # Take a random sentence as an input
    input_document = tokenizer.texts_to_sequences(["a random sentence"])
    input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=encoder_maxlen, padding='post', truncating='post')
    encoder_input = tf.expand_dims(input_document[0], 0)

    # Take the start of sentence token as the only token in the output to predict the next word
    output = tf.expand_dims([tokenizer.word_index["[SOS]"]], 0)

    print(transformer)

    # predict the next word with your function
    predicted_token = next_word(transformer, encoder_input, output)
    print(f"Predicted token: {predicted_token}")

    predicted_word = tokenizer.sequences_to_texts(predicted_token.numpy())[0]
    print(f"Predicted word: {predicted_word}")

    """##### __Expected Output__

    ```
    Predicted token: [[14859]]
    Predicted word: masses
    ```
    """

################################################################################
#@title Provided Functions: Part IV
################################################################################

def summarize(model, input_document, tokenizer, encoder_maxlen = 150, decoder_maxlen = 50):
    """
    A function for summarization using the transformer model
    Arguments:
        input_document (tf.Tensor): Input data to summarize
    Returns:
        _ (str): The summary of the input_document
    """
    input_document = tokenizer.texts_to_sequences([input_document])
    input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=encoder_maxlen, padding='post', truncating='post')
    encoder_input = tf.expand_dims(input_document[0], 0)

    output = tf.expand_dims([tokenizer.word_index["[SOS]"]], 0)

    for i in range(decoder_maxlen):
        predicted_id = next_word(model, encoder_input, output)
        output = tf.concat([output, predicted_id], axis=-1)

        if predicted_id == tokenizer.word_index["[EOS]"]:
            break

    return tokenizer.sequences_to_texts(output.numpy())[0]  # since there is just one translated document

"""Now you can already summarize a sentence! But beware, since the model was not yet trained at all, it will just produce nonsense."""

# training_set_example = 0

# #### Need to edit this
# train_data, test_data = utils.get_train_test_data('corpus')

# document, summary = utils.preprocess(train_data)
# document_test, summary_test = utils.preprocess(test_data)


# # Check a summary of a document from the training set
# print('Training set example:')
# print(document[training_set_example])
# print('\nHuman written summary:')
# print(summary[training_set_example])
# print('\nModel written summary:')
# summarize(transformer, document[training_set_example])

#@title Train the Model

"""<a name='12'></a>
# 12 - Train the model

Now you can finally train the model. Below is a loop that will train your model for 20 epochs. note that it should take about 30 seconds per epoch (with the exception of the first few epochs which can take a few minutes each).

Note that after each epoch you perform the summarization on one of the sentences in the test set and print it out, so you can see how your model is improving.
"""

def train_model(data_folder):

    # Initialize the Model
    #
    # Now that you have defined the model, you can initialize and train it. First you 
    # can initialize the model with the parameters below. Note that generally these 
    # models are much larger and you are using a smaller version to fit this environment 
    # and to be able to train it in just a few minutes. The base model described in the 
    # original Transformer paper used `num_layers=6`, `embedding_dim=512`, and 
    # `fully_connected_dim=2048`.


    encoder_maxlen = 150
    decoder_maxlen = 50

    dataset, tokenizer = preprocess_data(data_folder, encoder_maxlen = 150, decoder_maxlen = 50)

    vocab_size = tokenizer
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocab size: {vocab_size}")

    # Define the model parameters
    num_layers = 2
    embedding_dim = 128
    fully_connected_dim = 128
    num_heads = 2
    positional_encoding_length = 256

    # Initialize the model
    transformer = Transformer(
        num_layers,
        embedding_dim,
        num_heads,
        fully_connected_dim,
        vocab_size,
        vocab_size,
        positional_encoding_length,
        positional_encoding_length,
    )

    # Prepare for Training the Model
    # 
    # The original transformer paper uses Adam optimizer with custom learning rate scheduling, 
    # which we define in the cell below. This was empirically shown to produce faster 
    # convergence.

    learning_rate = CustomSchedule(embedding_dim)
    optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    # Here you will store the losses, so you can later plot them
    losses = []

    train_data, test_data = utils.get_train_test_data('corpus')

    document, summary = utils.preprocess(train_data)
    document_test, summary_test = utils.preprocess(test_data)

    # Take an example from the test set, to monitor it during training
    test_example = 0
    true_summary = summary_test[test_example]
    true_document = document_test[test_example]

    # Define the number of epochs
    epochs = 20

    # Training loop
    for epoch in range(epochs):

        start = time.time()
        train_loss.reset_state()
        number_of_batches=len(list(enumerate(dataset)))

        for (batch, (inp, tar)) in enumerate(dataset):
            print(f'Epoch {epoch+1}, Batch {batch+1}/{number_of_batches}', end='\r')
            train_step(transformer, inp, tar, train_loss, optimizer)

        print (f'Epoch {epoch+1}, Loss {train_loss.result():.4f}')
        losses.append(train_loss.result())

        print (f'Time taken for one epoch: {time.time() - start} sec')
        print('Example summarization on the test set:')
        print('  True summarization:')
        print(f'    {true_summary}')
        print('  Predicted summarization:')
        print(f'    {summarize(transformer, true_document, tokenizer, 
            encoder_maxlen = encoder_maxlen, decoder_maxlen = decoder_maxlen)}\n')

    """Plot the loss funtion."""

    plt.plot(losses)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    return transformer

"""<a name='13'></a>
# 13 - Summarize some Sentences!

Below you can see an example of summarization of a sentence from the training set and a sentence from the test set. See if you notice anything interesting about them!
"""
def print_transformer_outputs(document):
    training_set_example = 0

    # Check a summary of a document from the training set
    print('Training set example:')
    print(document[training_set_example])
    print('\nHuman written summary:')
    print(summary[training_set_example])
    print('\nModel written summary:')
    print(summarize(transformer, document[training_set_example]))

    test_set_example = 3

    # Check a summary of a document from the test set
    print('Test set example:')
    print(document_test[test_set_example])
    print('\nHuman written summary:')
    print(summary_test[test_set_example])
    print('\nModel written summary:')
    print(summarize(transformer, document_test[test_set_example]))

"""
If you critically examine the output of the model, you can notice a few things:
 - In the training set the model output is (almost) identical to the real output
   (already after 20 epochs and even more so with more epochs). This might be
   because the training set is relatively small and the model is relatively big
   and has thus learned the sentences in the training set by heart (overfitting).
 - While the performance on the training set looks amazing, it is not so good on
   the test set. The model overfits, but fails to generalize. Again an easy
   candidate to blame is the small training set and a comparatively large model,
   but there might be a variety of other factors.
 - Look at the test set example 3 and its summarization. Would you summarize it
   the same way as it is written here? Sometimes the data may be ambiguous. And
   the training of **your model can only be as good as your data**.

Here you only use a small dataset, to show that something can be learned in a
reasonable amount of time in a relatively small environment. Generally, large
transformers are trained on more than one task and on very large quantities of
data to achieve superb performance. You will learn more about this in the rest
of this course.

**Congratulations on finishing this week's assignment!** You did a lot of work and now you should have a better understanding of the Transformers and their building blocks (encoder and decoder) and how they can be used for text summarization. And remember: you dont need to change much to use the same model for a translator, just change the dataset and it should work!

**Keep it up!**
"""

if __name__ == '__main__':

    # def print_usage():
    #     print("Exected six argmuents. Got ", len(sys.argv))
    #     print("Usage: ")
    #     print("$> python3 assignment6.py <train-sentences-path> <val-sentences-path> <test-sentences-path> \\")
    #     print("                          <train-labels-path> <val-labels-path> <test-label-path")
    #     print("")
    #     print("")
    #     print("Example: ")
    #     print("$> python3 assignment6.py data/large/train/sentences.txt data/large/val/sentences.txt data/large/test/sentences.txt \\")
    #     print("                          data/large/train/labels.txt data/large/val/labels.txt data/large/test/labels.txt")
    #     return

    transformer = train_model('corpus')
    print_transformer_outputs()