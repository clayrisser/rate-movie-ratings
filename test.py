from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import pydash as _
import nltk
import sys
import re

nltk.download('stopwords')

class Dataset():
    raw_data = []
    sanitized = []
    tokenized = []
    padded = []
    x = []
    valid_x = []
    y = []
    valid_y = []

def main():
    train_dataset = pd.read_csv('train.tsv', delimiter='\t')
    test_dataset = pd.read_csv('test.tsv', delimiter='\t')
    train, test = load_datasets(
        train_dataset.review,
        train_dataset.sentiment,
        test_dataset.review
    )
    stats('train', train)
    stats('test', test)

def load_datasets(train_dataset_data, train_dataset_scores, test_dataset_data):
    train = Dataset()
    test = Dataset()
    train.raw_data = train_dataset_data
    test.raw_data = test_dataset_data
    train.sanitized, test.sanitized = sanitize(train_dataset_data, test_dataset_data)
    train.tokenized, test.tokenized = tokenize(train.sanitized, test.sanitized)
    train.padded, test.padded = pad(train.tokenized, test.tokenized)
    train.x, train.valid_x, train.y, train.valid_y = split(train.padded, train_dataset_scores)
    return (train, test)

def sanitize(train_data, test_data):
    train_sanitized = [sanitize_text(text) for text in train_data]
    test_sanitized = [sanitize_text(text) for text in test_data]
    print('::: SANITIZED :::')
    sys.stdout.flush()
    return (train_sanitized, test_sanitized)

def tokenize(train_sanitized, test_sanitized):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_sanitized + test_sanitized)
    train_tokenized = tokenizer.texts_to_sequences(train_sanitized)
    test_tokenized = tokenizer.texts_to_sequences(test_sanitized)
    print('::: TOKENIZED :::')
    sys.stdout.flush()
    return (train_tokenized, test_tokenized)

def pad(train_tokenized, test_tokenized, maxlen=200):
    train_padded = pad_sequences(train_tokenized, maxlen=maxlen)
    test_padded = pad_sequences(test_tokenized, maxlen=maxlen)
    print('::: PADDED :::')
    sys.stdout.flush()
    return (train_padded, test_padded)

def split(train_padded, train_scores, test_size=0.15, random_state=2):
    train_x, train_valid_x, train_y, train_valid_y = train_test_split(
        train_padded,
        train_scores,
        test_size=test_size,
        random_state=random_state
    )
    print('::: SPLIT :::')
    sys.stdout.flush()
    return (train_x, train_valid_x, train_y, train_valid_y)

def stats(name, dataset):
    if len(dataset.raw_data) > 0:
        print('\n')
        print(name + ' raw_data: ' + str(len(dataset.raw_data)))
        print('example length: ' + str(len(dataset.raw_data[0].split(' '))))
        print('******* EXAMPLE ********')
        print(dataset.raw_data[0])
    if len(dataset.sanitized) > 0:
        print('\n')
        print(name + ' sanitized: ' + str(len(dataset.sanitized)))
        print('example length: ' + str(len(dataset.sanitized[0].split(' '))))
        print('******* EXAMPLE ********')
        print(dataset.sanitized[0])
    if len(dataset.tokenized) > 0:
        print('\n')
        print(name + ' tokenized: ' + str(len(dataset.tokenized)))
        print('example length: ' + str(len(dataset.tokenized[0])))
        print('******* EXAMPLE ********')
        print(dataset.tokenized[0])
    if len(dataset.padded) > 0:
        print('\n')
        print(name + ' padded: ' + str(len(dataset.padded)))
        print('example length: ' + str(len(dataset.padded[0])))
        print('******* EXAMPLE ********')
        print(dataset.padded[0])
    if len(dataset.x) > 0:
        print('\n')
        print(name + ' x: ' + str(len(dataset.x)))
        print('example length: ' + str(len(dataset.x[_.keys(dataset.x)[0]])))
        print('******* EXAMPLE ********')
        print(dataset.x[_.keys(dataset.x)[0]])
    if len(dataset.valid_x) > 0:
        print('\n')
        print(name + ' valid_x: ' + str(len(dataset.valid_x)))
        print('example length: ' + str(len(dataset.valid_x[_.keys(dataset.valid_x)[0]])))
        print('******* EXAMPLE ********')
        print(dataset.valid_x[_.keys(dataset.valid_x)[0]])
    if len(dataset.y) > 0:
        print('\n')
        print(name + ' y: ' + str(len(dataset.y)))
        print('******* EXAMPLE ********')
        print(dataset.y[_.keys(dataset.y)[0]])
    if len(dataset.valid_y) > 0:
        print('\n')
        print(name + ' valid_y: ' + str(len(dataset.valid_y)))
        print('******* EXAMPLE ********')
        print(dataset.valid_y[_.keys(dataset.valid_y)[0]])

def sanitize_text(text):
    stops = set(nltk.corpus.stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'[^a-z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([w for w in text.split() if not w in stops])
    return text

def build_rnn(n_words, embed_size, batch_size, lstm_size, num_layers, dropout, learning_rate, multiple_fc, fc_units):
    print('building rnn')
    tf.reset_default_graph()
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    with tf.name_scope('labels'):
        labels = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, [None, None], name='keep_prob')
    with tf.name_scope('embeddings'):
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs)
    with tf.name_scope('RNN_layers'):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    with tf.name_scope('RNN_init_state'):
        initial_state = cell.zero_state(batch_size, tf.float32)
    with tf.name_scope('RNN_forward'):
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)
    with tf.name_scope('fully_connected'):
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        dense = tf.contrib.layers.fully_connected(
            outputs[:, -1],
            num_outputs = fc_units,
            activation_fn = tf.sigmoid,
            weights_initializer = weights,
            biases_initializer = biases
        )
        dense = tf.contrib.layers.dropout(dense, keep_prob)
        if multiple_fc == True:
            dense = tf.contrib.layers.fully_connected(
                dense,
                num_outputs = fc_units,
                activation_fn = tf.sigmoid,
                weights_initializer = weights,
                biases_initializer = biases
            )
            dense = tf.contrib.layers.dropout(dense, keep_prob)
    with tf.name_scope('predictions'):
        predictions = tf.contrib.layers.fully_connected(
            dense,
            num_outputs = 1,
            activation_fn=tf.sigmoid,
            weights_initializer = weights,
            biases_initializer = biases
        )
        tf.summary.histogram('predictions', predictions)
    with tf.name_scope('cost'):
        cost = tf.losses.mean_squared_error(labels, predictions)
        tf.summary.scalar('cost', cost)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(
            tf.cast(tf.round(predictions), tf.int32),
            labels
        )
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    export_nodes = [
        'inputs', 'labels', 'keep_prob','initial_state',
        'final_state','accuracy', 'predictions', 'cost',
        'optimizer', 'merged'
    ]
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])
    return graph

main()
