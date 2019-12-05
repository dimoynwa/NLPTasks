import numpy as np
import pandas as pd
import tensorflow as tf

EMBEDDING_DIM = 100
EMBEDDINGS_FILE_PATH = './embeddings/glove/glove.6B.%sd.txt' % EMBEDDING_DIM
TRAIN_DATA_FILE = './data/jigsaw-toxic-comment-classification-challenge/train.csv'
MAX_VOCAB_SIZE = 20000
"""Loading pre-trained embeddings"""


def load_embeddings():
    print('Load embeddings...')
    word_index = {}
    with open(EMBEDDINGS_FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            split = line.split()
            word = split[0]
            embeddings = split[1:]
            word_index[word] = np.asarray(embeddings, dtype='float32')
    print('Embeddings loaded.')
    return word_index


def load_data():
    # Loading the data from csv
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    sentences = train_df['comment_text'].fillna('DUMMY').values

    # Columns for possible labels
    possible_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    targets = train_df[possible_labels].values

    print('Maximum sentence length : ', max(len(x) for x in sentences))
    return sentences, targets


def fit_tokenizer(sentences):
    # Tokenizing sentences
    print('Tokenizer fitting...')
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(sentences)
    print('Tokenizer fitting done.')
    return tokenizer


def process_sentences(tokenizer, sentences):
    print('Texts to sequences...')
    sequences = tokenizer.texts_to_sequences(sentences)
    print('Texts to sequences done.')

    print('Number of sequences: ', len(sequences))
    print('Max sequence length: ', max([len(x) for x in sequences]))

    print('Number of tokens found in sentences: ', len(tokenizer.word_index))

    # Padding sequences
    print('Padding sequences...')
    data = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    print('Padding done.')
    print('Padded data shape: ', data.shape)
    return data


def load_embedding_matrix(tokenizer, word_index):
    word2idx = tokenizer.word_index

    vocab_size = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
    print('Vocabulary size: ', vocab_size)
    embedding_matrix = np.zeros(shape=(vocab_size, EMBEDDING_DIM))

    print('Filling embedding matrix...')
    for word, idx in word2idx.items():
        if idx < vocab_size:
            emb_vec = word_index.get(word)
            if emb_vec is not None:
                embedding_matrix[idx] = word_index.get(word)
    print('Embedding matrix done.')
    return embedding_matrix


def pre_process_data():
    word_index = load_embeddings()
    sentences, targets = load_data()
    tokenizer = fit_tokenizer(sentences)
    data = process_sentences(tokenizer, sentences)
    embedding_matrix = load_embedding_matrix(tokenizer, word_index)
    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)
    config = {
        'data': data,
        'targets': targets,
        'tokenizer': tokenizer,
        'embedding_matrix': embedding_matrix,
        'vocab_size': vocab_size,
        'embedding_dim': EMBEDDING_DIM
    }
    return config
