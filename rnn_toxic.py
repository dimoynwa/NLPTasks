import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from preprocess import pre_process_data


BATCH_SIZE = 64
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EPOCHS = 10
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
HIDDEN_UNITS = 128


def build_model(conf):
    print('Building the model...')
    embedding_layer = tf.keras.layers.Embedding(conf['vocab_size'], conf['embedding_dim'],
                                                input_length=conf['data'].shape[1],
                                                weights=[conf['embedding_matrix']],
                                                trainable=False)

    inp_ = tf.keras.layers.Input(shape=(conf['data'].shape[1],))
    x = embedding_layer(inp_)
    x = tf.keras.layers.LSTM(HIDDEN_UNITS, return_sequences=True)(x)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    x = tf.keras.layers.Dense(conf['targets'].shape[1], activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inp_, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

    print('Model built.')

    return model


if __name__ == "__main__":
    config = pre_process_data()
    model = build_model(config)
    print('Training the model...')

    history = model.fit(x=config['data'], y=config['targets'], epochs=EPOCHS, validation_split=VALIDATION_SPLIT,
                        batch_size=BATCH_SIZE)
    print('Training done.')

    model.save('./saved_models/rnn_toxic.h5')

    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.legend()
    plt.show()

    # ROC_AUC_CURVE calculate
    print('Going to calculate ROC_AUC score...')
    predictions = model.predict(config['data'])
    aucs = []
    for j in range(config['targets'].shape[1]):
        auc = roc_auc_score(config['targets'][:, j], predictions[:, j])
        aucs.append(auc)
    print('ROC_AUC scores: ', aucs)
    print('ROC_AUC mean: ', np.mean(aucs))
