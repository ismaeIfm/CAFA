import logging
from random import shuffle

import tensorflow as tf
from keras.callbacks import EarlyStopping

from mlom.datasets.create_CAFATrainingData import create_go_cafa_dataset
from mlom.encoders.input.text import TokenizerWrapper
from mlom.encoders.output import OutputEncoder
from mlom.mlom import MLOM
from mlom.models import get_lstm_model
from mlom.utils import AUCCallback, get_data

if __name__ == '__main__':

    logging.info('*** Loading data...')
    data = create_go_cafa_dataset(cafa_version=2) + create_go_cafa_dataset(
        cafa_version=3)
    shuffle(data)

    X_name = 'seq'
    y_names = ['C', 'F']

    logging.info('*** Formatting data...')
    data = get_data(data=data, X_name=X_name, y_names=y_names)

    valid_dataset = data
    label_encoders = {}
    for l in y_names:
        label_encoders[l] = OutputEncoder()
        label_encoders[l].fit(data[l])

    ############################################################################
    tknzr_char_enc = TokenizerWrapper(
        nb_words=500, max_input_length=500, char_level=True)
    tknzr_char_enc.fit(data[X_name])

    lstm_model = get_lstm_model(input_encoder=tknzr_char_enc, name='lstm')
    lstm = {'model': lstm_model, 'input_encoder': tknzr_char_enc}
    ############################################################################

    m = MLOM(
        name='lstm_vae',
        models=[seqVec],
        format_={
            'X':
            X_name,
            'y': [{
                'name': label,
                'activation': 'sigmoid',
                'encoder': label_encoders[label],
                'vae': True
            } for label in y_names]
        },
        verbose=1)

    m.compile(optimizer='adamax', loss='binary_crossentropy')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10),
        AUCCallback(
            dirname='roc/cafa/%s' % m.name,
            data=valid_dataset,
            X_name=X_name,
            y_names=y_names,
            save=False,
            label_encoders=label_encoders,
            input_encoders=m.get_input_encoders())
    ]

    m.fit_generator(
        data=data,
        encoded=False,
        validation_data=valid_dataset,
        batch_size=512,
        callbacks=callbacks,
        epochs=100000,
        pretrain_epochs=1000,
        verbose=1)
