import logging
from random import shuffle

import tensorflow as tf
from keras.callbacks import EarlyStopping

from mlom.datasets.create_CAFATrainingData import create_go_cafa_dataset
from mlom.encoders.input.text import CrepeEncoder
from mlom.encoders.output import OutputEncoder
from mlom.mlom import MLOM
from mlom.models import get_crepe_model
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
    crepe_enc = CrepeEncoder(maxlen=1024)
    crepe_model = get_crepe_model(input_encoder=crepe_enc)

    crepe = {'model': crepe_model, 'input_encoder': crepe_enc}
    ############################################################################

    m = MLOM(
        name='crepe',
        models=[crepe],
        format_={
            'X':
            X_name,
            'y': [
                {
                    'name': label,
                    'activation': 'sigmoid',
                    'encoder': label_encoders[label],
                    #'vae': True
                } for label in y_names
            ]
        },
        verbose=1)

    m.compile(optimizer='adamax', loss='binary_crossentropy')
    m.save('roc/cafa/%s/mlom/%s' % (m.name, m.name), weights=False)

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
        batch_size=128,
        callbacks=callbacks,
        epochs=1000,
        pretrain_epochs=1000,
        verbose=1)
