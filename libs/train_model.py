import tensorflow as tf
import pandas as pd
from dataset import load_dataset, ds_shape
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from dataset import WindowGenerator
import json
import argparse

def _import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def load_model_from_config(conf, session):
    session = config[session]
    _loss = eval(session["loss"])

    model = session["model_name"]
    model_args = session["model_args"]
    model = _import(model)(**model_args)

    model.compile(
        loss=_loss,
        optimizer=tf.optimizers.RMSprop(learning_rate=1e-4),
        metrics=[tf.metrics.MeanAbsoluteError()],
        run_eagerly=True,
    )

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='train_model')
    parser.add_argument('--session')

    args = parser.parse_args()

    with open("defmodel.json") as f :
        config = json.load(f)
    session = config[args.session]

    model = load_model_from_config(config, args.session)
    ref_days = session["dataset"]["ref_days"]

    tr, tr_y, vd, tt = load_dataset(session["input_file"],days=int(ref_days))
    window_generator = session["window_generator"]
    w = WindowGenerator(train_df=tr, valid_df=vd, test_df=tt, **window_generator)

    ds = w.make_dataset(tr)
    patience = 3
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min')

    h = model.fit(w.train, epochs=10, verbose=0, validation_data=w.val, callbacks=[early_stopping]) 
    model.save("vn_1_s1.md")

