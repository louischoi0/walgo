import tensorflow as tf
from dataset import WindowGenerator, load_dataset
from matplotlib import pyplot
import pandas as pd

if __name__ == "__main__":
    tr, tr_y, vd, tt = load_dataset(start_date="2022-01-01", days=20)
    w = WindowGenerator(input_width=9, label_width=1, train_df=tr, valid_df=vd, test_df=tt, shift=1, label_col_index=0)
    print(w.test)
    
    model = tf.keras.models.load_model('vn_1_s1.md')
    predictions = model.predict(w.test)
    print(predictions.shape)
    predictions = predictions.reshape((-1,))
    print(predictions.shape)
    print(tt.values.shape)

    s = 30.731748653620663
    m = 54.40985132047694 

    predictions = list(x*s+m for x in predictions)

    mse, mae = model.evaluate(w.test, verbose=0)
    print(mse, mae)
    pyplot.plot(predictions)
    pyplot.plot(list(x[0]*s +m for x in tt.values[10:]))
    pyplot.show() 
