import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime
from flask import Flask, request, jsonify
import tensorflow as tf

global graph
graph = tf.get_default_graph() 

app = Flask(__name__)
model = None

def onehot(inpt, base):
    out = np.zeros((inpt.shape[0], base))
    i = 0
    for location_id in np.nditer(inpt):
        out[i, location_id-1] = 1
        i += 1
    return out.astype(int)

def prepare_date(df_date):
    weekdays_one_hot_df = pd.DataFrame(onehot(df_date.map(lambda x: x.dayofweek).values, 7))
    time_flat_series = df_date.map(lambda a: (a.hour-10)/10 + a.minute/600)
    
    weekdays_one_hot_df['time'] = time_flat_series
    return weekdays_one_hot_df

def prepare_data(df, sort=False):
    if sort:
        df = df.sort_values(by=['date']).reset_index().drop(['index'], axis = 1)
    dates = prepare_date(df['date'])
    places_dummies = pd.DataFrame(onehot(df['location_id'].values, 85)) # 85 locations
    
    return pd.concat([dates, places_dummies], axis=1).values, (df['pickup_time']/90).values if 'pickup_time' in df.columns else False#, dates, places_dummies  


def load_my_model():
    global model
    DIRECTORY = './input/'
    LEARN = False

    if LEARN:
        locations = pd.read_csv(DIRECTORY + 'locations.csv')
        pickup_times = pd.read_csv(DIRECTORY + 'pickup_times.csv')
        pickup_times['date'] = pd.to_datetime(pickup_times['iso_8601_timestamp'])
        pickup_times = pickup_times.drop(['iso_8601_timestamp'], axis=1)

        X, y = prepare_data(pickup_times)

        model = Sequential()

        model.add(Dense(124, kernel_initializer='normal', activation='selu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, kernel_initializer='normal', activation='selu'))
        model.add(Dropout(0.2))
        model.add(Dense(512, kernel_initializer='normal', activation='selu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, kernel_initializer='normal', activation='selu'))
        model.add(Dropout(0.2))
        model.add(Dense(124, kernel_initializer='normal', activation='selu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

        history = model.fit(X, y, epochs=200, validation_split = 0.2, batch_size=128, verbose=1).history
        model.save(DIRECTORY + "wolt_model.h5")
        with open(DIRECTORY + 'wolt_learning_history.pkl','wb') as f:
            pickle.dump(history, f)
    else:
        model = load_model(DIRECTORY + 'wolt_model.h5')
        with open(DIRECTORY + 'wolt_learning_history.pkl','rb') as f:
            history = pickle.load(f)

def exec_model(l, t):
    inpt = pd.DataFrame([[int(l), datetime.strptime(t, "%Y-%m-%dT%H:%M:%S")]], columns = ['location_id', 'date'])
    X, _ = prepare_data(inpt)

    with graph.as_default():
        result = (model.predict(X)*90).astype(int)[0][0]
    return str(result)

@app.route('/predict')
def predict():
    location_id_str = request.args.get('location_id')
    time = request.args.get('time')

    answer = exec_model(location_id_str, time)
    answer = {'estimated_time':answer}
    return jsonify(answer)

if __name__ == '__main__':
    load_my_model()
    app.run()





