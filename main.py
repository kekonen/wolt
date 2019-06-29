import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
app = Flask(__name__)

DIRECTORY = './'

locations = pd.read_csv(    DIRECTORY + 'locations.csv')
pickup_times = pd.read_csv( DIRECTORY + 'pickup_times.csv')
pickup_times['date'] = pd.to_datetime(pickup_times['iso_8601_timestamp'])
pickup_times = pickup_times.drop(['iso_8601_timestamp'], axis=1)

def get_median_int(p, s, e):
    dates = p[(p['date']> s) & (p['date']< e) & (p['location_id']==12)] # '2019-01-09 12:00:00'
    median = int(np.median(dates.sort_values(by=['pickup_time'])['pickup_time'].values))
    return median

@app.route('/median_pickup_time')
def median_pickup_time():
    location_id_str = request.args.get('location_id')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')

    answer = get_median_int(pickup_times, start_time, end_time)
    answer = {'median':answer}
    return jsonify(answer)

app.run()






