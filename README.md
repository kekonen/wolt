# Hello!

I made a mistake and did a little more than was asked and understood it too late...

There are 2 solutions in here. **Please, check out both of them.**

**But before all that please put your copies of files locations.csv and pickup_times.csv into 'input' folder**

## First (boring and easy) solution
First, is what was asked. It is a Flask server which is returning a median in the requested format. Easy, just do:
1. Install Python3 and PIP
2. Then install with pip/pip3 (depends of the way you installed pip) ```pip3 install pandas numpy flask```
3. Run `FLASK_APP=easy.py flask run`
4. Make a GET request `http://127.0.0.1:5000/median_pickup_time?location_id=12&start_time=2019-01-09T11:00:00&end_time=2019-01-09T12:00:00`

## Second (the interesting) solution
I didn't read the task properly and having a look at data which was provided in such a big amount I started preparing for **Predicting**. And I chose Neural Networks for that.
I set up a baseline model for predicting `pickup_time` depending on `pickup location` and `time`. 

[I am Jupyter Notebook, don't forget to check me out!](https://github.com/kekonen/wolt/blob/master/HelloWolt.ipynb)

You can go through comments and little analyse with Jupyter Notebook by link above.
Or you can just run the server:
1. Install Python3 and PIP
2. Then install with pip/pip3 (depends of the way you installed pip) ```pip3 install pandas numpy sklearn matplotlib flask tensorflow keras```
3. Run `FLASK_APP=main.py flask run`
4. Make a GET request `http://127.0.0.1:5000/predict?location_id=12&time=2019-01-09T11:00:00`

**Important!** To be able to use this solution with your data you need to teach the model first. For that replace `pickup_times.csv` in the input folder with actual data AND in the source code replace the flag `LEARN = False` to `True`.
Be prepared that it will eat some of your time
