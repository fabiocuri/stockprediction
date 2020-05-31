import sys
import pyrebase
from get_data import *
from models import *
from firebase_actions import *
from test import *

if '__main__' == __name__:

    # Configuration
    stock, index, params_f = str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])

    with open(params_f) as f:
        parameters = json.load(f)
        for key in parameters:
            globals()[key] = parameters[key]

    config = {"apiKey": firebase_key,
              "authDomain": "{}.firebaseapp.com".format(firebase_path),
              "databaseURL": "https://{}.firebaseio.com".format(firebase_path),
              "storageBucket": "{}.appspot.com".format(firebase_path)}

    firebase_app_ = pyrebase.initialize_app(config)
    db = firebase_app_.database()
    
    index = index.split("'")[1].split("<")

    # Get stock data
    stock_data = test_output(get_historical_data, stock, years, period, features, alphavantage_key)

    # Hyper-parameter tuning
    if dt.datetime.today().weekday() in [tuning_day, tuning_day+1]:
        test_only(hyperparameter_tuning, stock, stock_data, length_backtesting, features, steps, training, period, years, num_exps, db, index)

    # Train current day model and perform predictions
    if tuning_day == -1:
        params = []
    else:
        params = test_output(retrieve_params_firebase, stock, db, index)
    test_only(predict_tomorrow, stock, stock_data, num_exps, steps, training, db, params, index)
