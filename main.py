import sys
import pyrebase
from get_data import *
from models import *
from firebase_actions import *
from test import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if '__main__' == __name__:
    
    # Configuration
    stock, index, params_f = str(sys.argv[1]), str(
        sys.argv[2]), str(sys.argv[3])

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
    stock_data = get_historical_data(stock=stock, years=years)

    # Missing values imputation
    stock_data = impute_missing_values(stock_data=stock_data)
    
    # Hyper-parameter tuning
    hyperparameter_tuning(stock=stock, stock_data=stock_data, years=years,
                          length_backtesting=length_backtesting, steps=steps, training=training, db=db, index=index)

    # Train current day model and perform predictions
    params = retrieve_params_firebase(stock=stock, db=db, index=index)
    predict_tomorrow(stock=stock, stock_data=stock_data, steps=steps,
                     training=training, db=db, index=index, params=params)
