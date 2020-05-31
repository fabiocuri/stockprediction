import sys
import json
import time
import pyrebase

if '__main__' == __name__:

    # Configuration
    params_f = str(sys.argv[1])

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
    
    for folder in ['DJIA', 'NASDAQ', 'SP500', 'WILSHIRE']:

        users = db.child('{}_Params'.format(folder)).get()

        avg, count = 0, 0
        for user in users.each():
            for key in user.val():
                avg += float(user.val()[key]['Back-testing accuracy'])
                count += 1

        data = {'Back-testing accuracy': round(avg/count,2)}
        db.child('{}_Accuracy'.format(folder)).remove()
        db.child('{}_Accuracy'.format(folder)).push(data)
        print('Back-testing accuracy exported to Firebase!')
