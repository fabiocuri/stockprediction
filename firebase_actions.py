def export_firebase(data, stock, db, folder):
    ''' Exports data to Google Firebase '''

    db.child(folder).child(stock).remove()
    db.child(folder).child(stock).push(data)

def retrieve_params_firebase(stock, db, index):
    ''' Retrieves hyper-parameters from Google Firebase for a stock already tuned '''

    users = db.child('{}_Params'.format(index[0])).child(stock).get()
    params = {}
    for i in users.val():
        params['lstm_size'] = users.val()[i]['LSTM size']
        params['batch_size'] = users.val()[i]['Batch size']
        params['learning_rate'] = users.val()[i]['Learning rate']
        params['selected_features'] = users.val()[i]['Selected features']

    return params

if '__main__' == __name__:
    print('')
