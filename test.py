def test_only(function, *args):

    args = list([*args])

    try:
        function_state = True
        function(*args)
    except Exception as e:
        print('Error with: {}'.format(args[0]))
        function_state = False
        print(e)

    assert function_state == True

def test_output(function, *args):

    args = list([*args])

    try:
        function_state = True
        output = function(*args)
    except Exception as e:
        print('Error with: {}'.format(args[0]))
        function_state = False
        print(e)

    assert function_state == True

    return output

if '__main__' == __name__:
    print('')
