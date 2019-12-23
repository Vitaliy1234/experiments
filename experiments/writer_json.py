import json


def write(data, file_name):
    file = json.dumps(data)
    with open(file_name, 'w') as hfile:
        hfile.write(file)


def read(file_name):
    with open(file_name, 'r') as hfile:
        data = json.load(hfile)

    return data
