from flask import Flask
from flask import request
import joblib

app = Flask(__name__)

# TODO:
# wire in svg to png logic
# test predict route
# write end point the takes and entire board
@app.route('/', methods=['GET'])
def perdict():
    X = request.args.get('X')
    clf = joblib.load('classifier.pkl')
    return clf.predict(X)
