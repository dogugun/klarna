from flask import Flask, request
from flask_cors import CORS
import json
# from diagnose import get_historical_summary
# from get_plot_data import get_historical_data
# from get_upper_and_lower_limits import get_bounds
# from optimize_results import optimize
from klarna.service import estimations

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/deneme')
def deneme():
    return 'deneme'

@app.route('/defaultscore')
def get_default_score():
    p_id = request.args.get("uuid")
    return estimations.get_default_score(p_id)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)
