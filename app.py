import os

import spacy
from flask import Flask, render_template, jsonify, request

from components import BingSearch

app = Flask(__name__)
# subscription key
bing_subscription_key = "adca0bb823d44e98a1f14c3c185d7934"

bingSearch = BingSearch(bing_subscription_key)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/answer-question', methods=['POST'])
def analyzer():
    # question = request.data
    data = request.get_json()
    question = data.get('question')
    print(question)
    answers = bingSearch.getAnswer(str(question))
    return jsonify(answers)


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000)
	app.run()
