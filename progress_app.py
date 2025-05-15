from flask import Flask, jsonify
from src.utils.shared_status import get_progress

app = Flask(__name__)

@app.route('/get_progress', methods=['GET'])
def get_progress_route():
    return jsonify(get_progress())

if __name__ == '__main__':
    app.run(port=5002, debug=False, use_reloader=False)
