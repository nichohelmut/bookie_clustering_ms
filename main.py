import os

from flask import Flask, jsonify
from aa import AA
from flask_cors import CORS
from flask_sslify import SSLify

app = Flask(__name__)
CORS(app, supports_credentials=True)
sslify = SSLify(app)


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    """healthcheck"""
    return "Still alive", 200


@app.route("/")
def main():
    aa = AA()
    response = aa.run()
    return jsonify(response), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
