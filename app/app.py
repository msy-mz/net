import os
from flask import Flask, send_from_directory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 向上到 net/
STATIC_DIR = os.path.join(BASE_DIR, 'frontend')

print("BASE_DIR:", BASE_DIR)
print("STATIC_DIR:", STATIC_DIR)
print("index.html exists:", os.path.exists(os.path.join(STATIC_DIR, 'index.html')))

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8888, debug=True)
