from flask import Flask, Response, stream_with_context, send_from_directory
import time
import json
import random
import os
from urllib.parse import quote

app = Flask(__name__, static_folder="static")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, 'index.html')


def _list_directory(dirpath, base_url):
    try:
        entries = sorted(os.listdir(dirpath))
    except FileNotFoundError:
        return f"<h3>Not found: {dirpath}</h3>", 404
    items = []
    for name in entries:
        full = os.path.join(dirpath, name)
        if os.path.isdir(full):
            items.append(f"<li>📁 <a href=\"{base_url}{quote(name)}/\">{name}/</a></li>")
        else:
            items.append(f"<li>📄 <a href=\"{base_url}{quote(name)}\">{name}</a></li>")
    body = "<h2>Listing: %s</h2><ul>%s</ul>" % (os.path.basename(dirpath) or dirpath, '\n'.join(items))
    return body


@app.route('/files/dataset/')
@app.route('/files/dataset/<path:sub>')
def files_dataset(sub=None):
    root = os.path.join(os.getcwd(), 'Dataset_Sample')
    if not sub:
        return _list_directory(root, '/files/dataset/')
    target = os.path.join(root, sub)
    if os.path.isdir(target):
        return _list_directory(target, '/files/dataset/')
    if os.path.isfile(target):
        return send_from_directory(root, sub)
    return ("Not found", 404)


@app.route('/files/output/')
@app.route('/files/output/<path:sub>')
def files_output(sub=None):
    root = os.path.join(os.getcwd(), 'output_analysis')
    if not sub:
        return _list_directory(root, '/files/output/')
    target = os.path.join(root, sub)
    if os.path.isdir(target):
        return _list_directory(target, '/files/output/')
    if os.path.isfile(target):
        return send_from_directory(root, sub)
    return ("Not found", 404)


@app.route('/stream')
def stream():
    def event_stream():
        while True:
            data = {"t": int(time.time() * 1000), "v": random.uniform(0, 1)}
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.5)

    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, threaded=True)
