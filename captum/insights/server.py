import logging
import socket
import threading
from typing import Optional

from flask import Flask, jsonify, render_template

app = Flask(
    __name__, static_folder="frontend/build/static", template_folder="frontend/build"
)


def namedtuple_to_dict(obj):
    if hasattr(obj, "_asdict"):  # detect namedtuple
        return dict(zip(obj._fields, (namedtuple_to_dict(item) for item in obj)))
    elif isinstance(obj, str):  # iterables - strings
        return obj
    elif hasattr(obj, "keys"):  # iterables - mapping
        return dict(
            zip(obj.keys(), (namedtuple_to_dict(item) for item in obj.values()))
        )
    elif hasattr(obj, "__iter__"):  # iterables - sequence
        return type(obj)((namedtuple_to_dict(item) for item in obj))
    else:  # non-iterable cannot contain namedtuples
        return obj


@app.route("/fetch", methods=["POST"])
def fetch():
    return jsonify(namedtuple_to_dict(visualizer.visualize()))


@app.route("/")
@app.route("/<int:id>")
def index(id=0):
    return render_template("index.html")


def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port


def start_server(_viz, port: Optional[int] = None, debug=True):
    global visualizer
    visualizer = _viz
    if port is None:
        port = get_free_tcp_port()

    print("starting server on port:", port)

    if not debug:
        log = logging.getLogger("werkzeug")
        log.disabled = True
        app.logger.disabled = True
        threading.Thread(target=app.run, kwargs={"port": port}).start()
    else:
        app.run(use_reloader=True, port=5000, debug=True, threaded=True)
    return port


if __name__ == "__main__":
    start_server()
