import logging
import socket
import threading
from time import sleep
from typing import Optional

from flask import Flask, jsonify, render_template

app = Flask(
    __name__, static_folder="frontend/build/static", template_folder="frontend/build"
)
visualizer = None
port = None


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
    visualizer_output = visualizer.visualize()
    clean_output = namedtuple_to_dict(visualizer_output)
    return jsonify(clean_output)


@app.route("/")
def index(id=0):
    return render_template("index.html")


def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port


def run_app():
    app.run(port=port, use_reloader=False)


def start_server(
    _viz, blocking: bool = False, debug: bool = False, _port: Optional[int] = None
):
    global visualizer
    visualizer = _viz

    global port
    if port is None:
        if not debug:
            log = logging.getLogger("werkzeug")
            log.disabled = True
            app.logger.disabled = True

        port = _port or get_free_tcp_port()
        print(f"\nFetch data and view Captum Insights at http://localhost:{port}/\n")
        # Start in a new thread to not block notebook execution
        t = threading.Thread(target=run_app).start()
        sleep(0.01)  # add a short delay to allow server to start up
        if blocking:
            t.join()

    return port
