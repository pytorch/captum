#!/usr/bin/env python3
import logging
import os
import socket
import threading
from time import sleep
from typing import Optional

from captum.log import log_usage
from flask import Flask, jsonify, render_template, request
from flask_compress import Compress
from torch import Tensor

app = Flask(
    __name__, static_folder="frontend/build/static", template_folder="frontend/build"
)
visualizer = None
port = None
Compress(app)


def namedtuple_to_dict(obj):
    if isinstance(obj, Tensor):
        return obj.item()
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


@app.route("/attribute", methods=["POST"])
def attribute():
    # force=True needed for Colab notebooks, which doesn't use the correct
    # Content-Type header when forwarding requests through the Colab proxy
    r = request.get_json(force=True)
    return jsonify(
        namedtuple_to_dict(
            visualizer._calculate_attribution_from_cache(
                r["inputIndex"], r["modelIndex"], r["labelIndex"]
            )
        )
    )


@app.route("/fetch", methods=["POST"])
def fetch():
    # force=True needed, see comment for "/attribute" route above
    visualizer._update_config(request.get_json(force=True))
    visualizer_output = visualizer.visualize()
    clean_output = namedtuple_to_dict(visualizer_output)
    return jsonify(clean_output)


@app.route("/init")
def init():
    return jsonify(visualizer.get_insights_config())


@app.route("/")
def index(id=0):
    return render_template("index.html")


def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port


def run_app(debug: bool = True, bind_all: bool = False):
    if bind_all:
        app.run(port=port, use_reloader=False, debug=debug, host="0.0.0.0")
    else:
        app.run(port=port, use_reloader=False, debug=debug)


@log_usage()
def start_server(
    _viz,
    blocking: bool = False,
    debug: bool = False,
    _port: Optional[int] = None,
    bind_all: bool = False,
):
    global visualizer
    visualizer = _viz

    global port
    if port is None:
        os.environ["WERKZEUG_RUN_MAIN"] = "true"  # hides starting message
        if not debug:
            log = logging.getLogger("werkzeug")
            log.disabled = True
            app.logger.disabled = True

        port = _port or get_free_tcp_port()
        # Start in a new thread to not block notebook execution
        t = threading.Thread(
            target=run_app, kwargs={"debug": debug, "bind_all": bind_all}
        )
        t.start()
        sleep(0.01)  # add a short delay to allow server to start up
        if blocking:
            t.join()

    print(f"\nFetch data and view Captum Insights at http://localhost:{port}/\n")
    return port
