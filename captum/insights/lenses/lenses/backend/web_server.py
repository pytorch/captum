import os
from abc import ABC
from pathlib import Path
from typing import Optional

import flask
import requests

this_filepath = Path(os.path.abspath(__file__))
this_dirpath = this_filepath.parent.parent


class ApiController(ABC):
    def __init__(self, name: str, import_name: str, url_prefix: str):
        self._blueprint = flask.Blueprint(name, import_name, url_prefix=url_prefix)
        self.add_url_rule = self._blueprint.add_url_rule
        self._name = name

    def _url_for(self, method_name: str, **kwargs) -> str:
        return flask.url_for(f".{method_name}", **kwargs)

    @property
    def blueprint(self) -> flask.Blueprint:
        return self._blueprint

    @property
    def name(self) -> str:
        return self._name


class WebServer:
    def __init__(self, service: ApiController, dev_frontend_host: str = None):
        app = flask.Flask(
            __name__, static_folder=str(this_dirpath.joinpath("frontend", "build"))
        )
        self.app = app
        self.dev_frontend_host = dev_frontend_host
        serve_func = self._serve_dev if dev_frontend_host else self._serve
        self.app.register_blueprint(service.blueprint)
        self.service = service

        app.add_url_rule("/lens", view_func=self._get_active_lens)
        app.add_url_rule("/", view_func=serve_func)
        app.add_url_rule("/<path:subpath>", view_func=serve_func)

    def _get_active_lens(self):
        return flask.jsonify({"name": self.service.name})

    def _serve_dev(self, subpath: Optional[str] = ""):
        return requests.get(f"{self.dev_frontend_host}/{subpath}").content

    def _serve(self, subpath: Optional[str] = "index.html"):
        return flask.send_from_directory(self.app.static_folder, subpath)

    # TODO: get any available port as default port
    def start(self, port: int = 8031):
        self.app.run(port=port)
