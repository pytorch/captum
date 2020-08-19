from lenses.backend.web_server import ApiController
from typing import Optional, cast
from .workspace import Workspace
from .sample import Sample
from PIL import Image
from io import BytesIO
import json
import flask
import werkzeug

default_thumbnail = Image.new("RGB", (32, 32))


class EmbeddingExplorerService(ApiController):
    def __init__(
        self,
        workspace: Workspace,
        sample_window_size: Optional[int] = None,
        max_sample_window_size: Optional[int] = None,
        min_sample_window_size: Optional[int] = None,
    ):
        super().__init__("embedding_explorer", __name__, url_prefix="/mbx")

        self.workspace = workspace
        self.sample_window_size = sample_window_size
        self.max_sample_window_size = max_sample_window_size
        self.min_sample_window_size = min_sample_window_size
        self._add_url_rules()

    def _add_url_rules(self):
        self.add_url_rule("/init", view_func=self._init, methods=["GET"])
        self.add_url_rule("/workspace", view_func=self._workspace)
        self.add_url_rule("/explorer", view_func=self._explorer)
        self.add_url_rule("/sample", view_func=self._sample)
        self.add_url_rule("/sample_thumbnail", view_func=self._sample_thumbnail)
        self.add_url_rule("/generic_sample_attr", view_func=self._generic_sample_attr)

    def _init(self):
        return flask.jsonify(
            {
                "workspace_id": self.workspace.id,
                "sample_window_size": self.sample_window_size,
                "max_sample_window_size": self.max_sample_window_size,
                "min_sample_window_size": self.min_sample_window_size,
            }
        )

    def _workspace(self):
        # TODO handle potential inconsistency between
        # self.workspace.id and workspace id from request args
        return flask.jsonify(self.workspace.to_dict())

    def _explorer(self):
        id = int(flask.request.args.get("id"))
        explorer = self.workspace.explorers[id]
        return flask.jsonify(explorer.to_dict(pc_samples=False, ic_samples=False))

    def request_args_to_sample(
        self, args: werkzeug.datastructures.ImmutableMultiDict
    ) -> Sample:
        explorer_id = int(cast(str, args.get("explorer_id")))
        sample_id = int(cast(str, args.get("sample_id")))
        explorer = self.workspace.explorers[explorer_id]
        sample = explorer.get_sample(sample_id)
        return sample

    def _sample(self):
        sample = self.request_args_to_sample(flask.request.args)
        if isinstance(sample, dict):
            sample_dict = {"payload": sample}
        else:
            sample_dict = sample.to_dict()
        # using json.dumps instead of flask.jsonify to avoid key sorting,
        # the order specified by the user is important
        str_data = json.dumps(sample_dict)
        r = flask.Response(str_data)
        r.headers["Content-Type"] = "application/json"
        return r

    def _sample_thumbnail(self):
        sample = self.request_args_to_sample(flask.request.args)
        if hasattr(sample, "to_thumbnail"):
            thumbnail = sample.to_thumbnail()
        else:
            thumbnail = default_thumbnail

        if isinstance(thumbnail, Image.Image):
            img_io = BytesIO()
            thumbnail.save(img_io, format="PNG")
            img_io.seek(0)
            return flask.send_file(img_io, mimetype="image/png")
        elif isinstance(thumbnail, BytesIO):
            return flask.send_file(thumbnail, mimetype="image/png")
        elif isinstance(thumbnail, flask.Response):
            return thumbnail
        else:
            raise ValueError(f"invalid thumbnail type {type(thumbnail)}")

    def _generic_sample_attr(self):
        sample = self.request_args_to_sample(flask.request.args)
        attr_name: str = cast(str, flask.request.args.get("attr_name"))
        return sample.attr_request_handlers[attr_name](self, sample)
