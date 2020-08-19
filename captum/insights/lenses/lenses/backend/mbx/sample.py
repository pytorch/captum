from PIL import Image
from enum import Enum
import flask
from io import BytesIO
import numpy as np
from scipy.io import wavfile
from . import audio_utils
from typing import TYPE_CHECKING, Dict, Any, Union, Callable

if TYPE_CHECKING:
    from .embedding_explorer_service import EmbeddingExplorerService


default_thumbnail = Image.new("RGB", (32, 32))


class Sample:
    def to_thumbnail(self) -> Union[Image.Image, BytesIO, flask.Response]:
        return default_thumbnail

    def to_dict(self) -> Dict[str, Any]:
        return {}


class GenericSampleAttrType(Enum):
    IMAGE = "image"
    AUDIO = "audio"


class GenericSampleAttrRequestHandler:
    def __init__(
        self,
        attr_type: GenericSampleAttrType,
        callback: Callable[["EmbeddingExplorerService", Sample], flask.Response],
    ):
        self.attr_type = attr_type
        self.callback = callback

    def __call__(self, service, sample):
        return self.callback(service, sample)


class GenericSampleAttrRequestHandlers:
    def __init__(self):
        self.handlers: Dict[str, GenericSampleAttrRequestHandler] = {}

    def add(
        self,
        attr_name: str,
        attr_type: GenericSampleAttrType,
        callback: Callable[["EmbeddingExplorerService", Sample], flask.Response],
    ):
        if attr_name in self.handlers:
            raise RuntimeError(f"attr request handler for {attr_name} already defined")
        handler = GenericSampleAttrRequestHandler(attr_type, callback)
        self.handlers[attr_name] = handler

    def items(self):
        return self.handlers.items()

    def __getitem__(self, key: str):
        return self.handlers[key]

    def __contains__(self, key: str):
        return key in self.handlers

    def enable_image_from_file(self, attr_name: str, dirname: str, filename: str):
        def callback(service, sample):
            return flask.send_from_directory(dirname, filename)

        self.add(attr_name, GenericSampleAttrType.IMAGE, callback)

    def enable_image_from_memory(
        self, attr_name: str, img: Union[Image.Image, BytesIO]
    ):
        def callback(service, sample):
            if isinstance(img, Image.Image):
                img_io = BytesIO()
                img.save(img_io, format="PNG")
                img_io.seek(0)
            else:
                img_io = img
            return flask.send_file(
                img_io, mimetype="image/png"
            )  # TODO support other formats?

        self.add(attr_name, GenericSampleAttrType.IMAGE, callback)

    def enable_audio_from_data(self, attr_name: str, data: np.array, rate: int):
        def callback(service, sample):
            bytes_io = BytesIO()
            wavfile.write(bytes_io, rate, data)
            return flask.send_file(bytes_io, mimetype="image/wav")

        self.add(attr_name, GenericSampleAttrType.AUDIO, callback)

    def enable_spectrogram_from_data(
        self, attr_name: str, data: np.array, rate: int, size: int = 32
    ):
        img_io = audio_utils.signal_to_spectrogram(data, rate, size)
        self.enable_image_from_memory(attr_name, img_io)


class GenericSample(Sample):
    def __init__(self):
        self.attr_request_handlers: Dict[
            str, Callable[[Sample], flask.Response]
        ] = GenericSampleAttrRequestHandlers()

    def to_payload(self) -> Dict[str, Any]:
        return {}

    def process_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        processed_payload = {}
        for key, value in payload.items():
            if key in self.attr_request_handlers:
                handler = self.attr_request_handlers[key]
                processed_value = {"type": handler.attr_type.value}
                processed_payload[key] = processed_value
            else:
                processed_payload[key] = value
        return processed_payload

    def to_dict(self) -> Dict[str, Any]:
        return {"payload": self.process_payload(self.to_payload())}
