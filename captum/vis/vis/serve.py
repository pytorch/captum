from flask import Flask, render_template, jsonify


print("Starting webserver")

app = Flask(
    __name__, static_folder="frontend/build/static", template_folder="frontend/build"
)


def convert_img_base64(img, denormalize=False):
    if denormalize:
        img = img / 2 + 0.5

    buff = BytesIO()

    plt.imsave(buff, img)
    base64img = base64.b64encode(buff.getvalue()).decode("utf-8")
    return base64img


def namedtuple_asdict(obj):
    if hasattr(obj, "_asdict"):  # detect namedtuple
        return dict(zip(obj._fields, (namedtuple_asdict(item) for item in obj)))
    elif isinstance(obj, str):  # iterables - strings
        return obj
    elif hasattr(obj, "keys"):  # iterables - mapping
        return dict(zip(obj.keys(), (namedtuple_asdict(item) for item in obj.values())))
    elif hasattr(obj, "__iter__"):  # iterables - sequence
        return type(obj)((namedtuple_asdict(item) for item in obj))
    else:  # non-iterable cannot contain namedtuples
        return obj


# TODO remove GET method when done testing
@app.route("/fetch", methods=["POST", "GET"])
def fetch():
    print("fetched!")
    return jsonify(namedtuple_asdict(visualizer.visualize(2)))


@app.route("/")
@app.route("/<int:id>")
def index(id=0):
    return render_template("index.html")
    # return "".join([img_html(out.base), img_html(out.modified)])


def start_server(_viz, port: int = 5000):
    global visualizer  # is there a better way of passing the visualizer object?
    visualizer = _viz
    app.run(use_reloader=True, port=port, threaded=True)


if __name__ == "__main__":
    start_server()
