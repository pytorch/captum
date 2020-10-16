# import os
from pprint import pprint

import numpy as np
import tensorflow as tf
import torch
from clarity.pytorch.inception_v1 import GS_SAVED_WEIGHTS_URL, GoogLeNet, googlenet
from lucid.misc.io import load

# from lucid.misc.io.loading import _load_graphdef_protobuf
from lucid.misc.io.writing import write_handle
from lucid.modelzoo.vision_models import InceptionV1
from lucid.optvis.render import import_model

# tf_path = os.path.abspath('./inception_v1.pb')  # Path to our TensorFlow checkpoint
# with open(tf_path, 'rb') as f:
#     graph_def = _load_graphdef_protobuf(f)
# pprint(tf_vars)

inception_v1_tf = InceptionV1()

# better ds?
node_info = dict((n.name, n) for n in inception_v1_tf.graph_def.node)

# interactive
op_types = set()
for node in inception_v1_tf.graph_def.node:
    op_types.add(node.op)
pprint(op_types)

aconst = None
for node in inception_v1_tf.graph_def.node:
    if node.op == "Const":
        print(node.name, node.op)
        aconst = node
        break

sess = tf.InteractiveSession()
tf.import_graph_def(inception_v1_tf.graph_def)
graph = tf.get_default_graph()


for op in graph.get_operations():
    if op.type == "Const":
        print(op.name, op.type)


# Testing our reimplementation
img_tf = load(
    "https://lucid-static.storage.googleapis.com/building-blocks/examples/dog_cat.png"
)
img_pt = torch.as_tensor(img_tf.transpose(2, 0, 1))[None, ...]

fresh_import = True

if fresh_import:
    net = GoogLeNet(transform_input=True)
    net.import_weights_from_tf(inception_v1_tf)

    tmp_dst = "/tmp/inceptionv1_weights.pth"
    torch.save(net.state_dict(), tmp_dst)
    with write_handle(GS_SAVED_WEIGHTS_URL, "wb") as handle:
        with open(tmp_dst, "rb") as tmp_file:
            handle.write(tmp_file.read())
else:
    net = googlenet(pretrained=True)

# forward pass PyTorch
out_pt = net(img_pt).detach()

latest_op_name = "softmax2"
# forward pass TF

with tf.Graph().as_default(), tf.Session() as sess:
    t_img = tf.placeholder("float32", [None, None, None, 3])
    T = import_model(inception_v1_tf, t_img, t_img)
    out_tf = T(latest_op_name).eval(feed_dict={t_img: img_tf[None]})

# diagnostic
print(f"\nDiagnostics… evaluating at '{latest_op_name}'")
print(
    f"PyTorch: {tuple(out_pt.shape)} µ: {out_pt.mean().item():.3f}, "
    f"↓: {out_pt.min().item():.1f}, ↑: {out_pt.max().item():8.3f}"
)
print(
    f"TnsrFlw: {tuple(out_tf.shape)} µ: {out_tf.mean().item():.3f}, "
    f"↓: {out_tf.min().item():.1f}, ↑: {out_tf.max().item():8.3f}"
)

if len(out_pt.shape) == 4:
    mean_error = np.abs(out_tf.transpose(0, 3, 1, 2) - out_pt.numpy()).mean()
else:
    mean_error = np.abs(out_tf - out_pt.numpy()).mean()
print(f"Mean Error: {mean_error:.5f}")
