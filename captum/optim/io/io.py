# TODO: redo show using display or register handler for jupyter display directly
# maybe we could even have subtypes of tensors that are "ImageTensors" or "ActivationTensors" etc
from lucid.misc.io import show as lucid_show


def show(thing):
    if len(thing.shape) == 3:
        numpy_thing = thing.cpu().detach().numpy().transpose(1, 2, 0)
    elif len(thing.shape) == 4:
        numpy_thing = thing.cpu().detach().numpy()[0].transpose(1, 2, 0)
    lucid_show(numpy_thing)
