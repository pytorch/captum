#!/usr/bin/env python3
import ipywidgets as widgets

from collections import namedtuple
from IPython.core.display import display
from random import shuffle


GLOBAL_PADDING = widgets.Layout(height="500px", overflow_y="auto")


class EmptyDatasetException(Exception):
    """Raised when dataset is empty on widget load"""

    pass


class Widget:
    """Parent Class representing a widget

    This is the parent class for all widgets under the captum.widgets module
        and it implements a set of shared methods common to all widgets. Any
        class that wishes to inherit this class must define the property
        self.final_view which must contain an ipywidgets.Widget.
    """

    def __init__(self):
        pass

    def _validate_input(self, wrapped_model, test_sets):
        """Makes sure that the model's output and prediction_classes are the same
            size"""
        if len(test_sets) == 0:
            raise EmptyDatasetException
        for test_set in test_sets:
            if len(test_set) == 0:
                raise EmptyDatasetException
            sample_input = test_set[0]
            input_size = len(sample_input.raw_inputs)
            feature_count = len(wrapped_model.get_feature_categories())
            feature_name_count = len(wrapped_model.get_feature_names())

            if input_size != feature_count:
                raise Exception(
                    "Model feature categories don't match input \
                    length: {} != {}".format(
                        feature_count, input_size
                    )
                )
            if input_size != feature_name_count:
                raise Exception(
                    "Model feature names don't match input \
                    length: {} != {}".format(
                        feature_name_count, input_size
                    )
                )

    def render(self):
        """Renders the widget to the notebook"""
        display(self.final_view)

    def get_root(self):
        """Returns an ipywidgets.Widget that represents the object"""
        return self.final_view


class ModelWrapper:
    """Wrapper for a general model for widget use

    Models come in a variety of shapes and sizes, and in order for the Widgets
        defined here to work properly, we need a common interface. That
        interface is defined here and all models passed to widgets must
        define the following methods:
            get_prediction
            get_prediction_classes
            get_feature_categories
            get_feature_names
        and the following attributes:
            model (a pytorch model)
    The user may otherwise define the wrapper as they see fit to accomedate
        the needs of their specific model.
    """

    def __init__(self, pytorch_model):
        self.model = pytorch_model

    def get_prediction(self, input, meta_data):
        """Gets the output for a given input

        Args:
            input (tuple of tensors): The input will come from whatever the user
                passes into the input section of the DataEntry class defined below.
            meta_data (any): Any extra arguments comming from the data_entries that
                the user created. See data_entry below.

        Returns:
            a torch.Tensor of softmax values that line up with the
                prediction classes returned by get_prediction_classes.
            For example:
                get_prediction(x) -> [.9, .05, .1, .0]
                get_prediction_classes -> ["Dog", "Cat", "Baby", "Turtle"]
                meaning the model predicts Dog to be the right label

        If the user needs to manipulate the input data stored in the DataEntry
            before it is passed to the model, that can be done here since
            the inputs will be DataEntry.inputs
        """
        raise NotImplementedError

    def get_prediction_classes(self):
        """Gets the prediction classes or labels for the predictions of the
        model

        Returns a list of strings that correspond to the different predictions
            the model can make. For example a binary classifier may be wrapped
            as follows:
                get_prediciton_classes() -> ["Dog", "Not Dog"]
        """
        raise NotImplementedError

    def get_prediction_indices(self, prediction):
        """Gets the positive predicition indices from a prediction

        Args:
            prediction (torch tensor): The output of self.get_prediction. The user
                needs to return the indices where their model is providing a
                positive classification
        Returns: list of indices where the model is making a positive classification

        Ex:
            get_prediction() -> [1.0, .67, .02, .20]
            get_prediction_indices(**) -> [0, 1]

        If the user has different cuts at different classes, here is the place to
            implement that.
        """
        raise NotImplementedError

    def get_feature_categories(self):
        """Gets the feature categories for all of the inputs along axis 0 of
        the input tensor

        Returns a list of strings corresponding to the types of each feature, so
            that they can be displayed properly. Valid types are:
            "image", "text_tokens", "text_characters", "scalar", "default"

        Example:
            If the input tensor looks like this
                [[Image_Tensor], [Text]]
            Then get_feature_categories() should return ["image", "text"]
        """
        raise NotImplementedError

    def get_feature_names(self):
        """Gets the feature names for all of the inputs along the 0th dim

        Returns a list of names for the 0th order features that are specified
            in get_feature_categories

        Example:
            If the input tensor looks like this
                [[Image_Tensor], [Text]]
            Then get_feature_categories() could return ["post image", "comment"]
        """
        raise NotImplementedError

    def process_attributions(self, attributions):
        """For advanced users who need to maniputlate the attributions to fit
        their raw data (e.g. pytext)

        Some models require special embedding formats, and to provide flexibility
            for this, one may specify this function to format them and return them
            as a list of attributions for the different outputs.

        Input: raw attribution Tensor
        Output: list of attributions in order of raw inputs
        """
        return attributions


"""Model Agnostic way to store data and pass it to the widget.

The user must process their own data (since it would be imposible for a general
    purpose widget to do this alone), and place the data in the following
    class.

Parameters:
    raw_inputs (tuple of any): Unprocessed input. For example, text
    inputs (tuple of torch tensor): Tensorized inputs to the model.
        model(input, meta_data) should produce a prediction
    target_indeces (list of ints): the indices of the true label in the models output
    meta_data (any): any extra parameters to be passed to the model. Will be passed to
        along to the wrapper that the user defines as shown above.

Example Use:
    data = []
    with open("test.tsv") as test_set:
        line = test_set.readline()
        (raw_input, raw_label) = line.split('\t')
        input = embeddings(raw_input)
        d = DataEntry(raw_input, input, 1, None)
        data.append(d)
"""
DataEntry = namedtuple(
    "DataEntry", ["raw_inputs", "inputs", "target_indices", "meta_data"]
)


class WidgetDataset:
    """A data container optimized for fetching widget datarecords"""

    def __init__(self, wrapped_model, keep_empty=False, name="dataset"):
        self.items = []
        self._predicted_indices = []
        self.name = name
        self.wrapped_model = wrapped_model
        self.class_names = wrapped_model.get_prediction_classes()
        self.target_lookup = [set() for i in range(len(self.class_names))]
        self.prediction_lookup = [set() for i in range(len(self.class_names))]
        self.type = {}
        self.keep_empty = keep_empty

    def add(self, data_entry):
        """Add a data entry to the dataset

        Args:
            data_entry: a DataEntry object or list of DataEntry objects as
                defined in helpers.py
        """
        if isinstance(data_entry, list):
            for item in data_entry:
                self.add(item)
            return
        prediction = self.wrapped_model.get_prediction(
            data_entry.inputs, meta=data_entry.meta_data
        )[0]
        p_indices = self.wrapped_model.get_prediction_indices(prediction)
        if not self.keep_empty and len(p_indices) + len(data_entry.target_indices) == 0:
            return
        self._predicted_indices.append(p_indices)
        self.items.append(data_entry)
        new_index = len(self.items) - 1
        for target in data_entry.target_indices:
            self.target_lookup[target].add(new_index)
        for prediction in self._predicted_indices[new_index]:
            self.prediction_lookup[prediction].add(new_index)
        # check if there are any indices that don't match up
        if len(data_entry.target_indices) != len(self._predicted_indices[new_index]):
            self.type[new_index] = "Incorrect"
            return

        l1 = sorted(data_entry.target_indices)
        l2 = sorted(self._predicted_indices[new_index])
        while len(l1) and len(l2):
            i1 = l1.pop()
            i2 = l2.pop()
            if i1 != i2:
                self.type[new_index] = "Incorrect"
                return
        self.type[new_index] = "Correct"

    def find(
        self,
        target_classes,
        predicted_classes,
        limit=20,
        random=True,
        exact=False,
        instance_type="All",
    ):
        """Method for fetching records from dataset

            Args:
                target_classes: a list of target classnames to get
                predicted_classes: a list of predicted classnames to get
                limit: number of records to return
                random: randomly sample records to return
                exact: return labels that fit restrictions exactly if true
                    return labels that at least fit the restrictions
                instance_type: one of "All", "Correct", "Incorrect". Determines
                    what type of instances to show.

            Returns: a list of indices that point to dataentries that
                fit the specifications of the parameters
        """
        target_indices = [
            self.class_names.index(class_name) for class_name in target_classes
        ]
        predicted_indices = [
            self.class_names.index(class_name) for class_name in predicted_classes
        ]
        valid_indices_sets = []
        if len(target_classes) > 0:
            valid_indices_sets += [self.target_lookup[i] for i in target_indices]
        if len(predicted_classes) > 0:
            valid_indices_sets += [self.prediction_lookup[i] for i in predicted_indices]
        if len(valid_indices_sets) == 0:
            if len(target_classes) + len(predicted_classes) == 0:
                valid_indices_sets = [set(range(len(self.items)))]
            else:
                return []
        valid_indices_set = valid_indices_sets[0].intersection(*valid_indices_sets)
        valid_indices = list(valid_indices_set)
        if exact:
            valid_indices = [
                valid_index
                for valid_index in valid_indices
                if len(self.items[valid_index].target_indices) == len(target_indices)
                and len(self._predicted_indices[valid_index]) == len(predicted_indices)
            ]
        if random:
            shuffle(valid_indices)

        if instance_type != "All":
            valid_indices = [
                valid_index
                for valid_index in valid_indices
                if self.type[valid_index] == instance_type
            ]

        if len(valid_indices) > limit:
            valid_indices = valid_indices[:limit]
        return valid_indices

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def wrap_dataset(
    data_iterator, unpacker, wrapped_model, limit=None, keep_empty=False, name="dataset"
):
    """Wrap a dataset for easy widget access

    Args:
        data_iterator: a iterator over a data input stream (file)
        unpacker: a function that converts an item from the iterator into
            a DataEntry object (defined in helpers.py)
        wrapped_model: a wrapped moded (defined in helpers.py)
        name (str): The display name of the dataset (to be rendered)

    Returns: a wrapped dataset (WidgetDataset defined in helpers.py) for internal
        widget use"""
    dataset = WidgetDataset(wrapped_model, keep_empty, name=name)
    for item in data_iterator:
        dataset.add(unpacker(item))
        if limit is not None and len(dataset) > limit:
            break
    return dataset
