### Integrated Gradients

This section of the documentation shows how to apply integrated gradients on
models with different types of parameters and inputs using Captum.

## Description

[Integrated gradients](https://arxiv.org/pdf/1703.01365.pdf) is a simple, yet powerful axiomatic attribution method that requires almost no modification of the original network. It can be used for augmenting accuracy metrics, model debugging and feature or rule extraction.

Captum provides a generic implementation of integrated gradients that can be used with any PyTorch model.
In this section of the tutorial we will describe how to apply integrated gradients for output predictions.
Here is an example code snippet that reproduces the results from the [original paper](https://arxiv.org/pdf/1703.01365.pdf) (page 10).

First, let's create a sample ToyModel.

```
import torch
import torch.nn as nn
import torch.nn.functional as F
class ToyModel(nn.Module):

"""
Example toy model from the original paper (page 10)

https://arxiv.org/pdf/1703.01365.pdf


f(x1, x2) = RELU(ReLU(x1) - 1 - ReLU(x2))
"""

def __init__(self):
    super().__init__()

def forward(self, input1, input2):
    relu_out1 = F.relu(input1)
    relu_out2 = F.relu(input2)
    return F.relu(relu_out1 - 1 - relu_out2)
```

Second, let's apply integrated gradients on the toy model's output layer using sample data.
Below code snippet computes the attribution of output with respect to the inputs.
`attribute` method of `IntegratedGradients` class returns input attributions which
have the same size and dimensionality as the inputs and an approximation error which
is computed based on the completeness property of the integrated gradients.

```
from captum.attr import IntegratedGradients
model = ToyModel()

# defining model input tensors
input1 = torch.tensor([3.0], requires_grad=True)
input2 = torch.tensor([1.0], requires_grad=True)

# defining baselines for each input tensor
baseline1 = torch.tensor([0.0])
baseline2 = torch.tensor([0.0])

# defining and applying integrated gradients on ToyModel and the
ig = IntegratedGradients(model)
attributions, approximation_error = ig.attribute((input1, input2),
                                                 baselines=(baseline1, baseline2),
                                                 method='gausslegendre')
output

...................

attributions: (tensor([1.5000], grad_fn=<MulBackward0>),
               tensor([-0.5000], grad_fn=<MulBackward0>))

approximation_error (aka delta): 1.1801719665527344e-05
```

Now let's look at a simple classification model. The network architecture of this
classification model is based on the network described in:
https://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class ToySoftmaxModel(nn.Module):
"""
Model architecture from:

https://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/
"""

def __init__(self, num_in, num_hidden, num_out):
    super().__init__()
    self.num_in = num_in
    self.num_hidden = num_hidden
    self.num_out = num_out
    self.lin1 = nn.Linear(num_in, num_hidden)
    self.lin2 = nn.Linear(num_hidden, num_hidden)
    self.lin3 = nn.Linear(num_hidden, num_out)
    self.softmax = nn.Softmax(dim=1)

def forward(self, input):
    lin1 = F.relu(self.lin1(input))
    lin2 = F.relu(self.lin2(lin1))
    lin3 = self.lin3(lin2)
    return self.softmax(lin3)
```

Now, let's apply integrated gradients on the toy classification model defined
above using inputs that contain a range of numbers. We also choose arbitrary
target class (target_class_index: 5) which we use to attribute our predictions to.
Similar to previous example the output of attribution is a tensor with the same
dimensionality as the inputs and an approximation error computed based on the
completeness property of integrated gradients.

```
from captum.attr import IntegratedGradients
num_in = 40
input = torch.arange(0.0, num_in * 1.0, requires_grad=True).unsqueeze(0)

# 10-class classification model
model = SoftmaxModel(num_in, 20, 10)

# attribution score will be computed with respect to target class
target_class_index = 5

# applying integrated gradients on the SoftmaxModel and input data point
ig = IntegratedGradients(model)
attributions, approximation_error = ig.attribute(input, target=target)

# The input and returned corresponding attribution have the same shape. The attributions
# are always returned in a tuples. Even if the input is a tensor it will internally
# be wrapped with a tuple for generality.

output

...................

attributions: (tensor([[ 0.0000,  0.0014,  0.0012,  0.0019,  0.0034,  0.0020, -0.0041,  
          0.0085, -0.0016,  0.0111, -0.0114, -0.0053, -0.0054, -0.0095,  0.0097, -0.0170,
          0.0067,  0.0036, -0.0296,  0.0244,  0.0091, -0.0287,  0.0270,  0.0073,
         -0.0287,  0.0008, -0.0150, -0.0188, -0.0328, -0.0080, -0.0337,  0.0422,
          0.0450,  0.0423, -0.0238,  0.0216, -0.0601,  0.0114,  0.0418, -0.0522]],
       grad_fn=<MulBackward0>),)

approximation_error (aka delta): 0.00013834238052368164

assert attributions.shape == input.shape
```

Now, let's look at a model that besides input tensors takes input arguments of
other types. For example, a single integer value.
Those arguments are passed as `additional_forward_args` to `attribute` method. In
the example below, we also demonstrate how to apply integrated gradients to a batch
of samples. The first dimension of the input corresponds to the batch size.
In this case batch size is equal to two.

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class ToyModel_With_Additional_Forward_Args(nn.Module):
    """
        Slightly modified example model from the paper
        https://arxiv.org/pdf/1703.01365.pdf
        f(x1, x2) = RELU(ReLU(x1 - 1) - ReLU(x2))
    """
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, index):
        relu_out1 = F.relu(input1 - 1)
        relu_out2 = F.relu(input2)
        return F.relu(relu_out1 - relu_out2)[:, index]
```

Now, let's apply integrated gradients on the model defined above and specify
`additional_forward_args` parameter in addition to others.

```
from captum.attr import IntegratedGradients

input1 = torch.tensor([[1.0, 3.0], [3.0, 5.0]], requires_grad=True)
input2 = torch.tensor([[1.0, 4.0], [0.0, 2.0]], requires_grad=True)
# Initializing our toy model
model = ToyModel_With_Additional_Forward_Args()
# Applying integrated gradients on the input
ig = IntegratedGradients(model)
(input1_attr, input2_attr), delta = ig.attribute((input1, input2), n_steps=100,
                                    additional_forward_args=1)
output
.........
input1_attr: tensor([[0.0000, 0.0000],
                     [0.0000, 3.3428]], grad_fn=<MulBackward0>)
input2_attr:  tensor([[ 0.0000,  0.0000],
                      [0.0000, -1.3371]], grad_fn=<MulBackward0>)
approximation_error (aka delta): 0.005693793296813965
```

## More details on how to apply integrated gradients on larger DNN networks can be found here

* [A simple classification model and CIFAR Dataset] (https://github.com/pytorch/captum/blob/master/tutorials/CIFAR_TorchVision_Interpret.ipynb)
* [Torchvision's ResNet18 Model using handpicked images] (TODO add link here)
* [Sentiment classification model using TorchText and IMDB Dataset] (https://github.com/pytorch/captum/blob/master/tutorials/IMDB_TorchText_Interpret.ipynb)
* [Visual Question Answering Model] (https://github.com/pytorch/captum/blob/master/tutorials/Multimodal_VQA_Interpret.ipynb)
* [A simple DNN with 2 hidden layers and Titanic Dataset] (https://github.com/pytorch/captum/blob/master/notebooks/Titanic_Basic_Interpret.ipynb)
