#!/usr/bin/env python3

import captum
import torch
import torchvision
from tests.helpers.basic import BaseTest


class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


class TinyAE(torch.nn.Module):
    def __init__(self):
        super(TinyAE, self).__init__()

        self.linear1 = torch.nn.Linear(100, 10)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, 100)

    def encode(self, x):
        x = self.linear1(x)
        return self.activation(x)

    def decode(self, z):
        return self.linear2(z)

    def forward(self, x):
        return self.decode(self.encode(x))

    
class ConvAE(torch.nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()
       
        self.conv1 = torch.nn.Conv2d(3, 16, 3)  
        self.conv2 = torch.nn.Conv2d(16, 4, 3)
       
        self.trans_conv1 = torch.nn.ConvTranspose2d(4, 16, 2)
        self.trans_conv2 = torch.nn.ConvTranspose2d(16, 3, 4)
        
        self.activation = torch.nn.ReLU()

    def encode(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x
        
    def decode(self, x):
        x = self.trans_conv1(x)   
        x = self.activation(x)
        x = self.trans_conv2(x)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))
    

class TestBasic(BaseTest):
    def test_basic_setup(self):

        model = TinyModel()
        ae = TinyAE()
        x = torch.randn(1, 100)

        batch_size = 1
        x = torch.randn(batch_size, 100)

        # Defining Latent Shift module
        attr = captum.attr.LatentShift(model, ae)

        # Computes counterfactual heatmap for class 3.
        outputs = attr.attribute(x, target=3, lambda_sweep_steps=10)
        assert (batch_size, 100) == outputs.shape
        
        # Computes counterfactual for class 3 and return counterfactuals.
        outputs = attr.attribute(x, target=3, lambda_sweep_steps=10, return_dicts=True)
        assert batch_size == len(outputs)
        for output in outputs:
            assert (100, ) == output["heatmap"].shape
            assert (10, 100) == output["generated_images"].shape
        
        
    def test_batches(self):

        model = TinyModel()
        ae = TinyAE()
        
        batch_size = 3
        x = torch.randn(batch_size, 100)

        # Defining Latent Shift module
        attr = captum.attr.LatentShift(model, ae)

        # Computes counterfactual heatmap for class 3.
        outputs = attr.attribute(x, target=3, lambda_sweep_steps=10)
        assert (batch_size, 100) == outputs.shape
        
        # Computes counterfactual for class 3 and return counterfactuals.
        outputs = attr.attribute(x, target=3, lambda_sweep_steps=10, return_dicts=True)
        assert batch_size == len(outputs)
        for output in outputs:
            assert (100, ) == output["heatmap"].shape
            assert (10, 100) == output["generated_images"].shape

        
class TestConv(BaseTest):
    def test_basic_setup(self):

        model = torchvision.models.resnet50(weights=None)
        ae = ConvAE()

        # Defining Latent Shift module
        attr = captum.attr.LatentShift(model, ae)
        
        batch_size = 1
        x = torch.randn(batch_size, 3, 200, 200)

        # Defining Latent Shift module
        attr = captum.attr.LatentShift(model, ae)

        # Computes counterfactual heatmap for class 3.
        outputs = attr.attribute(x, target=3, lambda_sweep_steps=10)
        assert (batch_size, 3, 200, 200) == outputs.shape
        
        # Computes counterfactual for class 3 and return counterfactuals.
        outputs = attr.attribute(x, target=3, lambda_sweep_steps=10, return_dicts=True)
        assert batch_size == len(outputs)
        for output in outputs:
            assert (3, 200, 200) == output["heatmap"].shape
            assert (10, 3, 200, 200) == output["generated_images"].shape

        
    def test_batches(self):

        model = torchvision.models.resnet50(weights=None)
        ae = ConvAE()

        # Defining Latent Shift module
        attr = captum.attr.LatentShift(model, ae)

        batch_size = 2
        x = torch.randn(batch_size, 3, 200, 200)

        # Defining Latent Shift module
        attr = captum.attr.LatentShift(model, ae)

        # Computes counterfactual heatmap for class 3.
        outputs = attr.attribute(x, target=3, lambda_sweep_steps=10)
        assert (batch_size, 3, 200, 200) == outputs.shape
        
        # Computes counterfactual for class 3 and return counterfactuals.
        outputs = attr.attribute(x, target=3, lambda_sweep_steps=10, return_dicts=True)
        assert batch_size == len(outputs)
        for output in outputs:
            assert (3, 200, 200) == output["heatmap"].shape
            assert (10, 3, 200, 200) == output["generated_images"].shape
        