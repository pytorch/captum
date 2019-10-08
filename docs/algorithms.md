---
id: algorithms
title: Algorithms
---

Captum is a library within which different interpretability methods can be implemented.  The Captum team welcomes any contributions in the form of algorithms, methods or library extensions!  

The algorithms in Captum are separated into three groups, general attribution, layer attribution and neuron attribution, which are defined as follows:
* General Attribution: Evaluates contribution of each input feature to the output of a model.
* Layer Attribution: Evaluates contribution of each neuron in a given layer to the output of the model.
* Neuron Attribution: Evaluates contribution of each input feature on the activation of a particular hidden neuron.

Below is a short summary of the various methods currently implemented for general, layer, and neuron attribution within Captum, as well as noise tunnel, which can be used to smooth the results of any attribution method.

## General Attribution
### Integrated Gradients
Integrated gradients represents the integral of gradients with respect to inputs along the path from a given baseline to input. The integral can be approximated using a Riemann Sum or Gauss Legendre quadrature rule. Formally, it can be described as follows:

![IG_eq1](/img/IG_eq1.png)
*Integrated Gradients along the i - th dimension of input X. Alpha is the scaling coefficient. The equations are copied from the [original paper](https://arxiv.org/abs/1703.01365).*

The cornerstones of this approach are two fundamental axioms, namely sensitivity and implementation invariance. More information regarding these axioms can be found in the original paper.

To learn more about Integrated Gradients, visit the following resources:  
- [Original paper](https://arxiv.org/abs/1703.01365)

### Gradient SHAP
Gradient SHAP is a gradient method to approximate SHAP values, which are based on Shapley values proposed in cooperative game theory. Gradient SHAP adds Gaussian noise to each input sample multiple times, selects a random point along the path between baseline and input, and computes the gradient of outputs with respect to those selected random points. The final SHAP values represent the expected value of gradients * (inputs - baselines).

The computed attributions approximate SHAP values under the assumptions that the input features are independent and that the model behaves linearly between the inputs and given baselines.

To learn more about GradientSHAP, visit the following resources:
- [SHAP paper](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
- [Original Implementation](https://github.com/slundberg/shap/#deep-learning-example-with-gradientexplainer-tensorflowkeraspytorch-models)

### DeepLIFT
DeepLIFT is a back-propagation based approach that attributes a change to inputs based on the differences between the inputs and corresponding references (or baselines) for non-linear activations.  As such, DeepLIFT seeks to explain the difference in the output from reference in terms of the difference in inputs from reference.  DeepLIFT uses the concept of multipliers to "blame" specific neurons for the difference in output.  The definition of a multiplier is as follows (from [original paper](https://arxiv.org/pdf/1704.02685.pdf)):
![deepLIFT_eq1](/img/deepLIFT_multipliers_eq1.png)
*x is the input neuron with a difference from reference Δx, and t is the target neuron with a difference from reference Δt. C is then the contribution of Δx to Δt.*

Like partial derivatives (gradients) used in back propagation, multipliers obey the Chain Rule. According to the formulations proposed in [this paper](https://openreview.net/pdf?id=Sy21R9JAW). DeepLIFT can be overwritten as the modified partial derivatives of output of non-linear activations with respect to their inputs.

Currently, we only support Rescale Rule of DeepLIFT Algorithms. RevealCancel Rule will be implemented in later releases.

To learn more about DeepLIFT, visit the following resources:  
- [Original paper](https://arxiv.org/abs/1704.02685)  
- [Explanatory videos attached to paper](https://www.youtube.com/playlist?list=PLJLjQOkqSRTP3cLB2cOOi_bQFw6KPGKML)
- [Towards Better Understanding of Gradient-Based Attribution Methods for Deep Neural Networks](https://openreview.net/pdf?id=Sy21R9JAW)

### DeepLIFT SHAP
DeepLIFT SHAP is a method extending DeepLIFT to approximate SHAP values, which are based on Shapley values proposed in cooperative game theory. DeepLIFT SHAP takes a distribution of baselines and computes the DeepLIFT attribution with respect to
each of these and averages the resulting attributions.

DeepLIFT's rules for non-linearities serve to linearize non-linear components of the network, and the method approximates SHAP values for the linearized version of the network. The method also assumes that the input features are independent.

To learn more about DeepLIFT SHAP, visit the following resources:  
- [SHAP paper](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)

### Saliency
Saliency is a simple approach for computing input attribution, returning the gradients with respect to input. This approach can be understood as taking a first-order Taylor approximation of the network at the input, and the gradients correspond to the coefficients of each feature in the linear representation of the model. The absolute value of these coefficients can be taken to represent feature importance.

To learn more about Saliency, visit the following resources:  
- [Original paper](https://arxiv.org/pdf/1312.6034.pdf)  

### Input X Gradient
Input X Gradient is an extension of the saliency approach, taking the gradients of the output with respect to the input and multiplying by the input feature values. One intuition for this approach considers a linear model; the gradients are simply the coefficients of each input, and the product of the input with a coefficient corresponds to the total contribution of the feature to the linear model's output.

### Guided Backpropagation
Guided backpropagation computes the gradient of the target output with respect the input, but backpropagation of ReLU functions is overridden so that only non-negative gradients are backpropagated (any negative gradients are set to 0). Guided backpropagation was proposed in the context of an all-convolutional network and is generally used for convolutional networks, although the approach can be applied generically.

To learn more about Guided Backpropagation, visit the following resources:  
- [Original paper](https://arxiv.org/abs/1412.6806)

### Guided GradCAM
Guided GradCAM computes the element-wise product of [guided backpropagation](###Guided-Backpropagation) attributions with upsampled (layer) [GradCAM](###GradCAM) attributions. GradCAM attributions are computed
with respect to a layer, and attributions are upsampled to match the input size.
This approach is designed for convolutional neural networks. The chosen layer is often the last convolutional layer in the network, but any layer that is spatially aligned with the input can be provided.


Guided GradCAM was proposed by the authors of GradCAM as a method to combine the high-resolution nature of Guided Backpropagation with the class-discriminative advantages of GradCAM, which has lower resolution due to upsampling from a convolutional layer.

To learn more about Guided GradCAM, visit the following resources:  
- [Original paper](https://arxiv.org/abs/1412.6806)
- [Website](http://gradcam.cloudcv.org/)

## Layer Attribution
### Layer Conductance
Conductance combines the neuron activation with the partial derivatives of both the neuron with respect to the input and the output with respect to the neuron to build a more complete picture of neuron importance.  

Conductance builds on Integrated Gradients (IG) by looking at the flow of IG attribution which occurs through the hidden neuron.  The formal definition of total conductance of a hidden neuron y (from the [original paper](https://arxiv.org/abs/1805.12233)) is as follows:  
![conductance_eq1](/img/conductance_eq_1.png)  

For more efficient computation of layer conductance, we use the idea presented in this [paper](https://arxiv.org/pdf/1807.09946.pdf) to avoid computing the gradient of each neuron with respect to the input.

To learn more about Conductance, visit the following resources:  
- [Original Paper](https://arxiv.org/abs/1805.12233)  
- [Computationally Efficient Measures of Internal Neuron Importance](https://arxiv.org/pdf/1807.09946.pdf)  

### Internal Influence
Internal influence approximates the integral of gradients with respect to a particular layer along the path from a baseline input to the given input. This method is similar to applying integrated gradients, integrating the gradient with respect to the layer (rather than the input).

To learn more about Internal Influence, visit the following resources:  
- [Original Paper](https://arxiv.org/pdf/1802.03788.pdf)

### Layer Activation
Layer Activation is a simple approach for computing layer attribution, returning the activation of each neuron in the identified layer.

### Layer Gradient X Activation
Layer Gradient X Activation is the analog of the Input X Gradient method for hidden layers in a network. It element-wise multiplies the layer's activation with the gradients of the target output with respect to the given layer.

### GradCAM

GradCAM is a layer attribution method designed for convolutional neural networks, and is usually applied to the last convolutional layer.
GradCAM computes the gradients of the target output with respect to the given layer, averages for each output channel (dimension 2 of output), and multiplies the average gradient for each channel by the
layer activations. The results are summed over all channels and a ReLU is applied to the output, returning only non-negative attributions.

This procedure sums over the second dimension (# of channels), so the output of GradCAM attributions will have a second dimension of 1, but all other dimensions will match that of the layer output.

Although GradCAM directly attributes the importance of different neurons in the target layer, GradCAM is often used as a general attribution method. To accomplish this, GradCAM attributions are upsampled and viewed as a mask to the input, since a convolutional layer output generally matches the input image spatially.

To learn more about GradCAM, visit the following resources:  
- [Original paper](https://arxiv.org/abs/1412.6806)
- [Website](http://gradcam.cloudcv.org/)

## Neuron Attribution
### Neuron Conductance
Conductance combines the neuron activation with the partial derivatives of both the neuron with respect to the input and the output with respect to the neuron to build a more complete picture of neuron importance.  

Conductance for a particular neuron builds on Integrated Gradients (IG) by looking at the flow of IG attribution from each input through the particular neuron.  The formal definition of conductance of neuron y for the attribution of input i (from the [original paper](https://openreview.net/forum?id=SylKoo0cKm)) is as follows:  
![conductance_eq2](/img/conductance_eq_2.png)  

Note that based on this definition, summing the neuron conductance (over all input features) always equals the layer conductance for the particular neuron.

To learn more about Conductance, visit the following resources:  
- [Original Paper](https://openreview.net/forum?id=SylKoo0cKm)  
- [Computationally Efficient Measures of Internal Neuron Importance](https://arxiv.org/pdf/1807.09946.pdf)  

### Neuron Gradient
Neuron gradient is the analog of the saliency method for a particular neuron in a network. It simply computes the gradient of the neuron output with respect to the model input. Like Saliency, this approach can be understood as taking a first-order Taylor approximation of the neuron's output at the given input, and the gradients correspond to the coefficients of each feature in the linear representation of the model.

### Neuron Integrated Gradients
Neuron Integrated Gradients approximates the integral of input gradients with respect to a particular neuron along the path from a baseline input to the given input. This method is equivalent to applying integrated gradients
      considering the output to be simply the output of the identified neuron.

To learn more about Integrated Gradients, visit the following resources:  
- [Original paper](https://arxiv.org/abs/1703.01365)

### Neuron Guided Backpropagation
Neuron guided backpropagation is the analog of guided backpropagation for a particular neuron. It computes the gradient of the target neuron with respect the input, but backpropagation of ReLU functions is overridden so that only non-negative gradients are backpropagated (any negative gradients are set to 0). Guided backpropagation was proposed in the context of an all-convolutional network and is generally used for neurons in convolutional networks, although the approach can be applied generically.

To learn more about Guided Backpropagation, visit the following resources:  
- [Original paper](https://arxiv.org/abs/1412.6806)

## Noise Tunnel
Noise Tunnel is a method that can be used on top of any of the given general, layer, or neuron attribution method. Noise tunnel computes attribution multiple times, adding Gaussian noise to the input each time, and combines the calculated attributions based on the chosen type. The supported types for noise tunnel are:
* Smoothgrad: The mean of the sampled attributions is returned. This approximates smoothing the given attribution method with a Gaussian Kernel.
* Smoothgrad Squared: The mean of the squared sample attributions is returned.
* Vargrad: The variance of the sample attributions is returned.

To learn more about Noise Tunnel methods, visit the following resources:
- [SmoothGrad Original paper](https://arxiv.org/abs/1706.03825)
- [VarGrad Original paper](https://arxiv.org/abs/1810.03307)
