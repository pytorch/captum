---
id: algorithms
title: Algorithms
---

Captum is a framework within which different interpretability methods can be implemented.  The Captum team welcomes any contributions in the form of algorithms, methods or framework extensions!  Below is a short summary of the various methods currently implemented within Captum and we will add descriptions of methods added by the community.

## Integrated Gradients
Integrated Gradients is an axiomatic attribution method that requires almost no modification of the original network. It can be used for augmenting accuracy metrics, model debugging and feature or rule extraction. The cornerstones of this approach are two fundamental axioms, namely sensitivity and implementation invariance. While other attribution methods fail to fulfill one or the other axiom, integrated gradients satisfies both.

Integrated gradients uses a baseline from which integration is done along the path from reference to the input in question.  As such, Integrated gradients represents the integral along the path from the baseline to input. Formally, it can be described as follows:

![IG_eq1](/img/IG_eq1.png)
Integrated Gradients along the i - th dimension of input X. Alpha is the scaling coefficient. The equations are copied from the [original paper](https://arxiv.org/abs/1703.01365).

To learn more about Integrated Gradients visit the following resources:  
- [Original paper](https://arxiv.org/abs/1703.01365)

## DeepLIFT
DeepLIFT is a back propagation based approach that attributes a change to inputs based on the differences between the inputs and corresponding references (or baselines) for non-linear activations.  As such, deepLIFT seeks to explain the difference in the output from reference in terms of the difference in inputs from reference.  DeepLIFT uses the concept of multipliers to "blame" specific neurons for the difference in output.  The definition of a multiplier is as follows (from [original paper](https://arxiv.org/pdf/1704.02685.pdf)):
![deepLIFT_eq1](/img/deepLIFT_multipliers_eq1.png)
x is the input neuron with a difference from reference Δx, and t is the target neuron with a difference from reference Δt. C is then the contribution of Δx to Δt.

Like partial derivatives (gradients) used in back propagation, multipliers obey the Chain Rule.  Implementations of DeepLIFT override the implementation of the activation function in use during back propagation in order to calculate multipliers rather than partial gradients.  

To learn more about DeepLIFT visit the following resources:  
- [Original paper](https://arxiv.org/abs/1704.02685)  
- [Explanatory videos attached to paper](https://www.youtube.com/playlist?list=PLJLjQOkqSRTP3cLB2cOOi_bQFw6KPGKML)
- [Towards Better Understanding of Gradient-Based Attribution Methods for Deep Neural Networks](https://openreview.net/pdf?id=Sy21R9JAW)

## Conductance
Conductance seeks to understand the importance of hidden layers or neurons within a deep neural network.  This is different from other interpretability algorithms which focus on the importance of input features.  Some existing methodologies to estimate neuron importance have used activation values, sometimes in combination with gradients, but this can be inaccurate.  Conductance combines the neuron activation with the partial derivatives of both the neuron with respect to the input and the output with respect to the neuron to build a more complete picture of neuron importance.  

Conductance builds on Integrated Gradients (IG) by looking at the flow of IG attribution which occurs through the hidden neuron.  The formal definition of total conductance (from the [original paper](https://openreview.net/forum?id=SylKoo0cKm)) is as follows:  
![conductance_eq1](/img/conductance_eq_1.png)  

To learn more about Conductance visit the following resources:  
- [Original Paper](https://openreview.net/forum?id=SylKoo0cKm)  
- [Computationally Efficient Measures of Internal Neuron Importance](https://arxiv.org/pdf/1807.09946.pdf)  
