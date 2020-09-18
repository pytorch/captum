# Awesome List

There is a lot of awesome research and development happening out in the interpretability community that we would like to share.  Here we will maintain a curated list of research, implementations and resources.  We would love to learn about more!  Please feel free to make a pull request to contribute to the list.


#### TorchRay: Visualization methods for deep CNNs
TorchRay focuses on attribution, namely the problem of determining which part of the input, usually an image, is responsible for the value computed by a neural network.
  - [https://github.com/facebookresearch/TorchRay](https://github.com/facebookresearch/TorchRay)


#### Score Cam: A gradient-free CAM extension
Score-CAM is a gradient-free visualization method extended from Grad-CAM and Grad-CAM++.  It provides score-weighted visual explanations for CNNs.
  - [Paper](https://arxiv.org/abs/1910.01279)
  - [https://github.com/haofanwang/Score-CAM](https://github.com/haofanwang/Score-CAM)


#### White Noise Analysis
White noise stimuli is fed to a classifier and the ones that are categorized into a particular class are averaged. It gives an estimate of the templates a classifier uses for classification, and is based on two popular and related methods in psychophysics and neurophysiology namely classification images and spike triggered analysis.
- [Paper](https://arxiv.org/abs/1912.12106)
- [https://github.com/aliborji/WhiteNoiseAnalysis.git](https://github.com/aliborji/WhiteNoiseAnalysis.git)


#### FastCAM: Multiscale Saliency Map with SMOE scale
An attribution method that uses information at the end of each network scale which is then combined into a single saliency map. 
- [Paper](https://arxiv.org/abs/1911.11293)
- [https://github.com/LLNL/fastcam](https://github.com/LLNL/fastcam)
- [pull request](https://github.com/pytorch/captum/pull/442)
- [jupyter notebook demo](https://github.com/LLNL/fastcam/blob/captum/demo-captum.ipynb)
