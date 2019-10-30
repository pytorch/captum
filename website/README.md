<a href="https://captum.ai">
  <img width="350" src="./captum_logo_lockup.svg" alt="Captum Logo" />
</a>

<hr/>

[![Conda](https://img.shields.io/conda/v/pytorch/captum.svg)](https://anaconda.org/pytorch/captum)
[![PyPI](https://img.shields.io/pypi/v/captum.svg)](https://pypi.org/project/captum)
[![CircleCI](https://circleci.com/gh/pytorch/captum.svg?style=shield)](https://circleci.com/gh/pytorch/captum)
[![Codecov](https://img.shields.io/codecov/c/github/pytorch/captum.svg)](https://codecov.io/github/pytorch/captum)

Captum is a model interpretability and understanding library for PyTorch.
Captum means comprehension in latin and contains general purpose implementations
of integrated gradients, saliency maps, smoothgrad, vargrad and others for
PyTorch models. It has quick integration for models built with domain-specific
libraries such as torchvision, torchtext, and others.

*Captum is currently in beta and under active development!*


#### Why Captum?

TODO: Add content here


#### Target Audience

TODO: Add content here


## Installation

**Installation Requirements**
- Python >= 3.6
- PyTorch >= 1.2


##### Installing the latest release

The latest release of Captum is easily installed either via
[Anaconda](https://www.anaconda.com/distribution/#download-section) (recommended):
```bash
conda install captum -c pytorch
```
or via `pip`:
```bash
pip install captum
```


##### Installing from latest master

If you'd like to try our bleeding edge features (and don't mind potentially
running into the occasional bug here or there), you can install the latest
master directly from GitHub:
```bash
pip install git+https://github.com/pytorch/captum.git
```

**Manual / Dev install**

Alternatively, you can do a manual install. For a basic install, run:
```bash
git clone https://github.com/pytorch/captum.git
cd captum
pip install -e .
```

To customize the installation, you can also run the following variants of the
above:
* `pip install -e .[dev]`: Also installs all tools necessary for development
  (testing, linting, docs building; see [Contributing](#contributing) below).
* `pip install -e .[tutorials]`: Also installs all packages necessary for running the tutorial notebooks.

To execute unit tests from a manual install, run:
```bash
# running a single unit test
python -m unittest -v tests.attributions.test_saliency
# running all unit tests
pytest -ra
```

## Getting Started

TODO: Add content here


## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.


## References

* [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)
* [Investigating the influence of noise and distractors on the interpretation of neural networks](https://arxiv.org/abs/1611.07270)
* [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)
* [Local Explanation Methods for Deep Neural Networks Lack Sensitivity to Parameter Values](https://arxiv.org/abs/1810.03307)
* [Investigating the influence of noise and distractors on the interpretation of neural networks](https://arxiv.org/abs/1611.07270)
* [Conductance - How Important is a neuron?](https://openreview.net/pdf?id=SylKoo0cKm)


## License
Captum is BSD licensed, as found in the [LICENSE](LICENSE) file.
