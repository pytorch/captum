/**
 * Copyright (c) 2019-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const React = require('react');

const CWD = process.cwd();

const CompLibrary = require(`${CWD}/node_modules/docusaurus/lib/core/CompLibrary.js`);
const Container = CompLibrary.Container;
const MarkdownBlock = CompLibrary.MarkdownBlock;

const TutorialSidebar = require(`${CWD}/core/TutorialSidebar.js`);

class TutorialHome extends React.Component {
  render() {
    return (
      <div className="docMainWrapper wrapper">
        <TutorialSidebar currentTutorialID={null} />
        <Container className="mainContainer documentContainer postContainer">
          <div className="post">
            <header className="postHeader">
              <h1 className="postHeaderTitle">Captum Tutorials</h1>
            </header>
            <body>
              <p>
                The tutorials here will help you understand and use Captum in
                your own work. They assume that you are familiar with both
                Bayesian optimization (BO) and PyTorch.
              </p>
              <p>
                If you are new to BO, we recommend you start with the{' '}
                <a href="https://ax.dev/docs/bayesopt">Ax docs</a> and the
                following{' '}
                <a href="https://arxiv.org/abs/1807.02811">tutorial paper</a>.
              </p>
              <p>
                If you are new to PyTorch, the easiest way to get started is
                with the{' '}
                <a href="https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py">
                  What is PyTorch?
                </a>{' '}
                tutorial.
              </p>
              <p>
                The Captum tutorials are grouped into the following four areas.
              </p>
              <p>
                <h4>Using Captum with Ax</h4>
                These tutorials give you an overview of how to leverage{' '}
                <a href="https://ax.dev">Ax</a>, a platform for sequential
                experimentation, in order to simplify the management of your BO
                loop. Doing so can help you focus on the main aspects of BO
                (models, acquisition functions, optimization of acquisition
                functions), rather than tedious loop control. See our{' '}
                <a href="https://captum.org/docs/captum_and_ax">
                  Documentation
                </a>{' '}
                for additional information.
                <h4>Full Optimization Loops</h4>
                In some situations (e.g. when working in a non-standard setting,
                or if you want to understand and control various details of the
                BO loop), then you may also consider working purely in Captum.
                The tutorials in this section illustrate this approach.
                <h4>Bite-Sized Tutorials</h4>
                Rather than guiding you through full end-to-end BO loops, the
                tutorials in this section focus on specific tasks that you will
                encounter in customizing your BO algorithms. For instance, you
                may want to{' '}
                <a href="https://captum.org/tutorials/custom_acquisition">
                  write a custom acquisition function
                </a>{' '}
                and then{' '}
                <a href="https://captum.org/tutorials/optimize_with_cmaes">
                  use a custom zero-th order optimizer
                </a>{' '}
                to optimize it.
                <h4>Advanced Usage</h4>
                Tutorials in this section showcase more advanced ways of using
                Captum. For instance,{' '}
                <a href="https://captum.org/tutorials/vae_mnist">
                  this tutorial
                </a>{' '}
                shows how to perform BO if your objective function is an image,
                by optimizing in the latent space of a variational auto-encoder
                (VAE).
              </p>
            </body>
          </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
