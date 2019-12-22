/**
 * Copyright (c) 2019-present, Facebook, Inc.
 *
 * This source code is licensed under the BSD license found in the
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
                The tutorials here will help you understand and use Captum. They assume that you are familiar with PyTorch and its basic features.
              </p>
              <p>
                If you are new to Captum, the easiest way to get started is
                with the{' '}
                <a href="Titanic_Basic_Interpret">
                  Getting started with Captum
                </a>{' '}
                tutorial.
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
                <h4>Getting started with Captum:</h4>
                In this tutorial we create and train a simple neural network on the Titanic survival dataset.
                We then use Integrated Gradients to analyze feature importance.  We then deep dive the network to assess layer and neuron importance
                using conductance.  Finally, we analyze a specific
                neuron to understand feature importance for that specific neuron.  Find the tutorial <a href="Titanic_Basic_Interpret">here</a>.

                <h4>Interpreting text models:</h4>
                In this tutorial we use a pre-trained CNN model for sentiment analysis on an IMDB dataset.
                We use Captum and Integrated Gradients to interpret model predictions by show which specific
                words have highest attribution to the model output.  Find the tutorial <a href="IMDB_TorchText_Interpret">here </a>.

                <h4>Interpreting vision with CIFAR:</h4>
                This tutorial demonstrates how to use Captum for interpreting vision focused models.
                First we create and train (or use a pre-trained) a simple CNN model on the CIFAR dataset.
                We then interpret the output of an example with a series of overlays using Integrated Gradients and DeepLIFT.
                Find the tutorial <a href="CIFAR_TorchVision_Interpret">here</a>.

                <h4>Interpreting vision with ResNet:</h4>
                Like the CIFAR based tutorial above, this tutorial demonstrates how to use Captum for interpreting vision-focused models.
                This tutorial begins with a pretrained resnet18 model and demonstrates how to use Intergrated Gradients along with Noise Tunnel.
                The tutorial finishes with a demonstration of how to use GradientShap.
                Find the tutorial <a href="Resnet_TorchVision_Interpret">here</a>.

                <h4>Feature ablation on images:</h4>
                This tutorial demonstrates feature ablation in Captum, applied on images as an example.
                It leverages segmentation masks to define ablation groups over the input features.
                We show how this kind of analysis helps understanding which parts of the input impacts a certain target in the model.
                Find the tutorial <a href="Resnet_TorchVision_Ablation">here</a>.
    
                <h4>Interpreting multimodal models:</h4>
                To demonstrate interpreting multimodal models we have chosen to look at an open source Visual Question Answer (VQA) model.
                Using Captum and Integrated Gradients we interpret the output of several test questions and analyze the attribution scores
                of the text and visual parts of the model. Find the tutorial <a href="Multimodal_VQA_Interpret">here</a>.

                <h4>[master only] Interpreting question answering with BERT:</h4>
                This tutorial demonstrates how to use Captum to interpret a BERT model for question answering.
                We use a pre-trained model from Hugging Face fine-tuned on the SQUAD dataset and show how to use hooks to
                examine and better understand embeddings, sub-embeddings, BERT, and attention layers.
                Find the tutorial <a href="Bert_SQUAD_Interpret">here</a>.

                <h4>Getting Started with Captum Insights:</h4>
                This tutorial demonstrates how to use Captum Insights for a vision model in a notebook setting.  A simple pretrained torchvision
                CNN model is loaded and then used on the CIFAR dataset.  Captum Insights is then loaded to visualize the interpretation of specific examples.
                Find the tutorial <a href="CIFAR_TorchVision_Captum_Insights">here</a>.
              </p>
            </body>
          </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
