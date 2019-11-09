/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the BSD license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

const MarkdownBlock = CompLibrary.MarkdownBlock;
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

const bash = (...args) => `~~~bash\n${String.raw(...args)}\n~~~`;

class HomeSplash extends React.Component {
  render() {
    const {siteConfig, language = ''} = this.props;
    const {baseUrl, docsUrl} = siteConfig;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

    const SplashContainer = props => (
      <div className="homeContainer">
        <div className="homeSplashFade">
          <div className="wrapper homeWrapper">{props.children}</div>
        </div>
      </div>
    );

    const Logo = props => (
      <div className="splashLogo">
        <img src={props.img_src} alt="Project Logo" className="primaryLogoImage"/>
      </div>
    );

    const ProjectTitle = () => (
      <h2 className="projectTitle">
        <small>{siteConfig.tagline}</small>
      </h2>
    );

    const PromoSection = props => (
      <div className="section promoSection">
        <div className="promoRow">
          <div className="pluginRowBlock">{props.children}</div>
        </div>
      </div>
    );

    const Button = props => (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={props.href} target={props.target}>
          {props.children}
        </a>
      </div>
    );

    return (
      <SplashContainer>
        <div className="productTitle">Captum</div>
        <div className="inner">
          <ProjectTitle siteConfig={siteConfig} />
          <PromoSection>
            <Button href={docUrl('introduction.html')}>Introduction</Button>
            <Button href={'#quickstart'}>Get Started</Button>
            <Button href={`${baseUrl}tutorials/`}>Tutorials</Button>
          </PromoSection>
        </div>
      </SplashContainer>
    );
  }
}

class Index extends React.Component {
  render() {
    const {config: siteConfig, language = ''} = this.props;
    const {baseUrl} = siteConfig;

    const Block = props => (
      <Container
        padding={['bottom', 'top']}
        id={props.id}
        background={props.background}>
        <GridBlock
          align="center"
          contents={props.children}
          layout={props.layout}
        />
      </Container>
    );

    const Description = () => (
      <Block background="light">
        {[
          {
            content:
              'This is another description of how this project is useful',
            image: `${baseUrl}img/captum-icon.png`,
            imageAlign: 'right',
            title: 'Description',
          },
        ]}
      </Block>
    );
    // getStartedSection
    const pre = '```';
    // Example for model fitting
    const createModelExample = `${pre}python
import numpy as np

import torch
import torch.nn as nn

from captum.attr import IntegratedGradients

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 3)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(3, 2)

        # initialize weights and biases
        self.lin1.weight = nn.Parameter(torch.arange(-4.0, 5.0).view(3, 3))
        self.lin1.bias = nn.Parameter(torch.zeros(1,3))
        self.lin2.weight = nn.Parameter(torch.arange(-3.0, 3.0).view(2, 3))
        self.lin2.bias = nn.Parameter(torch.ones(1,2))

    def forward(self, input):
        return self.lin2(self.relu(self.lin1(input)))


model = ToyModel()
model.eval()
    `;
    // Example for defining an acquisition function
    const defineInputBaseline = `${pre}python
input = torch.rand(2, 3)
baseline = torch.zeros(2, 3)
    `;

    const randomSeedsDefinition = `${pre}python
torch.manual_seed(123)
np.random.seed(123)
    `;
    // Example for optimizing candidates
    const instantiateApply = `${pre}python
ig = IntegratedGradients(model)
attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
print('IG Attributions: ', attributions, ' Convergence Delta: ', delta)
    `;

    const igOutput = `${pre}python
IG Attributions:  tensor([[-0.5922, -1.5497, -1.0067],
                          [ 0.0000, -0.2219, -5.1991]])
Convergence Delta: tensor([2.3842e-07, -4.7684e-07])
    `;
    //
    const QuickStart = () => (
      <div
        className="productShowcaseSection"
        id="quickstart"
        style={{textAlign: 'center'}}>
        <h2>Get Started</h2>
        <Container>
          <ol>
            <li>
              <h4>Install Captum:</h4>
              <a>via conda (recommended):</a>
              <MarkdownBlock>{bash`conda install captum -c pytorch`}</MarkdownBlock>
              <a>via pip:</a>
              <MarkdownBlock>{bash`pip install captum`}</MarkdownBlock>
            </li>
            <li>
              <h4>Create and prepare model:</h4>
              <MarkdownBlock>{createModelExample}</MarkdownBlock>
            </li>
            <li>
              <h4>To make computations deterministic, let's fix random seeds:</h4>
              <MarkdownBlock>{randomSeedsDefinition}</MarkdownBlock>
            </li>
            <li>
              <h4>Define input and baseline tensors:</h4>
              <MarkdownBlock>{defineInputBaseline}</MarkdownBlock>
            </li>
            <li>
              <h4>Select algorithm to instantiate and apply (Integrated Gradients in this example):</h4>
              <MarkdownBlock>{instantiateApply}</MarkdownBlock>
            </li>
            <li>
              <h4>View Output:</h4>
              <MarkdownBlock>{igOutput}</MarkdownBlock>
            </li>
          </ol>
        </Container>
      </div>
    );

    const Features = () => (
    <div className="productShowcaseSection" style={{textAlign: 'center'}}>
      <h2>Key Features</h2>
      <Block layout="threeColumn">
        {[
          {
            content:
              'Supports interpretability of models across modalities including vision, text, and more.',
            image: `${baseUrl}img/multi-modal.png`,
            imageAlign: 'top',
            title: 'Multi-Modal',
          },
          {
            content:
              'Supports most types of PyTorch models and can be used with minimal modification to the original neural network.',
            image: `${baseUrl}img/pytorch_logo.svg`,
            imageAlign: 'top',
            title: 'Built on PyTorch',
          },
          {
            content:
              'Open source, generic library for interpretability research. Easily implement and benchmark new algorithms. ',
            image: `${baseUrl}img/expanding_arrows.svg`,
            imageAlign: 'top',
            title: 'Extensible',
          },
        ]}
      </Block>
    </div>
  );

    const Showcase = () => {
      if ((siteConfig.users || []).length === 0) {
        return null;
      }

      const showcase = siteConfig.users
        .filter(user => user.pinned)
        .map(user => (
          <a href={user.infoLink} key={user.infoLink}>
            <img src={user.image} alt={user.caption} title={user.caption} />
          </a>
        ));

      const pageUrl = page => baseUrl + (language ? `${language}/` : '') + page;

      return (
        <div className="productShowcaseSection paddingBottom">
          <h2>Who is Using This?</h2>
          <p>This project is used by all these people</p>
          <div className="logos">{showcase}</div>
          <div className="more-users">
            <a className="button" href={pageUrl('users.html')}>
              More {siteConfig.title} Users
            </a>
          </div>
        </div>
      );
    };

    return (
      <div>
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="landingPage mainContainer">
          <Features />
          <QuickStart />
        </div>
      </div>
    );
  }
}

module.exports = Index;
