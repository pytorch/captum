# Embedding Explorer


Over successive layers, deep neural networks transform the data space so that its dimensions fit the task. Applying component analysis to the intermediate data spaces sheds light into the major features captured by the layers, and can visually surface potential deficiencies of the model or of the training data. 


### Idea: Probing the embedding space interactively

PCA helps us understand the major features that shape the embedding space. Probing samples along each Principal Component, in particular at its extremities, gives us ideas about what this component represents. Below is an example of the embedding space at the 3rd convolutional layer in AlexNet, trained to classify scenes from [MIT Places365](http://places2.csail.mit.edu/). Each column represents a principle component, and depicts samples at the extremities of its axis. The percentages are portions of the variance it explains.

<p align="center">
<img src="/sample_imgs/PCA_on_Places365_AlexNet.jpg" alt="Embedding of Layer3 in AlexNet trained on Places365" width="700" title="The top-5 Principal Components at the 3rd convolutional layer of AlexNet, trained to classify Places365 scenes. Each column represents a principle component, and depicts samples at the extremities of its axis. The percentages are portions of the variance it explains.
"/>
</p>

* The first component explains 15.6% of the variance. At one extreme of this component we can see vegetation scenes, while at the other extreme we can see indoor scenes containing horizontal and vertical edges. These two subsets are at odds of each other: Vegetation scenes are mainly outdoors and are composed primarily of edgeless texture.
* The second and third components seem to separate between different types of textures. The fourth component seems to separate scenes dominated by horizontal edges and those dominated by vertical ones.
* The fifth component seems sensitive to color saturation. 

The Embedding Explorer facilitates exploring embedding spaces via an interactive UI.
<p align="center">
<img src="/sample_imgs/PCA_on_ImageNet_ResNet_Layer1.png" alt="Embedding of Layer1 in ResNet trained on ImageNet" width="700" title="ProbingtheembeddingspacesofaResNet-18ImageNetclassifier.Directionsinthese spaces determined by PCA reveal significant features gradually learned by the model to discriminate between the classes. In the first layer the majority of the variance corresponds to image brightness, dominance of orange or blue pixels, and dominance of green or purple pixels."/>
</p>

## Example Notebook
Explore the embedding space of a protoypical image classifier.
[View Jupyter Notebook](https://colab.research.google.com/drive/1NdVAR4b1cwVeibxbh2_q6RVcO3ilaYca?usp=sharing#scrollTo=d4UkWTvB-B5N)


### What is the value of this analysis?
* **Understand the learned features - globally** <br>
There are numerous attribution methods that reveal which input features in a given sample activate a given neuron. These methods, however, do not expose collectively dominant input features, learned across multiple neurons. Dataset-level analysis is useful to surface these features by probing the activation space over the entire dataset.
* **Expose unreliable features and data deficiencies** <br>
Out of convenience or limited power, our model might propagate low-level input features to the final layer and rely on them in the task.
* **Model comparison** <br>
When comparing two architectures or two variations of the same models, it is informative to understand how the embeddings they compute differ. For example, Inception, ResNet, and VGGNet models deviate in the embeddings learned by early layers. Such analysis is helpful to shed light into why one model outperforms other models.  
* **Dataset comparison** <br>
Comparing the embedding space of two splits or two versions of a dataset is helpful to surface significant deviations in their distribution. It is possible to apply space probing using PCA directly to the dataset space, without having a model.


## Thank You

If you have any suggestions or feedback, we will be happy to hear from you!
