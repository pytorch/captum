#!/usr/bin/env python
# coding: utf-8

# # Feature Ablation to Inspect ResNet Decisions

# This notebook tutorial demonstrates how feature ablation in Captum can be applied to inspect computer vision models. 
# 
# **Task:** Classification into ImageNet-1k categories
# 
# **Model:** A ResNet18 trained on ImageNet-1k
# 
# **Data to inspect:** Samples from PASCAL VOC 2012
# 
# **Ablation based on:** Segmentation masks
# 
# We will use the visualization functions in Captum to show how each semantic part impacts the model output.
#   
# This tutorial assumes the following packages are installed: `captum`, `matplotlib`, `numpy`, `PIL`, `torch`, and `torchvision`.

# In[1]:


import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision import models

from captum.attr import visualization as viz
from captum.attr import FeatureAblation

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import os
import json


# ## 1. Loading the model and the dataset

# Let us load the pretrained `resnet18` model available in `torchvision` and set it to eval mode.
# 
# This model will serve as a classifier into the ImageNet-1k categories.

# In[2]:


resnet = models.resnet18(pretrained=True)
resnet = resnet.eval()


# A straightforward way to demonstrate feature ablation on images is to ablate semantic image areas.
# 
# Therefore, we will load sample images from PASCAL VOC, as these images come along with annotated segmentation masks.
# 
# **Note**: The VOC dataset is 2GB. If you do not want to download it, you can skip the next step and provide your own image and segmentation mask in the step next.

# In[3]:


root = "./VOC"
dataset = torchvision.datasets.VOCSegmentation(root, year='2012', image_set='train', download=False, transform=None, target_transform=None)


# Let us look at a sample image along with its segmentation mask:

# In[4]:


sample_ind = 91

img = Image.open(dataset.images[sample_ind])
plt.imshow(img); plt.axis('off'); plt.show()
mask_img = Image.open(dataset.masks[sample_ind])
plt.imshow(mask_img); plt.axis('off'); plt.show()


# According to the segmentation mask, the image contains three bottles, and two TV monitors, with the rest considered background. All of `background`, `bottle`, and `tvmonitor` are among the 20 categories in PASCAL VOC 2012. This dataset also features a `void` category, used to annotate pixels that are not considered part of any class. These pixels represent border between the objects in the above example.

# Let us also load ImageNet class labels to understand the output when we classify the samples using a classifier trained on ImageNet-1k.
# 
# **Note**: wget should be available as a command in your environment. You might need to install it. You can skip the next two steps if you are OK with class index as classification output (in that case, use `classify` in the next sections with `print_result`=`False`). 

# In[5]:


get_ipython().system('wget -P $HOME/.torch/models https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')


# In[6]:


labels_path = os.getenv("HOME") + '/.torch/models/imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)


# ## 2. Baseline classification

# Let us define a function for classifying images using our ResNet. This model produces logits as classification scores. To normalize these logits into probabilities, we process the output with a `softmax` layer.

# In[25]:


# uses torchvision transforms to convert a PIL image to a tensor and normalize it
img_to_resnet_input = T.Compose([
 T.ToTensor(),
 T.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )    
])

def classify(img, print_result=True):
    output = resnet(img_to_resnet_input(img).unsqueeze(0))
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    prediction_score.squeeze_()
    pred_label_idx.squeeze_()
    if print_result:
        labels_path = os.getenv("HOME") + '/.torch/models/imagenet_class_index.json'
        with open(labels_path) as json_data:
            idx_to_labels = json.load(json_data)
        predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
        print('Predicted:', predicted_label, "id =", pred_label_idx.item(), 'with a score of:', prediction_score.squeeze().item())
    return pred_label_idx.item(), prediction_score.item()



# Now, let us classify the image we loaded in the previous section:

# In[26]:


predicted_class, prediction_score = classify(img)


# Our model classifies the image as the ImageNet-1k category `wine_bottle`. Not bad.
# 
# <em>Note that the model is trained for classification, and expects images containing one object not a whole scene.</em>
# 
# Now, let us see how different parts of the image influence this output.

# ## 3. Inspecting influential image parts behind the classification output

# With the availability of a segmentation mask, we can quickly exclude parts of the image to see how they affect the classification output.
# 
# The Feature Ablation algorithm in Captum enables ablating a number of input features together as a group. This is very useful when dealing with images, where each color channel in each pixel is an input feature.
# To define our desired groups over input features, all we need is to provide a <em>feature mask</em>.
# 
# In case of an image input, the feature mask is also a 2D image of the same size, where each pixel in the mask indicates the feature group it belongs to via an integer value. Pixels of the same value define a group.
# 
# This means we can readily use segmentation masks as feature masks in Captum! 
# Let us see how:

# In[9]:


# convert the mask image to a numpy array of the expected shape (count, channels, height, width)
feature_mask = np.array(mask_img.getdata()).reshape(1, 1, mask_img.size[1], mask_img.size[0])


# Our `feature_mask` is basically ready to use. However, let us first check its unique values that define the group ids:

# In[10]:


print(np.unique(feature_mask))


# These ids correspond to the VOC labels for `background`, `bottle`, `tvmonitor` and `void`.
# 
# While they would work, Captum expects consecutive group ids and would hence consider that there are 256 feature groups (most of them empty). This would result in slow execution.
# 
# Let's instead map our groups to consecutive values as expected by Captum:
# 

# In[11]:


feature_mask[feature_mask == 5] = 1 # feature group for bottle
feature_mask[feature_mask == 20] = 2 # feature group for tvmonitor
feature_mask[feature_mask == 255] = 3 # feature group for void


# Now we can create our `FeatureAblation` object and use it to compute the influence of each feature group in our image on the target responsible for the `predicted_class` we computed in the previous section, which is `wine_bottle`

# In[12]:


ablator = FeatureAblation(resnet)

attribution_map = ablator.attribute(
    img_to_resnet_input(img).unsqueeze(0),
    target=predicted_class,
    feature_mask=torch.tensor(feature_mask))


# Let us visualize the resulting attribution map:

# In[13]:


attribution_map = attribution_map.squeeze().cpu().detach().numpy()
# adjust shape to height, width, channels 
attribution_map = np.transpose(attribution_map, (1,2,0))
_ = viz.visualize_image_attr(attribution_map,
                             method="heat_map",
                             sign="all",
                             show_colorbar=True)


# Captum has computed the influence of each feature groups on the predicted label `wine_bottle`:
# * The bottles are the most influential part of the image contributing positively towards this prediction. Ablating them would strongly impact the prediction.
# * The background and borders are less infuential.
# * The TV monitors have a slightly negative impact. Ablating them will actually boost the prediction score for `wine_bottle`, as this reduces the clutter in the input image w.r.t. to this class.
# 
# The attribution map has been computed with `target=predicted_class`. Captum can further compute attribution maps for alternative targets.

# ## 4. Inspecting influential image parts for alternative classification

# ImageNet does have a class for a TV monitor with `class_id = 664` according to the `idx_to_labels` map we loaded in section 1.
# 
# Let us compute an attribution map for this target in the same image:

# In[14]:


attribution_map = ablator.attribute(
    img_to_resnet_input(img).unsqueeze(0),
    target=664,
    feature_mask=torch.tensor(feature_mask))

attribution_map = attribution_map.squeeze().cpu().detach().numpy()
# adjust shape to height, width, channels 
attribution_map = np.transpose(attribution_map,  (1,2,0))

_ = viz.visualize_image_attr(attribution_map,
                             method="heat_map",
                             sign="all",
                             show_colorbar=True)



# Captum has computed the influence of each feature groups on the target `monitor`:
# * The TV monitors are expectedly the most influential part.
# * The background has no infuential.
# * The borders seem to have some negative impact. Ablating them slightly boosts the target, possibly because this helps framing the monitors in the image? 
# * The bottles have a considerable negative impact. Ablating them apparently makes the monitors stand out as the main object in the image, which in turns boosts the activation for this target.
# 
# You can verify this finding next if you are interested.

# ## 5. Sanity check: re-classifying after ablation

# Here we use OpenCV's `bitwise_and` to create an image with ablated bottles and borders:

# In[15]:


import cv2

cV2_mask = np.array(feature_mask)
cV2_mask[feature_mask == 0] = 1
cV2_mask[feature_mask == 1] = 0
cV2_mask[feature_mask == 2] = 1
cV2_mask[feature_mask == 3] = 0
cV2_mask = np.expand_dims(cV2_mask.squeeze(), axis=2).astype(np.uint8)

img_arr = np.transpose(T.ToTensor()(img).numpy(),  (1,2,0))
img_without_bottles = cv2.bitwise_and(img_arr, img_arr, mask=cV2_mask)
plt.imshow(img_without_bottles); plt.axis('off'); plt.show()


# Let us classify this ablated image:

# In[16]:


classify(img_without_bottles)


# The output is `desktop_computer`, another class in ImageNet whose images usuallly include a monitor and a keyboard. Not bad.
# 
# In fact, the `monitor` class (with id = 664) is among the top-5 guesses:

# In[17]:


output = resnet(img_to_resnet_input(img_without_bottles).unsqueeze(0))
output = F.softmax(output, dim=1)
torch.topk(output, 5)

