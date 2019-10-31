---
id: captum_insights
title: Captum Insights
---

Interpreting model output in complex models can be difficult. Even with interpretability libraries like Captum, it can be difficult to understand models without proper visualizations. Image and text input features can be especially difficult to understand without these visualizations.

Captum Insights is an interpretability visualization widget built on top of Captum to facilitate model understanding. Captum Insights works across images, text, and other features to help users understand feature attribution. Some examples of the widget are below.

Getting started with Captum Insights is easy. You can learn how to use Captum Insights with the [Getting started with Captum Insights](/tutorials/CIFAR_TorchVision_Captum_Insights) tutorial.  Alternatively, to analyze a sample model on CIFAR10 via Captum Insights execute the line below and navigate to the URL specified in the output.

```
python -m captum.insights.example
```


Below are some sample screenshots of Captum Insights:

![screenshot1](/img/captum_insights_screenshot.png)

![screenshot2](/img/captum_insights_screenshot_vqa.png)
