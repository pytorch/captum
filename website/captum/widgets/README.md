# Widgets!

# Welcome to widgets

>"What is the goal of the widgets project"

I'm glad you asked. The captum widgets tool is designed to make scanning your data for important features less about writing code, and more about, well, exploring!
We imagine that you use our tool to answer any of the hard questions that you currently don't have answers to.
For example, you want to know how your model is so good at predicting class X so that you can try and fix another class:
Load up the widget with a small sample of X and then see what exactly was important.

>How can I use widgets in my code?

Great question, all you need to do to get widgets up and running is to...

1. Have ipywidgets installed on your local machine

2. Have a working IPython notebook instance (jupyter, bento, for example)

3. Import the code, and get exploring!

> What do I need to call up a widget

The answer is, not much! Below is an example of how to get it up and running:

```
from captum.widgets import AttributionWidget

wrapped_model = ... # any pytorch classifier model wrapped by the Wrapped Model definition in helpers.py
wrapped_dataset = ... # your data, wrapped up in a WidgetDataset. definition and helpers in helpers.py

tb = ExploreWidget(wrapped_model, wrapped_dataset)
tb.render()
```
