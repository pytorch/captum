Utilities
==========

Visualization
^^^^^^^^^^^^^^

.. autofunction:: captum.attr.visualization.visualize_image_attr

.. autofunction:: captum.attr.visualization.visualize_image_attr_multiple


Interpretable Embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: captum.attr.InterpretableEmbeddingBase
    :members:

.. autofunction:: captum.attr.configure_interpretable_embedding_layer
.. autofunction:: captum.attr.remove_interpretable_embedding_layer


Token Reference Base
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: captum.attr.TokenReferenceBase
    :members:


Linear Models
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: captum._utils.models.model.Model
    :members:
.. autoclass:: captum._utils.models.linear_model.SkLearnLinearModel
    :members:
.. autoclass:: captum._utils.models.linear_model.SkLearnLinearRegression
    :members:
.. autoclass:: captum._utils.models.linear_model.SkLearnLasso
    :members:
.. autoclass:: captum._utils.models.linear_model.SkLearnRidge
    :members:
.. autoclass:: captum._utils.models.linear_model.SGDLinearModel
    :members:
.. autoclass:: captum._utils.models.linear_model.SGDLasso
    :members:
.. autoclass:: captum._utils.models.linear_model.SGDRidge
    :members:
