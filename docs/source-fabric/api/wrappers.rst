########################
Models wrapped by Fabric
########################

When you :doc:`set up <../api/fabric_methods>` a model in Fabric, it gets automatically wrapped by a new module, the ``FabricModule``:

.. code-block:: python

    import torch
    import lightning as L

    fabric = L.Fabric()
    model = torch.nn.Linear(10, 2)
    model = fabric.setup(model)

    print(type(model))  # <class 'lightning.fabric.wrappers._FabricModule'>

This wrapper module takes care of a few things for you, notably:

- Strategy: Handles strategy-specific logic for the forward method (DDP, FSDP, etc.).
- Precision: Inputs and outputs passed through ``forward`` get automatically converted to the right precision depending on the ``Fabric(precision=...)`` setting.
- Device: The wrapper remembers which device the model is on. You can access it with `model.device`.

.. note::
    The ``FabricModule`` wrapper is completely transparent and most users will never need to interact with it directly.

Below we describe a few functions and properties of the wrapper for advanced use cases.
This might be useful if you are building a custom Trainer using Fabric as the core.


----


********************************
Accessing methods and attributes
********************************

Access to methods and attributes gets redirected to the original model automatically:

.. code-block:: python

    import torch
    import lightning as L

    fabric = L.Fabric()
    model = torch.nn.Linear(10, 2)
    fabric_model = fabric.setup(model)

    # You can access attributes and methods normally
    print(fabric_model.weight is model.weight)  # True


----


********************
Unwrapping the model
********************

You can check whether a model is wrapped in a ``FabricModule`` with the ``is_wrapped`` utility function:

.. code-block:: python

    import torch
    import lightning as L
    from lightning.fabric import is_wrapped

    fabric = L.Fabric()
    model = torch.nn.Linear(10, 2)
    fabric_model = fabric.setup(model)

    print(is_wrapped(model))  # False
    print(is_wrapped(fabric_model))  # True


If you ever need to, you can access the original model explicitly via ``.module``:

.. code-block:: python

    # Access the original model explicitly
    original_model = fabric_model.module

    print(original_model is model)  # True


----


************************************************
Using methods other than forward for computation
************************************************

PyTorch's ``nn.Modules`` have a special contract you need to follow when using them for training: Your forward computation has to be defined in the **forward** method and you should call this forward method directly.
But sometimes your model may need to define different flavors of `forward`, like in this example below where the regular `forward` is used for training, but the `generate` method does something slightly different for inference:

.. code-block:: python

    import torch
    import lightning as L


    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 2)

        def forward(self, x):
            return self.layer(x)

        def generate(self):
            sample = torch.randn(10)
            return self(sample)


If you were to run this model in Fabric with multiple devices (DDP or FSDP), you would get an error:

.. code-block:: python

    fabric = L.Fabric(accelerator="cpu", devices=2)
    fabric.launch()
    model = MyModel()
    model = fabric.setup(model)

    # OK: Calling the model directly
    output = model(torch.randn(10))

    # OK: Calling the model's forward (equivalent to the above)
    output = model.forward(torch.randn(10))

    # ERROR: Calling another method that calls forward indirectly
    output = model.generate()

Fabric produces an error there informing the user about incorrect usage because this is normally not allowed in PyTorch and could potentially lead to silent correctness bugs.
If you want to use such methods, you need to mark them explicitly with ``.mark_forward_method()`` so that Fabric can do some rerouting behind the scenes for you to do the right thing:

.. code-block:: python

    # You must mark special forward methods explicitly:
    model.mark_forward_method(model.generate)

    # Passing just the name is also sufficient
    model.mark_forward_method("generate")

    # OK: Fabric will do some rerouting behind the scenes now
    output = model.generate()

|
