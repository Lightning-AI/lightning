.. _accelerator:

###########
Accelerator
###########

The Accelerator connects a Lightning Trainer to arbitrary hardware (CPUs, GPUs, TPUs, HPUs, MPS, ...).
Currently there are accelerators for:

- CPU
- :doc:`GPU <../accelerators/gpu>`
- :doc:`TPU <../accelerators/tpu>`
- :doc:`HPU <../integrations/hpu/index>`
- :doc:`MPS <../accelerators/mps>`

The Accelerator is part of the Strategy which manages communication across multiple devices (distributed communication).
Whenever the Trainer, the loops or any other component in Lightning needs to talk to hardware, it calls into the Strategy and the Strategy calls into the Accelerator.

.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/strategies/overview.jpeg
    :alt: Illustration of the Strategy as a composition of the Accelerator and several plugins

We expose Accelerators and Strategies mainly for expert users who want to extend Lightning to work with new
hardware and distributed training or clusters.


----------

Create a Custom Accelerator
---------------------------

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

Here is how you create a new Accelerator.
Let's pretend we want to integrate the fictional XPU accelerator and we have access to its hardware through a library
``xpulib``.

.. code-block:: python

    import torch
    import xpulib

    from functools import lru_cache
    from typing import Any, Dict, Union
    from lightning.pytorch.accelerators.accelerator import Accelerator

    from typing_extensions import override


    class XPUAccelerator(Accelerator):
        """Support for a hypothetical XPU, optimized for large-scale machine learning."""

        @override
        def setup_device(self, device: torch.device) -> None:
            """
            Raises:
                ValueError:
                    If the selected device is not of type hypothetical XPU.
            """
            if device.type != "xpu":
                raise ValueError(f"Device should be of type 'xpu', got '{device.type}' instead.")
            if device.index is None:
                device = torch.device("xpu", 0)
            xpulib.set_device(device.index)

        @override
        def teardown(self) -> None:
            xpulib.empty_cache()

        @staticmethod
        @override
        def parse_devices(devices: Any) -> Any:
            # Put parsing logic here how devices can be passed into the Trainer
            # via the `devices` argument
            return devices

        @staticmethod
        @override
        def get_parallel_devices(devices: Any) -> Any:
            # Here, convert the device indices to actual device objects
            return [torch.device("xpu", idx) for idx in devices]

        @staticmethod
        @override
        def auto_device_count() -> int:
            # Return a value for auto-device selection when `Trainer(devices="auto")`
            return xpulib.available_devices()

        @staticmethod
        @override
        def is_available() -> bool:
            return xpulib.is_available()

        def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
            # Return optional device statistics for loggers
            return {}

        @staticmethod
        @override
        def get_device_type() -> str:
            return "xpu"


Finally, add the XPUAccelerator to the Trainer:

.. code-block:: python

    from lightning.pytorch import Trainer
    from lightning.pytorch.strategies import DDPStrategy
    accelerator = XPUAccelerator()
    strategy = DDPStrategy(parallel_devices=accelerator.get_parallel_devices(2))
    trainer = Trainer(accelerator=accelerator, strategy=strategy, devices=2)


:doc:`Learn more about Strategies <../extensions/strategy>` and how they interact with the Accelerator.


----------

Registering Accelerators
------------------------

If you wish to switch to a custom accelerator from the CLI without code changes, you can implement the :meth:`~lightning.pytorch.accelerators.accelerator.Accelerator.register_accelerators` class method to register your new accelerator under a shorthand name like so:

.. code-block:: python

    class XPUAccelerator(Accelerator):
        ...

        @classmethod
        @override
        def register_accelerators(cls, accelerator_registry):
            accelerator_registry.register(
                "xpu",
                cls,
                description=f"XPU Accelerator - optimized for large-scale machine learning.",
            )

Now, this is possible:

.. code-block:: python

    trainer = Trainer(accelerator="xpu")

Or if you are using the Lightning CLI, for example:

.. code-block:: bash

    python train.py fit --trainer.accelerator=xpu --trainer.devices=2


----------

Accelerator API
---------------

.. currentmodule:: lightning.pytorch.accelerators

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    Accelerator
    CPUAccelerator
    CUDAAccelerator
    MPSAccelerator
    XLAAccelerator
