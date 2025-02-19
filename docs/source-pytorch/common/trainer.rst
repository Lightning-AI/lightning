.. role:: hidden
    :class: hidden-section

.. testsetup:: *

    import os
    from lightning.pytorch import Trainer, LightningModule, seed_everything

.. _trainer:

Trainer
=======

Once you've organized your PyTorch code into a :class:`~lightning.pytorch.core.LightningModule`, the ``Trainer`` automates everything else.

The ``Trainer`` achieves the following:

1. You maintain control over all aspects via PyTorch code in your :class:`~lightning.pytorch.core.LightningModule`.

2. The trainer uses best practices embedded by contributors and users
   from top AI labs such as Facebook AI Research, NYU, MIT, Stanford, etc...

3. The trainer allows disabling any key part that you don't want automated.

|

-----------

Basic use
---------

This is the basic use of the trainer:

.. code-block:: python

    model = MyLightningModule()

    trainer = Trainer()
    trainer.fit(model, train_dataloader, val_dataloader)

--------

Under the hood
--------------

The Lightning ``Trainer`` does much more than just "training". Under the hood, it handles all loop details for you, some examples include:

- Automatically enabling/disabling grads
- Running the training, validation and test dataloaders
- Calling the Callbacks at the appropriate times
- Putting batches and computations on the correct devices

Here's the pseudocode for what the trainer does under the hood (showing the train loop only)

.. code-block:: python

    # enable grads
    torch.set_grad_enabled(True)

    losses = []
    for batch in train_dataloader:
        # calls hooks like this one
        on_train_batch_start()

        # train step
        loss = training_step(batch)

        # clear gradients
        optimizer.zero_grad()

        # backward
        loss.backward()

        # update parameters
        optimizer.step()

        losses.append(loss)


--------

Trainer in Python scripts
-------------------------
In Python scripts, it's recommended you use a main function to call the Trainer.

.. code-block:: python

    from argparse import ArgumentParser


    def main(hparams):
        model = LightningModule()
        trainer = Trainer(accelerator=hparams.accelerator, devices=hparams.devices)
        trainer.fit(model)


    if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("--accelerator", default=None)
        parser.add_argument("--devices", default=None)
        args = parser.parse_args()

        main(args)

So you can run it like so:

.. code-block:: bash

    python main.py --accelerator 'gpu' --devices 2

.. note::

    Pro-tip: You don't need to define all flags manually.
    You can let the :doc:`LightningCLI <../cli/lightning_cli>` create the Trainer and model with arguments supplied from the CLI.


If you want to stop a training run early, you can press "Ctrl + C" on your keyboard.
The trainer will catch the ``KeyboardInterrupt`` and attempt a graceful shutdown. The trainer object will also set
an attribute ``interrupted`` to ``True`` in such cases. If you have a callback which shuts down compute
resources, for example, you can conditionally run the shutdown logic for only uninterrupted runs by overriding :meth:`lightning.pytorch.Callback.on_exception`.

------------

Validation
----------
You can perform an evaluation epoch over the validation set, outside of the training loop,
using :meth:`~lightning.pytorch.trainer.trainer.Trainer.validate`. This might be
useful if you want to collect new metrics from a model right at its initialization
or after it has already been trained.

.. code-block:: python

    trainer.validate(model=model, dataloaders=val_dataloaders)

------------

Testing
-------
Once you're done training, feel free to run the test set!
(Only right before publishing your paper or pushing to production)

.. code-block:: python

    trainer.test(dataloaders=test_dataloaders)

------------

Reproducibility
---------------

To ensure full reproducibility from run to run you need to set seeds for pseudo-random generators,
and set ``deterministic`` flag in ``Trainer``.

Example::

    from lightning.pytorch import Trainer, seed_everything

    seed_everything(42, workers=True)
    # sets seeds for numpy, torch and python.random.
    model = Model()
    trainer = Trainer(deterministic=True)


By setting ``workers=True`` in :func:`~lightning.pytorch.seed_everything`, Lightning derives
unique seeds across all dataloader workers and processes for :mod:`torch`, :mod:`numpy` and stdlib
:mod:`random` number generators. When turned on, it ensures that e.g. data augmentations are not repeated across workers.

-------

.. _trainer_flags:

Trainer flags
-------------

accelerator
^^^^^^^^^^^

Supports passing different accelerator types (``"cpu", "gpu", "tpu", "hpu", "auto"``)
as well as custom accelerator instances.

.. code-block:: python

    # CPU accelerator
    trainer = Trainer(accelerator="cpu")

    # Training with GPU Accelerator using 2 GPUs
    trainer = Trainer(devices=2, accelerator="gpu")

    # Training with TPU Accelerator using 8 tpu cores
    trainer = Trainer(devices=8, accelerator="tpu")

    # Training with GPU Accelerator using the DistributedDataParallel strategy
    trainer = Trainer(devices=4, accelerator="gpu", strategy="ddp")

.. note:: The ``"auto"`` option recognizes the machine you are on, and selects the appropriate ``Accelerator``.

.. code-block:: python

    # If your machine has GPUs, it will use the GPU Accelerator for training
    trainer = Trainer(devices=2, accelerator="auto")

You can also modify hardware behavior by subclassing an existing accelerator to adjust for your needs.

Example::

    class MyOwnAcc(CPUAccelerator):
        ...

    Trainer(accelerator=MyOwnAcc())

.. note::

    If the ``devices`` flag is not defined, it will assume ``devices`` to be ``"auto"`` and fetch the ``auto_device_count``
    from the accelerator.

    .. code-block:: python

        # This is part of the built-in `CUDAAccelerator`
        class CUDAAccelerator(Accelerator):
            """Accelerator for GPU devices."""

            @staticmethod
            def auto_device_count() -> int:
                """Get the devices when set to auto."""
                return torch.cuda.device_count()


        # Training with GPU Accelerator using total number of gpus available on the system
        Trainer(accelerator="gpu")

accumulate_grad_batches
^^^^^^^^^^^^^^^^^^^^^^^

Accumulates gradients over k batches before stepping the optimizer.

.. testcode::

    # default used by the Trainer (no accumulation)
    trainer = Trainer(accumulate_grad_batches=1)

Example::

    # accumulate every 4 batches (effective batch size is batch*4)
    trainer = Trainer(accumulate_grad_batches=4)

See also: :ref:`gradient_accumulation` to enable more fine-grained accumulation schedules.


benchmark
^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/benchmark.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/benchmark.jpg
    :width: 400
    :muted:

The value (``True`` or ``False``) to set ``torch.backends.cudnn.benchmark`` to. The value for
``torch.backends.cudnn.benchmark`` set in the current session will be used (``False`` if not manually set).
If :paramref:`~lightning.pytorch.trainer.trainer.Trainer.deterministic` is set to ``True``, this will default to ``False``.
You can read more about the interaction of ``torch.backends.cudnn.benchmark`` and ``torch.backends.cudnn.deterministic``
`here <https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking>`__

Setting this flag to ``True`` can increase the speed of your system if your input sizes don't
change. However, if they do, then it might make your system slower. The CUDNN auto-tuner will try to find the best
algorithm for the hardware when a new input size is encountered. This might also increase the memory usage.
Read more about it `here <https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936>`__.

Example::

    # Will use whatever the current value for torch.backends.cudnn.benchmark, normally False
    trainer = Trainer(benchmark=None)  # default

    # you can overwrite the value
    trainer = Trainer(benchmark=True)

deterministic
^^^^^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/deterministic.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/deterministic.jpg
    :width: 400
    :muted:

This flag sets the ``torch.backends.cudnn.deterministic`` flag.
Might make your system slower, but ensures reproducibility.

For more info check `PyTorch docs <https://pytorch.org/docs/stable/notes/randomness.html>`_.

Example::

    # default used by the Trainer
    trainer = Trainer(deterministic=False)

callbacks
^^^^^^^^^

This argument can be used to add a :class:`~lightning.pytorch.callbacks.callback.Callback` or a list of them.
Callbacks run sequentially in the order defined here
with the exception of :class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint` callbacks which run
after all others to ensure all states are saved to the checkpoints.

.. code-block:: python

    # single callback
    trainer = Trainer(callbacks=PrintCallback())

    # a list of callbacks
    trainer = Trainer(callbacks=[PrintCallback()])

Example::

    from lightning.pytorch.callbacks import Callback

    class PrintCallback(Callback):
        def on_train_start(self, trainer, pl_module):
            print("Training is started!")
        def on_train_end(self, trainer, pl_module):
            print("Training is done.")


Model-specific callbacks can also be added inside the ``LightningModule`` through
:meth:`~lightning.pytorch.core.LightningModule.configure_callbacks`.
Callbacks returned in this hook will extend the list initially given to the ``Trainer`` argument, and replace
the trainer callbacks should there be two or more of the same type.
:class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint` callbacks always run last.


check_val_every_n_epoch
^^^^^^^^^^^^^^^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/check_val_every_n_epoch.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/check_val_every_n_epoch.jpg
    :width: 400
    :muted:

Check val every n train epochs.

Example::

    # default used by the Trainer
    trainer = Trainer(check_val_every_n_epoch=1)

    # run val loop every 10 training epochs
    trainer = Trainer(check_val_every_n_epoch=10)


default_root_dir
^^^^^^^^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/default_root_dir.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/default%E2%80%A8_root_dir.jpg
    :width: 400
    :muted:

Default path for logs and weights when no logger or
:class:`lightning.pytorch.callbacks.ModelCheckpoint` callback passed.  On
certain clusters you might want to separate where logs and checkpoints are
stored. If you don't then use this argument for convenience. Paths can be local
paths or remote paths such as ``s3://bucket/path`` or ``hdfs://path/``. Credentials
will need to be set up to use remote filepaths.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(default_root_dir=os.getcwd())

devices
^^^^^^^

Number of devices to train on (``int``), which devices to train on (``list`` or ``str``), or ``"auto"``.

.. code-block:: python

    # Training with CPU Accelerator using 2 processes
    trainer = Trainer(devices=2, accelerator="cpu")

    # Training with GPU Accelerator using GPUs 1 and 3
    trainer = Trainer(devices=[1, 3], accelerator="gpu")

    # Training with TPU Accelerator using 8 tpu cores
    trainer = Trainer(devices=8, accelerator="tpu")

.. tip:: The ``"auto"`` option recognizes the devices to train on, depending on the ``Accelerator`` being used.

.. code-block:: python

    # Use whatever hardware your machine has available
    trainer = Trainer(devices="auto", accelerator="auto")

    # Training with CPU Accelerator using 1 process
    trainer = Trainer(devices="auto", accelerator="cpu")

    # Training with TPU Accelerator using 8 tpu cores
    trainer = Trainer(devices="auto", accelerator="tpu")

.. note::

    If the ``devices`` flag is not defined, it will assume ``devices`` to be ``"auto"`` and fetch the ``auto_device_count``
    from the accelerator.

    .. code-block:: python

        # This is part of the built-in `CUDAAccelerator`
        class CUDAAccelerator(Accelerator):
            """Accelerator for GPU devices."""

            @staticmethod
            def auto_device_count() -> int:
                """Get the devices when set to auto."""
                return torch.cuda.device_count()


        # Training with GPU Accelerator using total number of gpus available on the system
        Trainer(accelerator="gpu")

enable_checkpointing
^^^^^^^^^^^^^^^^^^^^

By default Lightning saves a checkpoint for you in your current working directory, with the state of your last training epoch,
Checkpoints capture the exact value of all parameters used by a model.
To disable automatic checkpointing, set this to `False`.

.. code-block:: python

    # default used by Trainer, saves the most recent model to a single checkpoint after each epoch
    trainer = Trainer(enable_checkpointing=True)

    # turn off automatic checkpointing
    trainer = Trainer(enable_checkpointing=False)


You can override the default behavior by initializing the :class:`~lightning.pytorch.callbacks.ModelCheckpoint`
callback, and adding it to the :paramref:`~lightning.pytorch.trainer.trainer.Trainer.callbacks` list.
See :doc:`Saving and Loading Checkpoints <../common/checkpointing>` for how to customize checkpointing.

.. testcode::

    from lightning.pytorch.callbacks import ModelCheckpoint

    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    # Add your callback to the callbacks list
    trainer = Trainer(callbacks=[checkpoint_callback])

fast_dev_run
^^^^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/fast_dev_run.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/fast_dev_run.jpg
    :width: 400
    :muted:

Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es) to ensure your code will execute without errors. This
applies to fitting, validating, testing, and predicting. This flag is **only** recommended for debugging purposes and
should not be used to limit the number of batches to run.

.. code-block:: python

    # default used by the Trainer
    trainer = Trainer(fast_dev_run=False)

    # runs only 1 training and 1 validation batch and the program ends
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(...)

    # runs 7 predict batches and program ends
    trainer = Trainer(fast_dev_run=7)
    trainer.predict(...)

This argument is different from ``limit_{train,val,test,predict}_batches`` because side effects are avoided to reduce the
impact to subsequent runs. These are the changes enabled:

- Sets ``Trainer(max_epochs=1)``.
- Sets ``Trainer(max_steps=...)`` to 1 or the number passed.
- Sets ``Trainer(num_sanity_val_steps=0)``.
- Sets ``Trainer(val_check_interval=1.0)``.
- Sets ``Trainer(check_every_n_epoch=1)``.
- Disables all loggers.
- Disables passing logged metrics to loggers.
- The :class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint` callbacks will not trigger.
- The :class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping` callbacks will not trigger.
- Sets ``limit_{train,val,test,predict}_batches`` to 1 or the number passed.
- Disables the tuning callbacks (:class:`~lightning.pytorch.callbacks.batch_size_finder.BatchSizeFinder`, :class:`~lightning.pytorch.callbacks.lr_finder.LearningRateFinder`).
- If using the CLI, the configuration file is not saved.


gradient_clip_val
^^^^^^^^^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/gradient_clip_val.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/gradient+_clip_val.jpg
    :width: 400
    :muted:

Gradient clipping value

.. testcode::

    # default used by the Trainer
    trainer = Trainer(gradient_clip_val=None)

limit_train_batches
^^^^^^^^^^^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/limit_batches.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/limit_train_batches.jpg
    :width: 400
    :muted:

How much of training dataset to check.
Useful when debugging or testing something that happens at the end of an epoch.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(limit_train_batches=1.0)

Example::

    # default used by the Trainer
    trainer = Trainer(limit_train_batches=1.0)

    # run through only 25% of the training set each epoch
    trainer = Trainer(limit_train_batches=0.25)

    # run through only 10 batches of the training set each epoch
    trainer = Trainer(limit_train_batches=10)

limit_test_batches
^^^^^^^^^^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/limit_batches.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/limit_test_batches.jpg
    :width: 400
    :muted:

How much of test dataset to check.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(limit_test_batches=1.0)

    # run through only 25% of the test set each epoch
    trainer = Trainer(limit_test_batches=0.25)

    # run for only 10 batches
    trainer = Trainer(limit_test_batches=10)

In the case of multiple test dataloaders, the limit applies to each dataloader individually.

limit_val_batches
^^^^^^^^^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/limit_batches.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/limit_val_batches.jpg
    :width: 400
    :muted:

How much of validation dataset to check.
Useful when debugging or testing something that happens at the end of an epoch.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(limit_val_batches=1.0)

    # run through only 25% of the validation set each epoch
    trainer = Trainer(limit_val_batches=0.25)

    # run for only 10 batches
    trainer = Trainer(limit_val_batches=10)

    # disable validation
    trainer = Trainer(limit_val_batches=0)

In the case of multiple validation dataloaders, the limit applies to each dataloader individually.

log_every_n_steps
^^^^^^^^^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/log_every_n_steps.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/log_every_n_steps.jpg
    :width: 400
    :muted:

How often to add logging rows (does not write to disk)

.. testcode::

    # default used by the Trainer
    trainer = Trainer(log_every_n_steps=50)

See Also:
    - :doc:`logging <../extensions/logging>`

logger
^^^^^^

:doc:`Logger <../visualize/loggers>` (or iterable collection of loggers) for experiment tracking. A ``True`` value uses the default ``TensorBoardLogger`` shown below. ``False`` will disable logging.

.. testcode::
    :skipif: not _TENSORBOARD_AVAILABLE and not _TENSORBOARDX_AVAILABLE

    from lightning.pytorch.loggers import TensorBoardLogger

    # default logger used by trainer (if tensorboard is installed)
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
    Trainer(logger=logger)

max_epochs
^^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/min_max_epochs.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/max_epochs.jpg
    :width: 400
    :muted:

Stop training once this number of epochs is reached

.. testcode::

    # default used by the Trainer
    trainer = Trainer(max_epochs=1000)

If both ``max_epochs`` and ``max_steps`` aren't specified, ``max_epochs`` will default to ``1000``.
To enable infinite training, set ``max_epochs = -1``.

min_epochs
^^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/min_max_epochs.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/min_epochs.jpg
    :width: 400
    :muted:

Force training for at least these many epochs

.. testcode::

    # default used by the Trainer
    trainer = Trainer(min_epochs=1)

max_steps
^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/min_max_steps.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/max_steps.jpg
    :width: 400
    :muted:

Stop training after this number of :ref:`global steps <common/trainer:global_step>`.
Training will stop if max_steps or max_epochs have reached (earliest).

.. testcode::

    # Default (disabled)
    trainer = Trainer(max_steps=-1)

    # Stop after 100 steps
    trainer = Trainer(max_steps=100)

If ``max_steps`` is not specified, ``max_epochs`` will be used instead (and ``max_epochs`` defaults to
``1000`` if ``max_epochs`` is not specified). To disable this default, set ``max_steps = -1``.

min_steps
^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/min_max_steps.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/min_steps.jpg
    :width: 400
    :muted:

Force training for at least this number of :ref:`global steps <common/trainer:global_step>`.
Trainer will train model for at least min_steps or min_epochs (latest).

.. testcode::

    # Default (disabled)
    trainer = Trainer(min_steps=None)

    # Run at least for 100 steps (disable min_epochs)
    trainer = Trainer(min_steps=100, min_epochs=0)

max_time
^^^^^^^^

Set the maximum amount of time for training. Training will get interrupted mid-epoch.
For customizable options use the :class:`~lightning.pytorch.callbacks.timer.Timer` callback.

.. testcode::

    # Default (disabled)
    trainer = Trainer(max_time=None)

    # Stop after 12 hours of training or when reaching 10 epochs (string)
    trainer = Trainer(max_time="00:12:00:00", max_epochs=10)

    # Stop after 1 day and 5 hours (dict)
    trainer = Trainer(max_time={"days": 1, "hours": 5})

In case ``max_time`` is used together with ``min_steps`` or ``min_epochs``, the ``min_*`` requirement
always has precedence.

num_nodes
^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/num_nodes.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/num_nodes.jpg
    :width: 400
    :muted:

Number of GPU nodes for distributed training.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(num_nodes=1)

    # to train on 8 nodes
    trainer = Trainer(num_nodes=8)


num_sanity_val_steps
^^^^^^^^^^^^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/num_sanity_val_steps.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/num_sanity%E2%80%A8_val_steps.jp
    :width: 400
    :muted:

Sanity check runs n batches of val before starting the training routine.
This catches any bugs in your validation without having to wait for the first validation check.
The Trainer uses 2 steps by default. Turn it off or modify it here.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(num_sanity_val_steps=2)

    # turn it off
    trainer = Trainer(num_sanity_val_steps=0)

    # check all validation data
    trainer = Trainer(num_sanity_val_steps=-1)


This option will reset the validation dataloader unless ``num_sanity_val_steps=0``.

overfit_batches
^^^^^^^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/overfit_batches.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/overfit_batches.jpg
    :width: 400
    :muted:

Uses this much data of the training & validation set.
If the training & validation dataloaders have ``shuffle=True``, Lightning will automatically disable it.

Useful for quickly debugging or trying to overfit on purpose.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(overfit_batches=0.0)

    # use only 1% of the train & val set
    trainer = Trainer(overfit_batches=0.01)

    # overfit on 10 of the same batches
    trainer = Trainer(overfit_batches=10)

plugins
^^^^^^^

:ref:`Plugins` allow you to connect arbitrary backends, precision libraries, clusters etc. For example:

- :ref:`Checkpoint IO <checkpointing_expert>`
- `TorchElastic <https://pytorch.org/elastic/0.2.2/index.html>`_
- :ref:`Precision Plugins <precision_expert>`

To define your own behavior, subclass the relevant class and pass it in. Here's an example linking up your own
:class:`~lightning.pytorch.plugins.environments.ClusterEnvironment`.

.. code-block:: python

    from lightning.pytorch.plugins.environments import ClusterEnvironment


    class MyCluster(ClusterEnvironment):
        def main_address(self):
            return your_main_address

        def main_port(self):
            return your_main_port

        def world_size(self):
            return the_world_size


    trainer = Trainer(plugins=[MyCluster()], ...)

precision
^^^^^^^^^

There are two different techniques to set the mixed precision. "True" precision and "Mixed" precision.

Lightning supports doing floating point operations in 64-bit precision ("double"), 32-bit precision ("full"), or 16-bit ("half") with both regular and `bfloat16 <https://pytorch.org/docs/1.10.0/generated/torch.Tensor.bfloat16.html>`_).
This selected precision will have a direct impact in the performance and memory usage based on your hardware.
Automatic mixed precision settings are denoted by a ``"-mixed"`` suffix, while "true" precision settings have a ``"-true"`` suffix:

.. code-block:: python

    # Default used by the Trainer
    fabric = Fabric(precision="32-true", devices=1)

    # the same as:
    trainer = Trainer(precision="32", devices=1)

    # 16-bit mixed precision (model weights remain in torch.float32)
    trainer = Trainer(precision="16-mixed", devices=1)

    # 16-bit bfloat mixed precision (model weights remain in torch.float32)
    trainer = Trainer(precision="bf16-mixed", devices=1)

    # 8-bit mixed precision via TransformerEngine (model weights get cast to torch.bfloat16)
    trainer = Trainer(precision="transformer-engine", devices=1)

    # 16-bit precision (model weights get cast to torch.float16)
    trainer = Trainer(precision="16-true", devices=1)

    # 16-bit bfloat precision (model weights get cast to torch.bfloat16)
    trainer = Trainer(precision="bf16-true", devices=1)

    # 64-bit (double) precision (model weights get cast to torch.float64)
    trainer = Trainer(precision="64-true", devices=1)


See the :doc:`N-bit precision guide <../common/precision>` for more details.


profiler
^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/profiler.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/profiler.jpg
    :width: 400
    :muted:

To profile individual steps during training and assist in identifying bottlenecks.

See the :doc:`profiler documentation <../tuning/profiler>` for more details.

.. testcode::

    from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

    # default used by the Trainer
    trainer = Trainer(profiler=None)

    # to profile standard training events, equivalent to `profiler=SimpleProfiler()`
    trainer = Trainer(profiler="simple")

    # advanced profiler for function-level stats, equivalent to `profiler=AdvancedProfiler()`
    trainer = Trainer(profiler="advanced")

enable_progress_bar
^^^^^^^^^^^^^^^^^^^

Whether to enable or disable the progress bar. Defaults to True.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(enable_progress_bar=True)

    # disable progress bar
    trainer = Trainer(enable_progress_bar=False)

reload_dataloaders_every_n_epochs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/reload_dataloaders_every_epoch.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/reload_%E2%80%A8dataloaders_%E2%80%A8every_epoch.jpg
    :width: 400
    :muted:

Set to a positive integer to reload dataloaders every n epochs from your currently used data source.
DataSource can be a ``LightningModule`` or a ``LightningDataModule``.


.. code-block:: python

    # if 0 (default)
    train_loader = model.train_dataloader()
    # or if using data module: datamodule.train_dataloader()
    for epoch in epochs:
        for batch in train_loader:
            ...

    # if a positive integer
    for epoch in epochs:
        if not epoch % reload_dataloaders_every_n_epochs:
            train_loader = model.train_dataloader()
            # or if using data module: datamodule.train_dataloader()
        for batch in train_loader:
            ...

The pseudocode applies also to the ``val_dataloader``.

.. _replace-sampler-ddp:

use_distributed_sampler
^^^^^^^^^^^^^^^^^^^^^^^

See :paramref:`lightning.pytorch.trainer.Trainer.params.use_distributed_sampler`.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(use_distributed_sampler=True)

By setting to False, you have to add your own distributed sampler:

.. code-block:: python

    # in your LightningModule or LightningDataModule
    def train_dataloader(self):
        dataset = ...
        # default used by the Trainer
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
        return dataloader


strategy
^^^^^^^^

Supports passing different training strategies with aliases (ddp, fsdp, etc) as well as configured strategies.

.. code-block:: python

    # Data-parallel training with the DDP strategy on 4 GPUs
    trainer = Trainer(strategy="ddp", accelerator="gpu", devices=4)

    # Model-parallel training with the FSDP strategy on 4 GPUs
    trainer = Trainer(strategy="fsdp", accelerator="gpu", devices=4)

Additionally, you can pass a strategy object.

.. code-block:: python

    from lightning.pytorch.strategies import DDPStrategy

    trainer = Trainer(strategy=DDPStrategy(static_graph=True), accelerator="gpu", devices=2)

See Also:
    - :ref:`Multi GPU Training <multi_gpu>`.
    - :doc:`Model Parallel GPU training guide <../advanced/model_parallel>`.
    - :doc:`TPU training guide <../accelerators/tpu>`.


sync_batchnorm
^^^^^^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/sync_batchnorm.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/sync_batchnorm.jpg
    :width: 400
    :muted:

Enable synchronization between batchnorm layers across all GPUs.

.. testcode::

    trainer = Trainer(sync_batchnorm=True)


val_check_interval
^^^^^^^^^^^^^^^^^^

.. video:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/val_check_interval.mp4
    :poster: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/thumb/val_check_interval.jpg
    :width: 400
    :muted:

How often within one training epoch to check the validation set.
Can specify as float or int.

- pass a ``float`` in the range [0.0, 1.0] to check after a fraction of the training epoch.
- pass an ``int`` to check after a fixed number of training batches. An ``int`` value can only be higher than the number of training
  batches when ``check_val_every_n_epoch=None``, which validates after every ``N`` training batches across epochs or iteration-based training.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(val_check_interval=1.0)

    # check validation set 4 times during a training epoch
    trainer = Trainer(val_check_interval=0.25)

    # check validation set every 1000 training batches in the current epoch
    trainer = Trainer(val_check_interval=1000)

    # check validation set every 1000 training batches across complete epochs or during iteration-based training
    # use this when using iterableDataset and your dataset has no length
    # (ie: production cases with streaming data)
    trainer = Trainer(val_check_interval=1000, check_val_every_n_epoch=None)


.. code-block:: python

    # Here is the computation to estimate the total number of batches seen within an epoch.

    # Find the total number of train batches
    total_train_batches = total_train_samples // (train_batch_size * world_size)

    # Compute how many times we will call validation during the training loop
    val_check_batch = max(1, int(total_train_batches * val_check_interval))
    val_checks_per_epoch = total_train_batches / val_check_batch

    # Find the total number of validation batches
    total_val_batches = total_val_samples // (val_batch_size * world_size)

    # Total number of batches run
    total_fit_batches = total_train_batches + total_val_batches


enable_model_summary
^^^^^^^^^^^^^^^^^^^^

Whether to enable or disable the model summarization. Defaults to True.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(enable_model_summary=True)

    # disable summarization
    trainer = Trainer(enable_model_summary=False)

    # enable custom summarization
    from lightning.pytorch.callbacks import ModelSummary

    trainer = Trainer(enable_model_summary=True, callbacks=[ModelSummary(max_depth=-1)])


inference_mode
^^^^^^^^^^^^^^

Whether to use :func:`torch.inference_mode` or :func:`torch.no_grad` mode during evaluation
(``validate``/``test``/``predict``)

.. testcode::

    # default used by the Trainer
    trainer = Trainer(inference_mode=True)

    # Use `torch.no_grad` instead
    trainer = Trainer(inference_mode=False)


With :func:`torch.inference_mode` disabled, you can enable the grad of your model layers if required.

.. code-block:: python

    class LitModel(LightningModule):
        def validation_step(self, batch, batch_idx):
            preds = self.layer1(batch)
            with torch.enable_grad():
                grad_preds = preds.requires_grad_()
                preds2 = self.layer2(grad_preds)


    model = LitModel()
    trainer = Trainer(inference_mode=False)
    trainer.validate(model)

enable_autolog_hparams
^^^^^^^^^^^^^^^^^^^^^^

Whether to log hyperparameters at the start of a run. Defaults to True.

.. testcode::

    # default used by the Trainer
    trainer = Trainer(enable_autolog_hparams=True)

    # disable logging hyperparams
    trainer = Trainer(enable_autolog_hparams=False)

With the parameter set to false, you can add custom code to log hyperparameters.

.. code-block:: python

    model = LitModel()
    trainer = Trainer(enable_autolog_hparams=False)
    for logger in trainer.loggers:
        if isinstance(logger, lightning.pytorch.loggers.CSVLogger):
            logger.log_hyperparams(hparams_dict_1)
        else:
            logger.log_hyperparams(hparams_dict_2)

You can also use `self.logger.log_hyperparams(...)` inside `LightningModule` to log.

-----

Trainer class API
-----------------

Methods
^^^^^^^

init
****

.. automethod:: lightning.pytorch.trainer.Trainer.__init__
   :noindex:

fit
****

.. automethod:: lightning.pytorch.trainer.Trainer.fit
   :noindex:

validate
********

.. automethod:: lightning.pytorch.trainer.Trainer.validate
   :noindex:

test
****

.. automethod:: lightning.pytorch.trainer.Trainer.test
   :noindex:

predict
*******

.. automethod:: lightning.pytorch.trainer.Trainer.predict
   :noindex:


Properties
^^^^^^^^^^

callback_metrics
****************

The metrics available to callbacks.

This includes metrics logged via :meth:`~lightning.pytorch.core.LightningModule.log`.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        self.log("a_val", 2.0)


    callback_metrics = trainer.callback_metrics
    assert callback_metrics["a_val"] == 2.0

logged_metrics
**************

The metrics sent to the loggers.

This includes metrics logged via :meth:`~lightning.pytorch.core.LightningModule.log` with the
:paramref:`~lightning.pytorch.core.LightningModule.log.logger` argument set.

progress_bar_metrics
********************

The metrics sent to the progress bar.

This includes metrics logged via :meth:`~lightning.pytorch.core.LightningModule.log` with the
:paramref:`~lightning.pytorch.core.LightningModule.log.prog_bar` argument set.

current_epoch
*************

The current epoch, updated after the epoch end hooks are run.

datamodule
**********

The current datamodule, which is used by the trainer.

.. code-block:: python

    used_datamodule = trainer.datamodule

is_last_batch
*************

Whether trainer is executing the last batch.

global_step
***********

The number of optimizer steps taken (does not reset each epoch).

This includes multiple optimizers (if enabled).

logger
*******

The first :class:`~lightning.pytorch.loggers.logger.Logger` being used.

loggers
********

The list of :class:`~lightning.pytorch.loggers.logger.Logger` used.

.. code-block:: python

    for logger in trainer.loggers:
        logger.log_metrics({"foo": 1.0})

log_dir
*******

The directory for the current experiment. Use this to save images to, etc...

.. code-block:: python

    def training_step(self, batch, batch_idx):
        img = ...
        save_img(img, self.trainer.log_dir)

is_global_zero
**************

Whether this process is the global zero in multi-node training.

.. code-block:: python

    def training_step(self, batch, batch_idx):
        if self.trainer.is_global_zero:
            print("in node 0, accelerator 0")

estimated_stepping_batches
**************************

The estimated number of batches that will ``optimizer.step()`` during training.

This accounts for gradient accumulation and the current trainer configuration. This might sets up your training
dataloader if hadn't been set up already.

.. code-block:: python

    def configure_optimizers(self):
        optimizer = ...
        stepping_batches = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=stepping_batches)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

state
*****

The current state of the Trainer, including the current function that is running, the stage of
execution within that function, and the status of the Trainer.

.. code-block:: python

    # fn in ("fit", "validate", "test", "predict")
    trainer.state.fn
    # status in ("initializing", "running", "finished", "interrupted")
    trainer.state.status
    # stage in ("train", "sanity_check", "validate", "test", "predict")
    trainer.state.stage

should_stop
***********

If you want to terminate the training during ``.fit``, you can set ``trainer.should_stop=True`` to terminate the training
as soon as possible. Note that, it will respect the arguments ``min_steps`` and ``min_epochs`` to check whether to stop. If these
arguments are set and the ``current_epoch`` or ``global_step`` don't meet these minimum conditions, training will continue until
both conditions are met. If any of these arguments is not set, it won't be considered for the final decision.


.. code-block:: python

    # setting `trainer.should_stop` at any point of training will terminate it
    class LitModel(LightningModule):
        def training_step(self, *args, **kwargs):
            self.trainer.should_stop = True


    trainer = Trainer()
    model = LitModel()
    trainer.fit(model)

.. code-block:: python

    # setting `trainer.should_stop` will stop training only after at least 5 epochs have run
    class LitModel(LightningModule):
        def training_step(self, *args, **kwargs):
            if self.current_epoch == 2:
                self.trainer.should_stop = True


    trainer = Trainer(min_epochs=5, max_epochs=100)
    model = LitModel()
    trainer.fit(model)

.. code-block:: python

    # setting `trainer.should_stop` will stop training only after at least 5 steps have run
    class LitModel(LightningModule):
        def training_step(self, *args, **kwargs):
            if self.global_step == 2:
                self.trainer.should_stop = True


    trainer = Trainer(min_steps=5, max_epochs=100)
    model = LitModel()
    trainer.fit(model)

.. code-block:: python

    # setting `trainer.should_stop` at any until both min_steps and min_epochs are satisfied
    class LitModel(LightningModule):
        def training_step(self, *args, **kwargs):
            if self.global_step == 7:
                self.trainer.should_stop = True


    trainer = Trainer(min_steps=5, min_epochs=5, max_epochs=100)
    model = LitModel()
    trainer.fit(model)

sanity_checking
***************

Indicates if the trainer is currently running sanity checking. This property can be useful to disable some hooks,
logging or callbacks during the sanity checking.

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        ...
        if not self.trainer.sanity_checking:
            self.log("value", value)

num_training_batches
********************

The number of training batches that will be used during ``trainer.fit()``.

num_sanity_val_batches
**********************

The number of validation batches that will be used during the sanity-checking part of ``trainer.fit()``.

num_val_batches
***************

The number of validation batches that will be used during ``trainer.fit()`` or ``trainer.validate()``.

num_test_batches
****************

The number of test batches that will be used during ``trainer.test()``.

num_predict_batches
*******************

The number of prediction batches that will be used during ``trainer.predict()``.

train_dataloader
****************

The training dataloader(s) used during ``trainer.fit()``.

val_dataloaders
***************

The validation dataloader(s) used during ``trainer.fit()`` or ``trainer.validate()``.

test_dataloaders
****************

The test dataloader(s) used during ``trainer.test()``.

predict_dataloaders
*******************

The prediction dataloader(s) used during ``trainer.predict()``.
