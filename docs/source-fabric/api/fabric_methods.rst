##############
Fabric Methods
##############

launch
======

With :meth:`~lightning.fabric.fabric.Fabric.launch` you can conveniently launch your script or a function
into multiple processes for distributed training on a single machine.

.. code-block:: python

    # Launch the script on 2 devices and init distributed backend
    fabric = Fabric(devices=2)
    fabric.launch()

The same can be done with code inside a function:

.. code-block:: python

    def run(fabric):
        # Your distributed code here
        ...


    # Launch a function on 2 devices and init distributed backend
    fabric = Fabric(devices=2)
    fabric.launch(run)

For example, you can use the latter for multi-GPU training inside a :doc:`Jupyter notebook <../fundamentals/notebooks>`.
For launching distributed training with the CLI, multi-node cluster, or cloud, see :doc:`../fundamentals/launch`.

setup
=====

Set up a model and corresponding optimizer(s). If you need to set up multiple models, call ``setup()`` on each of them.
Moves the model and optimizer to the correct device automatically.

.. code-block:: python

    model = nn.Linear(32, 64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)

    # Set up model and optimizer for accelerated training
    model, optimizer = fabric.setup(model, optimizer)

    # If you don't want Fabric to set the device
    model, optimizer = fabric.setup(model, optimizer, move_to_device=False)

    # If you want to additionally register a learning rate scheduler with compatible strategies such as DeepSpeed
    model, optimizer, scheduler = fabric.setup(model, optimizer, scheduler)


The setup method also prepares the model for the selected precision choice so that operations during ``forward()`` get
cast automatically. Advanced users should read :doc:`the notes on models wrapped by Fabric <../api/wrappers>`.

setup_dataloaders
=================

Set up one or multiple data loaders for accelerated operation. If you run a distributed strategy (e.g., DDP), Fabric
automatically replaces the sampler. In addition, the data loader will be configured to move the returned
data tensors to the correct device automatically.

.. code-block:: python

    train_data = torch.utils.DataLoader(train_dataset, ...)
    test_data = torch.utils.DataLoader(test_dataset, ...)

    train_data, test_data = fabric.setup_dataloaders(train_data, test_data)

    # If you don't want Fabric to move the data to the device
    train_data, test_data = fabric.setup_dataloaders(train_data, test_data, move_to_device=False)

    # If you don't want Fabric to replace the sampler in the context of distributed training
    train_data, test_data = fabric.setup_dataloaders(train_data, test_data, use_distributed_sampler=False)


backward
========

This replaces any occurrences of ``loss.backward()`` and makes your code accelerator and precision agnostic.

.. code-block:: python

    output = model(input)
    loss = loss_fn(output, target)

    # loss.backward()
    fabric.backward(loss)


clip_gradients
==============

Clip the gradients of the model to a given max value or max norm.
This is useful if your model experiences *exploding gradients* during training.

.. code-block:: python

    # Clip gradients to a max value of +/- 0.5
    fabric.clip_gradients(model, optimizer, clip_val=0.5)

    # Clip gradients such that their total norm is no bigger than 2.0
    fabric.clip_gradients(model, optimizer, max_norm=2.0)

    # By default, clipping by norm uses the 2-norm
    fabric.clip_gradients(model, optimizer, max_norm=2.0, norm_type=2)

    # You can also choose the infinity-norm, which clips the largest
    # element among all
    fabric.clip_gradients(model, optimizer, max_norm=2.0, norm_type="inf")

The :meth:`~lightning.fabric.fabric.Fabric.clip_gradients` method is agnostic to the precision and strategy being used.
If you pass `max_norm` as the argument, ``clip_gradients`` will return the total norm of the gradients (before clipping was applied) as a scalar tensor.


to_device
=========

Use :meth:`~lightning.fabric.fabric.Fabric.to_device` to move models, tensors, or collections of tensors to
the current device. By default :meth:`~lightning.fabric.fabric.Fabric.setup` and
:meth:`~lightning.fabric.fabric.Fabric.setup_dataloaders` already move the model and data to the correct
device, so calling this method is only necessary for manual operation when needed.

.. code-block:: python

    data = torch.load("dataset.pt")
    data = fabric.to_device(data)


seed_everything
===============

Make your code reproducible by calling this method at the beginning of your run.

.. code-block:: python

    # Instead of `torch.manual_seed(...)`, call:
    fabric.seed_everything(1234)


This covers PyTorch, NumPy, and Python random number generators. In addition, Fabric takes care of properly initializing
the seed of data loader worker processes (can be turned off by passing ``workers=False``).

init_module
===========

Instantiating a ``nn.Module`` in PyTorch creates all parameters on CPU in float32 precision by default.
To speed up initialization, you can force PyTorch to create the model directly on the target device and with the desired precision without changing your model code.

.. code-block:: python

    fabric = Fabric(accelerator="cuda", precision="16-true")

    with fabric.init_module():
        # models created here will be on GPU and in float16
        model = MyModel()

This eliminates the waiting time to transfer the model parameters from the CPU to the device.
For strategies that handle large sharded models (FSDP, DeepSpeed), the :meth:`~lightning.fabric.fabric.Fabric.init_module` method will allocate the model parameters on the meta device first before sharding.
This makes it possible to work with models that are larger than the memory of a single device.

See also: :doc:`../advanced/model_init`


autocast
========

Let the precision backend autocast the block of code under this context manager. This is optional and already done by
Fabric for the model's forward method (once the model was :meth:`~lightning.fabric.fabric.Fabric.setup`).
You need this only if you wish to autocast more operations outside the ones in model forward:

.. code-block:: python

    model, optimizer = fabric.setup(model, optimizer)

    # Fabric handles precision automatically for the model
    output = model(inputs)

    with fabric.autocast():  # optional
        loss = loss_function(output, target)

    fabric.backward(loss)
    ...

See also: :doc:`../fundamentals/precision`


print
=====

Print to the console via the built-in print function, but only on the main process.
This avoids excessive printing and logs when running on multiple devices/nodes.


.. code-block:: python

    # Print only on the main process
    fabric.print(f"{epoch}/{num_epochs}| Train Epoch Loss: {loss}")


save
====

Save the state of objects to a checkpoint file.
Replaces all occurrences of ``torch.save(...)`` in your code.
Fabric will handle the saving part correctly, whether running a single device, multi-devices, or multi-nodes.

.. code-block:: python

    # Define the state of your program/loop
    state = {
        "model1": model1,
        "model2": model2,
        "optimizer": optimizer,
        "iteration": iteration,
    }

    # Instead of `torch.save(...)`
    fabric.save("path/to/checkpoint.ckpt", state)

You should pass the model and optimizer objects directly into the dictionary so Fabric can unwrap them and automatically retrieve their *state-dict*.

See also: :doc:`../guide/checkpoint/index`


load
====

Load checkpoint contents from a file and restore the state of objects in your program.
Replaces all occurrences of ``torch.load(...)`` in your code.
Fabric will handle the loading part correctly, whether running a single device, multi-device, or multi-node.

.. code-block:: python

    # Define the state of your program/loop
    state = {
        "model1": model1,
        "model2": model2,
        "optimizer": optimizer,
        "iteration": iteration,
    }

    # Restore the state of objects (in-place)
    fabric.load("path/to/checkpoint.ckpt", state)

    # Or load everything and restore your objects manually
    checkpoint = fabric.load("./checkpoints/version_2/checkpoint.ckpt")
    model.load_state_dict(checkpoint["model"])
    ...


To load the state of your model or optimizer from a raw PyTorch checkpoint (not saved with Fabric), use :meth:`~lightning.fabric.fabric.Fabric.load_raw` instead.
See also: :doc:`../guide/checkpoint/index`


load_raw
========

Load the state-dict of a model or optimizer from a raw PyTorch checkpoint not saved by Fabric.

.. code-block:: python

    model = MyModel()

    # A model weights file saved by your friend who doesn't use Fabric
    fabric.load_raw("path/to/model.pt", model)

    # Equivalent to this:
    # model.load_state_dict(torch.load("path/to/model.pt"))


See also: :doc:`../guide/checkpoint/index`


barrier
=======

Call this if you want all processes to wait and synchronize. Once all processes have entered this call,
execution continues. Useful for example, when you want to download data on one process and make all others wait until
the data is written to disk.

.. code-block:: python

    if fabric.global_rank == 0:
        print("Downloading dataset. This can take a while ...")
        download_dataset("http://...")

    # All other processes wait here until rank 0 is done with downloading:
    fabric.barrier()

    # After everyone reached the barrier, they can access the downloaded files:
    load_dataset()

See also: :doc:`../advanced/distributed_communication`


all_gather, all_reduce, broadcast
=================================

You can send tensors and other data between processes using collective operations.
The three most common ones, :meth:`~lightning.fabric.fabric.Fabric.broadcast`, :meth:`~lightning.fabric.fabric.Fabric.all_gather` and :meth:`~lightning.fabric.fabric.Fabric.all_reduce` are available directly on the Fabric object for convenience:

- :meth:`~lightning.fabric.fabric.Fabric.broadcast`: Send a tensor from one process to all others.
- :meth:`~lightning.fabric.fabric.Fabric.all_gather`: Gather tensors from every process and stack them.
- :meth:`~lightning.fabric.fabric.Fabric.all_reduce`: Apply a reduction function on tensors across processes (sum, mean, etc.).

.. code-block:: python

    # Send the value of a tensor from rank 0 to all others
    result = fabric.broadcast(tensor, src=0)

    # Every process gets the stack of tensors from everybody else
    all_tensors = fabric.all_gather(tensor)

    # Sum a tensor across processes (everyone gets the result)
    reduced_tensor = fabric.all_reduce(tensor, reduce_op="sum")

    # Also works with a collection of tensors (dict, list, tuple):
    collection = {"loss": torch.tensor(...), "data": ...}
    gathered_collection = fabric.all_gather(collection, ...)
    reduced_collection = fabric.all_reduce(collection, ...)


.. important::

    Every process needs to enter the collective calls, and tensors need to have the same shape across all processes.
    Otherwise, the program will hang!

Learn more about :doc:`distributed communication <../advanced/distributed_communication>`.


no_backward_sync
================

Use this context manager when performing gradient accumulation and using a distributed strategy (e.g., DDP).
It will speed up your training loop by cutting redundant communication between processes during the accumulation phase.

.. code-block:: python

    # Accumulate gradient 8 batches at a time
    is_accumulating = batch_idx % 8 != 0

    with fabric.no_backward_sync(model, enabled=is_accumulating):
        output = model(input)
        loss = ...
        fabric.backward(loss)
        ...

    # Step the optimizer every 8 batches
    if not is_accumulating:
        optimizer.step()
        optimizer.zero_grad()

Both the model's `.forward()` and the `fabric.backward()` call need to run under this context as shown in the example above.
For single-device strategies, it is a no-op. Some strategies don't support this:

- deepspeed
- dp
- xla

For these, the context manager falls back to a no-op and emits a warning.


call
====

Use this to run all registered callback hooks with a given name and inputs.
It is useful when building a Trainer that allows the user to run arbitrary code at fixed points in the training loop.

.. code-block:: python

    class MyCallback:
        def on_train_start(self):
            ...

        def on_train_epoch_end(self, model, results):
            ...


    fabric = Fabric(callbacks=[MyCallback()])

    # Call any hook by name
    fabric.call("on_train_start")

    # Pass in additional arguments that the hook requires
    fabric.call("on_train_epoch_end", model=..., results={...})

    # Only the callbacks that have this method defined will be executed
    fabric.call("undefined")


See also: :doc:`../guide/callbacks`


log and log_dict
================

These methods allow you to send scalar metrics to a logger registered in Fabric.

.. code-block:: python

    # Set the logger in Fabric
    fabric = Fabric(loggers=TensorBoardLogger(...))

    # Anywhere in your training loop or model:
    fabric.log("loss", loss)

    # Or send multiple metrics at once:
    fabric.log_dict({"loss": loss, "accuracy": acc})

If no loggers are given to Fabric (default), ``log`` and ``log_dict`` won't do anything.
Here is what's happening under the hood (pseudo code) when you call ``.log()`` or ``log_dict``:

.. code-block:: python

    # When you call .log() or .log_dict(), we do this:
    for logger in fabric.loggers:
        logger.log_metrics(metrics=metrics, step=step)

See also: :doc:`../guide/logging`
