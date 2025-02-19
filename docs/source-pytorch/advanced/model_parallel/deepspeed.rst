:orphan:

.. _deepspeed_advanced:

*********
DeepSpeed
*********

`DeepSpeed <https://github.com/microsoft/DeepSpeed>`__ is a deep learning training optimization library, providing the means to train massive billion parameter models at scale.
Using the DeepSpeed strategy, we were able to **train model sizes of 10 Billion parameters and above**, with a lot of useful information in this `benchmark <https://github.com/huggingface/transformers/issues/9996>`_ and the `DeepSpeed docs <https://www.deepspeed.ai/tutorials/megatron/>`__.
DeepSpeed also offers lower level training optimizations, and efficient optimizers such as `1-bit Adam <https://www.deepspeed.ai/tutorials/onebit-adam/>`_. We recommend using DeepSpeed in environments where speed and memory optimizations are important (such as training large billion parameter models).

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

Below is a summary of all the configurations of DeepSpeed.

* :ref:`deepspeed-zero-stage-1` - **Shard optimizer states**, remains at speed parity with DDP whilst providing memory improvement

* :ref:`deepspeed-zero-stage-2` - **Shard optimizer states and gradients**, remains at speed parity with DDP whilst providing even more memory improvement

* :ref:`deepspeed-zero-stage-2-offload` - **Offload optimizer states and gradients to CPU**. Increases distributed communication volume and GPU-CPU device transfer, but provides significant memory improvement

* :ref:`deepspeed-zero-stage-3` - **Shard optimizer states, gradients, parameters and optionally activations**. Increases distributed communication volume, but provides even more memory improvement

* :ref:`deepspeed-zero-stage-3-offload` - **Offload optimizer states, gradients, parameters and optionally activations to CPU**. Increases distributed communication volume and GPU-CPU device transfer, but even more significant memory improvement.

* :ref:`deepspeed-activation-checkpointing` - **Free activations after forward pass**. Increases computation, but provides memory improvement for all stages.

To use DeepSpeed, you first need to install DeepSpeed using the commands below.

.. code-block:: bash

    pip install deepspeed

If you run into an issue with the install or later in training, ensure that the CUDA version of the PyTorch you've installed matches your locally installed CUDA (you can see which one has been recognized by running ``nvcc --version``).

.. note::

    DeepSpeed currently only supports single optimizer, single scheduler within the training loop.

    When saving a checkpoint we rely on DeepSpeed which saves a directory containing the model and various components.


.. _deepspeed-zero-stage-1:

DeepSpeed ZeRO Stage 1
======================

`DeepSpeed ZeRO Stage 1 <https://www.deepspeed.ai/tutorials/zero/#zero-overview>`_ partitions your optimizer states (Stage 1) across your GPUs to reduce memory.

It is recommended to skip Stage 1 and use Stage 2, which comes with larger memory improvements and still remains efficient. Stage 1 is useful to pair with certain optimizations such as `Torch ORT <https://github.com/pytorch/ort>`__.

.. code-block:: python

    from lightning.pytorch import Trainer

    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_1", precision=16)
    trainer.fit(model)


.. _deepspeed-zero-stage-2:

DeepSpeed ZeRO Stage 2
======================

`DeepSpeed ZeRO Stage 2 <https://www.deepspeed.ai/tutorials/zero/#zero-overview>`_ partitions your optimizer states (Stage 1) and your gradients (Stage 2) across your GPUs to reduce memory. In most cases, this is more efficient or at parity with DDP, primarily due to the optimized custom communications written by the DeepSpeed team.
As a result, benefits can also be seen on a single GPU. Do note that the default bucket sizes allocate around ``3.6GB`` of VRAM to use during distributed communications, which can be tweaked when instantiating the strategy described in a few sections below.

.. code-block:: python

    from lightning.pytorch import Trainer

    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_2", precision=16)
    trainer.fit(model)

.. code-block:: bash

    python train.py --strategy deepspeed_stage_2 --precision 16 --accelerator 'gpu' --devices 4


.. _deepspeed-zero-stage-2-offload:

DeepSpeed ZeRO Stage 2 Offload
------------------------------

Below we show an example of running `ZeRO-Offload <https://www.deepspeed.ai/tutorials/zero-offload/>`_. ZeRO-Offload leverages the host CPU to offload optimizer memory/computation, reducing the overall memory consumption.

.. code-block:: python

    from lightning.pytorch import Trainer

    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_2_offload", precision=16)
    trainer.fit(model)


This can also be done via the command line using a PyTorch Lightning script:

.. code-block:: bash

    python train.py --strategy deepspeed_stage_2_offload --precision 16 --accelerator 'gpu' --devices 4


You can also modify the ZeRO-Offload parameters via the strategy as below.

.. code-block:: python

    from lightning.pytorch import Trainer
    from lightning.pytorch.strategies import DeepSpeedStrategy

    model = MyModel()
    trainer = Trainer(
        accelerator="gpu",
        devices=4,
        strategy=DeepSpeedStrategy(offload_optimizer=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8),
        precision=16,
    )
    trainer.fit(model)


.. note::
    We suggest tuning the ``allgather_bucket_size`` parameter and ``reduce_bucket_size`` parameter to find optimum parameters based on your model size.
    These control how large a buffer we limit the model to using when reducing gradients/gathering updated parameters. Smaller values will result in less memory, but tradeoff with speed.

    DeepSpeed allocates a reduce buffer size `multiplied by 1.5x <https://github.com/microsoft/DeepSpeed/blob/fead387f7837200fefbaba3a7b14709072d8d2cb/deepspeed/runtime/zero/stage_1_and_2.py#L2188>`_ so take that into consideration when tweaking the parameters.

    The strategy sets a reasonable default of ``2e8``, which should work for most low VRAM GPUs (less than ``7GB``), allocating roughly ``3.6GB`` of VRAM as buffer. Higher VRAM GPUs should aim for values around ``5e8``.

For even more speed benefit, DeepSpeed offers an optimized CPU version of ADAM called `DeepSpeedCPUAdam <https://deepspeed.readthedocs.io/en/latest/optimizers.html#adam-cpu>`_ to run the offloaded computation, which is faster than the standard PyTorch implementation.

.. code-block:: python

    from lightning.pytorch import LightningModule, Trainer
    from deepspeed.ops.adam import DeepSpeedCPUAdam


    class MyModel(LightningModule):
        ...

        def configure_optimizers(self):
            # DeepSpeedCPUAdam provides 5x to 7x speedup over torch.optim.adam(w)
            return DeepSpeedCPUAdam(self.parameters())


    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_2_offload", precision=16)
    trainer.fit(model)


.. _deepspeed-zero-stage-3:

DeepSpeed ZeRO Stage 3
======================

DeepSpeed ZeRO Stage 3 shards the optimizer states, gradients and the model parameters (also optionally activations). Sharding model parameters and activations comes with an increase in distributed communication, however allows you to scale your models massively from one GPU to multiple GPUs.
**The DeepSpeed team report the ability to fine-tune models with over 40B parameters on a single GPU and over 2 Trillion parameters on 512 GPUs.** For more information we suggest checking the `DeepSpeed ZeRO-3 Offload documentation <https://www.deepspeed.ai/2021/03/07/zero3-offload.html>`__.

We've ran benchmarks for all these features and given a simple example of how all these features work in Lightning, which you can see at `minGPT <https://github.com/SeanNaren/minGPT/tree/stage3>`_.

To reach the highest memory efficiency or model size, you must:

1. Use the DeepSpeed strategy with the stage 3 parameter
2. Use CPU Offloading to offload weights to CPU, plus have a reasonable amount of CPU RAM to offload onto
3. Use DeepSpeed Activation Checkpointing to shard activations

Below we describe how to enable all of these to see benefit. **With all these improvements we reached 45 Billion parameters training a GPT model on 8 GPUs with ~1TB of CPU RAM available**.

Also please have a look at our :ref:`deepspeed-zero-stage-3-tips` which contains a lot of helpful information when configuring your own models.

.. note::

    When saving a model using DeepSpeed and Stage 3, model states and optimizer states will be saved in separate sharded states (based on the world size). See :ref:`deepspeed-zero-stage-3-single-file` to obtain a single checkpoint file.

.. code-block:: python

    from lightning.pytorch import Trainer
    from deepspeed.ops.adam import FusedAdam


    class MyModel(LightningModule):
        ...

        def configure_optimizers(self):
            return FusedAdam(self.parameters())


    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3", precision=16)
    trainer.fit(model)

    trainer.test()
    trainer.predict()


You can also use the Lightning Trainer to run predict or evaluate with DeepSpeed once the model has been trained.

.. code-block:: python

    from lightning.pytorch import Trainer


    class MyModel(LightningModule):
        ...


    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3", precision=16)
    trainer.test(ckpt_path="my_saved_deepspeed_checkpoint.ckpt")


Shard Model Instantly to Reduce Initialization Time/Memory
----------------------------------------------------------

When instantiating really large models, it is sometimes necessary to shard the model layers instantly.

This is the case if layers may not fit on one single machines CPU or GPU memory, but would fit once sharded across multiple machines.
We expose a hook that layers initialized within the hook will be sharded instantly on a per layer basis, allowing you to instantly shard models.

This reduces the time taken to initialize very large models, as well as ensure we do not run out of memory when instantiating larger models. For more information you can refer to the DeepSpeed docs for `Constructing Massive Models <https://deepspeed.readthedocs.io/en/latest/zero3.html>`_.

.. code-block:: python

    import torch.nn as nn
    from lightning.pytorch import Trainer
    from deepspeed.ops.adam import FusedAdam


    class MyModel(LightningModule):
        ...

        def configure_model(self):
            # Created within sharded model context, modules are instantly sharded across processes
            # as soon as they are made.
            self.block = nn.Sequential(nn.Linear(32, 32), nn.ReLU())

        def configure_optimizers(self):
            return FusedAdam(self.parameters())


    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3", precision=16)
    trainer.fit(model)

    trainer.test()
    trainer.predict()


See also: See also: :doc:`../model_init`


.. _deepspeed-zero-stage-3-offload:

DeepSpeed ZeRO Stage 3 Offload
------------------------------

DeepSpeed ZeRO Stage 3 Offloads optimizer state, gradients to the host CPU to reduce memory usage as ZeRO Stage 2 does, however additionally allows you to offload the parameters as well for even more memory saving.

.. note::

    When saving a model using DeepSpeed and Stage 3, model states and optimizer states will be saved in separate sharded states (based on the world size). See :ref:`deepspeed-zero-stage-3-single-file` to obtain a single checkpoint file.

.. code-block:: python

    from lightning.pytorch import Trainer
    from lightning.pytorch.strategies import DeepSpeedStrategy

    # Enable CPU Offloading
    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3_offload", precision=16)
    trainer.fit(model)

    # Enable CPU Offloading, and offload parameters to CPU
    model = MyModel()
    trainer = Trainer(
        accelerator="gpu",
        devices=4,
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
        ),
        precision=16,
    )
    trainer.fit(model)


DeepSpeed Infinity (NVMe Offloading)
------------------------------------

Additionally, DeepSpeed supports offloading to NVMe drives for even larger models, utilizing the large memory space found in NVMes. DeepSpeed `reports <https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/>`__ the ability to fine-tune 1 Trillion+ parameters using NVMe Offloading on one 8 GPU machine. Below shows how to enable this, assuming the NVMe drive is mounted in a directory called ``/local_nvme``.

.. code-block:: python

    from lightning.pytorch import Trainer
    from lightning.pytorch.strategies import DeepSpeedStrategy

    # Enable CPU Offloading
    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3_offload", precision=16)
    trainer.fit(model)

    # Enable CPU Offloading, and offload parameters to CPU
    model = MyModel()
    trainer = Trainer(
        accelerator="gpu",
        devices=4,
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            remote_device="nvme",
            offload_params_device="nvme",
            offload_optimizer_device="nvme",
            nvme_path="/local_nvme",
        ),
        precision=16,
    )
    trainer.fit(model)

When offloading to NVMe you may notice that the speed is slow. There are parameters that need to be tuned based on the drives that you are using. Running the `aio_bench_perf_sweep.py <https://github.com/microsoft/DeepSpeed/blob/master/csrc/aio/py_test/aio_bench_perf_sweep.py>`__ script can help you to find optimum parameters. See the `issue <https://github.com/deepspeedai/DeepSpeed/issues/998>`__ for more information on how to parse the information.

.. _deepspeed-activation-checkpointing:

DeepSpeed Activation Checkpointing
----------------------------------

Activation checkpointing frees activations from memory as soon as they are not needed during the forward pass.
They are then re-computed for the backwards pass as needed.

Activation checkpointing is very useful when you have intermediate layers that produce large activations.

This saves memory when training larger models, however requires using a checkpoint function to run modules as shown below.

.. warning::

    Ensure to not wrap the entire model with activation checkpointing. This is not the intended usage of activation checkpointing, and will lead to failures as seen in `this discussion <https://github.com/Lightning-AI/lightning/discussions/9144>`__.

.. code-block:: python

    from lightning.pytorch import Trainer
    import deepspeed


    class MyModel(LightningModule):
        ...

        def __init__(self):
            super().__init__()
            self.block_1 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
            self.block_2 = torch.nn.Linear(32, 2)

        def forward(self, x):
            # Use the DeepSpeed checkpointing function instead of calling the module directly
            # checkpointing self.block_1 means the activations are deleted after use,
            # and re-calculated during the backward passes
            x = deepspeed.checkpointing.checkpoint(self.block_1, x)
            return self.block_2(x)


.. code-block:: python

    from lightning.pytorch import Trainer
    from lightning.pytorch.strategies import DeepSpeedStrategy
    import deepspeed


    class MyModel(LightningModule):
        ...

        def configure_model(self):
            self.block_1 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
            self.block_2 = torch.nn.Linear(32, 2)

        def forward(self, x):
            # Use the DeepSpeed checkpointing function instead of calling the module directly
            x = deepspeed.checkpointing.checkpoint(self.block_1, x)
            return self.block_2(x)


    model = MyModel()

    trainer = Trainer(accelerator="gpu", devices=4, strategy="deepspeed_stage_3_offload", precision=16)

    # Enable CPU Activation Checkpointing
    trainer = Trainer(
        accelerator="gpu",
        devices=4,
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,  # Enable CPU Offloading
            cpu_checkpointing=True,  # (Optional) offload activations to CPU
        ),
        precision=16,
    )
    trainer.fit(model)


.. _deepspeed-zero-stage-3-tips:

DeepSpeed ZeRO Stage 3 Tips
---------------------------

Here is some helpful information when setting up DeepSpeed ZeRO Stage 3 with Lightning.

* If you're using Adam or AdamW, ensure to use FusedAdam or DeepSpeedCPUAdam (for CPU Offloading) rather than the default torch optimizers as they come with large speed benefits
* Treat your GPU/CPU memory as one large pool. In some cases, you may not want to offload certain things (like activations) to provide even more space to offload model parameters
* When offloading to the CPU, make sure to bump up the batch size as GPU memory will be freed
* We also support sharded checkpointing. By passing ``save_full_weights=False`` to the ``DeepSpeedStrategy``, we'll save shards of the model which allows you to save extremely large models. However to load the model and run test/validation/predict you must use the Trainer object.
* DeepSpeed provides `MiCS support <https://deepspeed.readthedocs.io/en/latest/zero3.html#deepspeed.runtime.zero.config.DeepSpeedZeroConfig.mics_shard_size>`_ which allows you to control how model parameters are sharded across GPUs. For example, with 16 GPUs, ZeRO-3 will shard the model into 16 pieces by default. Instead with ``mics_shard_size=8``, every 8 GPUs will keep a full copy of the model weights, reducing the communication overhead. You can set ``"zero_optimization":  {"stage": 3, "mics_shard_size": (shards num), ...}`` in a DeepSpeed config file to take advantage of this feature.

.. _deepspeed-zero-stage-3-single-file:

Collating Single File Checkpoint for DeepSpeed ZeRO Stage 3
-----------------------------------------------------------

After training using ZeRO Stage 3, you'll notice that your checkpoints are a directory of sharded model and optimizer states. If you'd like to collate a single file from the checkpoint directory please use the below command, which handles all the Lightning states additionally when collating the file.

.. code-block:: python

    from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

    # lightning deepspeed has saved a directory instead of a file
    save_path = "lightning_logs/version_0/checkpoints/epoch=0-step=0.ckpt/"
    output_path = "lightning_model.pt"
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)


.. warning::

    This single file checkpoint does not include the optimizer/lr-scheduler states. This means we cannot restore training via the ``trainer.fit(ckpt_path=)`` call. Ensure to keep the sharded checkpoint directory if this is required.

Custom DeepSpeed Config
=======================

In some cases you may want to define your own DeepSpeed Config, to access all parameters defined. We've exposed most of the important parameters, however, there may be debugging parameters to enable. Also, DeepSpeed allows the use of custom DeepSpeed optimizers and schedulers defined within a config file that is supported.

.. note::
    All strategy default parameters will be ignored when a config object is passed.
    All compatible arguments can be seen in the `DeepSpeed docs <https://www.deepspeed.ai/docs/config-json/>`_.

.. code-block:: python

    from lightning.pytorch import Trainer
    from lightning.pytorch.strategies import DeepSpeedStrategy

    deepspeed_config = {
        "zero_allow_untested_optimizer": True,
        "optimizer": {
            "type": "OneBitAdam",
            "params": {
                "lr": 3e-5,
                "betas": [0.998, 0.999],
                "eps": 1e-5,
                "weight_decay": 1e-9,
                "cuda_aware": True,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "last_batch_iteration": -1,
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 100,
            },
        },
        "zero_optimization": {
            "stage": 2,  # Enable Stage 2 ZeRO (Optimizer/Gradient state partitioning)
            "offload_optimizer": {"device": "cpu"},  # Enable Offloading optimizer state/calculation to the host CPU
            "contiguous_gradients": True,  # Reduce gradient fragmentation.
            "overlap_comm": True,  # Overlap reduce/backward operation of gradients for speed.
            "allgather_bucket_size": 2e8,  # Number of elements to all gather at once.
            "reduce_bucket_size": 2e8,  # Number of elements we reduce/allreduce at once.
        },
    }

    model = MyModel()
    trainer = Trainer(accelerator="gpu", devices=4, strategy=DeepSpeedStrategy(config=deepspeed_config), precision=16)
    trainer.fit(model)


We support taking the config as a json formatted file:

.. code-block:: python

    from lightning.pytorch import Trainer
    from lightning.pytorch.strategies import DeepSpeedStrategy

    model = MyModel()
    trainer = Trainer(
        accelerator="gpu", devices=4, strategy=DeepSpeedStrategy(config="/path/to/deepspeed_config.json"), precision=16
    )
    trainer.fit(model)


You can use also use an environment variable via your PyTorch Lightning script:

.. code-block:: bash

    PL_DEEPSPEED_CONFIG_PATH=/path/to/deepspeed_config.json python train.py --strategy deepspeed
