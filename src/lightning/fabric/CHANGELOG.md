# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [2.5.0] - 2024-12-19

### Added

- Added `step` parameter to `TensorBoardLogger.log_hyperparams` to visualize changes during training ([#20176](https://github.com/Lightning-AI/pytorch-lightning/pull/20176))
- Added timeout to DeepSpeedStrategy ([#20474](https://github.com/Lightning-AI/pytorch-lightning/pull/20474))
- Added FP8 + FSDP2 + torch.compile examples for Fabric ([#20440](https://github.com/Lightning-AI/pytorch-lightning/pull/20440))
- Added RTX 4080 super to chips dictionary ([#20285](https://github.com/Lightning-AI/pytorch-lightning/pull/20285))
- Added device property to lazy load functionality ([#20183](https://github.com/Lightning-AI/pytorch-lightning/pull/20183))
- Added `ddp_find_unused_parameters_true` alias in Fabric's DDPStrategy ([#20125](https://github.com/Lightning-AI/pytorch-lightning/pull/20125))

### Changed

- Changed seeding NumPy using `np.random.SeedSequence()` in `pl_worker_init_function()` to robustly seed NumPy-dependent dataloader workers ([#20369](https://github.com/Lightning-AI/pytorch-lightning/pull/20369))
- Bumped PyTorch to version `2.5` ([#20351](https://github.com/Lightning-AI/pytorch-lightning/pull/20351))
- Update BitsAndBytes version ([#20313](https://github.com/Lightning-AI/pytorch-lightning/pull/20313))

### Fixed

- Fixed use of `convert_module` in FSDP to avoid using more memory than necessary during initialization ([#20323](https://github.com/Lightning-AI/pytorch-lightning/pull/20323))

## [2.4.0] - 2024-08-06

### Added

- Made saving non-distributed checkpoints fully atomic ([#20011](https://github.com/Lightning-AI/pytorch-lightning/pull/20011))
- Added a flag `verbose` to the `seed_everything()` function ([#20108](https://github.com/Lightning-AI/pytorch-lightning/pull/20108))
- Added support for PyTorch 2.4 ([#20028](https://github.com/Lightning-AI/pytorch-lightning/pull/20028))
- Added support for Python 3.12 ([20078](https://github.com/Lightning-AI/pytorch-lightning/pull/20078))

### Changed

- Changed the implementation of how seeds are chosen for dataloader workers when using `seed_everything(..., workers=True)` ([#20055](https://github.com/Lightning-AI/pytorch-lightning/pull/20055))
- NumPy is no longer a required dependency ([#20090](https://github.com/Lightning-AI/pytorch-lightning/issues/20090))

### Removed

- Removed support for PyTorch 2.1 ([#20009](https://github.com/Lightning-AI/lightning/pull/20009))
- Removed support for Python 3.8 ([#20071](https://github.com/Lightning-AI/lightning/pull/20071))

### Fixed

- Fixed an attribute error when loading a checkpoint into a quantized model using the `_lazy_load()` function ([#20121](https://github.com/Lightning-AI/lightning/pull/20121))
- Fixed `_optimizer_to_device` logic for special 'step' key in optimizer state causing performance regression ([#20019](https://github.com/Lightning-AI/lightning/pull/20019))


## [2.3.0] - 2024-06-13

### Added

- Added sanitization for classes before logging them as hyperparameters ([#19771](https://github.com/Lightning-AI/pytorch-lightning/pull/19771))
- Added logging support for list of dicts without collapsing to a single key ([#19957](https://github.com/Lightning-AI/pytorch-lightning/issues/19957))
- Enabled consolidating distributed checkpoints through `fabric consolidate` in the new CLI ([#19560](https://github.com/Lightning-AI/pytorch-lightning/pull/19560))
- Added the ability to explicitly mark forward methods in Fabric via `_FabricModule.mark_forward_method()` ([#19690](https://github.com/Lightning-AI/pytorch-lightning/pull/19690))
- Added support for PyTorch 2.3 ([#19708](https://github.com/Lightning-AI/pytorch-lightning/pull/19708))
- Added `ModelParallelStrategy` to support 2D parallelism ([#19846](https://github.com/Lightning-AI/pytorch-lightning/pull/19846), [#19852](https://github.com/Lightning-AI/pytorch-lightning/pull/19852), [#19870](https://github.com/Lightning-AI/pytorch-lightning/pull/19870), [#19872](https://github.com/Lightning-AI/pytorch-lightning/pull/19872))
- Added a call to `torch.distributed.destroy_process_group` in atexit handler if process group needs destruction ([#19931](https://github.com/Lightning-AI/pytorch-lightning/pull/19931))
- Added support for configuring hybrid-sharding by passing a tuple for the `FSDPStrategy(device_mesh=...)` argument ([#19504](https://github.com/Lightning-AI/pytorch-lightning/pull/19504))

### Changed

- Renamed `lightning run model` to `fabric run` ([#19442](https://github.com/Lightning-AI/pytorch-lightning/pull/19442), [#19527](https://github.com/Lightning-AI/pytorch-lightning/pull/19527))
- The `Fabric.rank_zero_first` context manager now uses a barrier without timeout to avoid long-running tasks to be interrupted ([#19448](https://github.com/Lightning-AI/lightning/pull/19448))
- Fabric now raises an error if you forget to call `fabric.backward()` when it is needed by the strategy or precision selection ([#19447](https://github.com/Lightning-AI/lightning/pull/19447), [#19493](https://github.com/Lightning-AI/lightning/pull/19493))
- `_BackwardSyncControl` can now control what to do when gradient accumulation is disabled ([#19577](https://github.com/Lightning-AI/lightning/pull/19577))

### Removed

- Removed support for PyTorch 1.13 ([#19706](https://github.com/Lightning-AI/lightning/pull/19706))

### Fixed

- Fixed a matrix shape mismatch issue when running a model loaded from a quantized checkpoint (bitsandbytes) ([#19886](https://github.com/Lightning-AI/lightning/pull/19886))


## [2.2.2] - 2024-04-11

### Fixed

- Fixed an issue causing a TypeError when using `torch.compile` as a decorator ([#19627](https://github.com/Lightning-AI/pytorch-lightning/pull/19627))
- Fixed issue where some model methods couldn't be monkeypatched after being Fabric wrapped ([#19705](https://github.com/Lightning-AI/pytorch-lightning/pull/19705))
- Fixed an issue causing weights to be reset in `Fabric.setup()` when using FSDP ([#19755](https://github.com/Lightning-AI/pytorch-lightning/pull/19755))


## [2.2.1] - 2024-03-04

### Fixed

- Fixed an issue with CSVLogger trying to append to file from a previous run when the version is set manually ([#19446](https://github.com/Lightning-AI/lightning/pull/19446))


## [2.2.0] - 2024-02-08

### Added

- Added `lightning.fabric.utilities.ThroughputMonitor` and `lightning.fabric.utilities.Throughput` to track throughput and log it ([#18848](https://github.com/Lightning-AI/lightning/pull/18848))
- Added `lightning.fabric.utilities.AttributeDict` for convenient dict-attribute access to represent state in script ([#18943](https://github.com/Lightning-AI/lightning/pull/18943))
- Added support for meta-device initialization and materialization of 4-bit Bitsandbytes layers ([#19150](https://github.com/Lightning-AI/lightning/pull/19150))
- Added `TransformerEnginePrecision(fallback_compute_dtype=)` to control the dtype of operations that don't support fp8 ([#19082](https://github.com/Lightning-AI/lightning/pull/19082))
- Added support for clipping gradients by value with FSDP ([#19236](https://github.com/Lightning-AI/lightning/pull/19236))
- Added a utility function and CLI to consolidate FSDP sharded checkpoints into a single file ([#19213](https://github.com/Lightning-AI/lightning/pull/19213))
- Added support for re-compiling the model inside `Fabric.setup()` over the FSDP/DDP wrappers ([#19280](https://github.com/Lightning-AI/lightning/pull/19280))

### Changed

- `seed_everything()` without passing in a seed no longer randomly selects a seed, and now defaults to `0` ([#18846](https://github.com/Lightning-AI/lightning/pull/18846))
- Changed the `TransformerEnginePrecision(dtype=)` argument to `weights_dtype` and made it required ([#19082](https://github.com/Lightning-AI/lightning/pull/19082))
- The columns in the `metrics.csv` file produced by `CSVLogger` are now sorted alphabetically ([#19159](https://github.com/Lightning-AI/lightning/pull/19159))

### Removed

- Removed support for PyTorch 1.12 ([#19300](https://github.com/Lightning-AI/lightning/pull/19300))

### Fixed

- Fixed parsing of v100s GPUs in `get_available_flops` ([#18952](https://github.com/Lightning-AI/lightning/pull/18952))
- Fixed issue where the `precision="transformer-engine"` argument would not replace layers by default ([#19082](https://github.com/Lightning-AI/lightning/pull/19082))
- Fixed the input validation logic in `FSDPStrategy` to accept a `device_mesh` ([#19392](https://github.com/Lightning-AI/lightning/pull/19392))


## [2.1.4] - 2024-01-31

### Fixed

- Fixed an issue preventing Fabric to run on CPU when the system's CUDA driver is outdated or broken ([#19234](https://github.com/Lightning-AI/lightning/pull/19234))
- Fixed typo in kwarg in SpikeDetection ([#19282](https://github.com/Lightning-AI/lightning/pull/19282))


## [2.1.3] - 2023-12-21

### Fixed

- Avoid moving the model to device if `move_to_device=False` is passed ([#19152](https://github.com/Lightning-AI/lightning/pull/19152))
- Fixed broadcast at initialization in `MPIEnvironment` ([#19074](https://github.com/Lightning-AI/lightning/pull/19074))


## [2.1.2] - 2023-11-15

### Fixed

- Fixed precision default from environment ([#18928](https://github.com/Lightning-AI/lightning/pull/18928))


## [2.1.1] - 2023-11-06

### Changed

- Calling a method other than `forward` that invokes submodules is now an error when the model is wrapped (e.g., with DDP) ([#18819](https://github.com/Lightning-AI/lightning/pull/18819))

### Fixed

- Fixed false-positive warnings about method calls on the Fabric-wrapped module ([#18819](https://github.com/Lightning-AI/lightning/pull/18819))
- Refined the FSDP saving logic and error messaging when path exists ([#18884](https://github.com/Lightning-AI/lightning/pull/18884))
- Fixed layer conversion under `Fabric.init_module()` context manager when using the `BitsandbytesPrecision` plugin ([#18914](https://github.com/Lightning-AI/lightning/pull/18914))


## [2.1.0] - 2023-10-11

### Added

- Added support for the TPU-v4 architecture ([#17227](https://github.com/Lightning-AI/lightning/pull/17227))
- Added support for XLA's new PJRT runtime ([#17352](https://github.com/Lightning-AI/lightning/pull/17352))
- Added support for Fully Sharded Data Parallel (FSDP) training with XLA ([#18126](https://github.com/Lightning-AI/lightning/pull/18126), [#18424](https://github.com/Lightning-AI/lightning/pull/18424), [#18430](https://github.com/Lightning-AI/lightning/pull/18430))
- Check for invalid TPU device inputs ([#17227](https://github.com/Lightning-AI/lightning/pull/17227))
- Added `XLAStrategy(sync_module_states=bool)` to control whether to broadcast the parameters to all devices ([#17522](https://github.com/Lightning-AI/lightning/pull/17522))
- Added support for joint setup of model and optimizer with FSDP ([#17305](https://github.com/Lightning-AI/lightning/pull/17305))
- Added support for handling multiple parameter groups in optimizers set up with FSDP ([#17305](https://github.com/Lightning-AI/lightning/pull/17305))
- Added support for saving and loading sharded model and optimizer state with `FSDPStrategy` ([#17323](https://github.com/Lightning-AI/lightning/pull/17323))
- Added a warning when calling methods on `_FabricModule` that bypass the strategy-specific wrappers ([#17424](https://github.com/Lightning-AI/lightning/pull/17424))
- Added `Fabric.init_tensor()` context manager to instantiate tensors efficiently directly on device and dtype ([#17488](https://github.com/Lightning-AI/lightning/pull/17488))
- Added `Fabric.init_module()` context manager to instantiate large models efficiently directly on device, dtype, and with sharding support ([#17462](https://github.com/Lightning-AI/lightning/pull/17462))
  * Creates the model parameters in the desired dtype (`torch.float32`, `torch.float64`, `torch.float16`, or `torch.bfloat16`) depending on the 'true' precision choice in `Fabric(precision='32-true'|'64-true'|'16-true'|'bf16-true')`
  * Handles initialization for FSDP models before wrapping and the Zero stage 3 initialization for DeepSpeed before sharding
- Added support for empty weight initialization with `Fabric.init_module(empty_init=True)` for checkpoint loading ([#17627](https://github.com/Lightning-AI/lightning/pull/17627))
- Added support for meta-device initialization with `Fabric.init_module(empty_init=True)` in FSDP ([#18122](https://github.com/Lightning-AI/lightning/pull/18122))
- Added `lightning.fabric.plugins.Precision.module_init_context()` and `lightning.fabric.strategies.Strategy.module_init_context()` context managers to control model and tensor instantiation ([#17462](https://github.com/Lightning-AI/lightning/pull/17462))
- `lightning.fabric.strategies.Strategy.tensor_init_context()` context manager to instantiate tensors efficiently directly on device and dtype ([#17607](https://github.com/Lightning-AI/lightning/pull/17607))
- Run the DDP wrapper in a CUDA stream ([#17334](https://github.com/Lightning-AI/lightning/pull/17334))
- Added support for true half-precision as `Fabric(precision="16-true"|"bf16-true")` ([#17287](https://github.com/Lightning-AI/lightning/pull/17287))
- Added support for mixed 8-bit precision as `Fabric(precision="transformer-engine")` using [Nvidia's Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine) ([#17597](https://github.com/Lightning-AI/lightning/pull/17597))
- Added support for linear layer quantization with `Fabric(plugins=BitsandbytesPrecision())` using [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) ([#18655](https://github.com/Lightning-AI/lightning/pull/18655))
- Added error messaging for missed `.launch()` when it is required ([#17570](https://github.com/Lightning-AI/lightning/pull/17570))
- Added support for saving checkpoints with either full state-dict or sharded state dict via `FSDPStrategy(state_dict_type="full"|"sharded")` ([#17526](https://github.com/Lightning-AI/lightning/pull/17526))
- Added support for loading a full-state checkpoint file into a sharded model ([#17623](https://github.com/Lightning-AI/lightning/pull/17623))
- Added support for calling hooks on a LightningModule via `Fabric.call` ([#17874](https://github.com/Lightning-AI/lightning/pull/17874))
- Added the parameter `Fabric.load(..., strict=True|False)` to enable non-strict loading of partial checkpoint state ([#17645](https://github.com/Lightning-AI/lightning/pull/17645))
- Added the parameter `Fabric.save(..., filter=...)` to enable saving a partial checkpoint state ([#17845](https://github.com/Lightning-AI/lightning/pull/17845))
- Added support for loading optimizer states from a full-state checkpoint file ([#17747](https://github.com/Lightning-AI/lightning/pull/17747))
- Automatically call `xla_model.mark_step()` before saving checkpoints with XLA ([#17882](https://github.com/Lightning-AI/lightning/pull/17882))
- Automatically call `xla_model.mark_step()` after `optimizer.step()` with XLA ([#17883](https://github.com/Lightning-AI/lightning/pull/17883))
- Added support for all half-precision modes in FSDP precision plugin ([#17807](https://github.com/Lightning-AI/lightning/pull/17807))
- Added `FSDPStrategy(activation_checkpointing_policy=...)` to customize the layer policy for automatic activation checkpointing (requires torch>=2.1) ([#18045](https://github.com/Lightning-AI/lightning/pull/18045))
- Added a callback for spike-detection ([#18014](https://github.com/Lightning-AI/lightning/pull/18014))
- Added the ability to set the `torch.distributed.fsdp.ShardingStrategy` via string in `FSDPStrategy` ([#18087](https://github.com/Lightning-AI/lightning/pull/18087))
- Improved error messages when attempting to load a DeepSpeed checkpoint at an invalid path ([#17795](https://github.com/Lightning-AI/lightning/pull/17795))
- Added `Fabric.load_raw()` for loading raw PyTorch state dict checkpoints for model or optimizer objects ([#18049](https://github.com/Lightning-AI/lightning/pull/18049))
- Allowed accessing rank information in the main process before processes are launched when using the `XLAStrategy` ([#18194](https://github.com/Lightning-AI/lightning/pull/18194))
- Added automatic process cleanup to avoid zombie child processes and stalls when exceptions are raised ([#18218](https://github.com/Lightning-AI/lightning/pull/18218))
- Added validation of user input for `devices` and `num_nodes` when running with `SLURM` or `TorchElastic` ([#18292](https://github.com/Lightning-AI/lightning/pull/18292))
- Improved the error messaging and instructions when handling custom batch samplers in distributed settings ([#18402](https://github.com/Lightning-AI/lightning/pull/18402))
- Added support for saving and loading stateful objects other than modules and optimizers ([#18513](https://github.com/Lightning-AI/lightning/pull/18513))
- Enabled the default process group configuration for FSDP's hybrid sharding ([#18583](https://github.com/Lightning-AI/lightning/pull/18583))
- Added `lightning.fabric.utilities.suggested_max_num_workers` to assist with setting a good value in distributed settings ([#18591](https://github.com/Lightning-AI/lightning/pull/18591))
- Added `lightning.fabric.utilities.is_shared_filesystem` utility function to automatically check whether the filesystem is shared between machines ([#18586](https://github.com/Lightning-AI/lightning/pull/18586))
- Removed support for PyTorch 1.11 ([#18691](https://github.com/Lightning-AI/lightning/pull/18691))
- Added support for passing the argument `.load_state_dict(..., assign=True|False)` on Fabric-wrapped modules in PyTorch 2.1 or newer ([#18690](https://github.com/Lightning-AI/lightning/pull/18690))

### Changed

- Allow using iterable-style datasets with TPUs ([#17331](https://github.com/Lightning-AI/lightning/pull/17331))
- Increased the minimum XLA requirement to 1.13 ([#17368](https://github.com/Lightning-AI/lightning/pull/17368))
- Fabric argument validation now only raises an error if conflicting settings are set through the CLI ([#17679](https://github.com/Lightning-AI/lightning/pull/17679))
- DataLoader re-instantiation is now only performed when a distributed sampler is required ([#18191](https://github.com/Lightning-AI/lightning/pull/18191))
- Improved the formatting of emitted warnings ([#18288](https://github.com/Lightning-AI/lightning/pull/18288))
- Broadcast and reduction of tensors with XLA-based strategies now preserve the input's device ([#18275](https://github.com/Lightning-AI/lightning/pull/18275))
- Due to lack of reliability, Fabric now only runs on one GPU instead of all GPUs in a Jupyter notebook if `devices="auto"` (default) ([#18291](https://github.com/Lightning-AI/lightning/pull/18291))
- Enabled launching via `torchrun` in a SLURM environment; the `TorchElasticEnvironment` now gets chosen over the `SLURMEnvironment` if both are detected ([#18618](https://github.com/Lightning-AI/lightning/pull/18618))
- If not set by the user, Lightning will set `OMP_NUM_THREADS` to `num_cpus / num_processes` when launching subprocesses (e.g. when DDP is used) to avoid system overload for CPU-intensive tasks ([#18677](https://github.com/Lightning-AI/lightning/pull/18677))

### Deprecated

- Deprecated the `DDPStrategy.is_distributed` property. This strategy is distributed by definition ([#17381](https://github.com/Lightning-AI/lightning/pull/17381))
- Deprecated the `SingleTPUStrategy` (`strategy="single_tpu"`) in favor of `SingleDeviceXLAStrategy` (`strategy="single_xla"`) ([#17383](https://github.com/Lightning-AI/lightning/pull/17383))
- Deprecated the `TPUAccelerator` in favor of `XLAAccelerator` ([#17383](https://github.com/Lightning-AI/lightning/pull/17383))
- Deprecated the `TPUPrecision` in favor of `XLAPrecision` ([#17383](https://github.com/Lightning-AI/lightning/pull/17383))
- Deprecated the `TPUBf16Precision` in favor of `XLABf16Precision` ([#17383](https://github.com/Lightning-AI/lightning/pull/17383))

### Removed

- Removed automatic sharding support with `Fabric.run` or using `fabric.launch(fn)`. This only impacts FSDP and DeepSpeed strategy users. Please instantiate your module under the newly added `fabric.init_module` context manager ([#17832](https://github.com/Lightning-AI/lightning/pull/17832))
- Removed the unsupported `checkpoint_io` argument from the `FSDPStrategy` ([#18192](https://github.com/Lightning-AI/lightning/pull/18192))

### Fixed

- Fixed issue where running on TPUs would select the wrong device index ([#17227](https://github.com/Lightning-AI/lightning/pull/17227))
- Removed the need to call `.launch()` when using the DP-strategy (`strategy="dp"`) ([#17931](https://github.com/Lightning-AI/lightning/pull/17931))
- Fixed FSDP re-applying activation checkpointing when the user had manually applied it already ([#18006](https://github.com/Lightning-AI/lightning/pull/18006))
- Fixed FSDP re-wrapping the module root when the user had manually wrapped the model ([#18054](https://github.com/Lightning-AI/lightning/pull/18054))
- Fixed issue where unexpected exceptions would leave the default torch dtype modified when using true precision settings ([#18500](https://github.com/Lightning-AI/lightning/pull/18500))
- Fixed redundant input-type casting in FSDP precision ([#18630](https://github.com/Lightning-AI/lightning/pull/18630))
- Fixed an issue with `find_usable_cuda_devices(0)` incorrectly returning a list of devices ([#18722](https://github.com/Lightning-AI/lightning/pull/18722))
- Fixed redundant file writes in `CSVLogger` ([#18567](https://github.com/Lightning-AI/lightning/pull/18567))


## [2.0.9] - 2023-09-14

### Fixed

- Fixed an issue causing the `_FabricOptimizer.state` to remain outdated after loading with `load_state_dict` ([#18488](https://github.com/Lightning-AI/lightning/pull/18488))


## [2.0.8] - 2023-08-29

### Changed

- On XLA, avoid setting the global rank before processes have been launched as this will initialize the PJRT computation client in the main process ([#16966](https://github.com/Lightning-AI/lightning/pull/16966))

### Fixed

- Fixed model parameters getting shared between processes when running with `strategy="ddp_spawn"` and `accelerator="cpu"`; this has a necessary memory impact, as parameters are replicated for each process now ([#18238](https://github.com/Lightning-AI/lightning/pull/18238))
- Removed false positive warning when using `fabric.no_backward_sync` with XLA strategies ([#17761](https://github.com/Lightning-AI/lightning/pull/17761))
- Fixed issue where Fabric would not initialize the global rank, world size, and rank-zero-only rank after initialization and before launch ([#16966](https://github.com/Lightning-AI/lightning/pull/16966))
- Fixed FSDP full-precision `param_dtype` training (`16-mixed`, `bf16-mixed` and `32-true` configurations) to avoid FSDP assertion errors with PyTorch < 2.0 ([#18278](https://github.com/Lightning-AI/lightning/pull/18278))



## [2.0.7] - 2023-08-14

### Changed

- Disabled the auto-detection of the Kubeflow environment ([#18137](https://github.com/Lightning-AI/lightning/pull/18137))

### Fixed

- Fixed issue where DDP subprocesses that used Hydra would set hydra's working directory to current directory ([#18145](https://github.com/Lightning-AI/lightning/pull/18145))
- Fixed an issue that would prevent the user to set the multiprocessing start method after importing lightning ([#18177](https://github.com/Lightning-AI/lightning/pull/18177))
- Fixed an issue with `Fabric.all_reduce()` not performing an inplace operation for all backends consistently ([#18235](https://github.com/Lightning-AI/lightning/pull/18235))


## [2.0.6] - 2023-07-20

### Fixed

- Fixed `TensorBoardLogger.log_graph` not unwrapping the `_FabricModule` ([#17844](https://github.com/Lightning-AI/lightning/pull/17844))


## [2.0.5] - 2023-07-07

### Added

- Added validation against misconfigured device selection when using the DeepSpeed strategy ([#17952](https://github.com/Lightning-AI/lightning/pull/17952))

### Changed

- Avoid info message when loading 0 entry point callbacks ([#17990](https://github.com/Lightning-AI/lightning/pull/17990))

### Fixed

- Fixed the emission of a false-positive warning when calling a method on the Fabric-wrapped module that accepts no arguments ([#17875](https://github.com/Lightning-AI/lightning/pull/17875))
- Fixed check for FSDP's flat parameters in all parameter groups ([#17914](https://github.com/Lightning-AI/lightning/pull/17914))
- Fixed automatic step tracking in Fabric's CSVLogger ([#17942](https://github.com/Lightning-AI/lightning/pull/17942))
- Fixed an issue causing the `torch.set_float32_matmul_precision` info message to show multiple times ([#17960](https://github.com/Lightning-AI/lightning/pull/17960))
- Fixed loading model state when `Fabric.load()` is called after `Fabric.setup()` ([#17997](https://github.com/Lightning-AI/lightning/pull/17997))


## [2.0.4] - 2023-06-22

### Fixed

- Fixed validation of parameters of `plugins.precision.MixedPrecision` ([#17687](https://github.com/Lightning-AI/lightning/pull/17687))
- Fixed an issue with hpu imports leading to performance degradation  ([#17788](https://github.com/Lightning-AI/lightning/pull/17788))


- Fixed computing the next version folder in `CSVLogger` ([#17139](https://github.com/Lightning-AI/lightning/pull/17139), [#17139](https://github.com/Lightning-AI/lightning/pull/17986))


## [2.0.3] - 2023-06-07

- Added support for `Callback` registration through entry points ([#17756](https://github.com/Lightning-AI/lightning/pull/17756))

### Changed

- Made type hints public ([#17100](https://github.com/Lightning-AI/lightning/pull/17100))
- Support compiling a module after it was set up by Fabric ([#17529](https://github.com/Lightning-AI/lightning/pull/17529))

### Fixed

- Fixed computing the next version folder in `CSVLogger` ([#17139](https://github.com/Lightning-AI/lightning/pull/17139))
- Fixed inconsistent settings for FSDP Precision ([#17670](https://github.com/Lightning-AI/lightning/issues/17670))


## [2.0.2] - 2023-04-24

### Changed

- Enabled precision autocast for LightningModule step methods in Fabric ([#17439](https://github.com/Lightning-AI/lightning/pull/17439))

### Fixed

- Fixed an issue with `LightningModule.*_step` methods bypassing the DDP/FSDP wrapper ([#17424](https://github.com/Lightning-AI/lightning/pull/17424))
- Fixed device handling in `Fabric.setup()` when the model has no parameters ([#17441](https://github.com/Lightning-AI/lightning/pull/17441))


## [2.0.1] - 2023-03-30

### Changed

- Generalized `Optimizer` validation to accommodate both FSDP 1.x and 2.x ([#16733](https://github.com/Lightning-AI/lightning/pull/16733))


## [2.0.0] - 2023-03-15

### Added

- Added `Fabric.all_reduce` ([#16459](https://github.com/Lightning-AI/lightning/pull/16459))
- Added support for saving and loading DeepSpeed checkpoints through `Fabric.save/load()` ([#16452](https://github.com/Lightning-AI/lightning/pull/16452))
- Added support for automatically calling `set_epoch` on the `dataloader.batch_sampler.sampler` ([#16841](https://github.com/Lightning-AI/lightning/pull/16841))
- Added support for writing logs to remote file systems with the `CSVLogger` ([#16880](https://github.com/Lightning-AI/lightning/pull/16880))
- Added support for frozen dataclasses in the optimizer state ([#16656](https://github.com/Lightning-AI/lightning/pull/16656))
- Added `lightning.fabric.is_wrapped` to check whether a module, optimizer, or dataloader was already wrapped by Fabric ([#16953](https://github.com/Lightning-AI/lightning/pull/16953))

### Changed

- Fabric now chooses `accelerator="auto", strategy="auto", devices="auto"` as defaults ([#16842](https://github.com/Lightning-AI/lightning/pull/16842))
- Checkpoint saving and loading redesign ([#16434](https://github.com/Lightning-AI/lightning/pull/16434))
  * Changed the method signatrue of `Fabric.save` and `Fabric.load`
  * Changed the method signature of `Strategy.save_checkpoint` and `Fabric.load_checkpoint`
  * `Fabric.save` accepts a state that can contain model and optimizer references
  * `Fabric.load` can now load state in-place onto models and optimizers
  * `Fabric.load` returns a dictionary of objects that weren't loaded into the state
  * `Strategy.save_checkpoint` and `Fabric.load_checkpoint` are now responsible for accessing the state of the model and optimizers
- `DataParallelStrategy.get_module_state_dict()` and `DDPStrategy.get_module_state_dict()` now correctly extracts the state dict without keys prefixed with 'module' ([#16487](https://github.com/Lightning-AI/lightning/pull/16487))
- "Native" suffix removal ([#16490](https://github.com/Lightning-AI/lightning/pull/16490))
  * `strategy="fsdp_full_shard_offload"` is now `strategy="fsdp_cpu_offload"`
  * `lightning.fabric.plugins.precision.native_amp` is now `lightning.fabric.plugins.precision.amp`
- Enabled all shorthand strategy names that can be supported in the CLI ([#16485](https://github.com/Lightning-AI/lightning/pull/16485))
- Renamed `strategy='tpu_spawn'` to `strategy='xla'` and `strategy='tpu_spawn_debug'` to `strategy='xla_debug'` ([#16781](https://github.com/Lightning-AI/lightning/pull/16781))
- Changed arguments for precision settings (from [64|32|16|bf16] to ["64-true"|"32-true"|"16-mixed"|"bf16-mixed"]) ([#16767](https://github.com/Lightning-AI/lightning/pull/16767))
- The selection `Fabric(strategy="ddp_spawn", ...)` no longer falls back to "ddp" when a cluster environment gets detected ([#16780](https://github.com/Lightning-AI/lightning/pull/16780))
- Renamed `setup_dataloaders(replace_sampler=...)` to `setup_dataloaders(use_distributed_sampler=...)` ([#16829](https://github.com/Lightning-AI/lightning/pull/16829))

### Removed

- Removed support for PyTorch 1.10 ([#16492](https://github.com/Lightning-AI/lightning/pull/16492))
- Removed support for Python 3.7 ([#16579](https://github.com/Lightning-AI/lightning/pull/16579))

### Fixed

- Fixed issue where the wrapped dataloader `iter()` would be called twice ([#16841](https://github.com/Lightning-AI/lightning/pull/16841))

- Improved the error message for installing tensorboard or tensorboardx ([#17053](https://github.com/Lightning-AI/lightning/pull/17053))


## [1.9.4] - 2023-03-01

### Added

- Added `Fabric(strategy="auto")` support ([#16916](https://github.com/Lightning-AI/lightning/pull/16916))

### Fixed

- Fixed edge cases in parsing device ids using NVML ([#16795](https://github.com/Lightning-AI/lightning/pull/16795))
- Fixed DDP spawn hang on TPU Pods ([#16844](https://github.com/Lightning-AI/lightning/pull/16844))
- Fixed an error when passing `find_usable_cuda_devices(num_devices=-1)` ([#16866](https://github.com/Lightning-AI/lightning/pull/16866))


## [1.9.3] - 2023-02-21

### Fixed

- Fixed an issue causing a wrong environment plugin to be selected when `accelerator=tpu` and `devices > 1` ([#16806](https://github.com/Lightning-AI/lightning/pull/16806))
- Fixed parsing of defaults for `--accelerator` and `--precision` in Fabric CLI when `accelerator` and `precision` are set to non-default values in the code ([#16818](https://github.com/Lightning-AI/lightning/pull/16818))


## [1.9.2] - 2023-02-15

### Fixed

- Fixed an attribute error and improved input validation for invalid strategy types being passed to Trainer ([#16693](https://github.com/Lightning-AI/lightning/pull/16693))


## [1.9.1] - 2023-02-10

### Fixed

- Fixed error handling for `accelerator="mps"` and `ddp` strategy pairing ([#16455](https://github.com/Lightning-AI/lightning/pull/16455))
- Fixed strict availability check for `torch_xla` requirement ([#16476](https://github.com/Lightning-AI/lightning/pull/16476))
- Fixed an issue where PL would wrap DataLoaders with XLA's MpDeviceLoader more than once ([#16571](https://github.com/Lightning-AI/lightning/pull/16571))
- Fixed the batch_sampler reference for DataLoaders wrapped with XLA's MpDeviceLoader ([#16571](https://github.com/Lightning-AI/lightning/pull/16571))
- Fixed an import error when `torch.distributed` is not available ([#16658](https://github.com/Lightning-AI/lightning/pull/16658))


## [1.9.0] - 2023-01-17

### Added

- Added `Fabric.launch()` to programmatically launch processes (e.g. in Jupyter notebook) ([#14992](https://github.com/Lightning-AI/lightning/pull/14992))
- Added the option to launch Fabric scripts from the CLI, without the need to wrap the code into the `run` method ([#14992](https://github.com/Lightning-AI/lightning/pull/14992))
- Added `Fabric.setup_module()` and `Fabric.setup_optimizers()` to support strategies that need to set up the model before an optimizer can be created ([#15185](https://github.com/Lightning-AI/lightning/pull/15185))
- Added support for Fully Sharded Data Parallel (FSDP) training in Lightning Lite ([#14967](https://github.com/Lightning-AI/lightning/pull/14967))
- Added `lightning.fabric.accelerators.find_usable_cuda_devices` utility function ([#16147](https://github.com/Lightning-AI/lightning/pull/16147))
- Added basic support for LightningModules ([#16048](https://github.com/Lightning-AI/lightning/pull/16048))
- Added support for managing callbacks via `Fabric(callbacks=...)` and emitting events through `Fabric.call()` ([#16074](https://github.com/Lightning-AI/lightning/pull/16074))
- Added Logger support ([#16121](https://github.com/Lightning-AI/lightning/pull/16121))
  * Added `Fabric(loggers=...)` to support different Logger frameworks in Fabric
  * Added `Fabric.log` for logging scalars using multiple loggers
  * Added `Fabric.log_dict` for logging a dictionary of multiple metrics at once
  * Added `Fabric.loggers` and `Fabric.logger` attributes to access the individual logger instances
  * Added support for calling `self.log` and `self.log_dict` in a LightningModule when using Fabric
  * Added access to `self.logger` and `self.loggers` in a LightningModule when using Fabric
- Added `lightning.fabric.loggers.TensorBoardLogger` ([#16121](https://github.com/Lightning-AI/lightning/pull/16121))
- Added `lightning.fabric.loggers.CSVLogger` ([#16346](https://github.com/Lightning-AI/lightning/pull/16346))
- Added support for a consistent `.zero_grad(set_to_none=...)` on the wrapped optimizer regardless of which strategy is used ([#16275](https://github.com/Lightning-AI/lightning/pull/16275))

### Changed

- Renamed the class `LightningLite` to `Fabric` ([#15932](https://github.com/Lightning-AI/lightning/pull/15932), [#15938](https://github.com/Lightning-AI/lightning/pull/15938))
- The `Fabric.run()` method is no longer abstract ([#14992](https://github.com/Lightning-AI/lightning/pull/14992))
- The `XLAStrategy` now inherits from `ParallelStrategy` instead of `DDPSpawnStrategy` ([#15838](https://github.com/Lightning-AI/lightning/pull/15838))
- Merged the implementation of `DDPSpawnStrategy` into `DDPStrategy` and removed `DDPSpawnStrategy` ([#14952](https://github.com/Lightning-AI/lightning/pull/14952))
- The dataloader wrapper returned from `.setup_dataloaders()` now calls `.set_epoch()` on the distributed sampler if one is used ([#16101](https://github.com/Lightning-AI/lightning/pull/16101))
- Renamed `Strategy.reduce` to `Strategy.all_reduce` in all strategies ([#16370](https://github.com/Lightning-AI/lightning/pull/16370))
- When using multiple devices, the strategy now defaults to "ddp" instead of "ddp_spawn" when none is set ([#16388](https://github.com/Lightning-AI/lightning/pull/16388))

### Removed

- Removed support for FairScale's sharded training (`strategy='ddp_sharded'|'ddp_sharded_spawn'`). Use Fully-Sharded Data Parallel instead (`strategy='fsdp'`) ([#16329](https://github.com/Lightning-AI/lightning/pull/16329))

### Fixed

- Restored sampling parity between PyTorch and Fabric dataloaders when using the `DistributedSampler` ([#16101](https://github.com/Lightning-AI/lightning/pull/16101))
- Fixes an issue where the error message wouldn't tell the user the real value that was passed through the CLI ([#16334](https://github.com/Lightning-AI/lightning/pull/16334))


## [1.8.6] - 2022-12-21

- minor cleaning


## [1.8.5] - 2022-12-15

- minor cleaning


## [1.8.4] - 2022-12-08

### Fixed

- Fixed `shuffle=False` having no effect when using DDP/DistributedSampler ([#15931](https://github.com/Lightning-AI/lightning/pull/15931))


## [1.8.3] - 2022-11-22

### Changed

- Temporarily removed support for Hydra multi-run ([#15737](https://github.com/Lightning-AI/lightning/pull/15737))


## [1.8.2] - 2022-11-17

### Fixed

- Fixed the automatic fallback from `LightningLite(strategy="ddp_spawn", ...)` to `LightningLite(strategy="ddp", ...)` when on an LSF cluster ([#15103](https://github.com/Lightning-AI/lightning/pull/15103))


## [1.8.1] - 2022-11-10

### Fixed

- Fix an issue with the SLURM `srun` detection causing permission errors ([#15485](https://github.com/Lightning-AI/lightning/pull/15485))
- Fixed the import of `lightning_lite` causing a warning 'Redirects are currently not supported in Windows or MacOs' ([#15610](https://github.com/Lightning-AI/lightning/pull/15610))
