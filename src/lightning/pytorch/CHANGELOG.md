# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [unreleased] - YYYY-MM-DD

### Added

- Allow LightningCLI to use a customized argument parser class ([#20596](https://github.com/Lightning-AI/pytorch-lightning/pull/20596))

### Changed

### Removed

### Fixed

## [2.5.0] - 2024-12-19

### Added

- Added `step` parameter to `TensorBoardLogger.log_hyperparams` to visualize changes during training ([#20176](https://github.com/Lightning-AI/pytorch-lightning/pull/20176))
- Added `str` method to datamodule ([#20301](https://github.com/Lightning-AI/pytorch-lightning/pull/20301))
- Added timeout to DeepSpeedStrategy ([#20474](https://github.com/Lightning-AI/pytorch-lightning/pull/20474))
- Added doc for Truncated Back-Propagation Through Time ([#20422](https://github.com/Lightning-AI/pytorch-lightning/pull/20422))
- Added FP8 + FSDP2 + torch.compile examples for PyTorch Lightning ([#20440](https://github.com/Lightning-AI/pytorch-lightning/pull/20440))
- Added profiling to `Trainer.save_checkpoint` ([#20405](https://github.com/Lightning-AI/pytorch-lightning/pull/20405))
- Added after_instantiate_classes hook to CLI ([#20401](https://github.com/Lightning-AI/pytorch-lightning/pull/20401))

### Changed

- Updated checkpointing documentation to mark `resume_from_checkpoint` as deprecated ([#20477](https://github.com/Lightning-AI/pytorch-lightning/pull/20477))
- Made plugin type checks more flexible ([#20186](https://github.com/Lightning-AI/pytorch-lightning/pull/20186))
- Changed seeding NumPy using `np.random.SeedSequence()` in `pl_worker_init_function()` to robustly seed NumPy-dependent dataloader workers ([#20369](https://github.com/Lightning-AI/pytorch-lightning/pull/20369))
- Allowed callbacks to be restored not just during training ([#20403](https://github.com/Lightning-AI/pytorch-lightning/pull/20403))
- Changed LightningCLI tests to account for future fix in jsonargparse ([#20372](https://github.com/Lightning-AI/pytorch-lightning/pull/20372))
- Bumped PyTorch to version `2.5` ([#20351](https://github.com/Lightning-AI/pytorch-lightning/pull/20351))
- Decoupled checkpoint artifact path from model artifact path ([#20325](https://github.com/Lightning-AI/pytorch-lightning/pull/20325))
- Updated BitsAndBytes version ([#20313](https://github.com/Lightning-AI/pytorch-lightning/pull/20313))
- Changed merging of hparams when logging to ignore parameter names that start with an underscore `_` ([#20221](https://github.com/Lightning-AI/pytorch-lightning/pull/20221))
- Re-enabled passing `BytesIO` as path in `.to_onnx()` ([#20172](https://github.com/Lightning-AI/pytorch-lightning/pull/20172))

### Removed

- Removed `List[int]` as input type for Trainer when `accelerator="cpu"` ([#20399](https://github.com/Lightning-AI/pytorch-lightning/pull/20399))

### Fixed

- Fixed UnboundLocalError when using the predict method with return_predictions=False. ([#20484](https://github.com/Lightning-AI/pytorch-lightning/pull/20484))
- Fixed use of `convert_module` in FSDP to avoid using more memory than necessary during initialization ([#20323](https://github.com/Lightning-AI/pytorch-lightning/pull/20323))
- Fixed TypeError in `configure_optimizers` when running with `ReduceLROnPlateau` ([#20471](https://github.com/Lightning-AI/pytorch-lightning/pull/20471))
- Fixed return type in `configure_optimizers` example ([#20420](https://github.com/Lightning-AI/pytorch-lightning/pull/20420))
- Fixed in ncorrect URI prefix stripping in MLFlowLogger ([#20365](https://github.com/Lightning-AI/pytorch-lightning/pull/20365))
- Fixed shuffling behavior when using a custom sampler in data module ([#20327](https://github.com/Lightning-AI/pytorch-lightning/pull/20327))
- Ensured restarting from checkpoints leads to consistent internal counters compared to uninterrupted training ([#20379](https://github.com/Lightning-AI/pytorch-lightning/pull/20379))
- Fixed LightningCLI failing when both module and data module save hyperparameters due to conflicting internal `_class_path` parameter ([#20221](https://github.com/Lightning-AI/pytorch-lightning/pull/20221))

## [2.4.0] - 2024-08-06

### Added

- Made saving non-distributed checkpoints fully atomic ([#20011](https://github.com/Lightning-AI/pytorch-lightning/pull/20011))
- Added `dump_stats` flag to `AdvancedProfiler` ([#19703](https://github.com/Lightning-AI/pytorch-lightning/issues/19703))
- Added a flag `verbose` to the `seed_everything()` function ([#20108](https://github.com/Lightning-AI/pytorch-lightning/pull/20108))
- Added support for PyTorch 2.4 ([#20010](https://github.com/Lightning-AI/pytorch-lightning/pull/20010))
- Added support for Python 3.12 ([20078](https://github.com/Lightning-AI/pytorch-lightning/pull/20078))
- The `TQDMProgressBar` now provides an option to retain prior training epoch bars ([#19578](https://github.com/Lightning-AI/pytorch-lightning/pull/19578))
- Added the count of modules in train and eval mode to the printed `ModelSummary` table ([#20159](https://github.com/Lightning-AI/pytorch-lightning/pull/20159))

### Changed

- Triggering KeyboardInterrupt (Ctrl+C) during `.fit()`, `.evaluate()`, `.test()` or `.predict()` now terminates all processes launched by the Trainer and exits the program ([#19976](https://github.com/Lightning-AI/pytorch-lightning/pull/19976))
- Changed the implementation of how seeds are chosen for dataloader workers when using `seed_everything(..., workers=True)` ([#20055](https://github.com/Lightning-AI/pytorch-lightning/pull/20055))
- NumPy is no longer a required dependency ([#20090](https://github.com/Lightning-AI/pytorch-lightning/issues/20090))

### Removed

- Removed support for PyTorch 2.1 ([#20009](https://github.com/Lightning-AI/lightning/pull/20009))
- Removed support for Python 3.8 ([#20071](https://github.com/Lightning-AI/lightning/pull/20071))

### Fixed

- Avoid LightningCLI saving hyperparameters with `class_path` and `init_args` since this would be a breaking change ([#20068](https://github.com/Lightning-AI/pytorch-lightning/pull/20068))
- Fixed an issue that would cause too many printouts of the seed info when using `seed_everything()` ([#20108](https://github.com/Lightning-AI/pytorch-lightning/pull/20108))
- Fixed `_LoggerConnector`'s `_ResultMetric` to move all registered keys to the device of the logged value if needed ([#19814](https://github.com/Lightning-AI/pytorch-lightning/issues/19814))
- Fixed `_optimizer_to_device` logic for special 'step' key in optimizer state causing performance regression ([#20019](https://github.com/Lightning-AI/lightning/pull/20019))
- Fixed parameter counts in `ModelSummary` when model has distributed parameters (DTensor) ([#20163](https://github.com/Lightning-AI/pytorch-lightning/pull/20163))
- Fixed PyTorch Lightning FSDP takes more memory than PyTorch FSDP ([#20323](https://github.com/Lightning-AI/pytorch-lightning/pull/20323))


## [2.3.0] - 2024-06-13

### Added

- The `ModelSummary` and `RichModelSummary` callbacks now display the training mode of each layer in the column "Mode" ([#19468](https://github.com/Lightning-AI/lightning/pull/19468))
- Added `load_from_checkpoint` support for `LightningCLI` when using dependency injection ([#18105](https://github.com/Lightning-AI/lightning/pull/18105))
- Added robust timer duration parsing with an informative error message when parsing fails ([#19513](https://github.com/Lightning-AI/pytorch-lightning/pull/19513))
- Added `on_exception` hook to `LightningDataModule` ([#19601](https://github.com/Lightning-AI/pytorch-lightning/pull/19601))
- Added support for PyTorch 2.3 ([#19708](https://github.com/Lightning-AI/pytorch-lightning/pull/19708))
- Added `ModelParallelStrategy` to support 2D parallelism ([#19878](https://github.com/Lightning-AI/pytorch-lightning/pull/19878), [#19888](https://github.com/Lightning-AI/pytorch-lightning/pull/19888))
- Added a call to `torch.distributed.destroy_process_group` in atexit handler if process group needs destruction ([#19931](https://github.com/Lightning-AI/pytorch-lightning/pull/19931))
- Added support for configuring hybrid-sharding by passing a tuple for the `FSDPStrategy(device_mesh=...)` argument ([#19504](https://github.com/Lightning-AI/pytorch-lightning/pull/19504))

### Changed

- The `prepare_data()` hook in `LightningModule` and `LightningDataModule` is now subject to a barrier without timeout to avoid long-running tasks to be interrupted ([#19448](https://github.com/Lightning-AI/lightning/pull/19448))
- Relaxed the requirement for custom batch samplers to expose `drop_last` for prediction ([#19678](https://github.com/Lightning-AI/pytorch-lightning/pull/19678))
- It is no longer allowed to skip `training_step()` by returning `None` in distributed training ([#19918](https://github.com/Lightning-AI/pytorch-lightning/pull/19918))

### Removed

- Removed the Bagua integration (`Trainer(strategy="bagua")`) ([#19445](https://github.com/Lightning-AI/lightning/pull/19445))
- Removed support for PyTorch 1.13 ([#19706](https://github.com/Lightning-AI/lightning/pull/19706))

### Fixed

- Fixed a matrix shape mismatch issue when running a model loaded from a quantized checkpoint (bitsandbytes) ([#19886](https://github.com/Lightning-AI/lightning/pull/19886))
- Fixed `WandbLogger.log_hyperparameters()` raising an error if hyperparameters are not JSON serializable ([#19769](https://github.com/Lightning-AI/pytorch-lightning/pull/19769))
- Fixed an issue with the LightningCLI not being able to set the `ModelCheckpoint(save_last=...)` argument ([#19808](https://github.com/Lightning-AI/pytorch-lightning/pull/19808))
- Fixed an issue causing ValueError for certain object such as TorchMetrics when dumping hyperparameters to YAML ([#19804](https://github.com/Lightning-AI/pytorch-lightning/pull/19804))
- Fixed resetting `epoch_loop.restarting` to avoid full validation run after `LearningRateFinder` ([#19818](https://github.com/Lightning-AI/pytorch-lightning/issues/19818))


## [2.2.2] - 2024-04-11

### Fixed

- Fixed a KeyError when saving a FSDP sharded checkpoint and setting `save_weights_only=True` ([#19524](https://github.com/Lightning-AI/pytorch-lightning/pull/19524))
- Fixed an issue causing a TypeError when using `torch.compile` as a decorator ([#19627](https://github.com/Lightning-AI/pytorch-lightning/pull/19627))


## [2.2.1] - 2024-03-04


### Fixed

- Fixed an issue with CSVLogger trying to append to file from a previous run when the version is set manually ([#19446](https://github.com/Lightning-AI/lightning/pull/19446))
- Fixed the divisibility check for `Trainer.accumulate_grad_batches` and `Trainer.log_every_n_steps` in ThroughputMonitor ([#19470](https://github.com/Lightning-AI/lightning/pull/19470))
- Fixed support for Remote Stop and Remote Abort with NeptuneLogger ([#19130](https://github.com/Lightning-AI/pytorch-lightning/pull/19130))
- Fixed infinite recursion error in precision plugin graveyard ([#19542](https://github.com/Lightning-AI/pytorch-lightning/pull/19542))


## [2.2.0] - 2024-02-08

### Added

- Added `lightning.pytorch.callbacks.ThroughputMonitor` to track throughput and log it ([#18848](https://github.com/Lightning-AI/lightning/pull/18848))
- The Trainer now restores the training mode set through `.train()` or `.eval()` on a submodule-level when switching from validation to training ([#18951](https://github.com/Lightning-AI/lightning/pull/18951))
- Added support for meta-device initialization and materialization of 4-bit Bitsandbytes layers ([#19150](https://github.com/Lightning-AI/lightning/pull/19150))
- Added `TransformerEnginePrecision(fallback_compute_dtype=)` to control the dtype of operations that don't support fp8 ([#19082](https://github.com/Lightning-AI/lightning/pull/19082))
- Added the option `ModelCheckpoint(save_last='link')` to create a symbolic link for the 'last.ckpt' file ([#19191](https://github.com/Lightning-AI/lightning/pull/19191))
- Added a utility function and CLI to consolidate FSDP sharded checkpoints into a single file ([#19213](https://github.com/Lightning-AI/lightning/pull/19213))
- The TQDM progress bar now respects the env variable `TQDM_MINITERS` for setting the refresh rate ([#19381](https://github.com/Lightning-AI/lightning/pull/19381))
- Added support for saving and loading stateful training DataLoaders ([#19361](https://github.com/Lightning-AI/lightning/pull/19361))
- Added shortcut name `strategy='deepspeed_stage_1_offload'` to the strategy registry ([#19075](https://github.com/Lightning-AI/lightning/pull/19075))
- Added support for non-strict state-dict loading in Trainer via the new `LightningModule.strict_loading = True | False` attribute ([#19404](https://github.com/Lightning-AI/lightning/pull/19404))


### Changed

- `seed_everything()` without passing in a seed no longer randomly selects a seed, and now defaults to `0` ([#18846](https://github.com/Lightning-AI/lightning/pull/18846))
- The `LightningModule.on_{validation,test,predict}_model_{eval,train}` now only get called if they are overridden by the user ([#18951](https://github.com/Lightning-AI/lightning/pull/18951))
- The `Trainer.fit()` loop no longer calls `LightningModule.train()` at the start; it now preserves the user's configuration of frozen layers ([#18951](https://github.com/Lightning-AI/lightning/pull/18951))
- The `LightningModule.load_from_checkpoint()` function now calls `.configure_model()` on the model if it is overridden, to ensure all layers can be loaded from the checkpoint ([#19036](https://github.com/Lightning-AI/lightning/pull/19036))
- Restored usage of `step` parameter when logging metrics with `NeptuneLogger` ([#19126](https://github.com/Lightning-AI/pytorch-lightning/pull/19126))
- Changed the `TransformerEnginePrecision(dtype=)` argument to `weights_dtype` and made it required ([#19082](https://github.com/Lightning-AI/lightning/pull/19082))
- The columns in the `metrics.csv` file produced by `CSVLogger` are now sorted alphabetically ([#19159](https://github.com/Lightning-AI/lightning/pull/19159))
- Reverted back to creating a checkpoint copy when `ModelCheckpoint(save_last=True)` instead of creating a symbolic link ([#19191](https://github.com/Lightning-AI/lightning/pull/19191))

### Deprecated

- Deprecated all precision plugin classes under `lightning.pytorch.plugins` with the suffix `Plugin` in the name ([#18840](https://github.com/Lightning-AI/lightning/pull/18840))

### Removed

- Removed support for PyTorch 1.12 ([#19300](https://github.com/Lightning-AI/lightning/pull/19300))

### Fixed

- Fixed issue where the `precision="transformer-engine"` argument would not replace layers by default ([#19082](https://github.com/Lightning-AI/lightning/pull/19082))
- Fixed issue where layers created in `LightningModule.setup` or `LightningModule.configure_model` wouldn't get converted when using the Bitsandbytes or TransformerEngine plugins ([#19061](https://github.com/Lightning-AI/lightning/pull/19061))
- Fixed the input validation logic in `FSDPStrategy` to accept a `device_mesh` ([#19392](https://github.com/Lightning-AI/lightning/pull/19392))


## [2.1.4] - 2024-01-31

### Fixed

- Fixed `Trainer` not expanding the `default_root_dir` if it has the `~` (home) prefix ([#19179](https://github.com/Lightning-AI/lightning/pull/19179))
- Fixed warning for Dataloader if `num_workers=1` and CPU count is 1 ([#19224](https://github.com/Lightning-AI/lightning/pull/19224))
- Fixed `WandbLogger.watch()` method annotation to accept `None` for the log parameter ([#19237](https://github.com/Lightning-AI/lightning/pull/19237))
- Fixed an issue preventing the Trainer to run on CPU when the system's CUDA driver is outdated or broken ([#19234](https://github.com/Lightning-AI/lightning/pull/19234))
- Fixed an issue with the ModelCheckpoint callback not saving relative symlinks with `ModelCheckpoint(save_last="link")` ([#19303](https://github.com/Lightning-AI/lightning/pull/19303))
- Fixed issue where the `_restricted_classmethod_impl` would incorrectly raise a TypeError on inspection rather than on call ([#19332](https://github.com/Lightning-AI/lightning/pull/19332))
- Fixed exporting `__version__` in `__init__` ([#19221](https://github.com/Lightning-AI/lightning/pull/19221))


## [2.1.3] - 2023-12-21

### Changed

- `LightningCLI` no longer allows setting a normal class instance as default. A `lazy_instance` can be used instead ([#18822](https://github.com/Lightning-AI/lightning/pull/18822))

### Fixed

- Fixed checks for local file protocol due to fsspec changes in 2023.10.0 ([#19023](https://github.com/Lightning-AI/lightning/pull/19023))
- Fixed automatic detection of 'last.ckpt' files to respect the extension when filtering ([#17072](https://github.com/Lightning-AI/lightning/pull/17072))
- Fixed an issue where setting `CHECKPOINT_JOIN_CHAR` or `CHECKPOINT_EQUALS_CHAR` would only work on the `ModelCheckpoint` class but not on an instance ([#19054](https://github.com/Lightning-AI/lightning/pull/19054))
- Fixed `ModelCheckpoint` not expanding the `dirpath` if it has the `~` (home) prefix ([#19058](https://github.com/Lightning-AI/lightning/pull/19058))
- Fixed handling checkpoint dirpath suffix in NeptuneLogger ([#18863](https://github.com/Lightning-AI/lightning/pull/18863))
- Fixed an edge case where `ModelCheckpoint` would alternate between versioned and unversioned filename ([#19064](https://github.com/Lightning-AI/lightning/pull/19064))
- Fixed broadcast at initialization in `MPIEnvironment` ([#19074](https://github.com/Lightning-AI/lightning/pull/19074))
- Fixed the tensor conversion in `self.log` to respect the default dtype ([#19046](https://github.com/Lightning-AI/pytorch-lightning/issues/19046))


## [2.1.2] - 2023-11-15

### Fixed

- Fixed an issue causing permission errors on Windows when attempting to create a symlink for the "last" checkpoint ([#18942](https://github.com/Lightning-AI/pytorch-lightning/issues/18942))
- Fixed an issue where Metric instances from `torchmetrics` wouldn't get moved to the device when using FSDP ([#18954](https://github.com/Lightning-AI/pytorch-lightning/issues/18954))
- Fixed an issue preventing the user to `Trainer.save_checkpoint()` an FSDP model when `Trainer.test/validate/predict()` ran after `Trainer.fit()` ([#18992](https://github.com/Lightning-AI/pytorch-lightning/issues/18992))


## [2.1.1] - 2023-11-06

### Fixed

- Fixed an issue when replacing an existing `last.ckpt` file with a symlink ([#18793](https://github.com/Lightning-AI/lightning/pull/18793))
- Fixed an issue when `BatchSizeFinder` `steps_per_trial` parameter ends up defining how many validation batches to run during the entire training ([#18394](https://github.com/Lightning-AI/pytorch-lightning/issues/18394))
- Fixed an issue saving the `last.ckpt` file when using `ModelCheckpoint` on a remote filesystem and no logger is used ([#18867](https://github.com/Lightning-AI/pytorch-lightning/issues/18867))
- Refined the FSDP saving logic and error messaging when path exists ([#18884](https://github.com/Lightning-AI/lightning/pull/18884))
- Fixed an issue parsing the version from folders that don't include a version number in `TensorBoardLogger` and `CSVLogger` ([#18897](https://github.com/Lightning-AI/pytorch-lightning/issues/18897))


## [2.1.0] - 2023-10-11

### Added

- Added `metrics_format` attribute to `RichProgressBarTheme` class ([#18373](https://github.com/Lightning-AI/lightning/pull/18373))
- Added `CHECKPOINT_EQUALS_CHAR` attribute to `ModelCheckpoint` class ([#17999](https://github.com/Lightning-AI/lightning/pull/17999))
- Added `**summarize_kwargs` to `ModelSummary` and `RichModelSummary` callbacks ([#16788](https://github.com/Lightning-AI/lightning/pull/16788))
- Added support for the `max_size_cycle|max_size|min_size` iteration modes during evaluation ([#17163](https://github.com/Lightning-AI/lightning/pull/17163))
- Added support for the TPU-v4 architecture ([#17227](https://github.com/Lightning-AI/lightning/pull/17227))
- Added support for XLA's new PJRT runtime ([#17352](https://github.com/Lightning-AI/lightning/pull/17352))
- Check for invalid TPU device inputs ([#17227](https://github.com/Lightning-AI/lightning/pull/17227))
- Added `XLAStrategy(sync_module_states=bool)` to control whether to broadcast the parameters to all devices ([#17522](https://github.com/Lightning-AI/lightning/pull/17522))
- Added support for multiple optimizer parameter groups when using the FSDP strategy ([#17309](https://github.com/Lightning-AI/lightning/pull/17309))
- Enabled saving the full model state dict when using the `FSDPStrategy` ([#16558](https://github.com/Lightning-AI/lightning/pull/16558))
- Update `LightningDataModule.from_datasets` to support arbitrary iterables ([#17402](https://github.com/Lightning-AI/lightning/pull/17402))
- Run the DDP wrapper in a CUDA stream ([#17334](https://github.com/Lightning-AI/lightning/pull/17334))
- Added `SaveConfigCallback.save_config` to ease use cases such as saving the config to a logger ([#17475](https://github.com/Lightning-AI/lightning/pull/17475))
- Enabled optional file versioning of model checkpoints ([#17320](https://github.com/Lightning-AI/lightning/pull/17320))
- Added the process group timeout argument `FSDPStrategy(timeout=...)` for the FSDP strategy ([#17274](https://github.com/Lightning-AI/lightning/pull/17274))
- Added `FSDPStrategy(activation_checkpointing_policy=...)` to customize the layer policy for automatic activation checkpointing (requires torch>=2.1) ([#18045](https://github.com/Lightning-AI/lightning/pull/18045))
- Added CLI option `--map-to-cpu` to the checkpoint upgrade script to enable converting GPU checkpoints on a CPU-only machine ([#17527](https://github.com/Lightning-AI/lightning/pull/17527))
- Added non-layer param count to the model summary ([#17005](https://github.com/Lightning-AI/lightning/pull/17005))
- Updated `LearningRateMonitor` to log monitored values to `trainer.callback_metrics` ([#17626](https://github.com/Lightning-AI/lightning/pull/17626))
- Added `log_weight_decay` argument to `LearningRateMonitor` callback ([#18439](https://github.com/Lightning-AI/lightning/pull/18439))
- Added `Trainer.print()` to print on local rank zero only ([#17980](https://github.com/Lightning-AI/lightning/pull/17980))
- Added `Trainer.init_module()` context manager to instantiate large models efficiently directly on device, dtype ([#18004](https://github.com/Lightning-AI/lightning/pull/18004))
  * Creates the model parameters in the desired dtype (`torch.float32`, `torch.float64`) depending on the 'true' precision choice in `Trainer(precision='32-true'|'64-true')`
- Added the `LightningModule.configure_model()` hook to instantiate large models efficiently directly on device, dtype, and with sharding support ([#18004](https://github.com/Lightning-AI/lightning/pull/18004))
  * Handles initialization for FSDP models before wrapping and the Zero stage 3 initialization for DeepSpeed before sharding
- Added support for meta-device initialization with `Trainer.init_module(empty_init=True)` in FSDP ([#18385](https://github.com/Lightning-AI/lightning/pull/18385))
- Added `lightning.pytorch.plugins.PrecisionPlugin.module_init_context()` and `lightning.pytorch.strategies.Strategy.tensor_init_context()` context managers to control model and tensor instantiation ([#18004](https://github.com/Lightning-AI/lightning/pull/18004))
- Automatically call `xla_model.mark_step()` before saving checkpoints with XLA ([#17882](https://github.com/Lightning-AI/lightning/pull/17882))
- Added a callback for spike-detection ([#18014](https://github.com/Lightning-AI/lightning/pull/18014))
- Added the ability to set the `torch.distributed.fsdp.ShardingStrategy` via string in `FSDPStrategy` ([#18087](https://github.com/Lightning-AI/lightning/pull/18087))
- Improved error messages when attempting to load a DeepSpeed checkpoint at an invalid path ([#17795](https://github.com/Lightning-AI/lightning/pull/17795))
- Allowed accessing rank information in the main process before processes are launched when using the `XLAStrategy` ([#18194](https://github.com/Lightning-AI/lightning/pull/18194))
- Added support for true half-precision training via `Trainer(precision="16-true"|"bf16-true")` ([#18193](https://github.com/Lightning-AI/lightning/pull/18193), [#18217](https://github.com/Lightning-AI/lightning/pull/18217), [#18213](https://github.com/Lightning-AI/lightning/pull/18213), [#18219](https://github.com/Lightning-AI/lightning/pull/18219))
- Added automatic process cleanup to avoid zombie child processes and stalls when exceptions are raised ([#18218](https://github.com/Lightning-AI/lightning/pull/18218))
- Added validation of user input for `devices` and `num_nodes` when running with `SLURM` or `TorchElastic` ([#18292](https://github.com/Lightning-AI/lightning/pull/18292))
- Added support for saving checkpoints with either full state-dict or sharded state dict via `FSDPStrategy(state_dict_type="full"|"sharded")` ([#18364](https://github.com/Lightning-AI/lightning/pull/18364))
- Added support for loading sharded/distributed checkpoints in FSDP ([#18358](https://github.com/Lightning-AI/lightning/pull/18358))
- Made the text delimiter in the rich progress bar configurable ([#18372](https://github.com/Lightning-AI/lightning/pull/18372))
- Improved the error messaging and instructions when handling custom batch samplers in distributed settings ([#18402](https://github.com/Lightning-AI/lightning/pull/18402))
- Added support for mixed 8-bit precision as `Trainer(precision="transformer-engine")` using [Nvidia's Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine) ([#18459](https://github.com/Lightning-AI/lightning/pull/18459))
- Added support for linear layer quantization with `Trainer(plugins=BitsandbytesPrecision())` using [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) ([#18655](https://github.com/Lightning-AI/lightning/pull/18655))
- Added support for passing the process group to the `FSDPStrategy` ([#18583](https://github.com/Lightning-AI/lightning/pull/18583))
- Enabled the default process group configuration for FSDP's hybrid sharding ([#18583](https://github.com/Lightning-AI/lightning/pull/18583))
- Added `lightning.pytorch.utilities.suggested_max_num_workers` to assist with setting a good value in distributed settings ([#18591](https://github.com/Lightning-AI/lightning/pull/18591))
- Improved the `num_workers` warning to give a more accurate upper limit on the `num_workers` suggestion ([#18591](https://github.com/Lightning-AI/lightning/pull/18591))
- Added `lightning.pytorch.utilities.is_shared_filesystem` utility function to automatically check whether the filesystem is shared between machines ([#18586](https://github.com/Lightning-AI/lightning/pull/18586))
- Added support for returning an object of type `Mapping` from `LightningModule.training_step()` ([#18657](https://github.com/Lightning-AI/lightning/pull/18657))
- Added the hook `LightningModule.on_validation_model_zero_grad()` to allow overriding the behavior of zeroing the gradients before entering the validation loop ([#18710](https://github.com/Lightning-AI/lightning/pull/18710))

### Changed

- Changed default metric formatting from `round(..., 3)` to `".3f"` format string in `MetricsTextColumn` class ([#18483](https://github.com/Lightning-AI/lightning/pull/18483))
- Removed the limitation to call `self.trainer.model.parameters()` in `LightningModule.configure_optimizers()` ([#17309](https://github.com/Lightning-AI/lightning/pull/17309))
- `Trainer(accelerator="tpu", devices=[i])"` now selects the i-th TPU core (0-based, previously it was 1-based) ([#17227](https://github.com/Lightning-AI/lightning/pull/17227))
- Allow using iterable-style datasets with TPUs ([#17331](https://github.com/Lightning-AI/lightning/pull/17331))
- Increased the minimum XLA requirement to 1.13 ([#17368](https://github.com/Lightning-AI/lightning/pull/17368))
- `self.log`ed tensors are now kept in the original device to reduce unnecessary host-to-device synchronizations ([#17334](https://github.com/Lightning-AI/lightning/pull/17334))
- Made the run initialization in `WandbLogger` lazy to avoid creating artifacts when the CLI is used ([#17573](https://github.com/Lightning-AI/lightning/pull/17573))
- Simplified redirection of `*_step` methods in strategies by removing the `_LightningModuleWrapperBase` wrapper module ([#17531](https://github.com/Lightning-AI/lightning/pull/17531))
- Support kwargs input for LayerSummary ([#17709](https://github.com/Lightning-AI/lightning/pull/17709))
- Dropped support for `wandb` versions older than 0.12.0 in `WandbLogger` ([#17876](https://github.com/Lightning-AI/lightning/pull/17876))
- During `LightningModule.setup()`, the `self.device` now returns the device the module will be placed on instead of `cpu` ([#18021](https://github.com/Lightning-AI/lightning/pull/18021))
- Increased the minimum supported `wandb` version for `WandbLogger` from 0.12.0 to 0.12.10 ([#18171](https://github.com/Lightning-AI/lightning/pull/18171))
- The input tensors now get cast to the right precision type before transfer to the device ([#18264](https://github.com/Lightning-AI/lightning/pull/18264))
- Improved the formatting of emitted warnings ([#18288](https://github.com/Lightning-AI/lightning/pull/18288))
- Broadcast and reduction of tensors with XLA-based strategies now preserve the input's device ([#18275](https://github.com/Lightning-AI/lightning/pull/18275))
- The `FSDPStrategy` now loads checkpoints after the `configure_model`/`configure_sharded_model` hook ([#18358](https://github.com/Lightning-AI/lightning/pull/18358))
- The `FSDPStrategy.load_optimizer_state_dict` and `FSDPStrategy.load_model_state_dict` are a no-op now ([#18358](https://github.com/Lightning-AI/lightning/pull/18358))
- The `Trainer.num_val_batches`, `Trainer.num_test_batches` and `Trainer.num_sanity_val_batches` now return a list of sizes per dataloader instead of a single integer ([#18441](https://github.com/Lightning-AI/lightning/pull/18441))
- The `*_step(dataloader_iter)` flavor now no longer takes the `batch_idx` in the signature ([#18390](https://github.com/Lightning-AI/lightning/pull/18390))
- Calling `next(dataloader_iter)` now returns a triplet `(batch, batch_idx, dataloader_idx)` ([#18390](https://github.com/Lightning-AI/lightning/pull/18390))
- Calling `next(combined_loader)` now returns a triplet `(batch, batch_idx, dataloader_idx)` ([#18390](https://github.com/Lightning-AI/lightning/pull/18390))
- Due to lack of reliability, Trainer now only runs on one GPU instead of all GPUs in a Jupyter notebook if `devices="auto"` (default) ([#18291](https://github.com/Lightning-AI/lightning/pull/18291))
- Made the `batch_idx` argument optional in `validation_step`, `test_step` and `predict_step` to maintain consistency with `training_step` ([#18512](https://github.com/Lightning-AI/lightning/pull/18512))
- The `TQDMProgressBar` now consistently shows it/s for the speed even when the iteration time becomes larger than one second ([#18593](https://github.com/Lightning-AI/lightning/pull/18593))
- The `LightningDataModule.load_from_checkpoint` and `LightningModule.load_from_checkpoint` methods now raise an error if they are called on an instance instead of the class ([#18432](https://github.com/Lightning-AI/lightning/pull/18432))
- Enabled launching via `torchrun` in a SLURM environment; the `TorchElasticEnvironment` now gets chosen over the `SLURMEnvironment` if both are detected ([#18618](https://github.com/Lightning-AI/lightning/pull/18618))
- If not set by the user, Lightning will set `OMP_NUM_THREADS` to `num_cpus / num_processes` when launching subprocesses (e.g. when DDP is used) to avoid system overload for CPU-intensive tasks ([#18677](https://github.com/Lightning-AI/lightning/pull/18677))
- The `ModelCheckpoint` no longer deletes files under the save-top-k mechanism when resuming from a folder that is not the same as the current checkpoint folder ([#18750](https://github.com/Lightning-AI/lightning/pull/18750))
- The `ModelCheckpoint` no longer deletes the file that was passed to `Trainer.fit(ckpt_path=...)` ([#18750](https://github.com/Lightning-AI/lightning/pull/18750))
- Calling `trainer.fit()` twice now raises an error with strategies that spawn subprocesses through `multiprocessing` (ddp_spawn, xla) ([#18776](https://github.com/Lightning-AI/lightning/pull/18776))
- The `ModelCheckpoint` now saves a symbolic link if `save_last=True` and `save_top_k != 0` ([#18748](https://github.com/Lightning-AI/lightning/pull/18748))

### Deprecated

- Deprecated the `SingleTPUStrategy` (`strategy="single_tpu"`) in favor of `SingleDeviceXLAStrategy` (`strategy="single_xla"`) ([#17383](https://github.com/Lightning-AI/lightning/pull/17383))
- Deprecated the `TPUAccelerator` in favor of `XLAAccelerator` ([#17383](https://github.com/Lightning-AI/lightning/pull/17383))
- Deprecated the `TPUPrecisionPlugin` in favor of `XLAPrecisionPlugin` ([#17383](https://github.com/Lightning-AI/lightning/pull/17383))
- Deprecated the `TPUBf16PrecisionPlugin` in favor of `XLABf16PrecisionPlugin` ([#17383](https://github.com/Lightning-AI/lightning/pull/17383))
- Deprecated the `Strategy.post_training_step` method ([#17531](https://github.com/Lightning-AI/lightning/pull/17531))
- Deprecated the `LightningModule.configure_sharded_model` hook in favor of `LightningModule.configure_model` ([#18004](https://github.com/Lightning-AI/lightning/pull/18004))
- Deprecated the `LightningDoublePrecisionModule` wrapper in favor of calling `Trainer.precision_plugin.convert_input()` ([#18209](https://github.com/Lightning-AI/lightning/pull/18209))

### Removed

- Removed the `XLAStrategy.is_distributed` property. It is always True ([#17381](https://github.com/Lightning-AI/lightning/pull/17381))
- Removed the `SingleTPUStrategy.is_distributed` property. It is always False ([#17381](https://github.com/Lightning-AI/lightning/pull/17381))
- Removed experimental support for `torchdistx` due to a lack of project maintenance ([#17995](https://github.com/Lightning-AI/lightning/pull/17995))
- Removed support for PyTorch 1.11 ([#18691](https://github.com/Lightning-AI/lightning/pull/18691))

### Fixed

- Fixed an issue with reusing the same model across multiple trainer stages when using the `DeepSpeedStrategy` ([#17531](https://github.com/Lightning-AI/lightning/pull/17531))
- Fixed the saving and loading of FSDP optimizer states ([#17819](https://github.com/Lightning-AI/lightning/pull/17819))
- Fixed FSDP re-applying activation checkpointing when the user had manually applied it already ([#18006](https://github.com/Lightning-AI/lightning/pull/18006))
- Fixed issue where unexpected exceptions would leave the default torch dtype modified when using true precision settings ([#18500](https://github.com/Lightning-AI/lightning/pull/18500))
- Fixed issue where not including the `batch_idx` argument in the `training_step` would disable gradient accumulation ([#18619](https://github.com/Lightning-AI/lightning/pull/18619))
- Fixed the replacement of callbacks returned in `LightningModule.configure_callbacks` when the callback was a subclass of an existing Trainer callback ([#18508](https://github.com/Lightning-AI/lightning/pull/18508))
- Fixed `Trainer.log_dir` not returning the correct directory for the `CSVLogger` ([#18548](https://github.com/Lightning-AI/lightning/pull/18548))
- Fixed redundant input-type casting in FSDP precision ([#18630](https://github.com/Lightning-AI/lightning/pull/18630))
- Fixed numerical issues when reducing values in low precision with `self.log` ([#18686](https://github.com/Lightning-AI/lightning/pull/18686))
- Fixed an issue that would cause the gradients to be erased if validation happened in the middle of a gradient accumulation phase ([#18710](https://github.com/Lightning-AI/lightning/pull/18710))
- Fixed redundant file writes in `CSVLogger` ([#18567](https://github.com/Lightning-AI/lightning/pull/18567))
- Fixed an issue that could lead to checkpoint files being deleted accidentally when resuming training ([#18750](https://github.com/Lightning-AI/lightning/pull/18750))


## [2.0.9] - 2023-09-14

### Fixed

- Fixed an issue that wouldn't prevent the user to set the `log_model` parameter in `WandbLogger` via the LightningCLI ([#18458](https://github.com/Lightning-AI/lightning/pull/18458))
- Fixed the display of `v_num` in the progress bar when running with `Trainer(fast_dev_run=True)` ([#18491](https://github.com/Lightning-AI/lightning/pull/18491))
- Fixed `UnboundLocalError` when running with `python -O` ([#18496](https://github.com/Lightning-AI/lightning/pull/18496))
- Fixed visual glitch with the TQDM progress bar leaving the validation bar incomplete before switching back to the training display ([#18503](https://github.com/Lightning-AI/lightning/pull/18503))
- Fixed false positive warning about logging interval when running with `Trainer(fast_dev_run=True)` ([#18550](https://github.com/Lightning-AI/lightning/pull/18550))


## [2.0.8] - 2023-08-29

### Changed

- On XLA, avoid setting the global rank before processes have been launched as this will initialize the PJRT computation client in the main process ([#16966](https://github.com/Lightning-AI/lightning/pull/16966))
- Fix inefficiency in rich progress bar ([#18369](https://github.com/Lightning-AI/lightning/pull/18369))

### Fixed

- Fixed FSDP full-precision `param_dtype` training (`16-mixed` and `bf16-mixed` configurations) to avoid FSDP assertion errors with PyTorch < 2.0 ([#18278](https://github.com/Lightning-AI/lightning/pull/18278))
- Fixed an issue that prevented the use of custom logger classes without an `experiment` property defined ([#18093](https://github.com/Lightning-AI/lightning/pull/18093))
- Fixed setting the tracking uri in `MLFlowLogger` for logging artifacts to the MLFlow server ([#18395](https://github.com/Lightning-AI/lightning/pull/18395))
- Fixed redundant `iter()` call to dataloader when checking dataloading configuration ([#18415](https://github.com/Lightning-AI/lightning/pull/18415))
- Fixed model parameters getting shared between processes when running with `strategy="ddp_spawn"` and `accelerator="cpu"`; this has a necessary memory impact, as parameters are replicated for each process now ([#18238](https://github.com/Lightning-AI/lightning/pull/18238))
- Properly manage `fetcher.done` with `dataloader_iter` ([#18376](https://github.com/Lightning-AI/lightning/pull/18376))


## [2.0.7] - 2023-08-14

### Added

- Added `LightningOptimizer.refresh()` to update the `__dict__` in case the optimizer it wraps has changed its internal state ([#18280](https://github.com/Lightning-AI/lightning/pull/18280))

### Changed

- Disabled the auto-detection of the Kubeflow environment ([#18137](https://github.com/Lightning-AI/lightning/pull/18137))

### Fixed

- Fixed a `Missing folder` exception when using a Google Storage URL as a `default_root_dir` ([#18088](https://github.com/Lightning-AI/lightning/pull/18088))
- Fixed an issue that would prevent the user to set the multiprocessing start method after importing lightning ([#18177](https://github.com/Lightning-AI/lightning/pull/18177))
- Fixed the gradient unscaling logic if the training step skipped backward (by returning `None`) ([#18267](https://github.com/Lightning-AI/lightning/pull/18267))
- Ensure that the closure running inside the optimizer step has gradients enabled, even if the optimizer step has it disabled ([#18268](https://github.com/Lightning-AI/lightning/pull/18268))
- Fixed an issue that could cause the `LightningOptimizer` wrapper returned by `LightningModule.optimizers()` have different internal state than the optimizer it wraps ([#18280](https://github.com/Lightning-AI/lightning/pull/18280))


## [2.0.6] - 2023-07-20

### Fixed

- `LightningCLI` not saving correctly `seed_everything` when `run=True` and `seed_everything=True` ([#18056](https://github.com/Lightning-AI/lightning/pull/18056))
- Fixed validation of non-PyTorch LR schedulers in manual optimization mode ([#18092](https://github.com/Lightning-AI/lightning/pull/18092))
- Fixed an attribute error for `_FaultTolerantMode` when loading an old checkpoint that pickled the enum ([#18094](https://github.com/Lightning-AI/lightning/pull/18094))


## [2.0.5] - 2023-07-07

### Fixed

- Fixed delayed creation of experiment metadata and checkpoint/log dir name when using `WandbLogger` ([#17818](https://github.com/Lightning-AI/lightning/pull/17818))
- Fixed incorrect parsing of arguments when augmenting exception messages in DDP ([#17948](https://github.com/Lightning-AI/lightning/pull/17948))
- Fixed an issue causing the `torch.set_float32_matmul_precision` info message to show multiple times ([#17960](https://github.com/Lightning-AI/lightning/pull/17960))
- Added missing `map_location` argument for the `LightningDataModule.load_from_checkpoint` function ([#17950](https://github.com/Lightning-AI/lightning/pull/17950))
- Fix support for `neptune-client` ([#17939](https://github.com/Lightning-AI/lightning/pull/17939))


## [2.0.4] - 2023-06-22

- Added validation against misconfigured device selection when using the DeepSpeed strategy ([#17952](https://github.com/Lightning-AI/lightning/pull/17952))

### Changed

- Changes to the `NeptuneLogger` ([#16761](https://github.com/Lightning-AI/lightning/pull/16761)):
  * It now supports neptune-client 0.16.16 and neptune >=1.0, and we have replaced the `log()` method with `append()` and `extend()`.
  * It now accepts a namespace `Handler` as an alternative to `Run` for the `run` argument. This means that you can call it like `NeptuneLogger(run=run["some/namespace"])` to log everything to the `some/namespace/` location of the run.

### Fixed

- Fixed validation of parameters of `plugins.precision.MixedPrecisionPlugin` ([#17687](https://github.com/Lightning-AI/lightning/pull/17687))
- Fixed deriving default map location in `LightningModule.load_from_checkpoint` when there is extra state ([#17812](https://github.com/Lightning-AI/lightning/pull/17812))


## [2.0.3] - 2023-06-07

### Changed

- Made type hints public ([#17100](https://github.com/Lightning-AI/lightning/pull/17100))


### Fixed

- `CombinedLoader` only starts DataLoader workers when necessary when operating in sequential mode ([#17639](https://github.com/Lightning-AI/lightning/pull/17639))
- Fixed a potential bug with uploading model checkpoints to Neptune.ai by uploading files from stream ([#17430](https://github.com/Lightning-AI/lightning/pull/17430))
- Fixed signature inspection of decorated hooks ([#17507](https://github.com/Lightning-AI/lightning/pull/17507))
- The `WandbLogger` no longer flattens dictionaries in the hyperparameters logged to the dashboard ([#17574](https://github.com/Lightning-AI/lightning/pull/17574))
- Fixed computing the next version folder in `CSVLogger` ([#17139](https://github.com/Lightning-AI/lightning/pull/17139))
- Fixed a formatting issue when the filename in `ModelCheckpoint` contained metrics that were substrings of each other ([#17610](https://github.com/Lightning-AI/lightning/pull/17610))
- Fixed `WandbLogger` ignoring the `WANDB_PROJECT` environment variable ([#16222](https://github.com/Lightning-AI/lightning/pull/16222))
- Fixed inconsistent settings for FSDP Precision ([#17670](https://github.com/Lightning-AI/lightning/pull/17670))
- Fixed an edge case causing overlapping samples in DDP when no global seed is set ([#17713](https://github.com/Lightning-AI/lightning/pull/17713))
- Fallback to module available check for mlflow ([#17467](https://github.com/Lightning-AI/lightning/pull/17467))
- Fixed LR finder max val batches ([#17636](https://github.com/Lightning-AI/lightning/pull/17636))
- Fixed multithreading checkpoint loading ([#17678](https://github.com/Lightning-AI/lightning/pull/17678))


## [2.0.2] - 2023-04-24

### Fixed

- Fixed issue where `Model.load_from_checkpoint("checkpoint.ckpt", map_location=map_location)` would always return model on CPU ([#17308](https://github.com/Lightning-AI/lightning/pull/17308))
- Fixed Sync module states during non-fit ([#17370](https://github.com/Lightning-AI/lightning/pull/17370))
- Fixed an issue that caused `num_nodes` not to be set correctly for `FSDPStrategy` ([#17438](https://github.com/Lightning-AI/lightning/pull/17438))


## [2.0.1] - 2023-03-30

### Changed

- Pickling the `LightningModule` no longer pickles the `Trainer` ([#17133](https://github.com/Lightning-AI/lightning/pull/17133))
- Generalized `Optimizer` validation to accommodate both FSDP 1.x and 2.x ([#16733](https://github.com/Lightning-AI/lightning/pull/16733))
- Disable `torch.inference_mode` with `torch.compile` in PyTorch 2.0 ([#17215](https://github.com/Lightning-AI/lightning/pull/17215))

### Fixed

- Fixed issue where pickling the module instance would fail with a DataLoader error ([#17130](https://github.com/Lightning-AI/lightning/pull/17130))
- Fixed WandbLogger not showing "best" aliases for model checkpoints when `ModelCheckpoint(save_top_k>0)` is used ([#17121](https://github.com/Lightning-AI/lightning/pull/17121))
- Fixed the availability check for `rich` that prevented Lightning to be imported in Google Colab ([#17156](https://github.com/Lightning-AI/lightning/pull/17156))
- Fixed parsing the precision config for inference in `DeepSpeedStrategy` ([#16973](https://github.com/Lightning-AI/lightning/pull/16973))
- Fixed issue where `torch.compile` would fail when logging to WandB ([#17216](https://github.com/Lightning-AI/lightning/pull/17216))
- Changed the `is_picklable` util function to handle the edge case that throws a `TypeError` ([#17270](https://github.com/Lightning-AI/lightning/pull/17270))


## [2.0.0] - 2023-03-15

### Added

- Added migration logic to warn about checkpoints with apex AMP state ([#16161](https://github.com/Lightning-AI/lightning/pull/16161))
- Added the `Trainer.ckpt_path = ...` setter to statefully set the checkpoint path to load. This can act as a replacement for the removed `Trainer(resume_from_checkpoint=...)` flag ([#16187](https://github.com/Lightning-AI/lightning/pull/16187))
- Added an argument `include_cuda` in `pl.utilities.seed.isolate_rng` to disable managing `torch.cuda`'s rng ([#16423](https://github.com/Lightning-AI/lightning/pull/16423))
- Added `Tuner.lr_find(attr_name=...)` to specify custom learning rate attribute names ([#16462](https://github.com/Lightning-AI/lightning/pull/16462))
- Added an `OnExceptionCheckpoint` callback to save a checkpoint on exception ([#16512](https://github.com/Lightning-AI/lightning/pull/16512))
- Added support for running the `MLFlowLogger` with the `mlflow-skinny` package ([16513](https://github.com/Lightning-AI/lightning/pull/16513))
- Added a `Trainer.received_sigterm` property to check whether a SIGTERM signal was received ([#16501](https://github.com/Lightning-AI/lightning/pull/16501))
- Added support for cascading a SIGTERM signal to launched processes after the launching process (rank 0) receives it ([#16525](https://github.com/Lightning-AI/lightning/pull/16525))
- Added a `kill` method to launchers to kill all launched processes ([#16525](https://github.com/Lightning-AI/lightning/pull/16525))
- Added suffix option to DDP strategy names to enable `find_unused_parameters=True`, for example `strategy="ddp_find_unused_parameters_true"` ([#16611](https://github.com/Lightning-AI/lightning/pull/16611))
- Added a new method `Strategy.on_exception` to the strategy base interface ([#16646](https://github.com/Lightning-AI/lightning/pull/16646))
- Added support for `predict_step(dataloader_iter, batch_index)` ([#16726](https://github.com/Lightning-AI/lightning/pull/16726))
- Added support for arbitrary iterables as dataloaders ([#16726](https://github.com/Lightning-AI/lightning/pull/16726))
- Added "sequential" mode support to `CombinedLoader` to consume multiple iterables in sequence ([#16743](https://github.com/Lightning-AI/lightning/pull/16743), [#16784](https://github.com/Lightning-AI/lightning/pull/16784))
- Added "max_size" mode support to `CombinedLoader` to consume multiple iterables entirely without cycling ([#16939](https://github.com/Lightning-AI/lightning/pull/16939)
- Added a `Trainer(barebones=True)` argument where all features that may impact raw speed are disabled ([#16854](https://github.com/Lightning-AI/lightning/pull/16854))
- Added support for writing logs remote file systems on `CSVLoggers`. ([#16880](https://github.com/Lightning-AI/lightning/pull/16880))
- Added `DDPStrategy(start_method=...)` argument, defaulting to 'popen' ([#16809](https://github.com/Lightning-AI/lightning/pull/16809))
- Added checks for whether the iterables used by the loops are valid ([#17007](https://github.com/Lightning-AI/lightning/pull/17007))

### Changed

- The Trainer's signal handlers are now registered for `trainer.{validate,test,predict}` ([#17017](https://github.com/Lightning-AI/lightning/pull/17017))
- Renamed `ProgressBarBase` to `ProgressBar` ([#17058](https://github.com/Lightning-AI/lightning/pull/17058))
- The `Trainer` now chooses `accelerator="auto", strategy="auto", devices="auto"` as defaults ([#16847](https://github.com/Lightning-AI/lightning/pull/16847))
- "Native" suffix removal ([#16490](https://github.com/Lightning-AI/lightning/pull/16490))
 * `strategy="fsdp_native"` is now `strategy="fsdp"`
 * `strategy="fsdp_native_full_shard_offload"` is now `strategy="fsdp_cpu_offload"`
 * `pl.strategies.fully_sharded_native.DDPFullyShardedNativeStrategy` is now `pl.strategies.fsdp.FSDPStrategy`
 * `pl.plugins.precision.fsdp_native_native_amp.FullyShardedNativeNativeMixedPrecisionPlugin` is now `pl.plugins.precision.fsdp.FSDPMixedPrecisionPlugin`
 * `pl.plugins.precision.native_amp` is now `pl.plugins.precision.amp`
 * `NativeSyncBatchNorm` is now `TorchSyncBatchNorm`
- Changed the default of `LearningRateFinder(update_attr=...)` and `Tuner.lr_find(update_attr=...)` to `True` ([#16462](https://github.com/Lightning-AI/lightning/pull/16462))
- Renamed the `pl.utilities.exceptions.GracefulExitException` to `SIGTERMException` ([#16501](https://github.com/Lightning-AI/lightning/pull/16501))
- The `Callback.on_train_epoch_end` hook now runs after the `LightningModule.on_train_epoch_end` hook for instances of `EarlyStopping` and `Checkpoint` callbacks ([#16567](https://github.com/Lightning-AI/lightning/pull/16567))
- The `LightningModule.{un}toggle_optimizer` methods no longer accept a `optimizer_idx` argument to select the relevant optimizer. Instead, the optimizer object can be passed in directly ([#16560](https://github.com/Lightning-AI/lightning/pull/16560))
- Manual optimization is now required for working with multiple optimizers ([#16539](https://github.com/Lightning-AI/lightning/pull/16539))
- DDP's `find_unused_parameters` now defaults to `False` ([#16611](https://github.com/Lightning-AI/lightning/pull/16611))
- The strategy selected by `accelerator="hpu"` now defaults to `find_unused_parameters=False` ([#16611](https://github.com/Lightning-AI/lightning/pull/16611))
- The main progress bar displayed during training no longer includes the combined progress for validation ([#16695](https://github.com/Lightning-AI/lightning/pull/16695))
- Renamed `TQDMProgressBar.main_progress_bar` to `TQDMProgressBar.train_progress_bar` ([#16695](https://github.com/Lightning-AI/lightning/pull/16695))
- Marked the progress tracking classes as protected ([#17009](https://github.com/Lightning-AI/lightning/pull/17009))
- Marked the `lightning.pytorch.trainer.configuration_validator.verify_loop_configurations` function as protected ([#17009](https://github.com/Lightning-AI/lightning/pull/17009))
- Marked the `lightning.pytorch.utiltiies.distributed.register_ddp_comm_hook` function as protected ([#17009](https://github.com/Lightning-AI/lightning/pull/17009))
- Marked `lightning.pytorch.utilities.supporters.CombinedDataset` as protected ([#16714](https://github.com/Lightning-AI/lightning/pull/16714))
- Marked the `{Accelerator,Signal,Callback,Checkpoint,Data,Logger}Connector` classes as protected ([#17008](https://github.com/Lightning-AI/lightning/pull/17008))
- Marked the `lightning.pytorch.trainer.connectors.signal_connector.HandlersCompose` class as protected ([#17008](https://github.com/Lightning-AI/lightning/pull/17008))
- Disabled strict loading in multiprocessing launcher ("ddp_spawn", etc.) when loading weights back into the main process ([#16365](https://github.com/Lightning-AI/lightning/pull/16365))
- Renamed `CombinedLoader.loaders` to `CombinedLoader.iterables` ([#16743](https://github.com/Lightning-AI/lightning/pull/16743))
- Renamed `Trainer(replace_sampler_ddp=...)` to `Trainer(use_distributed_sampler=...)` ([#16829](https://github.com/Lightning-AI/lightning/pull/16829))
- Moved the `CombinedLoader` class from `lightning.pytorch.trainer.supporters` to `lightning.pytorch.combined_loader` ([#16819](https://github.com/Lightning-AI/lightning/pull/16819))
- The top-level loops now own the data sources and combined dataloaders ([#16726](https://github.com/Lightning-AI/lightning/pull/16726))
- The `trainer.*_dataloader` properties now return what the user returned in their `LightningModule.*_dataloader()` hook ([#16726](https://github.com/Lightning-AI/lightning/pull/16726), [#16800](https://github.com/Lightning-AI/lightning/pull/16800))
- The `dataloader_idx` argument is now optional for the `on_{validation,test,predict}_batch_{start,end}` hooks. Remove it or default it to 0 if you don't use multiple dataloaders ([#16753](https://github.com/Lightning-AI/lightning/pull/16753))
- Renamed `TPUSpawnStrategy` to `XLAStrategy` ([#16781](https://github.com/Lightning-AI/lightning/pull/16781))
- Renamed `strategy='tpu_spawn'` to `strategy='xla'` and `strategy='tpu_spawn_debug'` to `strategy='xla_debug'` ([#16781](https://github.com/Lightning-AI/lightning/pull/16781))
- Changed arguments for precision settings (from [64|32|16|bf16] to ["64-true"|"32-true"|"16-mixed"|"bf16-mixed"]) ([#16783](https://github.com/Lightning-AI/lightning/pull/16783))
- When using multiple devices, the strategy now defaults to "ddp" instead of "ddp_spawn" when none is set ([#16780](https://github.com/Lightning-AI/lightning/pull/16780))
- The selection `Trainer(strategy="ddp_spawn", ...)` no longer falls back to "ddp" when a cluster environment gets detected ([#16780](https://github.com/Lightning-AI/lightning/pull/16780))
- Predict's custom BatchSampler that tracks the batch indices no longer consumes the entire batch sampler at the beginning ([#16826](https://github.com/Lightning-AI/lightning/pull/16826))
- Gradient norm tracking with `track_grad_norm` no longer rounds the norms to 4 digits, but instead logs them at full resolution ([#16877](https://github.com/Lightning-AI/lightning/pull/16877))
- Merged the `DDPSpawnStrategy` into `DDPStrategy` ([#16809](https://github.com/Lightning-AI/lightning/pull/16809))
- The `NeptuneLogger` now requires `neptune>=1.0.0` ([#16888](https://github.com/Lightning-AI/lightning/pull/16888))
- Changed minimum supported version of `rich` from `10.14.0` to `12.13.0` ([#16798](https://github.com/Lightning-AI/lightning/pull/16798))
- Removed the `lightning.pytorch.overrides.torch_distributed.broadcast_object_list` function ([#17011](https://github.com/Lightning-AI/lightning/pull/17011))
- The `ServableModule` is now an abstract interface ([#17000](https://github.com/Lightning-AI/lightning/pull/17000))
- The `psutil` package is now required for CPU monitoring ([#17010](https://github.com/Lightning-AI/lightning/pull/17010))
- The Trainer no longer accepts positional arguments to ([#17022](https://github.com/Lightning-AI/lightning/pull/17022))

### Removed

- Removed support for PyTorch 1.10 ([#16492](https://github.com/Lightning-AI/lightning/pull/16492))
- Removed support for Python 3.7 ([#16579](https://github.com/Lightning-AI/lightning/pull/16579))
- Removed the `pl.lite` module in favor of `lightning_fabric` ([#15953](https://github.com/Lightning-AI/lightning/pull/15953))
- `nvidia/apex` removal ([#16149](https://github.com/Lightning-AI/lightning/pull/16149))
  * Removed `pl.plugins.NativeMixedPrecisionPlugin` in favor of `pl.plugins.MixedPrecisionPlugin`
  * Removed the `LightningModule.optimizer_step(using_native_amp=...)` argument
  * Removed the `Trainer(amp_backend=...)` argument
  * Removed the `Trainer.amp_backend` property
  * Removed the `Trainer(amp_level=...)` argument
  * Removed the `pl.plugins.ApexMixedPrecisionPlugin` class
  * Removed the `pl.utilities.enums.AMPType` enum
  * Removed the `DeepSpeedPrecisionPlugin(amp_type=..., amp_level=...)` arguments
- Removed `Trainer(strategy='horovod')` support ([#16150](https://github.com/Lightning-AI/lightning/pull/16150))
- `FairScale` removal (in favor of PyTorch's FSDP implementation) ([#16400](https://github.com/Lightning-AI/lightning/pull/16400))
  * Removed the `pl.overrides.fairscale.LightningShardedDataParallel` class
  * Removed the `pl.plugins.precision.fully_sharded_native_amp.FullyShardedNativeMixedPrecisionPlugin` class
  * Removed the `pl.plugins.precision.sharded_native_amp.ShardedNativeMixedPrecisionPlugin` class
  * Removed the `pl.strategies.fully_sharded.DDPFullyShardedStrategy` (fsdp) class
  * Removed the `pl.strategies.sharded.DDPShardedStrategy` (ddp_sharded) class
  * Removed the `pl.strategies.sharded_spawn.DDPSpawnShardedStrategy` (ddp_sharded_spawn) class
- Removed legacy device arguments in Trainer ([#16171](https://github.com/Lightning-AI/lightning/pull/16171))
  * Removed the `Trainer(gpus=...)` argument
  * Removed the `Trainer(tpu_cores=...)` argument
  * Removed the `Trainer(ipus=...)` argument
  * Removed the `Trainer(num_processes=...)` argument
- Removed the deprecated `pl.utilities.AllGatherGrad` class ([#16360](https://github.com/Lightning-AI/lightning/pull/16360))
- Removed the deprecated `resume_from_checkpoint` Trainer argument ([#16167](https://github.com/Lightning-AI/lightning/pull/16167))
- Removed the deprecated `pl.profiler` module ([#16359](https://github.com/Lightning-AI/lightning/pull/16359))
- Removed deadlock detection / process reconciliation (`PL_RECONCILE_PROCESS=1`) ([#16204](https://github.com/Lightning-AI/lightning/pull/16204))
- Removed the `{training,validation,test}_epoch_end` hooks which would retain step outputs in memory. Alternative implementations are suggested by implementing their `on_*_epoch_end` hooks instead ([#16520](https://github.com/Lightning-AI/lightning/pull/16520))
- Removed the `outputs` argument from the `on_predict_epoch_end` hook. You can access them via `trainer.predict_loop.predictions` ([#16655](https://github.com/Lightning-AI/lightning/pull/16655))
- Removed support for the experimental `PL_FAULT_TOLERANT_TRAINING` environment flag ([#16516](https://github.com/Lightning-AI/lightning/pull/16516), [#16533](https://github.com/Lightning-AI/lightning/pull/16533))
- Removed the deprecated `LightningCLI` arguments ([#16380](https://github.com/Lightning-AI/lightning/pull/16380))
  * `save_config_filename`
  * `save_config_overwrite`
  * `save_config_multifile`
  * `description`
  * `env_prefix`
  * `env_parse`
- Removed the deprecated `pl.strategies.utils.on_colab_kaggle` function ([#16437](https://github.com/Lightning-AI/lightning/pull/16437))
- Removed the deprecated code in:
  * `pl.core.mixins` ([#16424](https://github.com/Lightning-AI/lightning/pull/16424))
  * `pl.utilities.distributed` ([#16390](https://github.com/Lightning-AI/lightning/pull/16390))
  * `pl.utilities.apply_func` ([#16413](https://github.com/Lightning-AI/lightning/pull/16413))
  * `pl.utilities.xla_device` ([#16404](https://github.com/Lightning-AI/lightning/pull/16404))
  * `pl.utilities.data` ([#16440](https://github.com/Lightning-AI/lightning/pull/16440))
  * `pl.utilities.device_parser` ([#16412](https://github.com/Lightning-AI/lightning/pull/16412))
  * `pl.utilities.optimizer` ([#16439](https://github.com/Lightning-AI/lightning/pull/16439))
  * `pl.utilities.seed` ([#16422](https://github.com/Lightning-AI/lightning/pull/16422))
  * `pl.utilities.cloud_io` ([#16438](https://github.com/Lightning-AI/lightning/pull/16438))
- Removed the deprecated `Accelerator.setup_environment` method ([#16436](https://github.com/Lightning-AI/lightning/pull/16436))
- Mark the `forward_module` argument as required ([#16386](https://github.com/Lightning-AI/lightning/pull/16386))
  * Removed the deprecated `pl_module` argument from the distributed module wrappers
  * Removed the deprecated `pl.overrides.base.unwrap_lightning_module` function
  * Removed the `pl.overrides.distributed.LightningDistributedModule` class
  * Removed the deprecated `pl.overrides.fairscale.unwrap_lightning_module_sharded` function
  * Removed the `pl.overrides.fairscale.LightningDistributedModule` class
- Removed the deprecated automatic GPU selection ([#16184](https://github.com/Lightning-AI/lightning/pull/16184))
  * Removed the `Trainer(auto_select_gpus=...)` argument
  * Removed the `pl.tuner.auto_gpu_select.{pick_single_gpu,pick_multiple_gpus}` functions
- Removed support for loop customization
  * Removed `Loop.replace()` ([#16361](https://github.com/Lightning-AI/lightning/pull/16361))
  * Removed `Loop.connect()` ([#16384](https://github.com/Lightning-AI/lightning/pull/16384))
  * Removed the `trainer.{fit,validate,test,predict}_loop` properties ([#16384](https://github.com/Lightning-AI/lightning/pull/16384))
  * Removed the default `Loop.run()` implementation ([#16384](https://github.com/Lightning-AI/lightning/pull/16384))
  * The loop classes are now marked as protected ([#16445](https://github.com/Lightning-AI/lightning/pull/16445))
  * The fetching classes are now marked as protected ([#16664](https://github.com/Lightning-AI/lightning/pull/16664))
- The `lightning.pytorch.overrides.distributed.IndexBatchSamplerWrapper` class is now marked as protected ([#16826](https://github.com/Lightning-AI/lightning/pull/16826))
- Removed the `DataLoaderLoop`, `EvaluationEpochLoop`, and `PredictionEpochLoop` classes ([#16726](https://github.com/Lightning-AI/lightning/pull/16726))
- Removed `trainer.reset_*_dataloader()` methods in favor of `Loop.setup_data()` for the top-level loops ([#16726](https://github.com/Lightning-AI/lightning/pull/16726))
- Removed special support for truncated backpropagation through time (TBPTT) ([#16172](https://github.com/Lightning-AI/lightning/pull/16172))
  * Removed the `LightningModule.truncated_bptt_steps` attribute
  * Removed the `LightningModule.tbptt_split_batch` hook
  * The `LightningModule.training_step` no longer accepts a `hiddens` argument
  * Removed the `pl.loops.batch.TrainingBatchLoop`
  * Removed the `FitLoop.split_idx` property
  * Removed the `LoggerConnector.on_train_split_start` method
- Removed the experimental `PL_INTER_BATCH_PARALLELISM` environment flag ([#16355](https://github.com/Lightning-AI/lightning/pull/16355))
- Removed the `Trainer(move_metrics_to_cpu=True)` argument ([#16358](https://github.com/Lightning-AI/lightning/pull/16358))
- Removed the `LightningModule.precision` attribute ([#16203](https://github.com/Lightning-AI/lightning/pull/16203))
- Removed the automatic addition of a moving average of the `training_step` loss in the progress bar. Use `self.log("loss", ..., prog_bar=True)` instead. ([#16192](https://github.com/Lightning-AI/lightning/pull/16192))
- Removed support for passing a dictionary value to `self.log()` ([#16389](https://github.com/Lightning-AI/lightning/pull/16389))
- Removed `Trainer.model` setter ([#16462](https://github.com/Lightning-AI/lightning/pull/16462))
- Removed the argument `Trainer(multiple_trainloader_mode=...)`. You can use `CombinedLoader(..., mode=...)` directly now ([#16800](https://github.com/Lightning-AI/lightning/pull/16800))
- Removed the unused `lightning.pytorch.utilities.finite_checks.print_nan_gradients` function ([#16682](https://github.com/Lightning-AI/lightning/pull/16682))
- Removed the unused `lightning.pytorch.utilities.finite_checks.detect_nan_parameters` function ([#16682](https://github.com/Lightning-AI/lightning/pull/16682))
- Removed the unused `lightning.pytorch.utilities.parsing.flatten_dict` function ([#16744](https://github.com/Lightning-AI/lightning/pull/16744))
- Removed the unused `lightning.pytorch.utilities.metrics.metrics_to_scalars` function ([#16681](https://github.com/Lightning-AI/lightning/pull/16681))
- Removed the unused `lightning.pytorch.utilities.supporters.{SharedCycleIteratorState,CombinedLoaderIterator}` classes ([#16714](https://github.com/Lightning-AI/lightning/pull/16714))
- Tuner removal
  * Removed the deprecated `trainer.tuning` property ([#16379](https://github.com/Lightning-AI/lightning/pull/16379))
  * Removed the deprecated `TrainerFn.TUNING` and `RunningStage.TUNING` enums ([#16379](https://github.com/Lightning-AI/lightning/pull/16379))
  * Removed `Trainer.tune()` in favor of `Tuner(trainer).{lr_find,scale_batch_size}` ([#16462](https://github.com/Lightning-AI/lightning/pull/16462))
  * Removed `Trainer(auto_scale_batch_size=...)` in favor of `Tuner(trainer).scale_batch_size()` ([#16462](https://github.com/Lightning-AI/lightning/pull/16462))
  * Removed `Trainer(auto_lr_find=...)` in favor of `Tuner(trainer).lr_find()` ([#16462](https://github.com/Lightning-AI/lightning/pull/16462))
- Removed the `on_tpu` argument from `LightningModule.optimizer_step` hook ([#16537](https://github.com/Lightning-AI/lightning/pull/16537))
- Removed the `using_lbfgs` argument from `LightningModule.optimizer_step` hook ([#16538](https://github.com/Lightning-AI/lightning/pull/16538))
- Removed the `Trainer.data_parallel` property. Use `isinstance(trainer.strategy, ParallelStrategy)` instead ([#16703](https://github.com/Lightning-AI/lightning/pull/16703))
- Removed the `Trainer.prediction_writer_callbacks` property ([#16759](https://github.com/Lightning-AI/lightning/pull/16759))
- Removed support for multiple optimizers in automatic optimization mode ([#16539](https://github.com/Lightning-AI/lightning/pull/16539))
  * Removed `opt_idx` argument from `BaseFinetuning.finetune_function` callback method
  * Removed `opt_idx` argument from `Callback.on_before_optimizer_step` callback method
  * Removed `optimizer_idx` as an optional argument in `LightningModule.training_step`
  * Removed `optimizer_idx` argument from `LightningModule.on_before_optimizer_step`
  * Removed `optimizer_idx` argument from `LightningModule.configure_gradient_clipping`
  * Removed `optimizer_idx` argument from `LightningModule.optimizer_step`
  * Removed `optimizer_idx` argument from `LightningModule.optimizer_zero_grad`
  * Removed `optimizer_idx` argument from `LightningModule.lr_scheduler_step`
  * Removed support for declaring optimizer frequencies in the dictionary returned from `LightningModule.configure_optimizers`
  * Removed arguments `optimizer` and `optimizer_idx` from `LightningModule.backward`
  * Removed `optimizer_idx` argument from `PrecisionPlugin.optimizer_step` and all of its overrides in subclasses
  * Removed `optimizer_idx` argument from `PrecisionPlugin.{optimizer_step,backward}` and all of its overrides in subclasses
  * Removed `optimizer_idx` argument from `Strategy.{optimizer_step,backward}` and all of its overrides in subclasses
  * Removed `Trainer.optimizer_frequencies` attribute
- Removed `Strategy.dispatch` ([#16618](https://github.com/Lightning-AI/lightning/pull/16618))
- Removed `PrecisionPlugin.dispatch` ([#16618](https://github.com/Lightning-AI/lightning/pull/16618))
- Removed legacy argparse utilities ([#16708](https://github.com/Lightning-AI/lightning/pull/16708))
  * Removed `LightningDataModule` methods: `add_argparse_args()`, `from_argparse_args()`, `parse_argparser()`, `get_init_arguments_and_types()`
  * Removed class methods from Trainer: `default_attributes()`, `from_argparse_args()`, `parse_argparser()`, `match_env_arguments()`, `add_argparse_args()`
  * Removed functions from `lightning.pytorch.utilities.argparse`: `from_argparse_args()`, `parse_argparser()`, `parse_env_variables()`, `get_init_arguments_and_types()`, `add_argparse_args()`
  * Removed functions from `lightning.pytorch.utilities.parsing`: `import str_to_bool()`, `str_to_bool_or_int()`, `str_to_bool_or_str()`
- Removed support for passing a scheduling dictionary to `Trainer(accumulate_grad_batches=...)` ([#16729](https://github.com/Lightning-AI/lightning/pull/16729))
- Removed support for `DataParallel` (`strategy='dp'`) and the `LightningParallelModule`-Wrapper, ([#16748](https://github.com/Lightning-AI/lightning/pull/16748))
- Removed the unused `lightning.pytorch.utilities.supporters.{SharedCycleIteratorState,CombinedLoaderIterator}` classes ([#16714](https://github.com/Lightning-AI/lightning/pull/16714))
- Removed `ProgressBarBase.{train_batch_idx,val_batch_idx,test_batch_idx,predict_batch_idx}` properties ([#16760](https://github.com/Lightning-AI/lightning/pull/16760))
- Removed the `fit_loop.{min,max}_steps` setters ([#16803](https://github.com/Lightning-AI/lightning/pull/16803))
- Removed the `Trainer(track_grad_norm=...)` argument ([#16745](https://github.com/Lightning-AI/lightning/pull/16745))
- Removed the `LightningModule.log_grad_norm()` hook method ([#16745](https://github.com/Lightning-AI/lightning/pull/16745))
- Removed the `QuantizationAwareTraining` callback ([#16750](https://github.com/Lightning-AI/lightning/pull/16750))
- Removed the `ColossalAIStrategy` and `ColossalAIPrecisionPlugin` in favor of the new [lightning-colossalai](https://github.com/Lightning-AI/lightning-colossalai) package ([#16757](https://github.com/Lightning-AI/lightning/pull/16757), [#16778](https://github.com/Lightning-AI/lightning/pull/16778))
- Removed the `training_step_end`, `validation_step_end`, and `test_step_end` hooks from the `LightningModule` in favor of the `*_batch_end` hooks ([#16791](https://github.com/Lightning-AI/lightning/pull/16791))
- Removed the `lightning.pytorch.strategies.DDPSpawnStrategy` in favor of `DDPStrategy(start_method='spawn')` (merged both classes) ([#16809](https://github.com/Lightning-AI/lightning/pull/16809))
- Removed registration of `ShardedTensor` state dict hooks in `LightningModule.__init__` with `torch>=2.1` ([#16892](https://github.com/Lightning-AI/lightning/pull/16892))
- Removed the `lightning.pytorch.core.saving.ModelIO` class interface ([#16999](https://github.com/Lightning-AI/lightning/pull/16999))
- Removed the unused `lightning.pytorch.utilities.memory.get_model_size_mb` function ([#17001](https://github.com/Lightning-AI/lightning/pull/17001))

### Fixed

- Fixed an issue where `DistributedSampler.set_epoch` wasn't getting called during `trainer.predict` ([#16785](https://github.com/Lightning-AI/lightning/pull/16785), [#16826](https://github.com/Lightning-AI/lightning/pull/16826))

- Fixed an issue with comparing torch versions when using a version of torch built from source ([#17030](https://github.com/Lightning-AI/lightning/pull/17030))


- Improved the error message for installing tensorboard or tensorboardx ([#17053](https://github.com/Lightning-AI/lightning/pull/17053))


## [1.9.4] - 2023-03-01

### Added

- Added `Fabric(strategy="auto")` support. It will choose DDP over DDP-spawn, contrary to `strategy=None` (default) ([#16916](https://github.com/Lightning-AI/lightning/pull/16916))

### Fixed

- Fixed DDP spawn hang on TPU Pods ([#16844](https://github.com/Lightning-AI/lightning/pull/16844))
- Fixed edge cases in parsing device ids using NVML ([#16795](https://github.com/Lightning-AI/lightning/pull/16795))
- Fixed backwards compatibility for `lightning.pytorch.utilities.parsing.get_init_args` ([#16851](https://github.com/Lightning-AI/lightning/pull/16851))


## [1.9.3] - 2023-02-21

### Fixed

- Fixed an issue causing a wrong environment plugin to be selected when `accelerator=tpu` and `devices > 1` ([#16806](https://github.com/Lightning-AI/lightning/pull/16806))


## [1.9.2] - 2023-02-15

### Fixed

- Fixed an attribute error and improved input validation for invalid strategy types being passed to Trainer ([#16693](https://github.com/Lightning-AI/lightning/pull/16693))
- Fixed early stopping triggering extra validation runs after reaching `min_epochs` or `min_steps` ([#16719](https://github.com/Lightning-AI/lightning/pull/16719))


## [1.9.1] - 2023-02-10

### Fixed

- Fixed an unintended limitation for calling `save_hyperparameters` on mixin classes that don't subclass `LightningModule`/`LightningDataModule` ([#16369](https://github.com/Lightning-AI/lightning/pull/16369))
- Fixed an issue with `MLFlowLogger` logging the wrong keys with `.log_hyperparams()` ([#16418](https://github.com/Lightning-AI/lightning/pull/16418))
- Fixed logging more than 100 parameters with `MLFlowLogger` and long values are truncated ([#16451](https://github.com/Lightning-AI/lightning/pull/16451))
- Fixed strict availability check for `torch_xla` requirement ([#16476](https://github.com/Lightning-AI/lightning/pull/16476))
- Fixed an issue where PL would wrap DataLoaders with XLA's MpDeviceLoader more than once ([#16571](https://github.com/Lightning-AI/lightning/pull/16571))
- Fixed the batch_sampler reference for DataLoaders wrapped with XLA's MpDeviceLoader ([#16571](https://github.com/Lightning-AI/lightning/pull/16571))
- Fixed an import error when `torch.distributed` is not available ([#16658](https://github.com/Lightning-AI/lightning/pull/16658))


## [1.9.0] - 2023-01-17

### Added

- Added support for native logging of `MetricCollection` with enabled compute groups ([#15580](https://github.com/Lightning-AI/lightning/pull/15580))
- Added support for custom artifact names in `pl.loggers.WandbLogger` ([#16173](https://github.com/Lightning-AI/lightning/pull/16173))
- Added support for DDP with `LRFinder` ([#15304](https://github.com/Lightning-AI/lightning/pull/15304))
- Added utilities to migrate checkpoints from one Lightning version to another ([#15237](https://github.com/Lightning-AI/lightning/pull/15237))
- Added support to upgrade all checkpoints in a folder using the `pl.utilities.upgrade_checkpoint` script ([#15333](https://github.com/Lightning-AI/lightning/pull/15333))
- Add an axes argument `ax` to the `.lr_find().plot()` to enable writing to a user-defined axes in a matplotlib figure ([#15652](https://github.com/Lightning-AI/lightning/pull/15652))
- Added `log_model` parameter to `MLFlowLogger` ([#9187](https://github.com/Lightning-AI/lightning/pull/9187))
- Added a check to validate that wrapped FSDP models are used while initializing optimizers ([#15301](https://github.com/Lightning-AI/lightning/pull/15301))
- Added a warning when `self.log(..., logger=True)` is called without a configured logger ([#15814](https://github.com/Lightning-AI/lightning/pull/15814))
- Added support for colossalai 0.1.11 ([#15888](https://github.com/Lightning-AI/lightning/pull/15888))
- Added `LightningCLI` support for optimizer and learning schedulers via callable type dependency injection ([#15869](https://github.com/Lightning-AI/lightning/pull/15869))
- Added support for activation checkpointing for the `DDPFullyShardedNativeStrategy` strategy ([#15826](https://github.com/Lightning-AI/lightning/pull/15826))
- Added the option to set `DDPFullyShardedNativeStrategy(cpu_offload=True|False)` via bool instead of needing to pass a configuration object ([#15832](https://github.com/Lightning-AI/lightning/pull/15832))
- Added info message for Ampere CUDA GPU users to enable tf32 matmul precision ([#16037](https://github.com/Lightning-AI/lightning/pull/16037))
- Added support for returning optimizer-like classes in `LightningModule.configure_optimizers` ([#16189](https://github.com/Lightning-AI/lightning/pull/16189))

### Changed

- Drop PyTorch 1.9 support ([#15347](https://github.com/Lightning-AI/lightning/pull/15347))
- Switch from `tensorboard` to `tensorboardx` in `TensorBoardLogger` ([#15728](https://github.com/Lightning-AI/lightning/pull/15728))
- From now on, Lightning Trainer and `LightningModule.load_from_checkpoint` automatically upgrade the loaded checkpoint if it was produced in an old version of Lightning ([#15237](https://github.com/Lightning-AI/lightning/pull/15237))
- `Trainer.{validate,test,predict}(ckpt_path=...)` no longer restores the `Trainer.global_step` and `trainer.current_epoch` value from the checkpoints - From now on, only `Trainer.fit` will restore this value ([#15532](https://github.com/Lightning-AI/lightning/pull/15532))
- The `ModelCheckpoint.save_on_train_epoch_end` attribute is now computed dynamically every epoch, accounting for changes to the validation dataloaders ([#15300](https://github.com/Lightning-AI/lightning/pull/15300))
- The Trainer now raises an error if it is given multiple stateful callbacks of the same time with colliding state keys ([#15634](https://github.com/Lightning-AI/lightning/pull/15634))
- `MLFlowLogger` now logs hyperparameters and metrics in batched API calls ([#15915](https://github.com/Lightning-AI/lightning/pull/15915))
- Overriding the `on_train_batch_{start,end}` hooks in conjunction with taking a `dataloader_iter` in the `training_step` no longer errors out and instead shows a warning ([#16062](https://github.com/Lightning-AI/lightning/pull/16062))
- Move `tensorboardX` to extra dependencies. Use the `CSVLogger` by default ([#16349](https://github.com/Lightning-AI/lightning/pull/16349))

### Deprecated

- Deprecated `description`, `env_prefix` and `env_parse` parameters in `LightningCLI.__init__` in favour of giving them through `parser_kwargs` ([#15651](https://github.com/Lightning-AI/lightning/pull/15651))
- Deprecated `pl.profiler` in favor of `pl.profilers` ([#16059](https://github.com/Lightning-AI/lightning/pull/16059))
- Deprecated `Trainer(auto_select_gpus=...)` in favor of `pl.accelerators.find_usable_cuda_devices` ([#16147](https://github.com/Lightning-AI/lightning/pull/16147))
- Deprecated `pl.tuner.auto_gpu_select.{pick_single_gpu,pick_multiple_gpus}` in favor of `pl.accelerators.find_usable_cuda_devices` ([#16147](https://github.com/Lightning-AI/lightning/pull/16147))
- `nvidia/apex` deprecation ([#16039](https://github.com/Lightning-AI/lightning/pull/16039))
  * Deprecated `pl.plugins.NativeMixedPrecisionPlugin` in favor of `pl.plugins.MixedPrecisionPlugin`
  * Deprecated the `LightningModule.optimizer_step(using_native_amp=...)` argument
  * Deprecated the `Trainer(amp_backend=...)` argument
  * Deprecated the `Trainer.amp_backend` property
  * Deprecated the `Trainer(amp_level=...)` argument
  * Deprecated the `pl.plugins.ApexMixedPrecisionPlugin` class
  * Deprecates the `pl.utilities.enums.AMPType` enum
  * Deprecates the `DeepSpeedPrecisionPlugin(amp_type=..., amp_level=...)` arguments
- `horovod` deprecation ([#16141](https://github.com/Lightning-AI/lightning/pull/16141))
  * Deprecated `Trainer(strategy="horovod")`
  * Deprecated the `HorovodStrategy` class
- Deprecated `pl.lite.LightningLite` in favor of `lightning.fabric.Fabric` ([#16314](https://github.com/Lightning-AI/lightning/pull/16314))
- `FairScale` deprecation (in favor of PyTorch's FSDP implementation) ([#16353](https://github.com/Lightning-AI/lightning/pull/16353))
  * Deprecated the `pl.overrides.fairscale.LightningShardedDataParallel` class
  * Deprecated the `pl.plugins.precision.fully_sharded_native_amp.FullyShardedNativeMixedPrecisionPlugin` class
  * Deprecated the `pl.plugins.precision.sharded_native_amp.ShardedNativeMixedPrecisionPlugin` class
  * Deprecated the `pl.strategies.fully_sharded.DDPFullyShardedStrategy` class
  * Deprecated the `pl.strategies.sharded.DDPShardedStrategy` class
  * Deprecated the `pl.strategies.sharded_spawn.DDPSpawnShardedStrategy` class


### Removed

- Removed deprecated `pl.utilities.memory.get_gpu_memory_map` in favor of `pl.accelerators.cuda.get_nvidia_gpu_stats` ([#15617](https://github.com/Lightning-AI/lightning/pull/15617))
- Temporarily removed support for Hydra multi-run ([#15737](https://github.com/Lightning-AI/lightning/pull/15737))
- Removed deprecated `pl.profiler.base.AbstractProfiler` in favor of `pl.profilers.profiler.Profiler` ([#15637](https://github.com/Lightning-AI/lightning/pull/15637))
- Removed deprecated `pl.profiler.base.BaseProfiler` in favor of `pl.profilers.profiler.Profiler` ([#15637](https://github.com/Lightning-AI/lightning/pull/15637))
- Removed deprecated code in `pl.utilities.meta` ([#16038](https://github.com/Lightning-AI/lightning/pull/16038))
- Removed the deprecated `LightningDeepSpeedModule` ([#16041](https://github.com/Lightning-AI/lightning/pull/16041))
- Removed the deprecated `pl.accelerators.GPUAccelerator` in favor of `pl.accelerators.CUDAAccelerator` ([#16050](https://github.com/Lightning-AI/lightning/pull/16050))
- Removed the deprecated `pl.profiler.*` classes in favor of `pl.profilers` ([#16059](https://github.com/Lightning-AI/lightning/pull/16059))
- Removed the deprecated `pl.utilities.cli` module in favor of `pl.cli` ([#16116](https://github.com/Lightning-AI/lightning/pull/16116))
- Removed the deprecated `pl.loggers.base` module in favor of `pl.loggers.logger` ([#16120](https://github.com/Lightning-AI/lightning/pull/16120))
- Removed the deprecated `pl.loops.base` module in favor of `pl.loops.loop` ([#16142](https://github.com/Lightning-AI/lightning/pull/16142))
- Removed the deprecated `pl.core.lightning` module in favor of `pl.core.module` ([#16318](https://github.com/Lightning-AI/lightning/pull/16318))
- Removed the deprecated `pl.callbacks.base` module in favor of `pl.callbacks.callback` ([#16319](https://github.com/Lightning-AI/lightning/pull/16319))
- Removed the deprecated `Trainer.reset_train_val_dataloaders()` in favor of `Trainer.reset_{train,val}_dataloader` ([#16131](https://github.com/Lightning-AI/lightning/pull/16131))
- Removed support for `LightningCLI(seed_everything_default=None)` ([#16131](https://github.com/Lightning-AI/lightning/pull/16131))
- Removed support in LightningLite for FairScale's sharded training (`strategy='ddp_sharded'|'ddp_sharded_spawn'`). Use Fully-Sharded Data Parallel instead (`strategy='fsdp'`) ([#16329](https://github.com/Lightning-AI/lightning/pull/16329))


### Fixed

- Enhanced `reduce_boolean_decision` to accommodate `any`-analogous semantics expected by the `EarlyStopping` callback ([#15253](https://github.com/Lightning-AI/lightning/pull/15253))
- Fixed the incorrect optimizer step synchronization when running across multiple TPU devices ([#16020](https://github.com/Lightning-AI/lightning/pull/16020))
- Fixed a type error when dividing the chunk size in the ColossalAI strategy ([#16212](https://github.com/Lightning-AI/lightning/pull/16212))
- Fixed bug where the ``interval`` key of the scheduler would be ignored during manual optimization, making the LearningRateMonitor callback fail to log the learning rate ([#16308](https://github.com/Lightning-AI/lightning/pull/16308))
- Fixed an issue with `MLFlowLogger` not finalizing correctly when status code 'finished' was passed ([#16340](https://github.com/Lightning-AI/lightning/pull/16340))


## [1.8.6] - 2022-12-21

- minor cleaning


## [1.8.5] - 2022-12-15

- Add function to remove checkpoint to allow override for extended classes ([#16067](https://github.com/Lightning-AI/lightning/pull/16067))


## [1.8.4] - 2022-12-08

### Changed

- Direct support for compiled models (
   [#15922](https://github.com/Lightning-AI/lightning/pull/15922),
   [#15957](https://github.com/Lightning-AI/lightning/pull/15957)
)

### Fixed

- Fixed issue with unsupported torch.inference_mode() on hpu backends ([#15918](https://github.com/Lightning-AI/lightning/pull/15918))
- Fixed LRScheduler import for PyTorch 2.0 ([#15940](https://github.com/Lightning-AI/lightning/pull/15940))
- Fixed `fit_loop.restarting` to be `False` for lr finder ([#15620](https://github.com/Lightning-AI/lightning/pull/15620))
- Fixed `torch.jit.script`-ing a LightningModule causing an unintended error message about deprecated `use_amp` property ([#15947](https://github.com/Lightning-AI/lightning/pull/15947))
- Fixed the `XLAProfiler` not recording anything due to mismatching of action names ([#15885](https://github.com/Lightning-AI/lightning/pull/15885))


## [1.8.3] - 2022-11-22

### Changed

- Temporarily removed support for Hydra multi-run ([#15737](https://github.com/Lightning-AI/lightning/pull/15737))
- Switch from `tensorboard` to `tensorboardx` in `TensorBoardLogger` ([#15728](https://github.com/Lightning-AI/lightning/pull/15728))


## [1.8.2] - 2022-11-17

### Fixed

- Make sure save_dir can be empty str ([#15638](https://github.com/Lightning-AI/lightning/pull/15638))
- Fixed the automatic fallback from `Trainer(strategy="ddp_spawn", ...)` to `Trainer(strategy="ddp", ...)` when on an LSF cluster ([#15103](https://github.com/Lightning-AI/lightning/pull/15103))



## [1.8.1] - 2022-11-10

### Added

- Added back the accidentally removed `pl.utilities.distributed.rank_zero_only` function ([#15536](https://github.com/Lightning-AI/lightning/pull/15536))

### Deprecated

- Deprecated `pl.utilities.distributed.rank_zero_only` in favor of `pl.utilities.rank_zero_only` ([#15536](https://github.com/Lightning-AI/lightning/pull/15536))

### Fixed

- Fixed `TensorBoardLogger` not validating the input array type when logging the model graph ([#15323](https://github.com/Lightning-AI/lightning/pull/15323))
- Fixed an attribute error in `ColossalAIStrategy` at import time when `torch.distributed` is not available ([#15535](https://github.com/Lightning-AI/lightning/pull/15535))
- Fixed an issue when calling `fs.listdir` with file URI instead of path in `CheckpointConnector` ([#15413](https://github.com/Lightning-AI/lightning/pull/15413))
- Fixed an issue with the `BaseFinetuning` callback not setting the `track_running_stats` attribute for batch normaliztion layers ([#15063](https://github.com/Lightning-AI/lightning/pull/15063))
- Fixed an issue with `WandbLogger(log_model=True|'all)` raising an error and not being able to serialize tensors in the metadata ([#15544](https://github.com/Lightning-AI/lightning/pull/15544))
- Fixed the gradient unscaling logic when using `Trainer(precision=16)` and fused optimizers such as `Adam(..., fused=True)` ([#15544](https://github.com/Lightning-AI/lightning/pull/15544))
- Fixed model state transfer in multiprocessing launcher when running multi-node ([#15567](https://github.com/Lightning-AI/lightning/pull/15567))
- Fixed manual optimization raising `AttributeError` with Bagua Strategy ([#12534](https://github.com/Lightning-AI/lightning/pull/12534))
- Fixed the import of `pytorch_lightning` causing a warning 'Redirects are currently not supported in Windows or MacOs' ([#15610](https://github.com/Lightning-AI/lightning/pull/15610))


## [1.8.0] - 2022-11-01

### Added

- Added support for requeueing slurm array jobs ([#15040](https://github.com/Lightning-AI/lightning/pull/15040))
- Added native AMP support for `ddp_fork` (and associated alias strategies) with CUDA GPUs ([#14983](https://github.com/Lightning-AI/lightning/pull/14983))
- Added `BatchSizeFinder` callback ([#11089](https://github.com/Lightning-AI/lightning/pull/11089))
- Added `LearningRateFinder` callback ([#13802](https://github.com/Lightning-AI/lightning/pull/13802))
- Tuner now supports a new `method` argument which will determine when to run the `BatchSizeFinder`: one of `fit`, `validate`, `test` or `predict` ([#11089](https://github.com/Lightning-AI/lightning/pull/11089))
- Added prefix to log message in `seed_everything` with rank info ([#14031](https://github.com/Lightning-AI/lightning/pull/14031))
- Added support for auto wrapping for `DDPFullyShardedNativeStrategy` ([#14252](https://github.com/Lightning-AI/lightning/pull/14252))
- Added support for passing extra init-parameters to the `LightningDataModule.from_datasets` ([#14185](https://github.com/Lightning-AI/lightning/pull/14185))
- Added support for saving sharded optimizer state dict outside of `DDPShardedStrategy` ([#14208](https://github.com/Lightning-AI/lightning/pull/14208))
- Added support for auto wrapping for `DDPFullyShardedStrategy` ([#14383](https://github.com/Lightning-AI/lightning/pull/14383))
- Integrate the `lightning_utilities` package (
  [#14475](https://github.com/Lightning-AI/lightning/pull/14475),
  [#14537](https://github.com/Lightning-AI/lightning/pull/14537),
  [#14556](https://github.com/Lightning-AI/lightning/pull/14556),
  [#14558](https://github.com/Lightning-AI/lightning/pull/14558),
  [#14575](https://github.com/Lightning-AI/lightning/pull/14575),
  [#14620](https://github.com/Lightning-AI/lightning/pull/14620))
- Added `args` parameter to `LightningCLI` to ease running from within Python ([#14596](https://github.com/Lightning-AI/lightning/pull/14596))
- Added `WandbLogger.download_artifact` and `WandbLogger.use_artifact` for managing artifacts with Weights and Biases ([#14551](https://github.com/Lightning-AI/lightning/pull/14551))
- Added an option to configure the signal SLURM sends when a job is preempted or requeued ([#14626](https://github.com/Lightning-AI/lightning/pull/14626))
- Added a warning when the model passed to `LightningLite.setup()` does not have all parameters on the same device ([#14822](https://github.com/Lightning-AI/lightning/pull/14822))
- The `CometLogger` now flags the Comet Experiments as being created from Lightning for analytics purposes ([#14906](https://github.com/Lightning-AI/lightning/pull/14906))
- Introduce `ckpt_path="hpc"` keyword for checkpoint loading ([#14911](https://github.com/Lightning-AI/lightning/pull/14911))
- Added a more descriptive error message when attempting to fork processes with pre-initialized CUDA context ([#14709](https://github.com/Lightning-AI/lightning/pull/14709))
- Added support for custom parameters in subclasses of `SaveConfigCallback` ([#14998](https://github.com/Lightning-AI/lightning/pull/14998))
- Added `inference_mode` flag to Trainer to let users enable/disable inference mode during evaluation ([#15034](https://github.com/Lightning-AI/lightning/pull/15034))
- Added `LightningLite.no_backward_sync` for control over efficient gradient accumulation with distributed strategies ([#14966](https://github.com/Lightning-AI/lightning/pull/14966))
- Added a sanity check that scripts are executed with the `srun` command in SLURM and that environment variables are not conflicting ([#15011](https://github.com/Lightning-AI/lightning/pull/15011))
- Added an error message when attempting to launch processes with `python -i` and an interactive-incompatible strategy ([#15293](https://github.com/Lightning-AI/lightning/pull/15293))

### Changed

- The `Trainer.{fit,validate,test,predict,tune}` methods now raise a useful error message if the input is not a `LightningModule` ([#13892](https://github.com/Lightning-AI/lightning/pull/13892))
- Raised a `MisconfigurationException` if batch transfer hooks are overridden with `IPUAccelerator` ([#13961](https://github.com/Lightning-AI/lightning/pull/13961))
- Replaced the unwrapping logic in strategies with direct access to unwrapped `LightningModule` ([#13738](https://github.com/Lightning-AI/lightning/pull/13738))
- Enabled `on_before_batch_transfer` for `DPStrategy` and `IPUAccelerator` ([#14023](https://github.com/Lightning-AI/lightning/pull/14023))
- When resuming training with Apex enabled, the `Trainer` will now raise an error ([#14341](https://github.com/Lightning-AI/lightning/pull/14341))
- Included `torch.cuda` rng state to the aggregate `_collect_rng_states()` and `_set_rng_states()` ([#14384](https://github.com/Lightning-AI/lightning/pull/14384))
- Changed `trainer.should_stop` to not stop in between an epoch and run until `min_steps/min_epochs` only ([#13890](https://github.com/Lightning-AI/lightning/pull/13890))
- The `pyDeprecate` dependency is no longer installed ([#14472](https://github.com/Lightning-AI/lightning/pull/14472))
- When using multiple loggers, by default checkpoints and profiler output now get saved to the log dir of the first logger in the list ([#14325](https://github.com/Lightning-AI/lightning/pull/14325))
- In Lightning Lite, state-dict access to the module wrapper now gets passed through to the original module reference ([#14629](https://github.com/Lightning-AI/lightning/pull/14629))
- Removed fall-back to `LightningEnvironment` when number of SLURM tasks does not correspond to number of processes in Trainer ([#14300](https://github.com/Lightning-AI/lightning/pull/14300))
- Aligned DDP and DDPSpawn strategies in setting up the environment ([#11073](https://github.com/Lightning-AI/lightning/pull/11073))
- Integrated the Lite Precision plugins into the PL Precision plugins - the base class in PL now extends the `lightning_lite.precision.Precision` base class ([#14798](https://github.com/Lightning-AI/lightning/pull/14798))
  * The `PrecisionPlugin.backward` signature changed: The `closure_loss` argument was renamed to `tensor`
  * The `PrecisionPlugin.{pre_,post_}backward` signature changed: The `closure_loss` argument was renamed to `tensor` and moved as the first argument
  * The `PrecisionPlugin.optimizer_step` signature changed: The `model`, `optimizer_idx` and `closure` arguments need to be passed as keyword arguments now
- Trainer queries the CUDA devices through NVML if available to avoid initializing CUDA before forking, which eliminates the need for the `PL_DISABLE_FORK` environment variable introduced in v1.7.4 ([#14631](https://github.com/Lightning-AI/lightning/pull/14631))
- The `MLFlowLogger.finalize()` now sets the status to `FAILED` when an exception occurred in `Trainer`, and sets the status to `FINISHED` on successful completion ([#12292](https://github.com/Lightning-AI/lightning/pull/12292))
- It is no longer needed to call `model.double()` when using `precision=64` in Lightning Lite ([#14827](https://github.com/Lightning-AI/lightning/pull/14827))
- HPC checkpoints are now loaded automatically only in slurm environment when no specific value for `ckpt_path` has been set ([#14911](https://github.com/Lightning-AI/lightning/pull/14911))
- The `Callback.on_load_checkpoint` now gets the full checkpoint dictionary and the `callback_state` argument was renamed `checkpoint` ([#14835](https://github.com/Lightning-AI/lightning/pull/14835))
- Moved the warning about saving nn.Module in `save_hyperparameters()` to before the deepcopy ([#15132](https://github.com/Lightning-AI/lightning/pull/15132))
- To avoid issues with forking processes, from PyTorch 1.13 and higher, Lightning will directly use the PyTorch NVML-based check for `torch.cuda.device_count` and from PyTorch 2.0 and higher, Lightning will configure PyTorch to use a NVML-based check for `torch.cuda.is_available`. ([#15110](https://github.com/Lightning-AI/lightning/pull/15110), [#15133](https://github.com/Lightning-AI/lightning/pull/15133))
- The `NeptuneLogger` now uses `neptune.init_run` instead of the deprecated `neptune.init` to initialize a run ([#15393](https://github.com/Lightning-AI/lightning/pull/15393))

### Deprecated

- Deprecated `LightningDeepSpeedModule` ([#14000](https://github.com/Lightning-AI/lightning/pull/14000))
- Deprecated `amp_level` from `Trainer` in favour of passing it explicitly via precision plugin ([#13898](https://github.com/Lightning-AI/lightning/pull/13898))
- Deprecated the calls to `pl.utiltiies.meta` functions in favor of built-in https://github.com/pytorch/torchdistx support ([#13868](https://github.com/Lightning-AI/lightning/pull/13868))
- Deprecated the `unwrap_lightning_module` and `unwrap_lightning_module_sharded` utility functions in favor of accessing the unwrapped `LightningModule` on the strategy directly ([#13738](https://github.com/Lightning-AI/lightning/pull/13738))
- Deprecated the `pl_module` argument in `LightningParallelModule`, `LightningDistributedModule`, `LightningShardedDataParallel`, `LightningBaguaModule` and `LightningDeepSpeedModule` wrapper classes ([#13738](https://github.com/Lightning-AI/lightning/pull/13738))
- Deprecated the `on_colab_kaggle` function ([#14247](https://github.com/Lightning-AI/lightning/pull/14247))
- Deprecated the internal `pl.core.mixins.DeviceDtypeModuleMixin` class ([#14511](https://github.com/Lightning-AI/lightning/pull/14511), [#14548](https://github.com/Lightning-AI/lightning/pull/14548))
- Deprecated all functions in `pl.utilities.xla_device` ([#14514](https://github.com/Lightning-AI/lightning/pull/14514), [#14550](https://github.com/Lightning-AI/lightning/pull/14550))
  * Deprecated the internal `inner_f` function
  * Deprecated the internal `pl_multi_process` function
  * Deprecated the internal `XLADeviceUtils.xla_available` staticmethod
  * Deprecated the `XLADeviceUtils.tpu_device_exists` staticmethod in favor of `pl.accelerators.TPUAccelerator.is_available()`
- Deprecated `pl.utilities.distributed.tpu_distributed` in favor of `lightning_lite.accelerators.tpu.tpu_distributed` ([#14550](https://github.com/Lightning-AI/lightning/pull/14550))
- Deprecated all functions in `pl.utilities.cloud_io` in favor of `lightning_lite.utilities.cloud_io` ([#14515](https://github.com/Lightning-AI/lightning/pull/14515))
- Deprecated the functions in `pl.utilities.apply_func` in favor of `lightning_utilities.core.apply_func` ([#14516](https://github.com/Lightning-AI/lightning/pull/14516), [#14537](https://github.com/Lightning-AI/lightning/pull/14537))
- Deprecated all functions in `pl.utilities.device_parser` ([#14492](https://github.com/Lightning-AI/lightning/pull/14492), [#14753](https://github.com/Lightning-AI/lightning/pull/14753))
  * Deprecated the `pl.utilities.device_parser.determine_root_gpu_device` in favor of `lightning_lite.utilities.device_parser.determine_root_gpu_device`
  * Deprecated the `pl.utilities.device_parser.parse_gpu_ids` in favor of `lightning_lite.utilities.device_parser.parse_gpu_ids`
  * Deprecated the `pl.utilities.device_parser.is_cuda_available` in favor of `lightning_lite.accelerators.cuda.is_cuda_available`
  * Deprecated the `pl.utilities.device_parser.num_cuda_devices` in favor of `lightning_lite.accelerators.cuda.num_cuda_devices`
  * Deprecated the `pl.utilities.device_parser.parse_cpu_cores` in favor of `lightning_lite.accelerators.cpu.parse_cpu_cores`
  * Deprecated the `pl.utilities.device_parser.parse_tpu_cores` in favor of `lightning_lite.accelerators.tpu.parse_tpu_cores`
  * Deprecated the `pl.utilities.device_parser.parse_hpus` in favor of `pl.accelerators.hpu.parse_hpus`
- Deprecated duplicate `SaveConfigCallback` parameters in `LightningCLI.__init__`: `save_config_kwargs`, `save_config_overwrite` and `save_config_multifile`. New `save_config_kwargs` parameter should be used instead ([#14998](https://github.com/Lightning-AI/lightning/pull/14998))
- Deprecated `TrainerFn.TUNING`, `RunningStage.TUNING` and `trainer.tuning` property ([#15100](https://github.com/Lightning-AI/lightning/pull/15100))
- Deprecated custom `pl.utilities.distributed.AllGatherGrad` implementation in favor of PyTorch's ([#15364](https://github.com/Lightning-AI/lightning/pull/15364))

### Removed

- Removed the deprecated `Trainer.training_type_plugin` property in favor of `Trainer.strategy` ([#14011](https://github.com/Lightning-AI/lightning/pull/14011))
- Removed all deprecated training type plugins ([#14011](https://github.com/Lightning-AI/lightning/pull/14011))
- Removed the deprecated `DDP2Strategy` ([#14026](https://github.com/Lightning-AI/lightning/pull/14026))
- Removed the deprecated `DistributedType` and `DeviceType` enum classes ([#14045](https://github.com/Lightning-AI/lightning/pull/14045))
- Removed deprecated support for passing the `rank_zero_warn` warning category positionally ([#14470](https://github.com/Lightning-AI/lightning/pull/14470))
- Removed the legacy and unused `Trainer.get_deprecated_arg_names()` ([#14415](https://github.com/Lightning-AI/lightning/pull/14415))
- Removed the deprecated `on_train_batch_end(outputs)` format when multiple optimizers are used and TBPTT is enabled ([#14373](https://github.com/Lightning-AI/lightning/pull/14373))
- Removed the deprecated `training_epoch_end(outputs)` format when multiple optimizers are used and TBPTT is enabled ([#14373](https://github.com/Lightning-AI/lightning/pull/14373))
- Removed the experimental `pl.utiltiies.meta` functions in favor of built-in https://github.com/pytorch/torchdistx support ([#13868](https://github.com/Lightning-AI/lightning/pull/13868))
- Removed the deprecated `LoggerCollection`; `Trainer.logger` and `LightningModule.logger` now returns the first logger when more than one gets passed to the Trainer ([#14283](https://github.com/Lightning-AI/lightning/pull/14283))
- Removed the deprecated the `trainer.lr_schedulers` ([#14408](https://github.com/Lightning-AI/lightning/pull/14408))
- Removed the deprecated `LightningModule.{on_hpc_load,on_hpc_save}` hooks in favor of the general purpose hooks `LightningModule.{on_load_checkpoint,on_save_checkpoint}` ([#14315](https://github.com/Lightning-AI/lightning/pull/14315))
- Removed deprecated support for old torchtext versions ([#14375](https://github.com/Lightning-AI/lightning/pull/14375))
- Removed deprecated support for the old `neptune-client` API in the `NeptuneLogger` ([#14727](https://github.com/Lightning-AI/lightning/pull/14727))
- Removed the deprecated `weights_save_path` Trainer argumnent and `Trainer.weights_save_path` property ([#14424](https://github.com/Lightning-AI/lightning/pull/14424))
- Removed the deprecated ([#14471](https://github.com/Lightning-AI/lightning/pull/14471))
  * `pl.utilities.distributed.rank_zero_only` in favor of `pl.utilities.rank_zero.rank_zero_only`
  * `pl.utilities.distributed.rank_zero_debug` in favor of `pl.utilities.rank_zero.rank_zero_debug`
  * `pl.utilities.distributed.rank_zero_info` in favor of `pl.utilities.rank_zero.rank_zero_info`
  * `pl.utilities.warnings.rank_zero_warn` in favor of `pl.utilities.rank_zero.rank_zero_warn`
  * `pl.utilities.warnings.rank_zero_deprecation` in favor of `pl.utilities.rank_zero.rank_zero_deprecation`
  * `pl.utilities.warnings.LightningDeprecationWarning` in favor of `pl.utilities.rank_zero.LightningDeprecationWarning`
- Removed deprecated `Trainer.num_processes` attribute in favour of `Trainer.num_devices` ([#14423](https://github.com/Lightning-AI/lightning/pull/14423))
- Removed the deprecated `Trainer.data_parallel_device_ids` hook in favour of `Trainer.device_ids` ([#14422](https://github.com/Lightning-AI/lightning/pull/14422))
- Removed the deprecated class `TrainerCallbackHookMixin` ([#14401](https://github.com/Lightning-AI/lightning/pull/14401))
- Removed the deprecated `BaseProfiler` and `AbstractProfiler` classes ([#14404](https://github.com/Lightning-AI/lightning/pull/14404))
- Removed the deprecated way to set the distributed backend via the environment variable `PL_TORCH_DISTRIBUTED_BACKEND`, in favor of setting the `process_group_backend` in the strategy constructor ([#14693](https://github.com/Lightning-AI/lightning/pull/14693))
- Removed deprecated callback hooks ([#14834](https://github.com/Lightning-AI/lightning/pull/14834))
  * `Callback.on_configure_sharded_model` in favor of `Callback.setup`
  * `Callback.on_before_accelerator_backend_setup` in favor of `Callback.setup`
  * `Callback.on_batch_start` in favor of `Callback.on_train_batch_start`
  * `Callback.on_batch_end` in favor of `Callback.on_train_batch_end`
  * `Callback.on_epoch_start` in favor of `Callback.on_{train,validation,test}_epoch_start`
  * `Callback.on_epoch_end` in favor of `Callback.on_{train,validation,test}_epoch_end`
  * `Callback.on_pretrain_routine_{start,end}` in favor of `Callback.on_fit_start`
- Removed the deprecated device attributes `Trainer.{devices,gpus,num_gpus,ipus,tpu_cores}` in favor of the accelerator-agnostic `Trainer.num_devices` ([#14829](https://github.com/Lightning-AI/lightning/pull/14829))
- Removed the deprecated `LightningIPUModule` ([#14830](https://github.com/Lightning-AI/lightning/pull/14830))
- Removed the deprecated `Logger.agg_and_log_metrics` hook in favour of `Logger.log_metrics` and the `agg_key_funcs` and `agg_default_func` arguments. ([#14840](https://github.com/Lightning-AI/lightning/pull/14840))
- Removed the deprecated precision plugin checkpoint hooks `PrecisionPlugin.on_load_checkpoint` and `PrecisionPlugin.on_save_checkpoint` ([#14833](https://github.com/Lightning-AI/lightning/pull/14833))
- Removed the deprecated `Trainer.root_gpu` attribute in favor of `Trainer.strategy.root_device` ([#14829](https://github.com/Lightning-AI/lightning/pull/14829))
- Removed the deprecated `Trainer.use_amp` and `LightningModule.use_amp` attributes ([#14832](https://github.com/Lightning-AI/lightning/pull/14832))
- Removed the deprecated callback hooks `Callback.on_init_start` and `Callback.on_init_end` ([#14867](https://github.com/Lightning-AI/lightning/pull/14867))
- Removed the deprecated `Trainer.run_stage` in favor of `Trainer.{fit,validate,test,predict}` ([#14870](https://github.com/Lightning-AI/lightning/pull/14870))
- Removed the deprecated `SimpleProfiler.profile_iterable` and `AdvancedProfiler.profile_iterable` attributes ([#14864](https://github.com/Lightning-AI/lightning/pull/14864))
- Removed the deprecated `Trainer.verbose_evaluate` ([#14884](https://github.com/Lightning-AI/lightning/pull/14884))
- Removed the deprecated `Trainer.should_rank_save_checkpoint` ([#14885](https://github.com/Lightning-AI/lightning/pull/14885))
- Removed the deprecated `TrainerOptimizersMixin` ([#14887](https://github.com/Lightning-AI/lightning/pull/14887))
- Removed the deprecated `Trainer.lightning_optimizers` ([#14889](https://github.com/Lightning-AI/lightning/pull/14889))
- Removed the deprecated `TrainerDataLoadingMixin` ([#14888](https://github.com/Lightning-AI/lightning/pull/14888))
- Removed the deprecated `Trainer.call_hook` in favor of `Trainer._call_callback_hooks`, `Trainer._call_lightning_module_hook`, `Trainer._call_ttp_hook`, and `Trainer._call_accelerator_hook` ([#14869](https://github.com/Lightning-AI/lightning/pull/14869))
- Removed the deprecated `Trainer.{validated,tested,predicted}_ckpt_path` ([#14897](https://github.com/Lightning-AI/lightning/pull/14897))
- Removed the deprecated `device_stats_monitor_prefix_metric_keys` ([#14890](https://github.com/Lightning-AI/lightning/pull/14890))
- Removed the deprecated `LightningDataModule.on_save/load_checkpoint` hooks ([#14909](https://github.com/Lightning-AI/lightning/pull/14909))
- Removed support for returning a value in `Callback.on_save_checkpoint` in favor of implementing `Callback.state_dict` ([#14835](https://github.com/Lightning-AI/lightning/pull/14835))

### Fixed

- Fixed an issue with `LightningLite.setup()` not setting the `.device` attribute correctly on the returned wrapper ([#14822](https://github.com/Lightning-AI/lightning/pull/14822))
- Fixed an attribute error when running the tuner together with the `StochasticWeightAveraging` callback ([#14836](https://github.com/Lightning-AI/lightning/pull/14836))
- Fixed MissingFieldException in offline mode for the `NeptuneLogger()` ([#14919](https://github.com/Lightning-AI/lightning/pull/14919))
- Fixed wandb `save_dir` is overridden by `None` `dir` when using CLI ([#14878](https://github.com/Lightning-AI/lightning/pull/14878))
- Fixed a missing call to `LightningDataModule.load_state_dict` hook while restoring checkpoint using `LightningDataModule.load_from_checkpoint` ([#14883](https://github.com/Lightning-AI/lightning/pull/14883))
- Fixed torchscript error with containers of LightningModules ([#14904](https://github.com/Lightning-AI/lightning/pull/14904))
- Fixed reloading of the last checkpoint on run restart ([#14907](https://github.com/Lightning-AI/lightning/pull/14907))
- `SaveConfigCallback` instances should only save the config once to allow having the `overwrite=False` safeguard when using `LightningCLI(..., run=False)` ([#14927](https://github.com/Lightning-AI/lightning/pull/14927))
- Fixed an issue with terminating the trainer profiler when a `StopIteration` exception is raised while using an `IterableDataset` ([#14940](https://github.com/Lightning-AI/lightning/pull/14945))
- Do not update on-plateau schedulers when reloading from an end-of-epoch checkpoint ([#14702](https://github.com/Lightning-AI/lightning/pull/14702))
- Fixed `Trainer` support for PyTorch built without distributed support ([#14971](https://github.com/Lightning-AI/lightning/pull/14971))
- Fixed batch normalization statistics calculation in `StochasticWeightAveraging` callback ([#14866](https://github.com/Lightning-AI/lightning/pull/14866))
- Avoided initializing optimizers during deepspeed inference ([#14944](https://github.com/Lightning-AI/lightning/pull/14944))
- Fixed `LightningCLI` parse_env and description in subcommands ([#15138](https://github.com/Lightning-AI/lightning/pull/15138))
- Fixed an exception that would occur when creating a `multiprocessing.Pool` after importing Lightning ([#15292](https://github.com/Lightning-AI/lightning/pull/15292))
- Fixed a pickling error when using `RichProgressBar` together with checkpointing ([#15319](https://github.com/Lightning-AI/lightning/pull/15319))
- Fixed the `RichProgressBar` crashing when used with distributed strategies ([#15376](https://github.com/Lightning-AI/lightning/pull/15376))
- Fixed an issue with `RichProgressBar` not resetting the internal state for the sanity check progress ([#15377](https://github.com/Lightning-AI/lightning/pull/15377))
- Fixed an issue with DataLoader re-instantiation when the attribute is an array and the default value of the corresponding argument changed ([#15409](https://github.com/Lightning-AI/lightning/pull/15409))


## [1.7.7] - 2022-09-22

### Fixed

- Fixed the availability check for the neptune-client package ([#14714](https://github.com/Lightning-AI/lightning/pull/14714))
- Break HPU Graphs into two parts (forward + backward as one and optimizer as another) for better performance ([#14656](https://github.com/Lightning-AI/lightning/pull/14656))
- Fixed torchscript error with ensembles of LightningModules ([#14657](https://github.com/Lightning-AI/lightning/pull/14657), [#14724](https://github.com/Lightning-AI/lightning/pull/14724))
- Fixed an issue with `TensorBoardLogger.finalize` creating a new experiment when none was created during the Trainer's execution ([#14762](https://github.com/Lightning-AI/lightning/pull/14762))
- Fixed `TypeError` on import when `torch.distributed` is not available ([#14809](https://github.com/Lightning-AI/lightning/pull/14809))


## [1.7.6] - 2022-09-13

### Changed

- Improved the error messaging when passing `Trainer.method(model, x_dataloader=None)` with no module-method implementations available ([#14614](https://github.com/Lightning-AI/lightning/pull/14614))

### Fixed

- Reset the dataloaders on OOM failure in batch size finder to use the last successful batch size ([#14372](https://github.com/Lightning-AI/lightning/pull/14372))
- Fixed an issue to keep downscaling the batch size in case there hasn't been even a single successful optimal batch size with `mode="power"` ([#14372](https://github.com/Lightning-AI/lightning/pull/14372))
- Fixed an issue where `self.log`-ing a tensor would create a user warning from PyTorch about cloning tensors ([#14599](https://github.com/Lightning-AI/lightning/pull/14599))
- Fixed compatibility when `torch.distributed` is not available ([#14454](https://github.com/Lightning-AI/lightning/pull/14454))


## [1.7.5] - 2022-09-06

### Fixed

- Squeezed tensor values when logging with `LightningModule.log` ([#14489](https://github.com/Lightning-AI/lightning/pull/14489))
- Fixed `WandbLogger` `save_dir` is not set after creation ([#14326](https://github.com/Lightning-AI/lightning/pull/14326))
- Fixed `Trainer.estimated_stepping_batches` when maximum number of epochs is not set ([#14317](https://github.com/Lightning-AI/lightning/pull/14317))


## [1.7.4] - 2022-08-31

### Added

- Added an environment variable `PL_DISABLE_FORK` that can be used to disable all forking in the Trainer ([#14319](https://github.com/Lightning-AI/lightning/pull/14319))

### Fixed

- Fixed `LightningDataModule` hparams parsing ([#12806](https://github.com/Lightning-AI/lightning/pull/12806))
- Reset epoch progress with batch size scaler ([#13846](https://github.com/Lightning-AI/lightning/pull/13846))
- Fixed restoring the trainer after using `lr_find()` so that the correct LR schedule is used for the actual training ([#14113](https://github.com/Lightning-AI/lightning/pull/14113))
- Fixed incorrect values after transferring data to an MPS device ([#14368](https://github.com/Lightning-AI/lightning/pull/14368))


## [1.7.3] - 2022-08-25

### Fixed

- Fixed an assertion error when using a `ReduceOnPlateau` scheduler with the Horovod strategy ([#14215](https://github.com/Lightning-AI/lightning/pull/14215))
- Fixed an `AttributeError` when accessing `LightningModule.logger` and the Trainer has multiple loggers ([#14234](https://github.com/Lightning-AI/lightning/pull/14234))
- Added back support for `log`ging in the `configure_gradient_clipping` hook after unintended removal in v1.7.2 ([#14298](https://github.com/Lightning-AI/lightning/pull/14298))
- Fixed wrong num padding for `RichProgressBar` ([#14296](https://github.com/Lightning-AI/lightning/pull/14296))
- Fixed an issue to avoid the impact of sanity check on `reload_dataloaders_every_n_epochs` for validation ([#13964](https://github.com/Lightning-AI/lightning/pull/13964))


## [1.7.2] - 2022-08-17

### Added

- Added `FullyShardedNativeNativeMixedPrecisionPlugin` to handle precision for `DDPFullyShardedNativeStrategy` ([#14092](https://github.com/Lightning-AI/lightning/pull/14092))
- Added profiling to these hooks: `on_before_batch_transfer`, `transfer_batch_to_device`, `on_after_batch_transfer`, `configure_gradient_clipping`, `clip_gradients` ([#14069](https://github.com/Lightning-AI/lightning/pull/14069))

### Changed

- The `WandbLogger.name` property no longer returns the name of the experiment, and instead returns the project's name ([#14145](https://github.com/Lightning-AI/lightning/pull/14145))
- The default project name in `WandbLogger` is now "lightning_logs" ([#14145](https://github.com/Lightning-AI/lightning/pull/14145))
- Updated compatibility for LightningLite to run with the latest DeepSpeed 0.7.0 ([13967](https://github.com/Lightning-AI/lightning/pull/13967))

### Fixed

- Fixed a bug that caused spurious `AttributeError` when multiple `DataLoader` classes are imported ([#14117](https://github.com/Lightning-AI/lightning/pull/14117))
- Fixed epoch-end logging results not being reset after the end of the epoch ([#14061](https://github.com/Lightning-AI/lightning/pull/14061))
- Fixed resuming from a checkpoint when using Stochastic Weight Averaging (SWA) ([#9938](https://github.com/Lightning-AI/lightning/pull/9938))
- Fixed the device placement when `LightningModule.cuda()` gets called without specifying a device index and the current cuda device was not 0 ([#14128](https://github.com/Lightning-AI/lightning/pull/14128))
- Avoided false positive warning about using `sync_dist` when using torchmetrics ([#14143](https://github.com/Lightning-AI/lightning/pull/14143))
- Avoid `metadata.entry_points` deprecation warning on Python 3.10 ([#14052](https://github.com/Lightning-AI/lightning/pull/14052))
- Fixed epoch-end logging results not being reset after the end of the epoch ([#14061](https://github.com/Lightning-AI/lightning/pull/14061))
- Avoid raising the sampler warning if num_replicas=1 ([#14097](https://github.com/Lightning-AI/lightning/pull/14097))
- Fixed saving hyperparameters in a composition where the parent class is not a `LightningModule` or `LightningDataModule` ([#14151](https://github.com/Lightning-AI/lightning/pull/14151))
- Avoided requiring the FairScale package to use precision with the fsdp native strategy ([#14092](https://github.com/Lightning-AI/lightning/pull/14092))
- Fixed an issue in which the default name for a run in `WandbLogger` would be set to the project name instead of a randomly generated string ([#14145](https://github.com/Lightning-AI/lightning/pull/14145))
- Fixed not preserving set attributes on `DataLoader` and `BatchSampler` when instantiated inside `*_dataloader` hooks ([#14212](https://github.com/Lightning-AI/lightning/pull/14212))


## [1.7.1] - 2022-08-09

### Fixed

- Casted only floating point tensors to fp16 with IPUs ([#13983](https://github.com/Lightning-AI/lightning/pull/13983))
- Casted tensors to fp16 before moving them to device with  `DeepSpeedStrategy` ([#14000](https://github.com/Lightning-AI/lightning/pull/14000))
- Fixed the `NeptuneLogger` dependency being unrecognized ([#13988](https://github.com/Lightning-AI/lightning/pull/13988))
- Fixed an issue where users would be warned about unset `max_epochs` even when `fast_dev_run` was set ([#13262](https://github.com/Lightning-AI/lightning/pull/13262))
- Fixed MPS device being unrecognized ([#13992](https://github.com/Lightning-AI/lightning/pull/13992))
- Fixed incorrect `precision="mixed"` being used with `DeepSpeedStrategy` and `IPUStrategy` ([#14041](https://github.com/Lightning-AI/lightning/pull/14041))
- Fixed dtype inference during gradient norm computation ([#14051](https://github.com/Lightning-AI/lightning/pull/14051))
- Fixed a bug that caused `ddp_find_unused_parameters` to be set `False`, whereas the intended default is `True` ([#14095](https://github.com/Lightning-AI/lightning/pull/14095))


## [1.7.0] - 2022-08-02

### Added

-  Added ``ServableModule`` and its associated callback called ``ServableModuleValidator`` to ensure the model can served ([#13614](https://github.com/Lightning-AI/lightning/pull/13614))
-  Converted validation loop config warnings to `PossibleUserWarning` ([#13377](https://github.com/Lightning-AI/lightning/pull/13377))
- Added a flag named `log_rank_zero_only` to `EarlyStopping` to disable logging to non-zero rank processes ([#13233](https://github.com/Lightning-AI/lightning/pull/13233))
- Added support for reloading the last checkpoint saved by passing `ckpt_path="last"` ([#12816](https://github.com/Lightning-AI/lightning/pull/12816))
- Added `LightningDataModule.load_from_checkpoint` to support loading datamodules directly from checkpoint ([#12550](https://github.com/Lightning-AI/lightning/pull/12550))
- Added a friendly error message when attempting to call `Trainer.save_checkpoint()` without a model attached ([#12772](https://github.com/Lightning-AI/lightning/pull/12772))
- Added a friendly error message when attempting to use `DeepSpeedStrategy` on unsupported accelerators ([#12699](https://github.com/Lightning-AI/lightning/pull/12699))
- Enabled `torch.inference_mode` for evaluation and prediction ([#12715](https://github.com/Lightning-AI/lightning/pull/12715))
- Added support for setting `val_check_interval` to a value higher than the amount of training batches when `check_val_every_n_epoch=None` ([#11993](https://github.com/Lightning-AI/lightning/pull/11993))
- Include the `pytorch_lightning` version as a header in the CLI config files ([#12532](https://github.com/Lightning-AI/lightning/pull/12532))
- Added support for `Callback` registration through entry points ([#12739](https://github.com/Lightning-AI/lightning/pull/12739))
- Added support for `Trainer(deterministic="warn")` to warn instead of fail when a non-deterministic operation is encountered ([#12588](https://github.com/Lightning-AI/lightning/pull/12588))
- Added profiling to the loops' dataloader `__next__` calls ([#12124](https://github.com/Lightning-AI/lightning/pull/12124))
- Hivemind Strategy
    * Added `CollaborativeStrategy` ([#12842](https://github.com/Lightning-AI/lightning/pull/12842))
    * Renamed `CollaborativeStrategy` to `HivemindStrategy` ([#13388](https://github.com/Lightning-AI/lightning/pull/13388))
    * Removed unnecessary endpoint logic, renamed `collaborative` to `hivemind` ([#13392](https://github.com/Lightning-AI/lightning/pull/13392))
- Include a version suffix for new "last" checkpoints of later runs in the same directory ([#12902](https://github.com/Lightning-AI/lightning/pull/12902))
- Show a better error message when a Metric that does not return a Tensor is logged ([#13164](https://github.com/Lightning-AI/lightning/pull/13164))
- Added missing `predict_dataset` argument in `LightningDataModule.from_datasets` to create predict dataloaders ([#12942](https://github.com/Lightning-AI/lightning/pull/12942))
- Added class name prefix to metrics logged by `DeviceStatsMonitor` ([#12228](https://github.com/Lightning-AI/lightning/pull/12228))
- Automatically wrap custom samplers under a distributed environment by using `DistributedSamplerWrapper` ([#12959](https://github.com/Lightning-AI/lightning/pull/12959))
- Added profiling of `LightningDataModule` hooks ([#12971](https://github.com/Lightning-AI/lightning/pull/12971))
- Added Native FSDP Strategy ([#12447](https://github.com/Lightning-AI/lightning/pull/12447))
- Added breaking of lazy graph across training, validation, test and predict steps when training with habana accelerators to ensure better performance ([#12938](https://github.com/Lightning-AI/lightning/pull/12938))
- Added `Checkpoint` class to inherit from ([#13024](https://github.com/Lightning-AI/lightning/pull/13024))
- Added CPU metric tracking to `DeviceStatsMonitor` ([#11795](https://github.com/Lightning-AI/lightning/pull/11795))
- Added `teardown()` method to `Accelerator` ([#11935](https://github.com/Lightning-AI/lightning/pull/11935))
- Added support for using custom Trainers that don't include callbacks using the CLI ([#13138](https://github.com/Lightning-AI/lightning/pull/13138))
- Added a `timeout` argument to `DDPStrategy` and `DDPSpawnStrategy`. ([#13244](https://github.com/Lightning-AI/lightning/pull/13244), [#13383](https://github.com/Lightning-AI/lightning/pull/13383))
- Added `XLAEnvironment` cluster environment plugin ([#11330](https://github.com/Lightning-AI/lightning/pull/11330))
- Added logging messages to notify when `FitLoop` stopping conditions are met ([#9749](https://github.com/Lightning-AI/lightning/pull/9749))
- Added support for calling unknown methods with `DummyLogger` ([#13224](https://github.com/Lightning-AI/lightning/pull/13224)
- Added support for recursively setting the `Trainer` reference for ensembles of `LightningModule`s ([#13638](https://github.com/Lightning-AI/lightning/pull/13638)
- Added Apple Silicon Support via `MPSAccelerator` ([#13123](https://github.com/Lightning-AI/lightning/pull/13123))
- Added support for DDP Fork ([#13405](https://github.com/Lightning-AI/lightning/pull/13405))
- Added support for async checkpointing ([#13658](https://github.com/Lightning-AI/lightning/pull/13658))
- Added support for HPU Device stats monitor ([#13819](https://github.com/Lightning-AI/lightning/pull/13819))

### Changed

- `accelerator="gpu"` now automatically selects an available GPU backend (CUDA and MPS currently) ([#13642](https://github.com/Lightning-AI/lightning/pull/13642))
- Enable validation during overfitting ([#12527](https://github.com/Lightning-AI/lightning/pull/12527))
- Added dataclass support to `extract_batch_size` ([#12573](https://github.com/Lightning-AI/lightning/pull/12573))
- Changed checkpoints save path in the case of one logger and user-provided weights_save_path from `weights_save_path/name/version/checkpoints` to `weights_save_path/checkpoints` ([#12372](https://github.com/Lightning-AI/lightning/pull/12372))
- Changed checkpoints save path in the case of multiple loggers and user-provided weights_save_path from `weights_save_path/name1_name2/version1_version2/checkpoints` to `weights_save_path/checkpoints` ([#12372](https://github.com/Lightning-AI/lightning/pull/12372))
- Marked `swa_lrs` argument in `StochasticWeightAveraging` callback as required ([#12556](https://github.com/Lightning-AI/lightning/pull/12556))
- `LightningCLI`'s shorthand notation changed to use jsonargparse native feature ([#12614](https://github.com/Lightning-AI/lightning/pull/12614))
- `LightningCLI` changed to use jsonargparse native support for list append ([#13129](https://github.com/Lightning-AI/lightning/pull/13129))
- Changed `seed_everything_default` argument in the `LightningCLI` to type `Union[bool, int]`. If set to `True` a seed is automatically generated for the parser argument `--seed_everything`. ([#12822](https://github.com/Lightning-AI/lightning/pull/12822), [#13110](https://github.com/Lightning-AI/lightning/pull/13110))
- Make positional arguments required for classes passed into the `add_argparse_args` function. ([#12504](https://github.com/Lightning-AI/lightning/pull/12504))
- Raise an error if there are insufficient training batches when using a float value of `limit_train_batches` ([#12885](https://github.com/Lightning-AI/lightning/pull/12885))
- `DataLoader` instantiated inside a `*_dataloader` hook will not set the passed arguments as attributes anymore ([#12981](https://github.com/Lightning-AI/lightning/pull/12981))
- When a multi-element tensor is logged, an error is now raised instead of silently taking the mean of all elements ([#13164](https://github.com/Lightning-AI/lightning/pull/13164))
- The `WandbLogger` will now use the run name in the logs folder if it is provided, and otherwise the project name  ([#12604](https://github.com/Lightning-AI/lightning/pull/12604))
- Enabled using any Sampler in distributed environment in Lite ([#13646](https://github.com/Lightning-AI/lightning/pull/13646))
- Raised a warning instead of forcing `sync_dist=True` on epoch end ([13364](https://github.com/Lightning-AI/lightning/pull/13364))
- Updated `val_check_interval`(int) to consider total train batches processed instead of `_batches_that_stepped` for validation check during training ([#12832](https://github.com/Lightning-AI/lightning/pull/12832)
- Updated Habana Accelerator's `auto_device_count`, `is_available` & `get_device_name` methods based on the latest torch habana package ([#13423](https://github.com/Lightning-AI/lightning/pull/13423))
- Disallowed using `BatchSampler` when running on multiple IPUs ([#13854](https://github.com/Lightning-AI/lightning/pull/13854))

### Deprecated

- Deprecated `pl.accelerators.gpu.GPUAccelerator` in favor of `pl.accelerators.cuda.CUDAAccelerator` ([#13636](https://github.com/Lightning-AI/lightning/pull/13636))
- Deprecated `pl.loggers.base.LightningLoggerBase` in favor of `pl.loggers.logger.Logger`, and deprecated `pl.loggers.base` in favor of `pl.loggers.logger` ([#120148](https://github.com/Lightning-AI/lightning/pull/12014))
- Deprecated `pl.callbacks.base.Callback` in favor of `pl.callbacks.callback.Callback` ([#13031](https://github.com/Lightning-AI/lightning/pull/13031))
- Deprecated `num_processes`, `gpus`, `tpu_cores,` and `ipus` from the `Trainer` constructor in favor of using the `accelerator` and `devices` arguments ([#11040](https://github.com/Lightning-AI/lightning/pull/11040))
- Deprecated setting `LightningCLI(seed_everything_default=None)` in favor of `False` ([#12804](https://github.com/Lightning-AI/lightning/pull/12804)).
- Deprecated `pl.core.lightning.LightningModule` in favor of `pl.core.module.LightningModule` ([#12740](https://github.com/Lightning-AI/lightning/pull/12740))
- Deprecated `pl.loops.base.Loop` in favor of `pl.loops.loop.Loop` ([#13043](https://github.com/Lightning-AI/lightning/pull/13043))
- Deprecated `Trainer.reset_train_val_dataloaders()` in favor of `Trainer.reset_{train,val}_dataloader` ([#12184](https://github.com/Lightning-AI/lightning/pull/12184))
- Deprecated LightningCLI's registries in favor of importing the respective package ([#13221](https://github.com/Lightning-AI/lightning/pull/13221))
- Deprecated public utilities in `pl.utilities.cli.LightningCLI` in favor of equivalent copies in `pl.cli.LightningCLI` ([#13767](https://github.com/Lightning-AI/lightning/pull/13767))
- Deprecated `pl.profiler.*` in favor of `pl.profilers` ([#12308](https://github.com/Lightning-AI/lightning/pull/12308))

### Removed

- Removed deprecated `IndexBatchSamplerWrapper.batch_indices` ([#13565](https://github.com/Lightning-AI/lightning/pull/13565))
- Removed the deprecated `LightningModule.add_to_queue` and `LightningModule.get_from_queue` method ([#13600](https://github.com/Lightning-AI/lightning/pull/13600))
- Removed deprecated `pl.core.decorators.parameter_validation` from `decorators` ([#13514](https://github.com/Lightning-AI/lightning/pull/13514))
- Removed the deprecated `Logger.close` method ([#13149](https://github.com/Lightning-AI/lightning/pull/13149))
- Removed the deprecated `weights_summary` argument from the `Trainer` constructor ([#13070](https://github.com/Lightning-AI/lightning/pull/13070))
- Removed the deprecated `flush_logs_every_n_steps` argument from the `Trainer` constructor ([#13074](https://github.com/Lightning-AI/lightning/pull/13074))
- Removed the deprecated `process_position` argument from the `Trainer` constructor ([13071](https://github.com/Lightning-AI/lightning/pull/13071))
- Removed the deprecated `checkpoint_callback` argument from the `Trainer` constructor ([#13027](https://github.com/Lightning-AI/lightning/pull/13027))
- Removed the deprecated `on_{train,val,test,predict}_dataloader` hooks from the `LightningModule` and `LightningDataModule` ([#13033](https://github.com/Lightning-AI/lightning/pull/13033))
- Removed the deprecated `TestTubeLogger` ([#12859](https://github.com/Lightning-AI/lightning/pull/12859))
- Removed the deprecated `pl.core.memory.LayerSummary` and `pl.core.memory.ModelSummary` ([#12593](https://github.com/Lightning-AI/lightning/pull/12593))
- Removed the deprecated `summarize` method from the `LightningModule` ([#12559](https://github.com/Lightning-AI/lightning/pull/12559))
- Removed the deprecated `model_size` property from the `LightningModule` class ([#12641](https://github.com/Lightning-AI/lightning/pull/12641))
- Removed the deprecated `stochastic_weight_avg` argument from the `Trainer` constructor ([#12535](https://github.com/Lightning-AI/lightning/pull/12535))
- Removed the deprecated `progress_bar_refresh_rate` argument from the `Trainer` constructor ([#12514](https://github.com/Lightning-AI/lightning/pull/12514))
- Removed the deprecated `prepare_data_per_node` argument from the `Trainer` constructor ([#12536](https://github.com/Lightning-AI/lightning/pull/12536))
- Removed the deprecated `pl.core.memory.{get_gpu_memory_map,get_memory_profile}` ([#12659](https://github.com/Lightning-AI/lightning/pull/12659))
- Removed the deprecated `terminate_on_nan` argument from the `Trainer` constructor ([#12553](https://github.com/Lightning-AI/lightning/pull/12553))
- Removed the deprecated `XLAStatsMonitor` callback ([#12688](https://github.com/Lightning-AI/lightning/pull/12688))
- Remove deprecated `pl.callbacks.progress.progress` ([#12658](https://github.com/Lightning-AI/lightning/pull/12658))
- Removed the deprecated `dim` and `size` arguments from the `LightningDataModule` constructor([#12780](https://github.com/Lightning-AI/lightning/pull/12780))
- Removed the deprecated `train_transforms` argument from the `LightningDataModule` constructor([#12662](https://github.com/Lightning-AI/lightning/pull/12662))
- Removed the deprecated `log_gpu_memory` argument from the `Trainer` constructor ([#12657](https://github.com/Lightning-AI/lightning/pull/12657))
- Removed the deprecated automatic logging of GPU stats by the logger connector ([#12657](https://github.com/Lightning-AI/lightning/pull/12657))
- Removed deprecated `GPUStatsMonitor` callback ([#12554](https://github.com/Lightning-AI/lightning/pull/12554))
- Removed support for passing strategy names or strategy instances to the accelerator Trainer argument ([#12696](https://github.com/Lightning-AI/lightning/pull/12696))
- Removed support for passing strategy names or strategy instances to the plugins Trainer argument ([#12700](https://github.com/Lightning-AI/lightning/pull/12700))
- Removed the deprecated `val_transforms` argument from the `LightningDataModule` constructor ([#12763](https://github.com/Lightning-AI/lightning/pull/12763))
- Removed the deprecated `test_transforms` argument from the `LightningDataModule` constructor ([#12773](https://github.com/Lightning-AI/lightning/pull/12773))
- Removed deprecated `Trainer(max_steps=None)` ([#13591](https://github.com/Lightning-AI/lightning/pull/13591))
- Removed deprecated `dataloader_idx` argument from `on_train_batch_start/end` hooks `Callback` and `LightningModule` ([#12769](https://github.com/Lightning-AI/lightning/pull/12769), [#12977](https://github.com/Lightning-AI/lightning/pull/12977))
- Removed deprecated `get_progress_bar_dict` property from `LightningModule` ([#12839](https://github.com/Lightning-AI/lightning/pull/12839))
- Removed sanity check for multi-optimizer support with habana backends ([#13217](https://github.com/Lightning-AI/lightning/pull/13217))
- Removed the need to explicitly load habana module ([#13338](https://github.com/Lightning-AI/lightning/pull/13338))
- Removed the deprecated `Strategy.post_dispatch()` hook ([#13461](https://github.com/Lightning-AI/lightning/pull/13461))
- Removed deprecated `pl.callbacks.lr_monitor.LearningRateMonitor.lr_sch_names` ([#13353](https://github.com/Lightning-AI/lightning/pull/13353))
- Removed deprecated `Trainer.slurm_job_id` in favor of `SLURMEnvironment.job_id` ([#13459](https://github.com/Lightning-AI/lightning/pull/13459))
- Removed support for the `DDP2Strategy` ([#12705](https://github.com/Lightning-AI/lightning/pull/12705))
- Removed deprecated `LightningDistributed` ([#13549](https://github.com/Lightning-AI/lightning/pull/13549))
- Removed deprecated ClusterEnvironment properties `master_address` and `master_port` in favor of `main_address` and `main_port` ([#13458](https://github.com/Lightning-AI/lightning/pull/13458))
- Removed deprecated ClusterEnvironment methods `KubeflowEnvironment.is_using_kubelfow()`, `LSFEnvironment.is_using_lsf()` and `TorchElasticEnvironment.is_using_torchelastic()` in favor of the `detect()` method ([#13458](https://github.com/Lightning-AI/lightning/pull/13458))
- Removed deprecated `Callback.on_keyboard_interrupt` ([#13438](https://github.com/Lightning-AI/lightning/pull/13438))
- Removed deprecated `LightningModule.on_post_move_to_device` ([#13548](https://github.com/Lightning-AI/lightning/pull/13548))
- Removed `TPUSpawnStrategy.{tpu_local_core_rank,tpu_global_core_rank}` attributes in favor of `TPUSpawnStrategy.{local_rank,global_rank}` ([#11163](https://github.com/Lightning-AI/lightning/pull/11163))
- Removed `SingleTPUStrategy.{tpu_local_core_rank,tpu_global_core_rank}` attributes in favor of `SingleTPUStrategy.{local_rank,global_rank}`([#11163](https://github.com/Lightning-AI/lightning/pull/11163))

### Fixed

- Improved support for custom `DataLoader`s when instantiated in `*_dataloader` hook ([#12981](https://github.com/Lightning-AI/lightning/pull/12981))
- Allowed custom `BatchSampler`s when instantiated in `*_dataloader` hook [#13640](https://github.com/Lightning-AI/lightning/pull/13640))
- Fixed an issue with unsupported torch.inference_mode() on hpu backends by making it use no_grad ([#13014](https://github.com/Lightning-AI/lightning/pull/13014))
- The model wrapper returned by `LightningLite.setup()` now properly supports pass-through when looking up attributes ([#12597](https://github.com/Lightning-AI/lightning/pull/12597))
- Fixed issue where the CLI fails with certain torch objects ([#13153](https://github.com/Lightning-AI/lightning/pull/13153))
- Fixed ``LightningCLI`` signature parameter resolving for some lightning classes ([#13283](https://github.com/Lightning-AI/lightning/pull/13283))
- Fixed Model Summary when using DeepSpeed Stage 3 ([#13427](https://github.com/Lightning-AI/lightning/pull/13427))
- Fixed `pl.utilities.distributed.gather_all_tensors` to handle tensors of different dimensions ([#12630](https://github.com/Lightning-AI/lightning/pull/12630))
- Fixed the input validation for the accelerator Trainer argument when passed as a string ([#13417](https://github.com/Lightning-AI/lightning/pull/13417))
- Fixed `Trainer.predict(return_predictions=False)` to track prediction's batch_indices ([#13629](https://github.com/Lightning-AI/lightning/pull/13629))
- Fixed and issue that prevented setting a custom `CheckpointIO` plugin with strategies ([#13785](https://github.com/Lightning-AI/lightning/pull/13785))
- Fixed main progress bar counter when `val_check_interval=int` and `check_val_every_n_epoch=None` ([#12832](https://github.com/Lightning-AI/lightning/pull/12832)
- Improved support for custom `ReduceLROnPlateau` scheduler if `reduce_on_plateau` is set by the user in scheduler config ([#13838](https://github.com/Lightning-AI/lightning/pull/13838))
- Used `global_step` while restoring logging step for old checkpoints ([#13645](https://github.com/Lightning-AI/lightning/pull/13645))
- When training with `precision=16` on IPU, the cast has been moved off the IPU onto the host, making the copies from host to IPU cheaper ([#13880](https://github.com/Lightning-AI/lightning/pull/13880))
- Fixed error handling in learning rate finder when not enough data points are available to give a good suggestion ([#13845](https://github.com/Lightning-AI/lightning/pull/13845))
- Fixed an issue that caused the learning rate finder to set the model's learning rate to None when no suggestion was possible ([#13845](https://github.com/Lightning-AI/lightning/pull/13845))
- Fixed an issue causing deterministic algorithms and other globals to get reset in spawned processes ([#13921](https://github.com/Lightning-AI/lightning/pull/13921))
- Fixed default `amp_level` for `DeepSpeedPrecisionPlugin` to `O2` ([#13897](https://github.com/Lightning-AI/lightning/pull/13897))
- Fixed Python 3.10 compatibility for truncated back-propagation through time (TBPTT) ([#13973](https://github.com/Lightning-AI/lightning/pull/13973))
- Fixed `TQDMProgressBar` reset and update to show correct time estimation (2/2) ([#13962](https://github.com/Lightning-AI/lightning/pull/13962))


## [1.6.5] - 2022-07-13

### Fixed

- Fixed `estimated_stepping_batches` requiring distributed comms in `configure_optimizers` for the `DeepSpeedStrategy` ([#13350](https://github.com/Lightning-AI/lightning/pull/13350))
- Fixed bug with Python version check that prevented use with development versions of Python ([#13420](https://github.com/Lightning-AI/lightning/pull/13420))
- The loops now call `.set_epoch()` also on batch samplers if the dataloader has one wrapped in a distributed sampler ([#13396](https://github.com/Lightning-AI/lightning/pull/13396))
- Fixed the restoration of log step during restart ([#13467](https://github.com/Lightning-AI/lightning/pull/13467))


## [1.6.4] - 2022-06-01

### Added

- Added all DDP params to be exposed through hpu parallel strategy ([#13067](https://github.com/Lightning-AI/lightning/pull/13067))

### Changed

- Keep `torch.backends.cudnn.benchmark=False` by default (unlike in v1.6.{0-3}) after speed and memory problems depending on the data used. Please consider tuning `Trainer(benchmark)` manually. ([#13154](https://github.com/Lightning-AI/lightning/pull/13154))
- Prevent modification of `torch.backends.cudnn.benchmark` when `Trainer(benchmark=...)` is not set ([#13154](https://github.com/Lightning-AI/lightning/pull/13154))

### Fixed

- Fixed an issue causing zero-division error for empty dataloaders ([#12885](https://github.com/Lightning-AI/lightning/pull/12885))
- Fixed mismatching default values for the types of some arguments in the DeepSpeed and Fully-Sharded strategies which made the CLI unable to use them ([#12989](https://github.com/Lightning-AI/lightning/pull/12989))
- Avoid redundant callback restore warning while tuning ([#13026](https://github.com/Lightning-AI/lightning/pull/13026))
- Fixed `Trainer(precision=64)` during evaluation which now uses the wrapped precision module ([#12983](https://github.com/Lightning-AI/lightning/pull/12983))
- Fixed an issue to use wrapped `LightningModule` for evaluation during `trainer.fit` for `BaguaStrategy` ([#12983](https://github.com/Lightning-AI/lightning/pull/12983))
- Fixed an issue wrt unnecessary usage of habana mixed precision package for fp32 types ([#13028](https://github.com/Lightning-AI/lightning/pull/13028))
- Fixed the number of references of `LightningModule` so it can be deleted ([#12897](https://github.com/Lightning-AI/lightning/pull/12897))
- Fixed `materialize_module` setting a module's child recursively ([#12870](https://github.com/Lightning-AI/lightning/pull/12870))
- Fixed issue where the CLI could not pass a `Profiler` to the `Trainer` ([#13084](https://github.com/Lightning-AI/lightning/pull/13084))
- Fixed torchelastic detection with non-distributed installations ([#13142](https://github.com/Lightning-AI/lightning/pull/13142))
- Fixed logging's step values when multiple dataloaders are used during evaluation ([#12184](https://github.com/Lightning-AI/lightning/pull/12184))
- Fixed epoch logging on train epoch end ([#13025](https://github.com/Lightning-AI/lightning/pull/13025))
- Fixed `DDPStrategy` and `DDPSpawnStrategy` to initialize optimizers only after moving the module to the device ([#11952](https://github.com/Lightning-AI/lightning/pull/11952))


## [1.6.3] - 2022-05-03

### Fixed

- Use only a single instance of `rich.console.Console` throughout codebase ([#12886](https://github.com/Lightning-AI/lightning/pull/12886))
- Fixed an issue to ensure all the checkpoint states are saved in a common filepath with `DeepspeedStrategy` ([#12887](https://github.com/Lightning-AI/lightning/pull/12887))
- Fixed `trainer.logger` deprecation message ([#12671](https://github.com/Lightning-AI/lightning/pull/12671))
- Fixed an issue where sharded grad scaler is passed in when using BF16 with the `ShardedStrategy` ([#12915](https://github.com/Lightning-AI/lightning/pull/12915))
- Fixed an issue wrt recursive invocation of DDP configuration in hpu parallel plugin ([#12912](https://github.com/Lightning-AI/lightning/pull/12912))
- Fixed printing of ragged dictionaries in `Trainer.validate` and `Trainer.test` ([#12857](https://github.com/Lightning-AI/lightning/pull/12857))
- Fixed threading support for legacy loading of checkpoints ([#12814](https://github.com/Lightning-AI/lightning/pull/12814))
- Fixed pickling of `KFoldLoop` ([#12441](https://github.com/Lightning-AI/lightning/pull/12441))
- Stopped `optimizer_zero_grad` from being called after IPU execution ([#12913](https://github.com/Lightning-AI/lightning/pull/12913))
- Fixed `fuse_modules` to be qat-aware for `torch>=1.11` ([#12891](https://github.com/Lightning-AI/lightning/pull/12891))
- Enforced eval shuffle warning only for default samplers in DataLoader ([#12653](https://github.com/Lightning-AI/lightning/pull/12653))
- Enable mixed precision in `DDPFullyShardedStrategy` when `precision=16` ([#12965](https://github.com/Lightning-AI/lightning/pull/12965))
- Fixed `TQDMProgressBar` reset and update to show correct time estimation (1/2) ([#12889](https://github.com/Lightning-AI/lightning/pull/12889))
- Fixed fit loop restart logic to enable resume using the checkpoint ([#12821](https://github.com/Lightning-AI/lightning/pull/12821))


## [1.6.2] - 2022-04-27

### Fixed

- Fixed `ImportError` when `torch.distributed` is not available. ([#12794](https://github.com/Lightning-AI/lightning/pull/12794))
- When using custom DataLoaders in LightningDataModule, multiple inheritance is resolved properly ([#12716](https://github.com/Lightning-AI/lightning/pull/12716))
- Fixed encoding issues on terminals that do not support unicode characters ([#12828](https://github.com/Lightning-AI/lightning/pull/12828))
- Fixed support for `ModelCheckpoint` monitors with dots ([#12783](https://github.com/Lightning-AI/lightning/pull/12783))


## [1.6.1] - 2022-04-13

### Changed

- Support `strategy` argument being case insensitive ([#12528](https://github.com/Lightning-AI/lightning/pull/12528))

### Fixed

- Run main progress bar updates independent of val progress bar updates in `TQDMProgressBar` ([#12563](https://github.com/Lightning-AI/lightning/pull/12563))
- Avoid calling `average_parameters` multiple times per optimizer step ([#12452](https://github.com/Lightning-AI/lightning/pull/12452))
- Properly pass some Logger's parent's arguments to `super().__init__()` ([#12609](https://github.com/Lightning-AI/lightning/pull/12609))
- Fixed an issue where incorrect type warnings appear when the overridden `LightningLite.run` method accepts user-defined arguments ([#12629](https://github.com/Lightning-AI/lightning/pull/12629))
- Fixed `rank_zero_only` decorator in LSF environments ([#12587](https://github.com/Lightning-AI/lightning/pull/12587))
- Don't raise a warning when `nn.Module` is not saved under hparams ([#12669](https://github.com/Lightning-AI/lightning/pull/12669))
- Raise `MisconfigurationException` when the accelerator is available but the user passes invalid `([]/0/"0")` values to the `devices` flag ([#12708](https://github.com/Lightning-AI/lightning/pull/12708))
- Support `auto_select_gpus` with the accelerator and devices API ([#12608](https://github.com/Lightning-AI/lightning/pull/12608))


## [1.6.0] - 2022-03-29

### Added

- Allow logging to an existing run ID in MLflow with `MLFlowLogger` ([#12290](https://github.com/Lightning-AI/lightning/pull/12290))
- Enable gradient accumulation using Horovod's `backward_passes_per_step` ([#11911](https://github.com/Lightning-AI/lightning/pull/11911))
- Add new `DETAIL` log level to provide useful logs for improving monitoring and debugging of batch jobs ([#11008](https://github.com/Lightning-AI/lightning/pull/11008))
- Added a flag `SLURMEnvironment(auto_requeue=True|False)` to control whether Lightning handles the requeuing ([#10601](https://github.com/Lightning-AI/lightning/pull/10601))
- Fault Tolerant Manual
    * Add `_Stateful` protocol to detect if classes are stateful ([#10646](https://github.com/Lightning-AI/lightning/pull/10646))
    * Add `_FaultTolerantMode` enum used to track different supported fault tolerant modes ([#10645](https://github.com/Lightning-AI/lightning/pull/10645))
    * Add a `_rotate_worker_indices` utility to reload the state according the latest worker ([#10647](https://github.com/Lightning-AI/lightning/pull/10647))
    * Add stateful workers ([#10674](https://github.com/Lightning-AI/lightning/pull/10674))
    * Add an utility to collect the states across processes ([#10639](https://github.com/Lightning-AI/lightning/pull/10639))
    * Add logic to reload the states across data loading components ([#10699](https://github.com/Lightning-AI/lightning/pull/10699))
    * Cleanup some fault tolerant utilities ([#10703](https://github.com/Lightning-AI/lightning/pull/10703))
    * Enable Fault Tolerant Manual Training ([#10707](https://github.com/Lightning-AI/lightning/pull/10707))
    * Broadcast the `_terminate_gracefully` to all processes and add support for DDP ([#10638](https://github.com/Lightning-AI/lightning/pull/10638))
- Added support for re-instantiation of custom (subclasses of) `DataLoaders` returned in the `*_dataloader()` methods, i.e., automatic replacement of samplers now works with custom types of `DataLoader` ([#10680](https://github.com/Lightning-AI/lightning/pull/10680))
- Added a function to validate if fault tolerant training is supported. ([#10465](https://github.com/Lightning-AI/lightning/pull/10465))
- Added a private callback to manage the creation and deletion of fault-tolerance checkpoints ([#11862](https://github.com/Lightning-AI/lightning/pull/11862))
- Show a better error message when a custom `DataLoader` implementation is not well implemented and we need to reconstruct it ([#10719](https://github.com/Lightning-AI/lightning/pull/10719))
- Show a better error message when frozen dataclass is used as a batch ([#10927](https://github.com/Lightning-AI/lightning/pull/10927))
- Save the `Loop`'s state by default in the checkpoint ([#10784](https://github.com/Lightning-AI/lightning/pull/10784))
- Added `Loop.replace` to easily switch one loop for another ([#10324](https://github.com/Lightning-AI/lightning/pull/10324))
- Added support for `--lr_scheduler=ReduceLROnPlateau` to the `LightningCLI` ([#10860](https://github.com/Lightning-AI/lightning/pull/10860))
- Added `LightningCLI.configure_optimizers` to override the `configure_optimizers` return value ([#10860](https://github.com/Lightning-AI/lightning/pull/10860))
- Added `LightningCLI(auto_registry)` flag to register all subclasses of the registerable components automatically ([#12108](https://github.com/Lightning-AI/lightning/pull/12108))
- Added a warning that shows when `max_epochs` in the `Trainer` is not set ([#10700](https://github.com/Lightning-AI/lightning/pull/10700))
- Added support for returning a single Callback from `LightningModule.configure_callbacks` without wrapping it into a list ([#11060](https://github.com/Lightning-AI/lightning/pull/11060))
- Added `console_kwargs` for `RichProgressBar` to initialize inner Console ([#10875](https://github.com/Lightning-AI/lightning/pull/10875))
- Added support for shorthand notation to instantiate loggers with the `LightningCLI` ([#11533](https://github.com/Lightning-AI/lightning/pull/11533))
- Added a `LOGGER_REGISTRY` instance to register custom loggers to the `LightningCLI` ([#11533](https://github.com/Lightning-AI/lightning/pull/11533))
- Added info message when the `Trainer` arguments `limit_*_batches`, `overfit_batches`, or `val_check_interval` are set to `1` or `1.0` ([#11950](https://github.com/Lightning-AI/lightning/pull/11950))
- Added a `PrecisionPlugin.teardown` method ([#10990](https://github.com/Lightning-AI/lightning/pull/10990))
- Added `LightningModule.lr_scheduler_step` ([#10249](https://github.com/Lightning-AI/lightning/pull/10249))
- Added support for no pre-fetching to `DataFetcher` ([#11606](https://github.com/Lightning-AI/lightning/pull/11606))
- Added support for optimizer step progress tracking with manual optimization ([#11848](https://github.com/Lightning-AI/lightning/pull/11848))
- Return the output of the `optimizer.step`. This can be useful for `LightningLite` users, manual optimization users, or users overriding `LightningModule.optimizer_step` ([#11711](https://github.com/Lightning-AI/lightning/pull/11711))
- Teardown the active loop and strategy on exception ([#11620](https://github.com/Lightning-AI/lightning/pull/11620))
- Added a `MisconfigurationException` if user provided `opt_idx` in scheduler config doesn't match with actual optimizer index of its respective optimizer ([#11247](https://github.com/Lightning-AI/lightning/pull/11247))
- Added a `loggers` property to `Trainer` which returns a list of loggers provided by the user ([#11683](https://github.com/Lightning-AI/lightning/pull/11683))
- Added a `loggers` property to `LightningModule` which retrieves the `loggers` property from `Trainer` ([#11683](https://github.com/Lightning-AI/lightning/pull/11683))
- Added support for DDP when using a `CombinedLoader` for the training data ([#11648](https://github.com/Lightning-AI/lightning/pull/11648))
- Added a warning when using `DistributedSampler` during validation/testing ([#11479](https://github.com/Lightning-AI/lightning/pull/11479))
- Added support for `Bagua` training strategy ([#11146](https://github.com/Lightning-AI/lightning/pull/11146))
- Added support for manually returning a `poptorch.DataLoader` in a `*_dataloader` hook ([#12116](https://github.com/Lightning-AI/lightning/pull/12116))
- Added `rank_zero` module to centralize utilities ([#11747](https://github.com/Lightning-AI/lightning/pull/11747))
- Added a `_Stateful` support for `LightningDataModule` ([#11637](https://github.com/Lightning-AI/lightning/pull/11637))
- Added `_Stateful` support for `PrecisionPlugin` ([#11638](https://github.com/Lightning-AI/lightning/pull/11638))
- Added `Accelerator.is_available` to check device availability ([#11797](https://github.com/Lightning-AI/lightning/pull/11797))
- Enabled static type-checking on the signature of `Trainer` ([#11888](https://github.com/Lightning-AI/lightning/pull/11888))
- Added utility functions for moving optimizers to devices ([#11758](https://github.com/Lightning-AI/lightning/pull/11758))
- Added a warning when saving an instance of `nn.Module` with `save_hyperparameters()` ([#12068](https://github.com/Lightning-AI/lightning/pull/12068))
- Added `estimated_stepping_batches` property to `Trainer` ([#11599](https://github.com/Lightning-AI/lightning/pull/11599))
- Added support for pluggable Accelerators ([#12030](https://github.com/Lightning-AI/lightning/pull/12030))
- Added profiling for `on_load_checkpoint`/`on_save_checkpoint` callback and LightningModule hooks ([#12149](https://github.com/Lightning-AI/lightning/pull/12149))
- Added `LayerSync` and `NativeSyncBatchNorm` plugins ([#11754](https://github.com/Lightning-AI/lightning/pull/11754))
- Added optional `storage_options` argument to `Trainer.save_checkpoint()` to pass to custom `CheckpointIO` implementations ([#11891](https://github.com/Lightning-AI/lightning/pull/11891))
- Added support to explicitly specify the process group backend for parallel strategies ([#11745](https://github.com/Lightning-AI/lightning/pull/11745))
- Added `device_ids` and `num_devices` property to `Trainer` ([#12151](https://github.com/Lightning-AI/lightning/pull/12151))
- Added `Callback.state_dict()` and `Callback.load_state_dict()` methods ([#12232](https://github.com/Lightning-AI/lightning/pull/12232))
- Added `AcceleratorRegistry` ([#12180](https://github.com/Lightning-AI/lightning/pull/12180))
- Added support for Habana Accelerator (HPU) ([#11808](https://github.com/Lightning-AI/lightning/pull/11808))
- Added support for dataclasses in `apply_to_collections` ([#11889](https://github.com/Lightning-AI/lightning/pull/11889))

### Changed

- Drop PyTorch 1.7 support ([#12191](https://github.com/Lightning-AI/lightning/pull/12191)), ([#12432](https://github.com/Lightning-AI/lightning/pull/12432))
- Make `benchmark` flag optional and set its value based on the deterministic flag ([#11944](https://github.com/Lightning-AI/lightning/pull/11944))
- Implemented a new native and rich format in `_print_results` method of the `EvaluationLoop` ([#11332](https://github.com/Lightning-AI/lightning/pull/11332))
- Do not print an empty table at the end of the `EvaluationLoop` ([#12427](https://github.com/Lightning-AI/lightning/pull/12427))
- Set the `prog_bar` flag to False in `LightningModule.log_grad_norm` ([#11472](https://github.com/Lightning-AI/lightning/pull/11472))
- Raised exception in `init_dist_connection()` when torch distributed is not available ([#10418](https://github.com/Lightning-AI/lightning/pull/10418))
- The `monitor` argument in the `EarlyStopping` callback is no longer optional ([#10328](https://github.com/Lightning-AI/lightning/pull/10328))
- Do not fail if batch size could not be inferred for logging when using DeepSpeed ([#10438](https://github.com/Lightning-AI/lightning/pull/10438))
- Raised `MisconfigurationException` when `enable_progress_bar=False` and a progress bar instance has been passed in the callback list ([#10520](https://github.com/Lightning-AI/lightning/pull/10520))
- Moved `trainer.connectors.env_vars_connector._defaults_from_env_vars` to `utilities.argsparse._defaults_from_env_vars` ([#10501](https://github.com/Lightning-AI/lightning/pull/10501))
- Changes in `LightningCLI` required for the new major release of jsonargparse v4.0.0 ([#10426](https://github.com/Lightning-AI/lightning/pull/10426))
- Renamed `refresh_rate_per_second` parameter to `refresh_rate` for `RichProgressBar` signature ([#10497](https://github.com/Lightning-AI/lightning/pull/10497))
- Moved ownership of the `PrecisionPlugin` into `TrainingTypePlugin` and updated all references ([#10570](https://github.com/Lightning-AI/lightning/pull/10570))
- Fault Tolerant relies on `signal.SIGTERM` to gracefully exit instead of `signal.SIGUSR1` ([#10605](https://github.com/Lightning-AI/lightning/pull/10605))
- `Loop.restarting=...` now sets the value recursively for all subloops ([#11442](https://github.com/Lightning-AI/lightning/pull/11442))
- Raised an error if the `batch_size` cannot be inferred from the current batch if it contained a string or was a custom batch object ([#10541](https://github.com/Lightning-AI/lightning/pull/10541))
- The validation loop is now disabled when `overfit_batches > 0` is set in the Trainer ([#9709](https://github.com/Lightning-AI/lightning/pull/9709))
- Moved optimizer related logics from `Accelerator` to `TrainingTypePlugin` ([#10596](https://github.com/Lightning-AI/lightning/pull/10596))
- Moved ownership of the lightning optimizers from the `Trainer` to the `Strategy` ([#11444](https://github.com/Lightning-AI/lightning/pull/11444))
- Moved ownership of the data fetchers from the DataConnector to the Loops ([#11621](https://github.com/Lightning-AI/lightning/pull/11621))
- Moved `batch_to_device` method from `Accelerator` to `TrainingTypePlugin` ([#10649](https://github.com/Lightning-AI/lightning/pull/10649))
- The `DDPSpawnPlugin` no longer overrides the `post_dispatch` plugin hook ([#10034](https://github.com/Lightning-AI/lightning/pull/10034))
- Integrate the progress bar implementation with progress tracking ([#11213](https://github.com/Lightning-AI/lightning/pull/11213))
- The `LightningModule.{add_to_queue,get_from_queue}` hooks no longer get a `torch.multiprocessing.SimpleQueue` and instead receive a list based queue ([#10034](https://github.com/Lightning-AI/lightning/pull/10034))
- Changed `training_step`, `validation_step`, `test_step` and `predict_step` method signatures in `Accelerator` and updated input from caller side ([#10908](https://github.com/Lightning-AI/lightning/pull/10908))
- Changed the name of the temporary checkpoint that the `DDPSpawnPlugin` and related plugins save ([#10934](https://github.com/Lightning-AI/lightning/pull/10934))
- `LoggerCollection` returns only unique logger names and versions ([#10976](https://github.com/Lightning-AI/lightning/pull/10976))
- Redesigned process creation for spawn-based plugins (`DDPSpawnPlugin`, `TPUSpawnPlugin`, etc.) ([#10896](https://github.com/Lightning-AI/lightning/pull/10896))
    * All spawn-based plugins now spawn processes immediately upon calling `Trainer.{fit,validate,test,predict}`
    * The hooks/callbacks `prepare_data`, `setup`, `configure_sharded_model` and `teardown` now run under initialized process group for spawn-based plugins just like their non-spawn counterparts
    * Some configuration errors that were previously raised as `MisconfigurationException`s will now be raised as `ProcessRaisedException` (torch>=1.8) or as `Exception` (torch<1.8)
    * Removed the `TrainingTypePlugin.pre_dispatch()` method and merged it with `TrainingTypePlugin.setup()` ([#11137](https://github.com/Lightning-AI/lightning/pull/11137))
- Changed profiler to index and display the names of the hooks with a new pattern [<base class>]<class>.<hook name> ([#11026](https://github.com/Lightning-AI/lightning/pull/11026))
- Changed `batch_to_device` entry in profiling from stage-specific to generic, to match profiling of other hooks ([#11031](https://github.com/Lightning-AI/lightning/pull/11031))
- Changed the info message for finalizing ddp-spawn worker processes to a debug-level message ([#10864](https://github.com/Lightning-AI/lightning/pull/10864))
- Removed duplicated file extension when uploading model checkpoints with `NeptuneLogger` ([#11015](https://github.com/Lightning-AI/lightning/pull/11015))
- Removed `__getstate__` and `__setstate__` of `RichProgressBar` ([#11100](https://github.com/Lightning-AI/lightning/pull/11100))
- The `DDPPlugin` and `DDPSpawnPlugin` and their subclasses now remove the `SyncBatchNorm` wrappers in `teardown()` to enable proper support at inference after fitting ([#11078](https://github.com/Lightning-AI/lightning/pull/11078))
- Moved ownership of the `Accelerator` instance to the `TrainingTypePlugin`; all training-type plugins now take an optional parameter `accelerator` ([#11022](https://github.com/Lightning-AI/lightning/pull/11022))
- Renamed the `TrainingTypePlugin` to `Strategy` ([#11120](https://github.com/Lightning-AI/lightning/pull/11120))
    * Renamed the `ParallelPlugin` to `ParallelStrategy` ([#11123](https://github.com/Lightning-AI/lightning/pull/11123))
    * Renamed the `DataParallelPlugin` to `DataParallelStrategy` ([#11183](https://github.com/Lightning-AI/lightning/pull/11183))
    * Renamed the `DDPPlugin` to `DDPStrategy` ([#11142](https://github.com/Lightning-AI/lightning/pull/11142))
    * Renamed the `DDP2Plugin` to `DDP2Strategy` ([#11185](https://github.com/Lightning-AI/lightning/pull/11185))
    * Renamed the `DDPShardedPlugin` to `DDPShardedStrategy` ([#11186](https://github.com/Lightning-AI/lightning/pull/11186))
    * Renamed the `DDPFullyShardedPlugin` to `DDPFullyShardedStrategy` ([#11143](https://github.com/Lightning-AI/lightning/pull/11143))
    * Renamed the `DDPSpawnPlugin` to `DDPSpawnStrategy` ([#11145](https://github.com/Lightning-AI/lightning/pull/11145))
    * Renamed the `DDPSpawnShardedPlugin` to `DDPSpawnShardedStrategy` ([#11210](https://github.com/Lightning-AI/lightning/pull/11210))
    * Renamed the `DeepSpeedPlugin` to `DeepSpeedStrategy` ([#11194](https://github.com/Lightning-AI/lightning/pull/11194))
    * Renamed the `HorovodPlugin` to `HorovodStrategy` ([#11195](https://github.com/Lightning-AI/lightning/pull/11195))
    * Renamed the `TPUSpawnPlugin` to `TPUSpawnStrategy` ([#11190](https://github.com/Lightning-AI/lightning/pull/11190))
    * Renamed the `IPUPlugin` to `IPUStrategy` ([#11193](https://github.com/Lightning-AI/lightning/pull/11193))
    * Renamed the `SingleDevicePlugin` to `SingleDeviceStrategy` ([#11182](https://github.com/Lightning-AI/lightning/pull/11182))
    * Renamed the `SingleTPUPlugin` to `SingleTPUStrategy` ([#11182](https://github.com/Lightning-AI/lightning/pull/11182))
    * Renamed the `TrainingTypePluginsRegistry` to `StrategyRegistry` ([#11233](https://github.com/Lightning-AI/lightning/pull/11233))
- Marked the `ResultCollection`, `ResultMetric`, and `ResultMetricCollection` classes as protected ([#11130](https://github.com/Lightning-AI/lightning/pull/11130))
- Marked `trainer.checkpoint_connector` as protected ([#11550](https://github.com/Lightning-AI/lightning/pull/11550))
- The epoch start/end hooks are now called by the `FitLoop` instead of the `TrainingEpochLoop` ([#11201](https://github.com/Lightning-AI/lightning/pull/11201))
- DeepSpeed does not require lightning module zero 3 partitioning ([#10655](https://github.com/Lightning-AI/lightning/pull/10655))
- Moved `Strategy` classes to the `strategies` directory ([#11226](https://github.com/Lightning-AI/lightning/pull/11226))
- Renamed `training_type_plugin` file to `strategy` ([#11239](https://github.com/Lightning-AI/lightning/pull/11239))
- Changed `DeviceStatsMonitor` to group metrics based on the logger's `group_separator` ([#11254](https://github.com/Lightning-AI/lightning/pull/11254))
- Raised `UserWarning` if evaluation is triggered with `best` ckpt and trainer is configured with multiple checkpoint callbacks ([#11274](https://github.com/Lightning-AI/lightning/pull/11274))
- `Trainer.logged_metrics` now always contains scalar tensors, even when a Python scalar was logged ([#11270](https://github.com/Lightning-AI/lightning/pull/11270))
- The tuner now uses the checkpoint connector to copy and restore its state ([#11518](https://github.com/Lightning-AI/lightning/pull/11518))
- Changed `MisconfigurationException` to `ModuleNotFoundError` when `rich` isn't available ([#11360](https://github.com/Lightning-AI/lightning/pull/11360))
- The `trainer.current_epoch` value is now increased by 1 during and after `on_train_end` ([#8578](https://github.com/Lightning-AI/lightning/pull/8578))
- The `trainer.global_step` value now accounts for multiple optimizers and TBPTT splits ([#11805](https://github.com/Lightning-AI/lightning/pull/11805))
- The `trainer.global_step` value is now increased right after the `optimizer.step()` call which will impact users who access it during an intra-training validation hook ([#11805](https://github.com/Lightning-AI/lightning/pull/11805))
- The filename of checkpoints created with `ModelCheckpoint(filename='{step}')` is different compared to previous versions. A checkpoint saved after 1 step will be named `step=1.ckpt` instead of `step=0.ckpt` ([#11805](https://github.com/Lightning-AI/lightning/pull/11805))
- Inherit from `ABC` for `Accelerator`: Users need to implement `auto_device_count` ([#11521](https://github.com/Lightning-AI/lightning/pull/11521))
- Changed `parallel_devices` property in `ParallelStrategy` to be lazy initialized ([#11572](https://github.com/Lightning-AI/lightning/pull/11572))
- Updated `TQDMProgressBar` to run a separate progress bar for each eval dataloader ([#11657](https://github.com/Lightning-AI/lightning/pull/11657))
- Sorted `SimpleProfiler(extended=False)` summary based on mean duration for each hook ([#11671](https://github.com/Lightning-AI/lightning/pull/11671))
- Avoid enforcing `shuffle=False` for eval dataloaders ([#11575](https://github.com/Lightning-AI/lightning/pull/11575))
- When using DP (data-parallel), Lightning will no longer automatically reduce all tensors returned in training_step; it will only reduce the loss unless `training_step_end` is overridden ([#11594](https://github.com/Lightning-AI/lightning/pull/11594))
- When using DP (data-parallel), the `training_epoch_end` hook will no longer receive reduced outputs from `training_step` and instead get the full tensor of results from all GPUs ([#11594](https://github.com/Lightning-AI/lightning/pull/11594))
- Changed default logger name to `lightning_logs` for consistency ([#11762](https://github.com/Lightning-AI/lightning/pull/11762))
- Rewrote `accelerator_connector` ([#11448](https://github.com/Lightning-AI/lightning/pull/11448))
- When manual optimization is used with DDP, we no longer force `find_unused_parameters=True` ([#12425](https://github.com/Lightning-AI/lightning/pull/12425))
- Disable loading dataloades if corresponding `limit_batches=0` ([#11576](https://github.com/Lightning-AI/lightning/pull/11576))
- Removed `is_global_zero` check in `training_epoch_loop` before `logger.save`. If you have a custom logger that implements `save` the Trainer will now call `save` on all ranks by default. To change this behavior add `@rank_zero_only` to your `save` implementation ([#12134](https://github.com/Lightning-AI/lightning/pull/12134))
- Disabled tuner with distributed strategies ([#12179](https://github.com/Lightning-AI/lightning/pull/12179))
- Marked `trainer.logger_connector` as protected ([#12195](https://github.com/Lightning-AI/lightning/pull/12195))
- Move `Strategy.process_dataloader` function call from `fit/evaluation/predict_loop.py` to `data_connector.py` ([#12251](https://github.com/Lightning-AI/lightning/pull/12251))
- `ModelCheckpoint(save_last=True, every_n_epochs=N)` now saves a "last" checkpoint every epoch (disregarding `every_n_epochs`) instead of only once at the end of training ([#12418](https://github.com/Lightning-AI/lightning/pull/12418))
- The strategies that support `sync_batchnorm` now only apply it when fitting ([#11919](https://github.com/Lightning-AI/lightning/pull/11919))
- Avoided fallback on CPU if no devices are provided for other accelerators ([#12410](https://github.com/Lightning-AI/lightning/pull/12410))
- Modified `supporters.py` so that in the accumulator element (for loss) is created directly on the device ([#12430](https://github.com/Lightning-AI/lightning/pull/12430))
- Removed `EarlyStopping.on_save_checkpoint` and `EarlyStopping.on_load_checkpoint` in favor of `EarlyStopping.state_dict` and `EarlyStopping.load_state_dict` ([#11887](https://github.com/Lightning-AI/lightning/pull/11887))
- Removed `BaseFinetuning.on_save_checkpoint` and `BaseFinetuning.on_load_checkpoint` in favor of `BaseFinetuning.state_dict` and `BaseFinetuning.load_state_dict` ([#11887](https://github.com/Lightning-AI/lightning/pull/11887))
- Removed `BackboneFinetuning.on_save_checkpoint` and `BackboneFinetuning.on_load_checkpoint` in favor of `BackboneFinetuning.state_dict` and `BackboneFinetuning.load_state_dict` ([#11887](https://github.com/Lightning-AI/lightning/pull/11887))
- Removed `ModelCheckpoint.on_save_checkpoint` and `ModelCheckpoint.on_load_checkpoint` in favor of `ModelCheckpoint.state_dict` and `ModelCheckpoint.load_state_dict` ([#11887](https://github.com/Lightning-AI/lightning/pull/11887))
- Removed `Timer.on_save_checkpoint` and `Timer.on_load_checkpoint` in favor of `Timer.state_dict` and `Timer.load_state_dict` ([#11887](https://github.com/Lightning-AI/lightning/pull/11887))
- Replaced PostLocalSGDOptimizer with a dedicated model averaging component ([#12378](https://github.com/Lightning-AI/lightning/pull/12378))

### Deprecated

- Deprecated `training_type_plugin` property in favor of `strategy` in `Trainer` and updated the references ([#11141](https://github.com/Lightning-AI/lightning/pull/11141))
- Deprecated `Trainer.{validated,tested,predicted}_ckpt_path` and replaced with read-only property `Trainer.ckpt_path` set when checkpoints loaded via `Trainer.{fit,validate,test,predict}` ([#11696](https://github.com/Lightning-AI/lightning/pull/11696))
- Deprecated `ClusterEnvironment.master_{address,port}` in favor of `ClusterEnvironment.main_{address,port}` ([#10103](https://github.com/Lightning-AI/lightning/pull/10103))
- Deprecated `DistributedType` in favor of `_StrategyType` ([#10505](https://github.com/Lightning-AI/lightning/pull/10505))
- Deprecated the `precision_plugin` constructor argument from `Accelerator` ([#10570](https://github.com/Lightning-AI/lightning/pull/10570))
- Deprecated `DeviceType` in favor of `_AcceleratorType` ([#10503](https://github.com/Lightning-AI/lightning/pull/10503))
- Deprecated the property `Trainer.slurm_job_id` in favor of the new `SLURMEnvironment.job_id()` method ([#10622](https://github.com/Lightning-AI/lightning/pull/10622))
- Deprecated the access to the attribute `IndexBatchSamplerWrapper.batch_indices` in favor of `IndexBatchSamplerWrapper.seen_batch_indices` ([#10870](https://github.com/Lightning-AI/lightning/pull/10870))
- Deprecated `on_init_start` and `on_init_end` callback hooks ([#10940](https://github.com/Lightning-AI/lightning/pull/10940))
- Deprecated `Trainer.call_hook` in favor of `Trainer._call_callback_hooks`, `Trainer._call_lightning_module_hook`, `Trainer._call_ttp_hook`, and `Trainer._call_accelerator_hook` ([#10979](https://github.com/Lightning-AI/lightning/pull/10979))
- Deprecated `TrainingTypePlugin.post_dispatch` in favor of `TrainingTypePlugin.teardown` ([#10939](https://github.com/Lightning-AI/lightning/pull/10939))
- Deprecated `ModelIO.on_hpc_{save/load}` in favor of `CheckpointHooks.on_{save/load}_checkpoint` ([#10911](https://github.com/Lightning-AI/lightning/pull/10911))
- Deprecated `Trainer.run_stage` in favor of `Trainer.{fit,validate,test,predict}` ([#11000](https://github.com/Lightning-AI/lightning/pull/11000))
- Deprecated `Trainer.lr_schedulers` in favor of `Trainer.lr_scheduler_configs` which returns a list of dataclasses instead of dictionaries ([#11443](https://github.com/Lightning-AI/lightning/pull/11443))
- Deprecated `Trainer.verbose_evaluate` in favor of `EvaluationLoop(verbose=...)` ([#10931](https://github.com/Lightning-AI/lightning/pull/10931))
- Deprecated `Trainer.should_rank_save_checkpoint` Trainer property ([#11068](https://github.com/Lightning-AI/lightning/pull/11068))
- Deprecated `Trainer.lightning_optimizers` ([#11444](https://github.com/Lightning-AI/lightning/pull/11444))
- Deprecated `TrainerOptimizersMixin` and moved functionality to `core/optimizer.py`([#11155](https://github.com/Lightning-AI/lightning/pull/11155))
- Deprecated the `on_train_batch_end(outputs)` format when multiple optimizers are used and TBPTT is enabled ([#12182](https://github.com/Lightning-AI/lightning/pull/12182))
- Deprecated the `training_epoch_end(outputs)` format when multiple optimizers are used and TBPTT is enabled ([#12182](https://github.com/Lightning-AI/lightning/pull/12182))
- Deprecated `TrainerCallbackHookMixin` ([#11148](https://github.com/Lightning-AI/lightning/pull/11148))
- Deprecated `TrainerDataLoadingMixin` and moved functionality to `Trainer` and `DataConnector` ([#11282](https://github.com/Lightning-AI/lightning/pull/11282))
- Deprecated function `pl.callbacks.device_stats_monitor.prefix_metric_keys` ([#11254](https://github.com/Lightning-AI/lightning/pull/11254))
- Deprecated `Callback.on_epoch_start` hook in favour of `Callback.on_{train/val/test}_epoch_start` ([#11578](https://github.com/Lightning-AI/lightning/pull/11578))
- Deprecated `Callback.on_epoch_end` hook in favour of `Callback.on_{train/val/test}_epoch_end` ([#11578](https://github.com/Lightning-AI/lightning/pull/11578))
- Deprecated `LightningModule.on_epoch_start` hook in favor of `LightningModule.on_{train/val/test}_epoch_start` ([#11578](https://github.com/Lightning-AI/lightning/pull/11578))
- Deprecated `LightningModule.on_epoch_end` hook in favor of `LightningModule.on_{train/val/test}_epoch_end` ([#11578](https://github.com/Lightning-AI/lightning/pull/11578))
- Deprecated `on_before_accelerator_backend_setup` callback hook in favour of `setup` ([#11568](https://github.com/Lightning-AI/lightning/pull/11568))
- Deprecated `on_batch_start` and `on_batch_end` callback hooks in favor of `on_train_batch_start` and `on_train_batch_end` ([#11577](https://github.com/Lightning-AI/lightning/pull/11577))
- Deprecated `on_configure_sharded_model` callback hook in favor of `setup` ([#11627](https://github.com/Lightning-AI/lightning/pull/11627))
- Deprecated `pl.utilities.distributed.rank_zero_only` in favor of `pl.utilities.rank_zero.rank_zero_only` ([#11747](https://github.com/Lightning-AI/lightning/pull/11747))
- Deprecated `pl.utilities.distributed.rank_zero_debug` in favor of `pl.utilities.rank_zero.rank_zero_debug` ([#11747](https://github.com/Lightning-AI/lightning/pull/11747))
- Deprecated `pl.utilities.distributed.rank_zero_info` in favor of `pl.utilities.rank_zero.rank_zero_info` ([#11747](https://github.com/Lightning-AI/lightning/pull/11747))
- Deprecated `pl.utilities.warnings.rank_zero_warn` in favor of `pl.utilities.rank_zero.rank_zero_warn` ([#11747](https://github.com/Lightning-AI/lightning/pull/11747))
- Deprecated `pl.utilities.warnings.rank_zero_deprecation` in favor of `pl.utilities.rank_zero.rank_zero_deprecation` ([#11747](https://github.com/Lightning-AI/lightning/pull/11747))
- Deprecated `pl.utilities.warnings.LightningDeprecationWarning` in favor of `pl.utilities.rank_zero.LightningDeprecationWarning` ([#11747](https://github.com/Lightning-AI/lightning/pull/11747))
- Deprecated `on_pretrain_routine_start` and `on_pretrain_routine_end` callback hooks in favor of `on_fit_start` ([#11794](https://github.com/Lightning-AI/lightning/pull/11794))
- Deprecated `LightningModule.on_pretrain_routine_start` and `LightningModule.on_pretrain_routine_end` hooks in favor of `on_fit_start` ([#12122](https://github.com/Lightning-AI/lightning/pull/12122))
- Deprecated `agg_key_funcs` and `agg_default_func` parameters from `LightningLoggerBase` ([#11871](https://github.com/Lightning-AI/lightning/pull/11871))
- Deprecated `LightningLoggerBase.update_agg_funcs` ([#11871](https://github.com/Lightning-AI/lightning/pull/11871))
- Deprecated `LightningLoggerBase.agg_and_log_metrics` in favor of `LightningLoggerBase.log_metrics` ([#11832](https://github.com/Lightning-AI/lightning/pull/11832))
- Deprecated passing `weights_save_path` to the `Trainer` constructor in favor of adding the `ModelCheckpoint` callback with `dirpath` directly to the list of callbacks ([#12084](https://github.com/Lightning-AI/lightning/pull/12084))
- Deprecated `pl.profiler.AbstractProfiler` in favor of `pl.profiler.Profiler` ([#12106](https://github.com/Lightning-AI/lightning/pull/12106))
- Deprecated `pl.profiler.BaseProfiler` in favor of `pl.profiler.Profiler` ([#12150](https://github.com/Lightning-AI/lightning/pull/12150))
- Deprecated `BaseProfiler.profile_iterable` ([#12102](https://github.com/Lightning-AI/lightning/pull/12102))
- Deprecated `LoggerCollection` in favor of `trainer.loggers` ([#12147](https://github.com/Lightning-AI/lightning/pull/12147))
- Deprecated `PrecisionPlugin.on_{save,load}_checkpoint` in favor of `PrecisionPlugin.{state_dict,load_state_dict}` ([#11978](https://github.com/Lightning-AI/lightning/pull/11978))
- Deprecated `LightningDataModule.on_save/load_checkpoint` in favor of `state_dict/load_state_dict` ([#11893](https://github.com/Lightning-AI/lightning/pull/11893))
- Deprecated `Trainer.use_amp` in favor of `Trainer.amp_backend` ([#12312](https://github.com/Lightning-AI/lightning/pull/12312))
- Deprecated `LightningModule.use_amp` in favor of `Trainer.amp_backend` ([#12315](https://github.com/Lightning-AI/lightning/pull/12315))
- Deprecated specifying the process group backend through the environment variable `PL_TORCH_DISTRIBUTED_BACKEND` ([#11745](https://github.com/Lightning-AI/lightning/pull/11745))
- Deprecated `ParallelPlugin.torch_distributed_backend` in favor of `DDPStrategy.process_group_backend` property ([#11745](https://github.com/Lightning-AI/lightning/pull/11745))
- Deprecated `ModelCheckpoint.save_checkpoint` in favor of `Trainer.save_checkpoint` ([#12456](https://github.com/Lightning-AI/lightning/pull/12456))
- Deprecated `Trainer.devices` in favor of `Trainer.num_devices` and `Trainer.device_ids` ([#12151](https://github.com/Lightning-AI/lightning/pull/12151))
- Deprecated `Trainer.root_gpu` in favor of `Trainer.strategy.root_device.index` when GPU is used ([#12262](https://github.com/Lightning-AI/lightning/pull/12262))
- Deprecated `Trainer.num_gpus` in favor of `Trainer.num_devices` when GPU is used ([#12384](https://github.com/Lightning-AI/lightning/pull/12384))
- Deprecated `Trainer.ipus` in favor of `Trainer.num_devices` when IPU is used ([#12386](https://github.com/Lightning-AI/lightning/pull/12386))
- Deprecated `Trainer.num_processes` in favor of `Trainer.num_devices` ([#12388](https://github.com/Lightning-AI/lightning/pull/12388))
- Deprecated `Trainer.data_parallel_device_ids` in favor of `Trainer.device_ids` ([#12072](https://github.com/Lightning-AI/lightning/pull/12072))
- Deprecated returning state from `Callback.on_save_checkpoint` in favor of returning state in `Callback.state_dict` for checkpointing ([#11887](https://github.com/Lightning-AI/lightning/pull/11887))
- Deprecated passing only the callback state to `Callback.on_load_checkpoint(callback_state)` in favor of passing the callback state to `Callback.load_state_dict` and in 1.8, passing the entire checkpoint dictionary to `Callback.on_load_checkpoint(checkpoint)` ([#11887](https://github.com/Lightning-AI/lightning/pull/11887))
- Deprecated `Trainer.gpus` in favor of `Trainer.device_ids` or `Trainer.num_devices` ([#12436](https://github.com/Lightning-AI/lightning/pull/12436))
- Deprecated `Trainer.tpu_cores` in favor of `Trainer.num_devices` ([#12437](https://github.com/Lightning-AI/lightning/pull/12437))

### Removed

- Removed deprecated parameter `method` in `pl.utilities.model_helpers.is_overridden` ([#10507](https://github.com/Lightning-AI/lightning/pull/10507))
- Remove deprecated method `ClusterEnvironment.creates_children` ([#10339](https://github.com/Lightning-AI/lightning/pull/10339))
- Removed deprecated `TrainerModelHooksMixin.is_function_implemented` and `TrainerModelHooksMixin.has_arg` ([#10322](https://github.com/Lightning-AI/lightning/pull/10322))
- Removed deprecated `pl.utilities.device_dtype_mixin.DeviceDtypeModuleMixin` in favor of `pl.core.mixins.device_dtype_mixin.DeviceDtypeModuleMixin` ([#10442](https://github.com/Lightning-AI/lightning/pull/10442))
- Removed deprecated `LightningModule.loaded_optimizer_states_dict` property ([#10346](https://github.com/Lightning-AI/lightning/pull/10346))
- Removed deprecated `Trainer.fit(train_dataloader=)`, `Trainer.validate(val_dataloaders=)`, and `Trainer.test(test_dataloader=)` ([#10325](https://github.com/Lightning-AI/lightning/pull/10325))
- Removed deprecated `has_prepared_data`, `has_setup_fit`, `has_setup_validate`, `has_setup_test`, `has_setup_predict`, `has_teardown_fit`, `has_teardown_validate`, `has_teardown_test` and `has_teardown_predict` datamodule lifecycle properties  ([#10350](https://github.com/Lightning-AI/lightning/pull/10350))
- Removed deprecated `every_n_val_epochs` parameter of ModelCheckpoint ([#10366](https://github.com/Lightning-AI/lightning/pull/10366))
- Removed deprecated `import pl.profiler.profilers` in favor of `import pl.profiler` ([#10443](https://github.com/Lightning-AI/lightning/pull/10443))
- Removed deprecated property `configure_slurm_dpp` from accelerator connector ([#10370](https://github.com/Lightning-AI/lightning/pull/10370))
- Removed deprecated arguments `num_nodes` and `sync_batchnorm` from `DDPPlugin`, `DDPSpawnPlugin`, `DeepSpeedPlugin` ([#10357](https://github.com/Lightning-AI/lightning/pull/10357))
- Removed deprecated property `is_slurm_managing_tasks` from AcceleratorConnector ([#10353](https://github.com/Lightning-AI/lightning/pull/10353))
- Removed deprecated `LightningModule.log(tbptt_reduce_fx, tbptt_reduce_token, sync_dist_op)` ([#10423](https://github.com/Lightning-AI/lightning/pull/10423))
- Removed deprecated `Plugin.task_idx` ([#10441](https://github.com/Lightning-AI/lightning/pull/10441))
- Removed deprecated method `master_params` from PrecisionPlugin ([#10372](https://github.com/Lightning-AI/lightning/pull/10372))
- Removed the automatic detachment of "extras" returned from `training_step`. For example, `return {'loss': ..., 'foo': foo.detach()}` will now be necessary if `foo` has gradients which you do not want to store ([#10424](https://github.com/Lightning-AI/lightning/pull/10424))
- Removed deprecated passthrough methods and properties from `Accelerator` base class:
  * ([#10403](https://github.com/Lightning-AI/lightning/pull/10403))
  * ([#10448](https://github.com/Lightning-AI/lightning/pull/10448))
- Removed deprecated signature for `transfer_batch_to_device` hook. The new argument `dataloader_idx` is now required ([#10480](https://github.com/Lightning-AI/lightning/pull/10480))
- Removed deprecated `utilities.distributed.rank_zero_{warn/deprecation}` ([#10451](https://github.com/Lightning-AI/lightning/pull/10451))
- Removed deprecated `mode` argument from `ModelSummary` class ([#10449](https://github.com/Lightning-AI/lightning/pull/10449))
- Removed deprecated `Trainer.train_loop` property in favor of `Trainer.fit_loop` ([#10482](https://github.com/Lightning-AI/lightning/pull/10482))
- Removed deprecated `Trainer.train_loop` property in favor of `Trainer.fit_loop` ([#10482](https://github.com/Lightning-AI/lightning/pull/10482))
- Removed deprecated `disable_validation` property from Trainer ([#10450](https://github.com/Lightning-AI/lightning/pull/10450))
- Removed deprecated `CheckpointConnector.hpc_load` property in favor of `CheckpointConnector.restore` ([#10525](https://github.com/Lightning-AI/lightning/pull/10525))
- Removed deprecated `reload_dataloaders_every_epoch` from `Trainer` in favour of `reload_dataloaders_every_n_epochs` ([#10481](https://github.com/Lightning-AI/lightning/pull/10481))
- Removed the `precision_plugin` attribute from `Accelerator` in favor of its equivalent attribute `precision_plugin` in the `TrainingTypePlugin` ([#10570](https://github.com/Lightning-AI/lightning/pull/10570))
- Removed `DeepSpeedPlugin.{precision,amp_type,amp_level}` properties ([#10657](https://github.com/Lightning-AI/lightning/pull/10657))
- Removed patching of `on_before_batch_transfer`, `transfer_batch_to_device` and `on_after_batch_transfer` hooks in `LightningModule` ([#10603](https://github.com/Lightning-AI/lightning/pull/10603))
- Removed argument `return_result` from the `DDPSpawnPlugin.spawn()` method ([#10867](https://github.com/Lightning-AI/lightning/pull/10867))
- Removed the property `TrainingTypePlugin.results` and corresponding properties in subclasses ([#10034](https://github.com/Lightning-AI/lightning/pull/10034))
- Removed the `mp_queue` attribute from `DDPSpawnPlugin` and `TPUSpawnPlugin` ([#10034](https://github.com/Lightning-AI/lightning/pull/10034))
- Removed unnecessary `_move_optimizer_state` method overrides from `TPUSpawnPlugin` and `SingleTPUPlugin` ([#10849](https://github.com/Lightning-AI/lightning/pull/10849))
- Removed `should_rank_save_checkpoint` property from `TrainingTypePlugin` ([#11070](https://github.com/Lightning-AI/lightning/pull/11070))
- Removed `model_sharded_context` method from `Accelerator` ([#10886](https://github.com/Lightning-AI/lightning/pull/10886))
- Removed method `pre_dispatch` from the `PrecisionPlugin` ([#10887](https://github.com/Lightning-AI/lightning/pull/10887))
- Removed method `setup_optimizers_in_pre_dispatch` from the `strategies` and achieve the same logic in `setup` and `pre_dispatch` methods ([#10906](https://github.com/Lightning-AI/lightning/pull/10906))
- Removed methods `pre_dispatch`, `dispatch` and `post_dispatch` from the `Accelerator` ([#10885](https://github.com/Lightning-AI/lightning/pull/10885))
- Removed method `training_step`, `test_step`, `validation_step` and `predict_step` from the `Accelerator` ([#10890](https://github.com/Lightning-AI/lightning/pull/10890))
- Removed `TrainingTypePlugin.start_{training,evaluating,predicting}` hooks and the same in all subclasses ([#10989](https://github.com/Lightning-AI/lightning/pull/10989), [#10896](https://github.com/Lightning-AI/lightning/pull/10896))
- Removed `Accelerator.on_train_start` ([#10999](https://github.com/Lightning-AI/lightning/pull/10999))
- Removed support for Python 3.6 ([#11117](https://github.com/Lightning-AI/lightning/pull/11117))
- Removed `Strategy.init_optimizers` in favor of `Strategy.setup_optimizers` ([#11236](https://github.com/Lightning-AI/lightning/pull/11236))
- Removed `profile("training_step_and_backward")` in `Closure` class since we already profile calls `training_step` and `backward` ([#11222](https://github.com/Lightning-AI/lightning/pull/11222))
- Removed `Strategy.optimizer_zero_grad` ([#11246](https://github.com/Lightning-AI/lightning/pull/11246))
- Removed `Strategy.on_gpu` ([#11537](https://github.com/Lightning-AI/lightning/pull/11537))
- Removed `Strategy.on_tpu` property ([#11536](https://github.com/Lightning-AI/lightning/pull/11536))
- Removed the abstract property `LightningLoggerBase.experiment` ([#11603](https://github.com/Lightning-AI/lightning/pull/11603))
- Removed `FitLoop.current_epoch` getter and setter ([#11562](https://github.com/Lightning-AI/lightning/pull/11562))
- Removed access to `_short_id` in `NeptuneLogger` ([#11517](https://github.com/Lightning-AI/lightning/pull/11517))
- Removed `log_text` and `log_image` from the `LightningLoggerBase` API ([#11857](https://github.com/Lightning-AI/lightning/pull/11857))
- Removed calls to `profile("model_forward")` in favor of profiling `training_step` ([#12032](https://github.com/Lightning-AI/lightning/pull/12032))
- Removed `get_mp_spawn_kwargs` from `DDPSpawnStrategy` and `TPUSpawnStrategy` in favor of configuration in the `_SpawnLauncher` ([#11966](https://github.com/Lightning-AI/lightning/pull/11966))
- Removed `_aggregate_metrics`, `_reduce_agg_metrics`, and `_finalize_agg_metrics` from `LightningLoggerBase` ([#12053](https://github.com/Lightning-AI/lightning/pull/12053))
- Removed the `AcceleratorConnector.device_type` property ([#12081](https://github.com/Lightning-AI/lightning/pull/12081))
- Removed `AcceleratorConnector.num_nodes` ([#12107](https://github.com/Lightning-AI/lightning/pull/12107))
- Removed `AcceleratorConnector.has_ipu` property ([#12111](https://github.com/Lightning-AI/lightning/pull/12111))
- Removed `AcceleratorConnector.use_ipu` property ([#12110](https://github.com/Lightning-AI/lightning/pull/12110))
- Removed `AcceleratorConnector.has_tpu` property ([#12109](https://github.com/Lightning-AI/lightning/pull/12109))
- Removed `AcceleratorConnector.use_dp` property ([#12112](https://github.com/Lightning-AI/lightning/pull/12112))
- Removed `configure_sync_batchnorm` from `ParallelStrategy` and all other strategies that inherit from it ([#11754](https://github.com/Lightning-AI/lightning/pull/11754))
- Removed public attribute `sync_batchnorm` from strategies ([#11754](https://github.com/Lightning-AI/lightning/pull/11754))
- Removed `AcceleratorConnector.root_gpu` property ([#12262](https://github.com/Lightning-AI/lightning/pull/12262))
- Removed `AcceleratorConnector.tpu_id` property ([#12387](https://github.com/Lightning-AI/lightning/pull/12387))
- Removed `AcceleratorConnector.num_gpus` property ([#12384](https://github.com/Lightning-AI/lightning/pull/12384))
- Removed `AcceleratorConnector.num_ipus` property ([#12386](https://github.com/Lightning-AI/lightning/pull/12386))
- Removed `AcceleratorConnector.num_processes` property ([#12388](https://github.com/Lightning-AI/lightning/pull/12388))
- Removed `AcceleratorConnector.parallel_device_ids` property ([#12072](https://github.com/Lightning-AI/lightning/pull/12072))
- Removed `AcceleratorConnector.devices` property ([#12435](https://github.com/Lightning-AI/lightning/pull/12435))
- Removed `AcceleratorConnector.parallel_devices` property ([#12075](https://github.com/Lightning-AI/lightning/pull/12075))
- Removed `AcceleratorConnector.tpu_cores` property ([#12437](https://github.com/Lightning-AI/lightning/pull/12437))

### Fixed

- Fixed an issue where `ModelCheckpoint` could delete last checkpoint from the old directory when `dirpath` has changed during resumed training ([#12225](https://github.com/Lightning-AI/lightning/pull/12225))
- Fixed an issue where `ModelCheckpoint` could delete older checkpoints when `dirpath` has changed during resumed training ([#12045](https://github.com/Lightning-AI/lightning/pull/12045))
- Fixed an issue where `HorovodStrategy.teardown()` did not complete gracefully if an exception was thrown during callback setup [#11752](https://github.com/Lightning-AI/lightning/pull/11752)
- Fixed security vulnerabilities CVE-2020-1747 and CVE-2020-14343 caused by the `PyYAML` dependency ([#11099](https://github.com/Lightning-AI/lightning/pull/11099))
- Fixed security vulnerability "CWE-94: Improper Control of Generation of Code (Code Injection)" ([#12212](https://github.com/Lightning-AI/lightning/pull/12212))
- Fixed logging on `{test,validation}_epoch_end` with multiple dataloaders ([#11132](https://github.com/Lightning-AI/lightning/pull/11132))
- Reset the validation progress tracking state after sanity checking ([#11218](https://github.com/Lightning-AI/lightning/pull/11218))
- Fixed double evaluation bug with fault-tolerance enabled where the second call was completely skipped ([#11119](https://github.com/Lightning-AI/lightning/pull/11119))
- Fixed an issue with the `TPUSpawnPlugin` handling the `XLA_USE_BF16` environment variable incorrectly ([#10990](https://github.com/Lightning-AI/lightning/pull/10990))
- Fixed wrong typehint for `Trainer.lightning_optimizers` ([#11155](https://github.com/Lightning-AI/lightning/pull/11155))
- Fixed the lr-scheduler state not being dumped to checkpoint when using the deepspeed strategy ([#11307](https://github.com/Lightning-AI/lightning/pull/11307))
- Fixed bug that forced overriding `configure_optimizers` with the CLI ([#11672](https://github.com/Lightning-AI/lightning/pull/11672))
- Fixed type promotion when tensors of higher category than float are logged ([#11401](https://github.com/Lightning-AI/lightning/pull/11401))
- Fixed `SimpleProfiler` summary ([#11414](https://github.com/Lightning-AI/lightning/pull/11414))
- No longer set a `DistributedSampler` to the `poptorch.DataLoader` when IPUs are used ([#12114](https://github.com/Lightning-AI/lightning/pull/12114))
- Fixed bug where progress bar was not being disabled when not in rank zero during predict ([#11377](https://github.com/Lightning-AI/lightning/pull/11377))
- Fixed the mid-epoch warning call while resuming training ([#11556](https://github.com/Lightning-AI/lightning/pull/11556))
- Fixed `LightningModule.{un,}toggle_model` when only 1 optimizer is used ([#12088](https://github.com/Lightning-AI/lightning/pull/12088))
- Fixed an issue in `RichProgressbar` to display the metrics logged only on main progress bar ([#11690](https://github.com/Lightning-AI/lightning/pull/11690))
- Fixed `RichProgressBar` progress when refresh rate does not evenly divide the total counter ([#11668](https://github.com/Lightning-AI/lightning/pull/11668))
- Fixed `RichProgressBar` progress validation bar total when using multiple validation runs within a single training epoch ([#11668](https://github.com/Lightning-AI/lightning/pull/11668))
- Configure native Deepspeed schedulers with interval='step' ([#11788](https://github.com/Lightning-AI/lightning/pull/11788)), ([#12031](https://github.com/Lightning-AI/lightning/pull/12031))
- Update `RichProgressBarTheme` styles after detecting light theme on colab ([#10993](https://github.com/Lightning-AI/lightning/pull/10993))
- Fixed passing `_ddp_params_and_buffers_to_ignore` ([#11949](https://github.com/Lightning-AI/lightning/pull/11949))
- Fixed an `AttributeError` when calling `save_hyperparameters` and no parameters need saving ([#11827](https://github.com/Lightning-AI/lightning/pull/11827))
- Fixed environment variable priority for global rank determination ([#11406](https://github.com/Lightning-AI/lightning/pull/11406))
- Fixed an issue that caused the Trainer to produce identical results on subsequent runs without explicit re-seeding ([#11870](https://github.com/Lightning-AI/lightning/pull/11870))
- Fixed an issue that caused the Tuner to affect the random state ([#11870](https://github.com/Lightning-AI/lightning/pull/11870))
- Fixed to avoid common hook warning if no hook is overridden ([#12131](https://github.com/Lightning-AI/lightning/pull/12131))
- Fixed deepspeed keeping old sub-folders in same ckpt path ([#12194](https://github.com/Lightning-AI/lightning/pull/12194))
- Fixed returning logged metrics instead of callback metrics during evaluation ([#12224](https://github.com/Lightning-AI/lightning/pull/12224))
- Fixed the case where `logger=None` is passed to the Trainer ([#12249](https://github.com/Lightning-AI/lightning/pull/12249))
- Fixed bug where the global step tracked by `ModelCheckpoint` was still set even if no checkpoint was saved ([#12418](https://github.com/Lightning-AI/lightning/pull/12418))
- Fixed bug where `ModelCheckpoint` was overriding the `epoch` and `step` logged values ([#12418](https://github.com/Lightning-AI/lightning/pull/12418))
- Fixed bug where monitoring the default `epoch` and `step` values with `ModelCheckpoint` would fail ([#12418](https://github.com/Lightning-AI/lightning/pull/12418))
- Fixed initializing optimizers unnecessarily in `DDPFullyShardedStrategy` ([#12267](https://github.com/Lightning-AI/lightning/pull/12267))
- Fixed check for horovod module ([#12377](https://github.com/Lightning-AI/lightning/pull/12377))
- Fixed logging to loggers with multiple eval dataloaders ([#12454](https://github.com/Lightning-AI/lightning/pull/12454))
- Fixed an issue with resuming from a checkpoint trained with QAT ([#11346](https://github.com/Lightning-AI/lightning/pull/11346))


## [1.5.10] - 2022-02-08

### Fixed

- Fixed an issue to avoid validation loop run on restart ([#11552](https://github.com/Lightning-AI/lightning/pull/11552))
- The `RichProgressBar` now correctly shows the `on_epoch` logged values on train epoch end ([#11689](https://github.com/Lightning-AI/lightning/pull/11689))
- Fixed an issue to make the `step` argument in `WandbLogger.log_image` work ([#11716](https://github.com/Lightning-AI/lightning/pull/11716))
- Fixed `restore_optimizers` for mapping states ([#11757](https://github.com/Lightning-AI/lightning/pull/11757))
- With `DPStrategy`, the batch is not explicitly moved to the device ([#11780](https://github.com/Lightning-AI/lightning/pull/11780))
- Fixed an issue to avoid val bar disappear after `trainer.validate()` ([#11700](https://github.com/Lightning-AI/lightning/pull/11700))
- Fixed supporting remote filesystems with `Trainer.weights_save_path` for fault-tolerant training ([#11776](https://github.com/Lightning-AI/lightning/pull/11776))
- Fixed check for available modules ([#11526](https://github.com/Lightning-AI/lightning/pull/11526))
- Fixed bug where the path for "last" checkpoints was not getting saved correctly which caused newer runs to not remove the previous "last" checkpoint ([#11481](https://github.com/Lightning-AI/lightning/pull/11481))
- Fixed bug where the path for best checkpoints was not getting saved correctly when no metric was monitored which caused newer runs to not use the best checkpoint ([#11481](https://github.com/Lightning-AI/lightning/pull/11481))


## [1.5.9] - 2022-01-20

### Fixed

- Pinned sphinx-autodoc-typehints with <v1.15 ([#11400](https://github.com/Lightning-AI/lightning/pull/11400))
- Skipped testing with PyTorch 1.7 and Python 3.9 on Ubuntu ([#11217](https://github.com/Lightning-AI/lightning/pull/11217))
- Fixed type promotion when tensors of higher category than float are logged ([#11401](https://github.com/Lightning-AI/lightning/pull/11401))
- Fixed the format of the configuration saved automatically by the CLI's `SaveConfigCallback` ([#11532](https://github.com/Lightning-AI/lightning/pull/11532))

### Changed
- Changed `LSFEnvironment` to use `LSB_DJOB_RANKFILE` environment variable instead of `LSB_HOSTS` for determining node rank and main address ([#10825](https://github.com/Lightning-AI/lightning/pull/10825))
- Disabled sampler replacement when using `IterableDataset` ([#11507](https://github.com/Lightning-AI/lightning/pull/11507))


## [1.5.8] - 2022-01-05

### Fixed

- Fixed `LightningCLI` race condition while saving the config ([#11199](https://github.com/Lightning-AI/lightning/pull/11199))
- Fixed the default value used with `log(reduce_fx=min|max)` ([#11310](https://github.com/Lightning-AI/lightning/pull/11310))
- Fixed data fetcher selection ([#11294](https://github.com/Lightning-AI/lightning/pull/11294))
- Fixed a race condition that could result in incorrect (zero) values being observed in prediction writer callbacks ([#11288](https://github.com/Lightning-AI/lightning/pull/11288))
- Fixed dataloaders not getting reloaded the correct amount of times when setting `reload_dataloaders_every_n_epochs` and `check_val_every_n_epoch` ([#10948](https://github.com/Lightning-AI/lightning/pull/10948))
- Fixed deepspeed strategy not restoring the lr-scheduler states when lr-scheduler(s) are configured through `LightningModule.configure_optimizer` ([#11322](https://github.com/Lightning-AI/lightning/pull/11322))


## [1.5.7] - 2021-12-21

### Fixed

- Fixed `NeptuneLogger` when using DDP ([#11030](https://github.com/Lightning-AI/lightning/pull/11030))
- Fixed a bug to disable logging hyperparameters in logger if there are no hparams ([#11105](https://github.com/Lightning-AI/lightning/pull/11105))
- Avoid the deprecated `onnx.export(example_outputs=...)` in torch 1.10 ([#11116](https://github.com/Lightning-AI/lightning/pull/11116))
- Fixed an issue when torch-scripting a `LightningModule` after training with `Trainer(sync_batchnorm=True)` ([#11078](https://github.com/Lightning-AI/lightning/pull/11078))
- Fixed an `AttributeError` occurring when using a `CombinedLoader` (multiple dataloaders) for prediction ([#11111](https://github.com/Lightning-AI/lightning/pull/11111))
- Fixed bug where `Trainer(track_grad_norm=..., logger=False)` would fail ([#11114](https://github.com/Lightning-AI/lightning/pull/11114))
- Fixed an incorrect warning being produced by the model summary when using `bf16` precision on CPU ([#11161](https://github.com/Lightning-AI/lightning/pull/11161))

### Changed

- DeepSpeed does not require lightning module zero 3 partitioning ([#10655](https://github.com/Lightning-AI/lightning/pull/10655))
- The `ModelCheckpoint` callback now saves and restores attributes `best_k_models`, `kth_best_model_path`, `kth_value`, and `last_model_path` ([#10995](https://github.com/Lightning-AI/lightning/pull/10995))


## [1.5.6] - 2021-12-15

### Fixed

- Fixed a bug where the DeepSpeedPlugin arguments `cpu_checkpointing` and `contiguous_memory_optimization` were not being forwarded to deepspeed correctly ([#10874](https://github.com/Lightning-AI/lightning/pull/10874))
- Fixed an issue with `NeptuneLogger` causing checkpoints to be uploaded with a duplicated file extension ([#11015](https://github.com/Lightning-AI/lightning/pull/11015))
- Fixed support for logging within callbacks returned from `LightningModule` ([#10991](https://github.com/Lightning-AI/lightning/pull/10991))
- Fixed running sanity check with `RichProgressBar` ([#10913](https://github.com/Lightning-AI/lightning/pull/10913))
- Fixed support for `CombinedLoader` while checking for warning raised with eval dataloaders ([#10994](https://github.com/Lightning-AI/lightning/pull/10994))
- The TQDM progress bar now correctly shows the `on_epoch` logged values on train epoch end ([#11069](https://github.com/Lightning-AI/lightning/pull/11069))
- Fixed bug where the TQDM updated the training progress bar during `trainer.validate` ([#11069](https://github.com/Lightning-AI/lightning/pull/11069))


## [1.5.5] - 2021-12-07

### Fixed

- Disabled batch_size extraction for torchmetric instances because they accumulate the metrics internally ([#10815](https://github.com/Lightning-AI/lightning/pull/10815))
- Fixed an issue with `SignalConnector` not restoring the default signal handlers on teardown when running on SLURM or with fault-tolerant training enabled ([#10611](https://github.com/Lightning-AI/lightning/pull/10611))
- Fixed `SignalConnector._has_already_handler` check for callable type ([#10483](https://github.com/Lightning-AI/lightning/pull/10483))
- Fixed an issue to return the results for each dataloader separately instead of duplicating them for each ([#10810](https://github.com/Lightning-AI/lightning/pull/10810))
- Improved exception message if `rich` version is less than `10.2.2` ([#10839](https://github.com/Lightning-AI/lightning/pull/10839))
- Fixed uploading best model checkpoint in NeptuneLogger ([#10369](https://github.com/Lightning-AI/lightning/pull/10369))
- Fixed early schedule reset logic in PyTorch profiler that was causing data leak ([#10837](https://github.com/Lightning-AI/lightning/pull/10837))
- Fixed a bug that caused incorrect batch indices to be passed to the `BasePredictionWriter` hooks when using a dataloader with `num_workers > 0` ([#10870](https://github.com/Lightning-AI/lightning/pull/10870))
- Fixed an issue with item assignment on the logger on rank > 0 for those who support it ([#10917](https://github.com/Lightning-AI/lightning/pull/10917))
- Fixed importing `torch_xla.debug` for `torch-xla<1.8` ([#10836](https://github.com/Lightning-AI/lightning/pull/10836))
- Fixed an issue with `DDPSpawnPlugin` and related plugins leaving a temporary checkpoint behind ([#10934](https://github.com/Lightning-AI/lightning/pull/10934))
- Fixed a `TypeError` occurring in the `SingalConnector.teardown()` method ([#10961](https://github.com/Lightning-AI/lightning/pull/10961))


## [1.5.4] - 2021-11-30

### Fixed

- Fixed support for `--key.help=class` with the `LightningCLI` ([#10767](https://github.com/Lightning-AI/lightning/pull/10767))
- Fixed `_compare_version` for python packages ([#10762](https://github.com/Lightning-AI/lightning/pull/10762))
- Fixed TensorBoardLogger `SummaryWriter` not close before spawning the processes ([#10777](https://github.com/Lightning-AI/lightning/pull/10777))
- Fixed a consolidation error in Lite when attempting to save the state dict of a sharded optimizer ([#10746](https://github.com/Lightning-AI/lightning/pull/10746))
- Fixed the default logging level for batch hooks associated with training from `on_step=False, on_epoch=True` to `on_step=True, on_epoch=False` ([#10756](https://github.com/Lightning-AI/lightning/pull/10756))

### Removed

- Removed PyTorch 1.6 support ([#10367](https://github.com/Lightning-AI/lightning/pull/10367), [#10738](https://github.com/Lightning-AI/lightning/pull/10738))


## [1.5.3] - 2021-11-24

### Fixed

- Fixed `ShardedTensor` state dict hook registration to check if torch distributed is available ([#10621](https://github.com/Lightning-AI/lightning/pull/10621))
- Fixed an issue with `self.log` not respecting a tensor's `dtype` when applying computations ([#10076](https://github.com/Lightning-AI/lightning/pull/10076))
- Fixed LigtningLite `_wrap_init` popping unexisting keys from DataLoader signature parameters ([#10613](https://github.com/Lightning-AI/lightning/pull/10613))
- Fixed signals being registered within threads ([#10610](https://github.com/Lightning-AI/lightning/pull/10610))
- Fixed an issue that caused Lightning to extract the batch size even though it was set by the user in `LightningModule.log` ([#10408](https://github.com/Lightning-AI/lightning/pull/10408))
- Fixed `Trainer(move_metrics_to_cpu=True)` not moving the evaluation logged results to CPU ([#10631](https://github.com/Lightning-AI/lightning/pull/10631))
- Fixed the `{validation,test}_step` outputs getting moved to CPU with `Trainer(move_metrics_to_cpu=True)` ([#10631](https://github.com/Lightning-AI/lightning/pull/10631))
- Fixed an issue with collecting logged test results with multiple dataloaders ([#10522](https://github.com/Lightning-AI/lightning/pull/10522))


## [1.5.2] - 2021-11-16

### Fixed

- Fixed `CombinedLoader` and `max_size_cycle` didn't receive a `DistributedSampler` ([#10374](https://github.com/Lightning-AI/lightning/pull/10374))
- Fixed an issue where class or init-only variables of dataclasses were passed to the dataclass constructor in `utilities.apply_to_collection` ([#9702](https://github.com/Lightning-AI/lightning/pull/9702))
- Fixed `isinstance` not working with `init_meta_context`, materialized model not being moved to the device ([#10493](https://github.com/Lightning-AI/lightning/pull/10493))
- Fixed an issue that prevented the Trainer to shutdown workers when execution is interrupted due to failure([#10463](https://github.com/Lightning-AI/lightning/pull/10463))
- Squeeze the early stopping monitor to remove empty tensor dimensions ([#10461](https://github.com/Lightning-AI/lightning/pull/10461))
- Fixed sampler replacement logic with `overfit_batches` to only replace the sample when `SequentialSampler` is not used ([#10486](https://github.com/Lightning-AI/lightning/pull/10486))
- Fixed scripting causing false positive deprecation warnings ([#10470](https://github.com/Lightning-AI/lightning/pull/10470), [#10555](https://github.com/Lightning-AI/lightning/pull/10555))
- Do not fail if batch size could not be inferred for logging when using DeepSpeed ([#10438](https://github.com/Lightning-AI/lightning/pull/10438))
- Fixed propagation of device and dtype information to submodules of LightningLite when they inherit from `DeviceDtypeModuleMixin` ([#10559](https://github.com/Lightning-AI/lightning/pull/10559))


## [1.5.1] - 2021-11-09

### Fixed

- Fixed `apply_to_collection(defaultdict)` ([#10316](https://github.com/Lightning-AI/lightning/pull/10316))
- Fixed failure when `DataLoader(batch_size=None)` is passed ([#10345](https://github.com/Lightning-AI/lightning/pull/10345))
- Fixed interception of `__init__` arguments for sub-classed DataLoader re-instantiation in Lite ([#10334](https://github.com/Lightning-AI/lightning/pull/10334))
- Fixed issue with pickling `CSVLogger` after a call to `CSVLogger.save` ([#10388](https://github.com/Lightning-AI/lightning/pull/10388))
- Fixed an import error being caused by `PostLocalSGD` when `torch.distributed` not available ([#10359](https://github.com/Lightning-AI/lightning/pull/10359))
- Fixed the logging with `on_step=True` in epoch-level hooks causing unintended side-effects. Logging with `on_step=True` in epoch-level hooks will now correctly raise an error ([#10409](https://github.com/Lightning-AI/lightning/pull/10409))
- Fixed deadlocks for distributed training with `RichProgressBar` ([#10428](https://github.com/Lightning-AI/lightning/pull/10428))
- Fixed an issue where the model wrapper in Lite converted non-floating point tensors to float ([#10429](https://github.com/Lightning-AI/lightning/pull/10429))
- Fixed an issue with inferring the dataset type in fault-tolerant training ([#10432](https://github.com/Lightning-AI/lightning/pull/10432))
- Fixed dataloader workers with `persistent_workers` being deleted on every iteration ([#10434](https://github.com/Lightning-AI/lightning/pull/10434))


## [1.5.0] - 2021-11-02

### Added

- Added support for monitoring the learning rate without schedulers in `LearningRateMonitor` ([#9786](https://github.com/Lightning-AI/lightning/pull/9786))
- Added registration of `ShardedTensor` state dict hooks in `LightningModule.__init__` if the PyTorch version supports `ShardedTensor` ([#8944](https://github.com/Lightning-AI/lightning/pull/8944))
- Added error handling including calling of `on_keyboard_interrupt()` and `on_exception()` for all entrypoints (fit, validate, test, predict) ([#8819](https://github.com/Lightning-AI/lightning/pull/8819))
- Added a flavor of `training_step` that takes `dataloader_iter` as an argument ([#8807](https://github.com/Lightning-AI/lightning/pull/8807))
- Added a `state_key` property to the `Callback` base class ([#6886](https://github.com/Lightning-AI/lightning/pull/6886))
- Added progress tracking to loops:
    * Integrated `TrainingEpochLoop.total_batch_idx` ([#8598](https://github.com/Lightning-AI/lightning/pull/8598))
    * Added `BatchProgress` and integrated `TrainingEpochLoop.is_last_batch` ([#9657](https://github.com/Lightning-AI/lightning/pull/9657))
    * Avoid optional `Tracker` attributes ([#9320](https://github.com/Lightning-AI/lightning/pull/9320))
    * Reset `current` progress counters when restarting an epoch loop that had already finished ([#9371](https://github.com/Lightning-AI/lightning/pull/9371))
    * Call `reset_on_restart` in the loop's `reset` hook instead of when loading a checkpoint ([#9561](https://github.com/Lightning-AI/lightning/pull/9561))
    * Use `completed` over `processed` in `reset_on_restart` ([#9656](https://github.com/Lightning-AI/lightning/pull/9656))
    * Renamed `reset_on_epoch` to `reset_on_run` ([#9658](https://github.com/Lightning-AI/lightning/pull/9658))
- Added `batch_size` and `rank_zero_only` arguments for `log_dict` to match `log` ([#8628](https://github.com/Lightning-AI/lightning/pull/8628))
- Added a check for unique GPU ids ([#8666](https://github.com/Lightning-AI/lightning/pull/8666))
- Added `ResultCollection` state_dict to the Loop `state_dict` and added support for distributed reload ([#8641](https://github.com/Lightning-AI/lightning/pull/8641))
- Added DeepSpeed collate checkpoint utility function ([#8701](https://github.com/Lightning-AI/lightning/pull/8701))
- Added a `handles_accumulate_grad_batches` property to the training type plugins ([#8856](https://github.com/Lightning-AI/lightning/pull/8856))
- Added a warning to `WandbLogger` when reusing a wandb run ([#8714](https://github.com/Lightning-AI/lightning/pull/8714))
- Added `log_graph` argument for `watch` method of `WandbLogger` ([#8662](https://github.com/Lightning-AI/lightning/pull/8662))
- `LightningCLI` additions:
  * Added `LightningCLI(run=False|True)` to choose whether to run a `Trainer` subcommand ([#8751](https://github.com/Lightning-AI/lightning/pull/8751))
  * Added support to call any trainer function from the `LightningCLI` via subcommands ([#7508](https://github.com/Lightning-AI/lightning/pull/7508))
  * Allow easy trainer re-instantiation ([#7508](https://github.com/Lightning-AI/lightning/pull/9241))
  * Automatically register all optimizers and learning rate schedulers ([#9565](https://github.com/Lightning-AI/lightning/pull/9565))
  * Allow registering custom optimizers and learning rate schedulers without subclassing the CLI ([#9565](https://github.com/Lightning-AI/lightning/pull/9565))
  * Support shorthand notation to instantiate optimizers and learning rate schedulers ([#9565](https://github.com/Lightning-AI/lightning/pull/9565))
  * Support passing lists of callbacks via command line ([#8815](https://github.com/Lightning-AI/lightning/pull/8815))
  * Support shorthand notation to instantiate models ([#9588](https://github.com/Lightning-AI/lightning/pull/9588))
  * Support shorthand notation to instantiate datamodules ([#10011](https://github.com/Lightning-AI/lightning/pull/10011))
  * Added `multifile` option to `LightningCLI` to enable/disable config saving to preserve multiple files structure ([#9073](https://github.com/Lightning-AI/lightning/pull/9073))
- Fault-tolerant training:
    * Added `FastForwardSampler` and `CaptureIterableDataset` injection to data loading utilities ([#8366](https://github.com/Lightning-AI/lightning/pull/8366))
    * Added `DataFetcher` to control fetching flow ([#8890](https://github.com/Lightning-AI/lightning/pull/8890))
    * Added `SharedCycleIteratorState` to prevent infinite loop ([#8889](https://github.com/Lightning-AI/lightning/pull/8889))
    * Added `CaptureMapDataset` for state management in map-style datasets ([#8891](https://github.com/Lightning-AI/lightning/pull/8891))
    * Added Fault Tolerant Training to `DataFetcher` ([#8891](https://github.com/Lightning-AI/lightning/pull/8891))
    * Replaced old prefetch iterator with new `DataFetcher` in training loop ([#8953](https://github.com/Lightning-AI/lightning/pull/8953))
    * Added partial support for global random state fault-tolerance in map-style datasets ([#8950](https://github.com/Lightning-AI/lightning/pull/8950))
    * Converted state to tuple explicitly when setting Python random state ([#9401](https://github.com/Lightning-AI/lightning/pull/9401))
    * Added support for restarting an optimizer loop (multiple optimizers) ([#9537](https://github.com/Lightning-AI/lightning/pull/9537))
    * Added support for restarting within Evaluation Loop ([#9563](https://github.com/Lightning-AI/lightning/pull/9563))
    * Added mechanism to detect that a signal has been sent so the Trainer can gracefully exit ([#9566](https://github.com/Lightning-AI/lightning/pull/9566))
    * Added support for skipping ahead to validation during the auto-restart of fitting ([#9681](https://github.com/Lightning-AI/lightning/pull/9681))
    * Added support for auto-restart if a fault-tolerant checkpoint is available ([#9722](https://github.com/Lightning-AI/lightning/pull/9722))
- Checkpoint saving and loading extensibility:
  * Added `CheckpointIO` plugin to expose checkpoint IO from training type plugin ([#8743](https://github.com/Lightning-AI/lightning/pull/8743))
  * Refactored `CheckpointConnector` to offload validation logic to the `CheckpointIO` plugin ([#9045](https://github.com/Lightning-AI/lightning/pull/9045))
  * Added `remove_checkpoint` to `CheckpointIO` plugin by moving the responsibility out of the `ModelCheckpoint` callback ([#9373](https://github.com/Lightning-AI/lightning/pull/9373))
  * Added `XLACheckpointIO` plugin ([#9972](https://github.com/Lightning-AI/lightning/pull/9972))
- Loop customization:
    * Added `Closure` and `AbstractClosure` classes ([#8642](https://github.com/Lightning-AI/lightning/pull/8642))
    * Refactored `TrainingBatchLoop` and extracted `OptimizerLoop`, splitting off automatic optimization into its own loop ([#9191](https://github.com/Lightning-AI/lightning/pull/9191))
    * Removed `TrainingBatchLoop.backward()`; manual optimization now calls directly into `Accelerator.backward()` and automatic optimization handles backward in new `OptimizerLoop` ([#9265](https://github.com/Lightning-AI/lightning/pull/9265))
    * Extracted `ManualOptimization` logic from `TrainingBatchLoop` into its own separate loop class ([#9266](https://github.com/Lightning-AI/lightning/pull/9266))
    * Added `OutputResult` and `ManualResult` classes ([#9437](https://github.com/Lightning-AI/lightning/pull/9437), [#9424](https://github.com/Lightning-AI/lightning/pull/9424))
    * Marked `OptimizerLoop.backward` as protected ([#9514](https://github.com/Lightning-AI/lightning/pull/9514))
    * Marked `FitLoop.should_accumulate` as protected ([#9515](https://github.com/Lightning-AI/lightning/pull/9515))
    * Marked several methods in `PredictionLoop` as protected: `on_predict_start`, `on_predict_epoch_end`, `on_predict_end`, `on_predict_model_eval` ([#9516](https://github.com/Lightning-AI/lightning/pull/9516))
    * Marked several methods in `EvaluationLoop` as protected: `get_max_batches`, `on_evaluation_model_eval`, `on_evaluation_model_train`, `on_evaluation_start`, `on_evaluation_epoch_start`, `on_evaluation_epoch_end`, `on_evaluation_end`, `reload_evaluation_dataloaders` ([#9516](https://github.com/Lightning-AI/lightning/pull/9516))
    * Marked several methods in `EvaluationEpochLoop` as protected: `on_evaluation_batch_start`, `evaluation_step`, `evaluation_step_end` ([#9516](https://github.com/Lightning-AI/lightning/pull/9516))
    * Added `yielding_training_step` example ([#9983](https://github.com/Lightning-AI/lightning/pull/9983))
- Added support for saving and loading state of multiple callbacks of the same type ([#7187](https://github.com/Lightning-AI/lightning/pull/7187))
- Added DeepSpeed Stage 1 support ([#8974](https://github.com/Lightning-AI/lightning/pull/8974))
- Added `Python dataclass` support for `LightningDataModule` ([#8272](https://github.com/Lightning-AI/lightning/pull/8272))
- Added sanitization of tensors when they get logged as hyperparameters in `TensorBoardLogger` ([#9031](https://github.com/Lightning-AI/lightning/pull/9031))
- Added `InterBatchParallelDataFetcher` ([#9020](https://github.com/Lightning-AI/lightning/pull/9020))
- Added `DataLoaderIterDataFetcher` ([#9020](https://github.com/Lightning-AI/lightning/pull/9020))
- Added `DataFetcher` within `Fit / Evaluation` Loop  ([#9047](https://github.com/Lightning-AI/lightning/pull/9047))
- Added a friendly error message when DDP attempts to spawn new distributed processes with rank > 0 ([#9005](https://github.com/Lightning-AI/lightning/pull/9005))
- Added Rich integration:
    * Added Rich progress bar ([#8929](https://github.com/Lightning-AI/lightning/pull/8929), [#9559](https://github.com/Lightning-AI/lightning/pull/9559))
    * Added Support for iterable datasets ([#9734](https://github.com/Lightning-AI/lightning/pull/9734))
    * Added `RichModelSummary` callback ([#9546](https://github.com/Lightning-AI/lightning/pull/9546))
    * Added `configure_columns` method to `RichProgressBar` ([#10288](https://github.com/Lightning-AI/lightning/pull/10288))
    * Added `leave` argument to `RichProgressBar` ([#10301](https://github.com/Lightning-AI/lightning/pull/10301))
- Added input validation logic for precision ([#9080](https://github.com/Lightning-AI/lightning/pull/9080))
- Added support for CPU AMP autocast ([#9084](https://github.com/Lightning-AI/lightning/pull/9084))
- Added `on_exception` callback hook ([#9183](https://github.com/Lightning-AI/lightning/pull/9183))
- Added a warning to DeepSpeed when inferring batch size ([#9221](https://github.com/Lightning-AI/lightning/pull/9221))
- Added `ModelSummary` callback ([#9344](https://github.com/Lightning-AI/lightning/pull/9344))
- Added `log_images`, `log_text` and `log_table` to `WandbLogger` ([#9545](https://github.com/Lightning-AI/lightning/pull/9545))
- Added `PL_RECONCILE_PROCESS` environment variable to enable process reconciliation regardless of cluster environment settings ([#9389](https://github.com/Lightning-AI/lightning/pull/9389))
- Added `get_device_stats` to the Accelerator interface and added its implementation for GPU and TPU ([#9586](https://github.com/Lightning-AI/lightning/pull/9586))
- Added a warning when an unknown key is encountered in the optimizer configuration, and when `OneCycleLR` is used with `"interval": "epoch"` ([#9666](https://github.com/Lightning-AI/lightning/pull/9666))
- Added `DeviceStatsMonitor` callback ([#9712](https://github.com/Lightning-AI/lightning/pull/9712))
- Added `enable_progress_bar` to the Trainer constructor ([#9664](https://github.com/Lightning-AI/lightning/pull/9664))
- Added `pl_legacy_patch` load utility for loading old checkpoints that have pickled legacy Lightning attributes ([#9166](https://github.com/Lightning-AI/lightning/pull/9166))
- Added support for `torch.use_deterministic_algorithms` ([#9121](https://github.com/Lightning-AI/lightning/pull/9121))
- Added automatic parameters tying for TPUs ([#9525](https://github.com/Lightning-AI/lightning/pull/9525))
- Added support for `torch.autograd.set_detect_anomaly` through `Trainer` constructor argument `detect_anomaly` ([#9848](https://github.com/Lightning-AI/lightning/pull/9848))
- Added `enable_model_summary` flag to Trainer ([#9699](https://github.com/Lightning-AI/lightning/pull/9699))
- Added `strategy` argument to Trainer ([#8597](https://github.com/Lightning-AI/lightning/pull/8597))
- Added `init_meta_context`, `materialize_module` utilities ([#9920](https://github.com/Lightning-AI/lightning/pull/9920))
- Added `TPUPrecisionPlugin` ([#10020](https://github.com/Lightning-AI/lightning/pull/#10020))
- Added `torch.bfloat16` support:
  * Added bfloat16 support for Lightning Trainer ([#9049](https://github.com/Lightning-AI/lightning/pull/9049))
  * Renamed `TPUHalfPrecisionPlugin` to `TPUBf16PrecisionPlugin` ([#10026](https://github.com/Lightning-AI/lightning/pull/10026))
  * Default to `precision=bf16` on CPU when `precision=16` is passed ([#10033](https://github.com/Lightning-AI/lightning/pull/10033))
  * Added support for `torch.autocast` ([#10053](https://github.com/Lightning-AI/lightning/pull/10053))
- Added `kfold` example for loop customization ([#9965](https://github.com/Lightning-AI/lightning/pull/9965))
- LightningLite:
    * Added `PrecisionPlugin.forward_context`, making it the default implementation for all `{train,val,test,predict}_step_context()` methods ([#9988](https://github.com/Lightning-AI/lightning/pull/9988))
    * Added `DDPSpawnPlugin.spawn()` for spawning new processes of a given function ([#10018](https://github.com/Lightning-AI/lightning/pull/10018), [#10022](https://github.com/Lightning-AI/lightning/pull/10022))
    * Added `TrainingTypePlugin.{_setup_model, _setup_optimizer}` methods ([#9994](https://github.com/Lightning-AI/lightning/pull/9994), [#10064](https://github.com/Lightning-AI/lightning/pull/10064))
    * Implemented `DataParallelPlugin._setup_model` ([#10010](https://github.com/Lightning-AI/lightning/pull/10010))
    * Implemented `DeepSpeedPlugin._setup_model_and_optimizers` ([#10009](https://github.com/Lightning-AI/lightning/pull/10009), [#10064](https://github.com/Lightning-AI/lightning/pull/10064))
    * Implemented `{DDPShardedPlugin,DDPShardedSpawnPlugin}._setup_model_and_optimizers` ([#10028](https://github.com/Lightning-AI/lightning/pull/10028), [#10064](https://github.com/Lightning-AI/lightning/pull/10064))
    * Added optional `model` argument to the `optimizer_step` methods in accelerators and plugins ([#10023](https://github.com/Lightning-AI/lightning/pull/10023))
    * Updated precision attributes in `DeepSpeedPlugin` ([#10164](https://github.com/Lightning-AI/lightning/pull/10164))
    * Added the ability to return a result from rank 0 in `DDPSpawnPlugin.spawn` ([#10162](https://github.com/Lightning-AI/lightning/pull/10162))
    * Added `pl.lite` package ([#10175](https://github.com/Lightning-AI/lightning/pull/10175))
    * Added `LightningLite` documentation ([#10043](https://github.com/Lightning-AI/lightning/pull/10043))
    * Added `LightningLite` examples ([#9987](https://github.com/Lightning-AI/lightning/pull/9987))
    * Make the `_LiteDataLoader` an iterator and add supports for custom dataloader ([#10279](https://github.com/Lightning-AI/lightning/pull/10279))
- Added `use_omegaconf` argument to `save_hparams_to_yaml` plugin ([#9170](https://github.com/Lightning-AI/lightning/pull/9170))
- Added `ckpt_path` argument for `Trainer.fit()` ([#10061](https://github.com/Lightning-AI/lightning/pull/10061))
- Added `auto_device_count` method to `Accelerators` ([#10222](https://github.com/Lightning-AI/lightning/pull/10222))
- Added support for `devices="auto"` ([#10264](https://github.com/Lightning-AI/lightning/pull/10264))
- Added a `filename` argument in `ModelCheckpoint.format_checkpoint_name` ([#9818](https://github.com/Lightning-AI/lightning/pull/9818))
- Added support for empty `gpus` list to run on CPU ([#10246](https://github.com/Lightning-AI/lightning/pull/10246))
- Added a warning if multiple batch sizes are found from ambiguous batch ([#10247](https://github.com/Lightning-AI/lightning/pull/10247))

### Changed

- Trainer now raises a `MisconfigurationException` when its methods are called with `ckpt_path="best"` but a checkpoint callback isn't configured ([#9841](https://github.com/Lightning-AI/lightning/pull/9841))
- Setting `Trainer(accelerator="ddp_cpu")` now does not spawn a subprocess if `num_processes` is kept `1` along with `num_nodes > 1` ([#9603](https://github.com/Lightning-AI/lightning/pull/9603))
- Module imports are now catching `ModuleNotFoundError` instead of `ImportError` ([#9867](https://github.com/Lightning-AI/lightning/pull/9867))
- `pl.loggers.neptune.NeptuneLogger` is now consistent with the new [neptune-client](https://github.com/neptune-ai/neptune-client) API; the old [neptune-client](https://github.com/neptune-ai/neptune-client) API is supported by `NeptuneClient` from the [neptune-contrib](https://github.com/neptune-ai/neptune-contrib) repo ([#6867](https://github.com/Lightning-AI/lightning/pull/6867))
- Parsing of `enums` type hyperparameters to be saved in the `haprams.yaml` file by TensorBoard and CSV loggers has been fixed and made in line with how OmegaConf parses it ([#9170](https://github.com/Lightning-AI/lightning/pull/9170))
- Parsing of the `gpus` Trainer argument has changed: `gpus="n"` (str) no longer selects the GPU index n and instead selects the first n devices ([#8770](https://github.com/Lightning-AI/lightning/pull/8770))
- `iteration_count` and other index attributes in the loops has been replaced with progress dataclasses ([#8477](https://github.com/Lightning-AI/lightning/pull/8477))
- The `trainer.lightning_module` reference is now properly set at the very beginning of a run ([#8536](https://github.com/Lightning-AI/lightning/pull/8536))
- The model weights now get loaded in all cases when the checkpoint path gets provided in validate/test/predict, regardless of whether the model instance is provided or not ([#8352](https://github.com/Lightning-AI/lightning/pull/8352))
- The `Trainer` functions `reset_{train,val,test,predict}_dataloader`, `reset_train_val_dataloaders`, and `request_dataloader` `model` argument is now optional ([#8536](https://github.com/Lightning-AI/lightning/pull/8536))
- Saved checkpoints will no longer use the type of a `Callback` as the key to avoid issues with unpickling ([#6886](https://github.com/Lightning-AI/lightning/pull/6886))
- Improved string conversion for `ResultCollection` ([#8622](https://github.com/Lightning-AI/lightning/pull/8622))
- `LightningCLI` changes:
    * `LightningCLI.init_parser` now returns the parser instance ([#8721](https://github.com/Lightning-AI/lightning/pull/8721))
    * `LightningCLI.add_core_arguments_to_parser`, `LightningCLI.parse_arguments` now take a `parser` argument ([#8721](https://github.com/Lightning-AI/lightning/pull/8721))
    * `LightningCLI.instantiate_trainer` now takes a config and a list of callbacks ([#8721](https://github.com/Lightning-AI/lightning/pull/8721))
    * Split `LightningCLI.add_core_arguments_to_parser` into `LightningCLI.add_default_arguments_to_parser` + `LightningCLI.add_core_arguments_to_parser` ([#8721](https://github.com/Lightning-AI/lightning/pull/8721))
- The accelerator and training type plugin `setup` hooks no longer have a `model` argument ([#8536](https://github.com/Lightning-AI/lightning/pull/8536))
- The accelerator and training type plugin `update_global_step` hook has been removed ([#8856](https://github.com/Lightning-AI/lightning/pull/8856))
- The coverage of `self.log`-ing in any `LightningModule` or `Callback` hook has been improved ([#8498](https://github.com/Lightning-AI/lightning/pull/8498))
- `self.log`-ing without a `Trainer` reference now raises a warning instead of an exception ([#9733](https://github.com/Lightning-AI/lightning/pull/9733))
- Removed restrictions in the Trainer that loggers can only log from rank 0; the existing logger behavior has not changed ([#8608](https://github.com/Lightning-AI/lightning/pull/8608))
- `Trainer.request_dataloader` now takes a `RunningStage` enum instance ([#8858](https://github.com/Lightning-AI/lightning/pull/8858))
- Changed `rank_zero_warn` to `NotImplementedError` in the `{train, val, test, predict}_dataloader` hooks that `Lightning(Data)Module` uses ([#9161](https://github.com/Lightning-AI/lightning/pull/9161))
- Moved `block_ddp_sync_behaviour` out of `TrainingBatchLoop` to loop utilities ([#9192](https://github.com/Lightning-AI/lightning/pull/9192))
- Executing the `optimizer_closure` is now required when overriding the `optimizer_step` hook ([#9360](https://github.com/Lightning-AI/lightning/pull/9360))
- Changed logging of `LightningModule` and `LightningDataModule` hyperparameters to raise an exception only if there are colliding keys with different values ([#9496](https://github.com/Lightning-AI/lightning/pull/9496))
- `seed_everything` now fails when an invalid seed value is passed instead of selecting a random seed ([#8787](https://github.com/Lightning-AI/lightning/pull/8787))
- The Trainer now calls `TrainingTypePlugin` collective APIs directly instead of going through the Accelerator reference ([#9677](https://github.com/Lightning-AI/lightning/pull/9677), [#9901](https://github.com/Lightning-AI/lightning/pull/9901))
- The tuner now uses a unique filename to save a temporary checkpoint ([#9682](https://github.com/Lightning-AI/lightning/pull/9682))
- Changed `HorovodPlugin.all_gather` to return a `torch.Tensor` instead of a list ([#9696](https://github.com/Lightning-AI/lightning/pull/9696))
- Changed Trainer connectors to be protected attributes:
    * Configuration Validator ([#9779](https://github.com/Lightning-AI/lightning/pull/9779))
- The `current_epoch` and `global_step` attributes now get restored irrespective of the Trainer task ([#9413](https://github.com/Lightning-AI/lightning/pull/9413))
- Trainer now raises an exception when requesting `amp_level` with native `amp_backend` ([#9755](https://github.com/Lightning-AI/lightning/pull/9755))
- Update the logic to check for accumulation steps with deepspeed ([#9826](https://github.com/Lightning-AI/lightning/pull/9826))
- `pl.utilities.grads.grad_norm` now raises an exception if parameter `norm_type <= 0` ([#9765](https://github.com/Lightning-AI/lightning/pull/9765))
- Updated error message for interactive incompatible plugins ([#9896](https://github.com/Lightning-AI/lightning/pull/9896))
- Moved the `optimizer_step` and `clip_gradients` hook from the `Accelerator` and `TrainingTypePlugin` into the `PrecisionPlugin` ([#10143](https://github.com/Lightning-AI/lightning/pull/10143), [#10029](https://github.com/Lightning-AI/lightning/pull/10029))
- `NativeMixedPrecisionPlugin` and its subclasses now take an optional `GradScaler` instance ([#10055](https://github.com/Lightning-AI/lightning/pull/10055))
- Trainer is now raising a `MisconfigurationException` instead of a warning if `Trainer.{validate/test}` is missing required methods ([#10016](https://github.com/Lightning-AI/lightning/pull/10016))
- Changed default value of the `max_steps` Trainer argument from `None` to -1 ([#9460](https://github.com/Lightning-AI/lightning/pull/9460))
- LightningModule now raises an error when calling `log(on_step=False, on_epoch=False)` ([#10227](https://github.com/Lightning-AI/lightning/pull/10227))
- Quantization aware training observers are now disabled by default during validating/testing/predicting stages ([#8540](https://github.com/Lightning-AI/lightning/pull/8540))
- Raised `MisconfigurationException` when total length of `dataloader` across ranks is zero, and give warning when total length is non-zero, but only local rank length is zero. ([#9827](https://github.com/Lightning-AI/lightning/pull/9827))
- Changed the model size calculation using `ByteCounter` ([#10123](https://github.com/Lightning-AI/lightning/pull/10123))
- Enabled `on_load_checkpoint` for `LightningDataModule` for all `trainer_fn` ([#10238](https://github.com/Lightning-AI/lightning/pull/10238))
- Allowed separate config files for parameters with class type when LightningCLI is in `subclass_mode=False` ([#10286](https://github.com/Lightning-AI/lightning/pull/10286))

### Deprecated

- Deprecated Trainer argument `terminate_on_nan` in favor of `detect_anomaly`([#9175](https://github.com/Lightning-AI/lightning/pull/9175))
- Deprecated `Trainer.terminate_on_nan` public attribute access ([#9849](https://github.com/Lightning-AI/lightning/pull/9849))
- Deprecated `LightningModule.summarize()` in favor of `pl.utilities.model_summary.summarize()` ([#8513](https://github.com/Lightning-AI/lightning/pull/8513))
- Deprecated `LightningModule.model_size` ([#8343](https://github.com/Lightning-AI/lightning/pull/8343))
- Deprecated `DataModule` properties: `train_transforms`, `val_transforms`, `test_transforms`, `size`, `dims` ([#8851](https://github.com/Lightning-AI/lightning/pull/8851))
- Deprecated `add_to_queue`, `get_from_queue` from `LightningModule` in favor of corresponding methods in the `DDPSpawnPlugin` ([#9118](https://github.com/Lightning-AI/lightning/pull/9118))
- Deprecated `LightningModule.get_progress_bar_dict` and `Trainer.progress_bar_dict` in favor of `pl.callbacks.progress.base.get_standard_metrics` and `ProgressBarBase.get_metrics` ([#8985](https://github.com/Lightning-AI/lightning/pull/8985))
- Deprecated `prepare_data_per_node` flag on Trainer and set it as a property of `DataHooks`, accessible in the `LightningModule` and `LightningDataModule` ([#8958](https://github.com/Lightning-AI/lightning/pull/8958))
- Deprecated the `TestTubeLogger` ([#9065](https://github.com/Lightning-AI/lightning/pull/9065))
- Deprecated `on_{train/val/test/predict}_dataloader()` from `LightningModule` and `LightningDataModule` ([#9098](https://github.com/Lightning-AI/lightning/pull/9098))
- Deprecated `on_keyboard_interrupt` callback hook in favor of new `on_exception` hook ([#9260](https://github.com/Lightning-AI/lightning/pull/9260))
- Deprecated passing `process_position` to the `Trainer` constructor in favor of adding the `ProgressBar` callback with `process_position` directly to the list of callbacks ([#9222](https://github.com/Lightning-AI/lightning/pull/9222))
- Deprecated passing `flush_logs_every_n_steps` as a Trainer argument, instead pass it to the logger init if supported ([#9366](https://github.com/Lightning-AI/lightning/pull/9366))
- Deprecated `LightningLoggerBase.close`, `LoggerCollection.close` in favor of `LightningLoggerBase.finalize`, `LoggerCollection.finalize` ([#9422](https://github.com/Lightning-AI/lightning/pull/9422))
- Deprecated passing `progress_bar_refresh_rate` to the `Trainer` constructor in favor of adding the `ProgressBar` callback with `refresh_rate` directly to the list of callbacks, or passing `enable_progress_bar=False` to disable the progress bar ([#9616](https://github.com/Lightning-AI/lightning/pull/9616))
- Deprecated `LightningDistributed` and moved the broadcast logic to `DDPPlugin` and `DDPSpawnPlugin` directly ([#9691](https://github.com/Lightning-AI/lightning/pull/9691))
- Deprecated passing `stochastic_weight_avg` to the `Trainer` constructor in favor of adding the `StochasticWeightAveraging` callback directly to the list of callbacks ([#8989](https://github.com/Lightning-AI/lightning/pull/8989))
- Deprecated Accelerator collective API `barrier`, `broadcast`, and `all_gather` in favor of calling the `TrainingTypePlugin` collective API directly ([#9677](https://github.com/Lightning-AI/lightning/pull/9677))
- Deprecated `checkpoint_callback` from the `Trainer` constructor in favor of `enable_checkpointing` ([#9754](https://github.com/Lightning-AI/lightning/pull/9754))
- Deprecated the `LightningModule.on_post_move_to_device` method ([#9525](https://github.com/Lightning-AI/lightning/pull/9525))
- Deprecated `pl.core.decorators.parameter_validation` in favor of `pl.utilities.parameter_tying.set_shared_parameters` ([#9525](https://github.com/Lightning-AI/lightning/pull/9525))
- Deprecated passing `weights_summary` to the `Trainer` constructor in favor of adding the `ModelSummary` callback with `max_depth` directly to the list of callbacks ([#9699](https://github.com/Lightning-AI/lightning/pull/9699))
- Deprecated `log_gpu_memory`, `gpu_metrics`, and util funcs in favor of `DeviceStatsMonitor` callback ([#9921](https://github.com/Lightning-AI/lightning/pull/9921))
- Deprecated `GPUStatsMonitor` and `XLAStatsMonitor` in favor of `DeviceStatsMonitor` callback ([#9924](https://github.com/Lightning-AI/lightning/pull/9924))
- Deprecated setting `Trainer(max_steps=None)`; To turn off the limit, set `Trainer(max_steps=-1)` (default) ([#9460](https://github.com/Lightning-AI/lightning/pull/9460))
- Deprecated access to the `AcceleratorConnector.is_slurm_managing_tasks` attribute and marked it as protected ([#10101](https://github.com/Lightning-AI/lightning/pull/10101))
- Deprecated access to the `AcceleratorConnector.configure_slurm_ddp` method and marked it as protected ([#10101](https://github.com/Lightning-AI/lightning/pull/10101))
- Deprecated passing `resume_from_checkpoint` to the `Trainer` constructor in favor of `trainer.fit(ckpt_path=)` ([#10061](https://github.com/Lightning-AI/lightning/pull/10061))
- Deprecated `ClusterEnvironment.creates_children()` in favor of `ClusterEnvironment.creates_processes_externally` (property) ([#10106](https://github.com/Lightning-AI/lightning/pull/10106))
- Deprecated `PrecisionPlugin.master_params()` in favor of `PrecisionPlugin.main_params()` ([#10105](https://github.com/Lightning-AI/lightning/pull/10105))
- Deprecated `lr_sch_names` from `LearningRateMonitor` ([#10066](https://github.com/Lightning-AI/lightning/pull/10066))
- Deprecated `ProgressBar` callback in favor of `TQDMProgressBar` ([#10134](https://github.com/Lightning-AI/lightning/pull/10134))

### Removed

- Removed deprecated `metrics` ([#8586](https://github.com/Lightning-AI/lightning/pull/8586/))
- Removed the deprecated `outputs` argument in both the `LightningModule.on_train_epoch_end` and `Callback.on_train_epoch_end` hooks ([#8587](https://github.com/Lightning-AI/lightning/pull/8587))
- Removed the deprecated `TrainerLoggingMixin` class ([#8609](https://github.com/Lightning-AI/lightning/pull/8609))
- Removed the deprecated `TrainerTrainingTricksMixin` class ([#8679](https://github.com/Lightning-AI/lightning/pull/8679))
- Removed the deprecated `optimizer_idx` from `training_step` as an accepted argument in manual optimization ([#8576](https://github.com/Lightning-AI/lightning/pull/8576))
- Removed support for the deprecated `on_save_checkpoint` signature. The hook now takes a `checkpoint` positional parameter ([#8697](https://github.com/Lightning-AI/lightning/pull/8697))
- Removed support for the deprecated `on_load_checkpoint` signature. The hook now takes a `pl_module` positional parameter ([#8697](https://github.com/Lightning-AI/lightning/pull/8697))
- Removed the deprecated `save_function` property in `ModelCheckpoint` ([#8680](https://github.com/Lightning-AI/lightning/pull/8680))
- Removed the deprecated `model` argument from `ModelCheckpoint.save_checkpoint` ([#8688](https://github.com/Lightning-AI/lightning/pull/8688))
- Removed the deprecated `sync_step` argument from `WandbLogger` ([#8763](https://github.com/Lightning-AI/lightning/pull/8763))
- Removed the deprecated `Trainer.truncated_bptt_steps` in favor of `LightningModule.truncated_bptt_steps` ([#8826](https://github.com/Lightning-AI/lightning/pull/8826))
- Removed `LightningModule.write_predictions` and `LightningModule.write_predictions_dict` ([#8850](https://github.com/Lightning-AI/lightning/pull/8850))
- Removed `on_reset_*_dataloader` hooks in TrainingType Plugins and Accelerators ([#8858](https://github.com/Lightning-AI/lightning/pull/8858))
- Removed deprecated `GradInformation` module in favor of `pl.utilities.grads` ([#8831](https://github.com/Lightning-AI/lightning/pull/8831/))
- Removed `TrainingTypePlugin.on_save` and `Accelerator.on_save` ([#9023](https://github.com/Lightning-AI/lightning/pull/9023))
- Removed `{Accelerator,TrainingTypePlugin,PrecisionPlugin}.post_optimizer_step` ([#9746](https://github.com/Lightning-AI/lightning/pull/9746))
- Removed deprecated `connect_precision_plugin` and `connect_training_type_plugin` from `Accelerator` ([#9019](https://github.com/Lightning-AI/lightning/pull/9019))
- Removed `on_train_epoch_end` from `Accelerator` ([#9035](https://github.com/Lightning-AI/lightning/pull/9035))
- Removed `InterBatchProcessor` in favor of `DataLoaderIterDataFetcher` ([#9052](https://github.com/Lightning-AI/lightning/pull/9052))
- Removed `Plugin` in `base_plugin.py` in favor of accessing `TrainingTypePlugin` and `PrecisionPlugin` directly instead ([#9066](https://github.com/Lightning-AI/lightning/pull/9066))
- Removed `teardown` from `ParallelPlugin` ([#8943](https://github.com/Lightning-AI/lightning/pull/8943))
- Removed deprecated `profiled_functions` argument from `PyTorchProfiler` ([#9178](https://github.com/Lightning-AI/lightning/pull/9178))
- Removed deprecated `pytorch_lightning.utilities.argparse_utils` module ([#9166](https://github.com/Lightning-AI/lightning/pull/9166))
- Removed deprecated property `Trainer.running_sanity_check` in favor of `Trainer.sanity_checking` ([#9209](https://github.com/Lightning-AI/lightning/pull/9209))
- Removed deprecated `BaseProfiler.output_filename` arg from it and its descendants in favor of `dirpath` and `filename` ([#9214](https://github.com/Lightning-AI/lightning/pull/9214))
- Removed deprecated property `ModelCheckpoint.period` in favor of `ModelCheckpoint.every_n_epochs` ([#9213](https://github.com/Lightning-AI/lightning/pull/9213))
- Removed deprecated `auto_move_data` decorator ([#9231](https://github.com/Lightning-AI/lightning/pull/9231))
- Removed deprecated property `LightningModule.datamodule` in favor of `Trainer.datamodule` ([#9233](https://github.com/Lightning-AI/lightning/pull/9233))
- Removed deprecated properties `DeepSpeedPlugin.cpu_offload*` in favor of `offload_optimizer`, `offload_parameters` and `pin_memory` ([#9244](https://github.com/Lightning-AI/lightning/pull/9244))
- Removed deprecated property `AcceleratorConnector.is_using_torchelastic` in favor of `TorchElasticEnvironment.is_using_torchelastic()` ([#9729](https://github.com/Lightning-AI/lightning/pull/9729))
- Removed `pl.utilities.debugging.InternalDebugger` ([#9680](https://github.com/Lightning-AI/lightning/pull/9680))
- Removed `call_configure_sharded_model_hook` property from `Accelerator` and `TrainingTypePlugin` ([#9612](https://github.com/Lightning-AI/lightning/pull/9612))
- Removed `TrainerProperties` mixin and moved property definitions directly into `Trainer` ([#9495](https://github.com/Lightning-AI/lightning/pull/9495))
- Removed a redundant warning with `ModelCheckpoint(monitor=None)` callback ([#9875](https://github.com/Lightning-AI/lightning/pull/9875))
- Remove `epoch` from `trainer.logged_metrics` ([#9904](https://github.com/Lightning-AI/lightning/pull/9904))
- Remove deprecated `distributed_backend` from `Trainer` ([#10017](https://github.com/Lightning-AI/lightning/pull/10017))
- Removed `process_idx` from the `{DDPSpawnPlugin,TPUSpawnPlugin}.new_process` methods ([#10022](https://github.com/Lightning-AI/lightning/pull/10022))
- Removed automatic patching of `{train,val,test,predict}_dataloader()` on the `LightningModule` ([#9764](https://github.com/Lightning-AI/lightning/pull/9764))
- Removed `pl.trainer.connectors.OptimizerConnector` ([#10120](https://github.com/Lightning-AI/lightning/pull/10120))

### Fixed

- Fixed ImageNet evaluation in example ([#10179](https://github.com/Lightning-AI/lightning/pull/10179))
- Fixed an issue with logger outputs not being finalized correctly after prediction runs ([#8685](https://github.com/Lightning-AI/lightning/pull/8685))
- Fixed `move_metrics_to_cpu` moving the loss to CPU while training on device ([#9308](https://github.com/Lightning-AI/lightning/pull/9308))
- Fixed incorrect main progress bar indicator when resuming training mid-epoch ([#9310](https://github.com/Lightning-AI/lightning/pull/9310))
- Fixed an issue with freeing memory of datafetchers during teardown ([#9387](https://github.com/Lightning-AI/lightning/pull/9387))
- Fixed a bug where the training step output needed to be `deepcopy`-ed ([#9349](https://github.com/Lightning-AI/lightning/pull/9349))
- Fixed an issue with freeing memory allocated by the data iterators in `Loop.on_run_end` ([#9386](https://github.com/Lightning-AI/lightning/pull/9386), [#9915](https://github.com/Lightning-AI/lightning/pull/9915))
- Fixed `BasePredictionWriter` not returning the batch indices in a non-distributed setting ([#9432](https://github.com/Lightning-AI/lightning/pull/9432))
- Fixed an error when running in XLA environments with no TPU attached ([#9572](https://github.com/Lightning-AI/lightning/pull/9572))
- Fixed check on torchmetrics logged whose `compute()` output is a multielement tensor ([#9582](https://github.com/Lightning-AI/lightning/pull/9582))
- Fixed gradient accumulation for `DDPShardedPlugin` ([#9122](https://github.com/Lightning-AI/lightning/pull/9122))
- Fixed missing DeepSpeed distributed call ([#9540](https://github.com/Lightning-AI/lightning/pull/9540))
- Fixed an issue with wrapped LightningModule during evaluation; The LightningModule no longer gets wrapped with data-parallel modules when not fitting in `DDPPlugin`, `DDPSpawnPlugin`, `DDPShardedPlugin`, `DDPSpawnShardedPlugin` ([#9096](https://github.com/Lightning-AI/lightning/pull/9096))
- Fixed `trainer.accumulate_grad_batches` to be an int on init. The default value for it is now `None` inside Trainer ([#9652](https://github.com/Lightning-AI/lightning/pull/9652))
- Fixed `broadcast` in `DDPPlugin` and `DDPSpawnPlugin` to respect the `src` input ([#9691](https://github.com/Lightning-AI/lightning/pull/9691))
- Fixed `self.log(on_epoch=True, reduce_fx=sum))` for the `on_batch_start` and `on_train_batch_start` hooks ([#9791](https://github.com/Lightning-AI/lightning/pull/9791))
- Fixed `self.log(on_epoch=True)` for the `on_batch_start` and `on_train_batch_start` hooks ([#9780](https://github.com/Lightning-AI/lightning/pull/9780))
- Fixed restoring training state during `Trainer.fit` only ([#9413](https://github.com/Lightning-AI/lightning/pull/9413))
- Fixed DeepSpeed and Lightning both calling the scheduler ([#9788](https://github.com/Lightning-AI/lightning/pull/9788))
- Fixed missing arguments when saving hyperparameters from the parent class but not from the child class ([#9800](https://github.com/Lightning-AI/lightning/pull/9800))
- Fixed DeepSpeed GPU device IDs ([#9847](https://github.com/Lightning-AI/lightning/pull/9847))
- Reset `val_dataloader` in `tuner/batch_size_scaling` ([#9857](https://github.com/Lightning-AI/lightning/pull/9857))
- Fixed use of `LightningCLI` in computer_vision_fine_tuning.py example ([#9934](https://github.com/Lightning-AI/lightning/pull/9934))
- Fixed issue with non-init dataclass fields in `apply_to_collection` ([#9963](https://github.com/Lightning-AI/lightning/pull/9963))
- Reset `val_dataloader` in `tuner/batch_size_scaling` for binsearch ([#9975](https://github.com/Lightning-AI/lightning/pull/9975))
- Fixed logic to check for spawn in dataloader `TrainerDataLoadingMixin._worker_check` ([#9902](https://github.com/Lightning-AI/lightning/pull/9902))
- Fixed `train_dataloader` getting loaded twice when resuming from a checkpoint during `Trainer.fit()` ([#9671](https://github.com/Lightning-AI/lightning/pull/9671))
- Fixed `LearningRateMonitor` logging with multiple param groups optimizer with no scheduler ([#10044](https://github.com/Lightning-AI/lightning/pull/10044))
- Fixed undesired side effects being caused by `Trainer` patching dataloader methods on the `LightningModule` ([#9764](https://github.com/Lightning-AI/lightning/pull/9764))
- Fixed gradients not being unscaled when clipping or logging the gradient norm ([#9287](https://github.com/Lightning-AI/lightning/pull/9287))
- Fixed `on_before_optimizer_step` getting called before the optimizer closure (including backward) has run ([#10167](https://github.com/Lightning-AI/lightning/pull/10167))
- Fixed monitor value in `ModelCheckpoint` getting moved to the wrong device in a special case where it becomes NaN ([#10118](https://github.com/Lightning-AI/lightning/pull/10118))
- Fixed creation of `dirpath` in `BaseProfiler` if it doesn't exist ([#10073](https://github.com/Lightning-AI/lightning/pull/10073))
- Fixed incorrect handling of sigterm ([#10189](https://github.com/Lightning-AI/lightning/pull/10189))
- Fixed bug where `log(on_step=True, on_epoch=True, sync_dist=True)` wouldn't reduce the value on step ([#10227](https://github.com/Lightning-AI/lightning/pull/10227))
- Fixed an issue with `pl.utilities.seed.reset_seed` converting the `PL_SEED_WORKERS` environment variable to `bool` ([#10099](https://github.com/Lightning-AI/lightning/pull/10099))
- Fixed iterating over a logger collection when `fast_dev_run > 0` ([#10232](https://github.com/Lightning-AI/lightning/pull/10232))
- Fixed `batch_size` in `ResultCollection` not being reset to 1 on epoch end ([#10242](https://github.com/Lightning-AI/lightning/pull/10242))
- Fixed `distrib_type` not being set when training plugin instances are being passed to the Trainer ([#10251](https://github.com/Lightning-AI/lightning/pull/10251))


## [1.4.9] - 2021-09-30

- Fixed `lr_find` to generate same results on multiple calls ([#9704](https://github.com/Lightning-AI/lightning/pull/9704))
- Fixed `reset` metrics on validation epoch end ([#9717](https://github.com/Lightning-AI/lightning/pull/9717))
- Fixed input validation for `gradient_clip_val`, `gradient_clip_algorithm`, `track_grad_norm` and `terminate_on_nan` Trainer arguments ([#9595](https://github.com/Lightning-AI/lightning/pull/9595))
- Reset metrics before each task starts ([#9410](https://github.com/Lightning-AI/lightning/pull/9410))


## [1.4.8] - 2021-09-22

- Fixed error reporting in DDP process reconciliation when processes are launched by an external agent ([#9389](https://github.com/Lightning-AI/lightning/pull/9389))
- Added PL_RECONCILE_PROCESS environment variable to enable process reconciliation regardless of cluster environment settings ([#9389](https://github.com/Lightning-AI/lightning/pull/9389))
- Fixed `add_argparse_args` raising `TypeError` when args are typed as `typing.Generic` in Python 3.6 ([#9554](https://github.com/Lightning-AI/lightning/pull/9554))
- Fixed back-compatibility for saving hyperparameters from a single container and inferring its argument name by reverting [#9125](https://github.com/Lightning-AI/lightning/pull/9125) ([#9642](https://github.com/Lightning-AI/lightning/pull/9642))


## [1.4.7] - 2021-09-14

- Fixed logging of nan parameters ([#9364](https://github.com/Lightning-AI/lightning/pull/9364))
- Fixed `replace_sampler` missing the batch size under specific conditions ([#9367](https://github.com/Lightning-AI/lightning/pull/9367))
- Pass init args to ShardedDataParallel ([#9483](https://github.com/Lightning-AI/lightning/pull/9483))
- Fixed collision of user argument when using ShardedDDP ([#9512](https://github.com/Lightning-AI/lightning/pull/9512))
- Fixed DeepSpeed crash for RNNs ([#9489](https://github.com/Lightning-AI/lightning/pull/9489))


## [1.4.6] - 2021-09-07

- Fixed an issues with export to ONNX format when a model has multiple inputs ([#8800](https://github.com/Lightning-AI/lightning/pull/8800))
- Removed deprecation warnings being called for `on_{task}_dataloader` ([#9279](https://github.com/Lightning-AI/lightning/pull/9279))
- Fixed save/load/resume from checkpoint for DeepSpeed Plugin (
    [#8397](https://github.com/Lightning-AI/lightning/pull/8397),
    [#8644](https://github.com/Lightning-AI/lightning/pull/8644),
    [#8627](https://github.com/Lightning-AI/lightning/pull/8627))
- Fixed `EarlyStopping` running on train epoch end when `check_val_every_n_epoch>1` is set ([#9156](https://github.com/Lightning-AI/lightning/pull/9156))
- Fixed an issue with logger outputs not being finalized correctly after prediction runs ([#8333](https://github.com/Lightning-AI/lightning/pull/8333))
- Fixed the Apex and DeepSpeed plugin closure running after the `on_before_optimizer_step` hook ([#9288](https://github.com/Lightning-AI/lightning/pull/9288))
- Fixed the Native AMP plugin closure not running with manual optimization ([#9288](https://github.com/Lightning-AI/lightning/pull/9288))
- Fixed bug where data-loading functions where not getting the correct running stage passed ([#8858](https://github.com/Lightning-AI/lightning/pull/8858))
- Fixed intra-epoch evaluation outputs staying in memory when the respective `*_epoch_end` hook wasn't overridden ([#9261](https://github.com/Lightning-AI/lightning/pull/9261))
- Fixed error handling in DDP process reconciliation when `_sync_dir` was not initialized ([#9267](https://github.com/Lightning-AI/lightning/pull/9267))
- Fixed PyTorch Profiler not enabled for manual optimization ([#9316](https://github.com/Lightning-AI/lightning/pull/9316))
- Fixed inspection of other args when a container is specified in `save_hyperparameters` ([#9125](https://github.com/Lightning-AI/lightning/pull/9125))
- Fixed signature of `Timer.on_train_epoch_end` and `StochasticWeightAveraging.on_train_epoch_end` to prevent unwanted deprecation warnings ([#9347](https://github.com/Lightning-AI/lightning/pull/9347))


## [1.4.5] - 2021-08-31

- Fixed reduction using `self.log(sync_dict=True, reduce_fx={mean,max})` ([#9142](https://github.com/Lightning-AI/lightning/pull/9142))
- Fixed not setting a default value for `max_epochs` if `max_time` was specified on the `Trainer` constructor ([#9072](https://github.com/Lightning-AI/lightning/pull/9072))
- Fixed the CometLogger, no longer modifies the metrics in place. Instead creates a copy of metrics before performing any operations ([#9150](https://github.com/Lightning-AI/lightning/pull/9150))
- Fixed `DDP` "CUDA error: initialization error" due to a `copy` instead of `deepcopy` on `ResultCollection` ([#9239](https://github.com/Lightning-AI/lightning/pull/9239))


## [1.4.4] - 2021-08-24

- Fixed a bug in the binary search mode of auto batch size scaling where exception was raised if the first trainer run resulted in OOM ([#8954](https://github.com/Lightning-AI/lightning/pull/8954))
- Fixed a bug causing logging with `log_gpu_memory='min_max'` not working ([#9013](https://github.com/Lightning-AI/lightning/pull/9013))


## [1.4.3] - 2021-08-17

- Fixed plateau scheduler stepping on incomplete epoch ([#8861](https://github.com/Lightning-AI/lightning/pull/8861))
- Fixed infinite loop with `CycleIterator` and multiple loaders ([#8889](https://github.com/Lightning-AI/lightning/pull/8889))
- Fixed `StochasticWeightAveraging` with a list of learning rates not applying them to each param group ([#8747](https://github.com/Lightning-AI/lightning/pull/8747))
- Restore original loaders if replaced by entrypoint ([#8885](https://github.com/Lightning-AI/lightning/pull/8885))
- Fixed lost reference to `_Metadata` object in `ResultMetricCollection` ([#8932](https://github.com/Lightning-AI/lightning/pull/8932))
- Ensure the existence of `DDPPlugin._sync_dir` in `reconciliate_processes` ([#8939](https://github.com/Lightning-AI/lightning/pull/8939))


## [1.4.2] - 2021-08-10

- Fixed recursive call for `apply_to_collection(include_none=False)` ([#8719](https://github.com/Lightning-AI/lightning/pull/8719))
- Fixed truncated backprop through time enablement when set as a property on the LightningModule and not the Trainer ([#8804](https://github.com/Lightning-AI/lightning/pull/8804/))
- Fixed comments and exception message for metrics_to_scalars ([#8782](https://github.com/Lightning-AI/lightning/pull/8782/))
- Fixed typo error in LightningLoggerBase.after_save_checkpoint docstring ([#8737](https://github.com/Lightning-AI/lightning/pull/8737/))


## [1.4.1] - 2021-08-03

- Fixed `trainer.fit_loop.split_idx` always returning `None` ([#8601](https://github.com/Lightning-AI/lightning/pull/8601))
- Fixed references for `ResultCollection.extra` ([#8622](https://github.com/Lightning-AI/lightning/pull/8622))
- Fixed reference issues during epoch end result collection ([#8621](https://github.com/Lightning-AI/lightning/pull/8621))
- Fixed horovod auto-detection when horovod is not installed and the launcher is `mpirun` ([#8610](https://github.com/Lightning-AI/lightning/pull/8610))
- Fixed an issue with `training_step` outputs not getting collected correctly for `training_epoch_end` ([#8613](https://github.com/Lightning-AI/lightning/pull/8613))
- Fixed distributed types support for CPUs ([#8667](https://github.com/Lightning-AI/lightning/pull/8667))
- Fixed a deadlock issue with DDP and torchelastic ([#8655](https://github.com/Lightning-AI/lightning/pull/8655))
- Fixed `accelerator=ddp` choice for CPU ([#8645](https://github.com/Lightning-AI/lightning/pull/8645))


## [1.4.0] - 2021-07-27

### Added

- Added `extract_batch_size` utility and corresponding tests to extract batch dimension from multiple batch types ([#8357](https://github.com/Lightning-AI/lightning/pull/8357/))
- Added support for named parameter groups in `LearningRateMonitor` ([#7987](https://github.com/Lightning-AI/lightning/pull/7987))
- Added `dataclass` support for `pl.utilities.apply_to_collection` ([#7935](https://github.com/Lightning-AI/lightning/pull/7935))
- Added support to `LightningModule.to_torchscript` for saving to custom filesystems with `fsspec` ([#7617](https://github.com/Lightning-AI/lightning/pull/7617))
- Added `KubeflowEnvironment` for use with the `PyTorchJob` operator in Kubeflow
- Added LightningCLI support for config files on object stores ([#7521](https://github.com/Lightning-AI/lightning/pull/7521))
- Added `ModelPruning(prune_on_train_epoch_end=True|False)` to choose when to apply pruning ([#7704](https://github.com/Lightning-AI/lightning/pull/7704))
- Added support for checkpointing based on a provided time interval during training ([#7515](https://github.com/Lightning-AI/lightning/pull/7515))
- Progress tracking
  * Added dataclasses for progress tracking ([#6603](https://github.com/Lightning-AI/lightning/pull/6603),
    [#7574](https://github.com/Lightning-AI/lightning/pull/7574),
    [#8140](https://github.com/Lightning-AI/lightning/pull/8140),
    [#8362](https://github.com/Lightning-AI/lightning/pull/8362))
  * Add `{,load_}state_dict` to the progress tracking dataclasses ([#8140](https://github.com/Lightning-AI/lightning/pull/8140))
  * Connect the progress tracking dataclasses to the loops ([#8244](https://github.com/Lightning-AI/lightning/pull/8244),
    [#8362](https://github.com/Lightning-AI/lightning/pull/8362))
  * Do not reset the progress tracking dataclasses total counters ([#8475](https://github.com/Lightning-AI/lightning/pull/8475))
- Added support for passing a `LightningDataModule` positionally as the second argument to `trainer.{validate,test,predict}` ([#7431](https://github.com/Lightning-AI/lightning/pull/7431))
- Added argument `trainer.predict(ckpt_path)` ([#7430](https://github.com/Lightning-AI/lightning/pull/7430))
- Added `clip_grad_by_value` support for TPUs ([#7025](https://github.com/Lightning-AI/lightning/pull/7025))
- Added support for passing any class to `is_overridden` ([#7918](https://github.com/Lightning-AI/lightning/pull/7918))
- Added `sub_dir` parameter to `TensorBoardLogger` ([#6195](https://github.com/Lightning-AI/lightning/pull/6195))
- Added correct `dataloader_idx` to batch transfer hooks ([#6241](https://github.com/Lightning-AI/lightning/pull/6241))
- Added `include_none=bool` argument to `apply_to_collection` ([#7769](https://github.com/Lightning-AI/lightning/pull/7769))
- Added `apply_to_collections` to apply a function to two zipped collections ([#7769](https://github.com/Lightning-AI/lightning/pull/7769))
- Added `ddp_fully_sharded` support ([#7487](https://github.com/Lightning-AI/lightning/pull/7487))
- Added `should_rank_save_checkpoint` property to Training Plugins ([#7684](https://github.com/Lightning-AI/lightning/pull/7684))
- Added `log_grad_norm` hook to `LightningModule` to customize the logging of gradient norms ([#7873](https://github.com/Lightning-AI/lightning/pull/7873))
- Added `save_config_filename` init argument to `LightningCLI` to ease resolving name conflicts ([#7741](https://github.com/Lightning-AI/lightning/pull/7741))
- Added `save_config_overwrite` init argument to `LightningCLI` to ease overwriting existing config files ([#8059](https://github.com/Lightning-AI/lightning/pull/8059))
- Added reset dataloader hooks to Training Plugins and Accelerators ([#7861](https://github.com/Lightning-AI/lightning/pull/7861))
- Added trainer stage hooks for Training Plugins and Accelerators ([#7864](https://github.com/Lightning-AI/lightning/pull/7864))
- Added the `on_before_optimizer_step` hook ([#8048](https://github.com/Lightning-AI/lightning/pull/8048))
- Added IPU Accelerator ([#7867](https://github.com/Lightning-AI/lightning/pull/7867))
- Fault-tolerant training
    * Added `{,load_}state_dict` to `ResultCollection` ([#7948](https://github.com/Lightning-AI/lightning/pull/7948))
    * Added `{,load_}state_dict` to `Loops` ([#8197](https://github.com/Lightning-AI/lightning/pull/8197))
    * Added `FastForwardSampler` and `CaptureIterableDataset` ([#8307](https://github.com/Lightning-AI/lightning/pull/8307))
    * Set `Loop.restarting=False` at the end of the first iteration ([#8362](https://github.com/Lightning-AI/lightning/pull/8362))
    * Save the loops state with the checkpoint (opt-in) ([#8362](https://github.com/Lightning-AI/lightning/pull/8362))
    * Save a checkpoint to restore the state on exception (opt-in) ([#8362](https://github.com/Lightning-AI/lightning/pull/8362))
    * Added `state_dict` and `load_state_dict` utilities for `CombinedLoader` + utilities for dataloader ([#8364](https://github.com/Lightning-AI/lightning/pull/8364))
- Added `rank_zero_only` to `LightningModule.log` function ([#7966](https://github.com/Lightning-AI/lightning/pull/7966))
- Added `metric_attribute` to `LightningModule.log` function ([#7966](https://github.com/Lightning-AI/lightning/pull/7966))
- Added a warning if `Trainer(log_every_n_steps)` is a value too high for the training dataloader ([#7734](https://github.com/Lightning-AI/lightning/pull/7734))
- Added LightningCLI support for argument links applied on instantiation ([#7895](https://github.com/Lightning-AI/lightning/pull/7895))
- Added LightningCLI support for configurable callbacks that should always be present ([#7964](https://github.com/Lightning-AI/lightning/pull/7964))
- Added DeepSpeed Infinity Support, and updated to DeepSpeed 0.4.0 ([#7234](https://github.com/Lightning-AI/lightning/pull/7234))
- Added support for `torch.nn.UninitializedParameter` in `ModelSummary` ([#7642](https://github.com/Lightning-AI/lightning/pull/7642))
- Added support `LightningModule.save_hyperparameters` when `LightningModule` is a dataclass ([#7992](https://github.com/Lightning-AI/lightning/pull/7992))
- Added support for overriding `optimizer_zero_grad` and `optimizer_step` when using accumulate_grad_batches ([#7980](https://github.com/Lightning-AI/lightning/pull/7980))
- Added `logger` boolean flag to `save_hyperparameters` ([#7960](https://github.com/Lightning-AI/lightning/pull/7960))
- Added support for calling scripts using the module syntax (`python -m package.script`) ([#8073](https://github.com/Lightning-AI/lightning/pull/8073))
- Added support for optimizers and learning rate schedulers to `LightningCLI` ([#8093](https://github.com/Lightning-AI/lightning/pull/8093))
- Added XLA Profiler ([#8014](https://github.com/Lightning-AI/lightning/pull/8014))
- Added `PrecisionPlugin.{pre,post}_backward` ([#8328](https://github.com/Lightning-AI/lightning/pull/8328))
- Added `on_load_checkpoint` and `on_save_checkpoint` hooks to the `PrecisionPlugin` base class ([#7831](https://github.com/Lightning-AI/lightning/pull/7831))
- Added `max_depth` parameter in `ModelSummary` ([#8062](https://github.com/Lightning-AI/lightning/pull/8062))
- Added `XLAStatsMonitor` callback ([#8235](https://github.com/Lightning-AI/lightning/pull/8235))
- Added `restore` function and `restarting` attribute to base `Loop` ([#8247](https://github.com/Lightning-AI/lightning/pull/8247))
- Added support for `save_hyperparameters` in `LightningDataModule` ([#3792](https://github.com/Lightning-AI/lightning/pull/3792))
- Added the `ModelCheckpoint(save_on_train_epoch_end)` to choose when to run the saving logic ([#8389](https://github.com/Lightning-AI/lightning/pull/8389))
- Added `LSFEnvironment` for distributed training with the LSF resource manager `jsrun` ([#5102](https://github.com/Lightning-AI/lightning/pull/5102))
- Added support for `accelerator='cpu'|'gpu'|'tpu'|'ipu'|'auto'` ([#7808](https://github.com/Lightning-AI/lightning/pull/7808))
- Added `tpu_spawn_debug` to plugin registry ([#7933](https://github.com/Lightning-AI/lightning/pull/7933))
- Enabled traditional/manual launching of DDP processes through `LOCAL_RANK` and `NODE_RANK` environment variable assignments ([#7480](https://github.com/Lightning-AI/lightning/pull/7480))
- Added `quantize_on_fit_end` argument to `QuantizationAwareTraining` ([#8464](https://github.com/Lightning-AI/lightning/pull/8464))
- Added experimental support for loop specialization ([#8226](https://github.com/Lightning-AI/lightning/pull/8226))
- Added support for `devices` flag to Trainer ([#8440](https://github.com/Lightning-AI/lightning/pull/8440))
- Added private `prevent_trainer_and_dataloaders_deepcopy` context manager on the `LightningModule` ([#8472](https://github.com/Lightning-AI/lightning/pull/8472))
- Added support for providing callables to the Lightning CLI instead of types ([#8400](https://github.com/Lightning-AI/lightning/pull/8400))

### Changed

- Decoupled device parsing logic from Accelerator connector to Trainer ([#8180](https://github.com/Lightning-AI/lightning/pull/8180))
- Changed the `Trainer`'s `checkpoint_callback` argument to allow only boolean values ([#7539](https://github.com/Lightning-AI/lightning/pull/7539))
- Log epoch metrics before the `on_evaluation_end` hook ([#7272](https://github.com/Lightning-AI/lightning/pull/7272))
- Explicitly disallow calling `self.log(on_epoch=False)` during epoch-only or single-call hooks ([#7874](https://github.com/Lightning-AI/lightning/pull/7874))
- Changed these `Trainer` methods to be protected: `call_setup_hook`, `call_configure_sharded_model`, `pre_dispatch`, `dispatch`, `post_dispatch`, `call_teardown_hook`, `run_train`, `run_sanity_check`, `run_evaluate`, `run_evaluation`, `run_predict`, `track_output_for_epoch_end`
- Changed `metrics_to_scalars` to work with any collection or value ([#7888](https://github.com/Lightning-AI/lightning/pull/7888))
- Changed `clip_grad_norm` to use `torch.nn.utils.clip_grad_norm_` ([#7025](https://github.com/Lightning-AI/lightning/pull/7025))
- Validation is now always run inside the training epoch scope ([#7357](https://github.com/Lightning-AI/lightning/pull/7357))
- `ModelCheckpoint` now runs at the end of the training epoch by default ([#8389](https://github.com/Lightning-AI/lightning/pull/8389))
- `EarlyStopping` now runs at the end of the training epoch by default ([#8286](https://github.com/Lightning-AI/lightning/pull/8286))
- Refactored Loops
    * Moved attributes `global_step`, `current_epoch`, `max/min_steps`, `max/min_epochs`, `batch_idx`, and `total_batch_idx` to TrainLoop ([#7437](https://github.com/Lightning-AI/lightning/pull/7437))
    * Refactored result handling in training loop ([#7506](https://github.com/Lightning-AI/lightning/pull/7506))
    * Moved attributes `hiddens` and `split_idx` to TrainLoop ([#7507](https://github.com/Lightning-AI/lightning/pull/7507))
    * Refactored the logic around manual and automatic optimization inside the optimizer loop ([#7526](https://github.com/Lightning-AI/lightning/pull/7526))
    * Simplified "should run validation" logic ([#7682](https://github.com/Lightning-AI/lightning/pull/7682))
    * Simplified logic for updating the learning rate for schedulers ([#7682](https://github.com/Lightning-AI/lightning/pull/7682))
    * Removed the `on_epoch` guard from the "should stop" validation check ([#7701](https://github.com/Lightning-AI/lightning/pull/7701))
    * Refactored internal loop interface; added new classes `FitLoop`, `TrainingEpochLoop`, `TrainingBatchLoop` ([#7871](https://github.com/Lightning-AI/lightning/pull/7871), [#8077](https://github.com/Lightning-AI/lightning/pull/8077))
    * Removed `pl.trainer.training_loop` ([#7985](https://github.com/Lightning-AI/lightning/pull/7985))
    * Refactored evaluation loop interface; added new classes `DataLoaderLoop`, `EvaluationLoop`, `EvaluationEpochLoop` ([#7990](https://github.com/Lightning-AI/lightning/pull/7990), [#8077](https://github.com/Lightning-AI/lightning/pull/8077))
    * Removed `pl.trainer.evaluation_loop` ([#8056](https://github.com/Lightning-AI/lightning/pull/8056))
    * Restricted public access to several internal functions ([#8024](https://github.com/Lightning-AI/lightning/pull/8024))
    * Refactored trainer `_run_*` functions and separate evaluation loops ([#8065](https://github.com/Lightning-AI/lightning/pull/8065))
    * Refactored prediction loop interface; added new classes `PredictionLoop`, `PredictionEpochLoop` ([#7700](https://github.com/Lightning-AI/lightning/pull/7700), [#8077](https://github.com/Lightning-AI/lightning/pull/8077))
    * Removed `pl.trainer.predict_loop` ([#8094](https://github.com/Lightning-AI/lightning/pull/8094))
    * Moved result teardown to the loops ([#8245](https://github.com/Lightning-AI/lightning/pull/8245))
    * Improve `Loop` API to better handle children `state_dict` and `progress` ([#8334](https://github.com/Lightning-AI/lightning/pull/8334))
- Refactored logging
    * Renamed and moved `core/step_result.py` to `trainer/connectors/logger_connector/result.py` ([#7736](https://github.com/Lightning-AI/lightning/pull/7736))
    * Dramatically simplify the `LoggerConnector` ([#7882](https://github.com/Lightning-AI/lightning/pull/7882))
    * `trainer.{logged,progress_bar,callback}_metrics` are now updated on-demand ([#7882](https://github.com/Lightning-AI/lightning/pull/7882))
    * Completely overhaul the `Result` object in favor of `ResultMetric` ([#7882](https://github.com/Lightning-AI/lightning/pull/7882))
    * Improve epoch-level reduction time and overall memory usage ([#7882](https://github.com/Lightning-AI/lightning/pull/7882))
    * Allow passing `self.log(batch_size=...)` ([#7891](https://github.com/Lightning-AI/lightning/pull/7891))
    * Each of the training loops now keeps its own results collection ([#7891](https://github.com/Lightning-AI/lightning/pull/7891))
    * Remove `EpochResultStore` and `HookResultStore` in favor of `ResultCollection` ([#7909](https://github.com/Lightning-AI/lightning/pull/7909))
    * Remove `MetricsHolder` ([#7909](https://github.com/Lightning-AI/lightning/pull/7909))
- Moved `ignore_scalar_return_in_dp` warning suppression to the DataParallelPlugin class ([#7421](https://github.com/Lightning-AI/lightning/pull/7421/))
- Changed the behaviour when logging evaluation step metrics to no longer append `/epoch_*` to the metric name ([#7351](https://github.com/Lightning-AI/lightning/pull/7351))
- Raised `ValueError` when a `None` value is `self.log`-ed ([#7771](https://github.com/Lightning-AI/lightning/pull/7771))
- Changed `resolve_training_type_plugins` to allow setting `num_nodes` and `sync_batchnorm` from `Trainer` setting ([#7026](https://github.com/Lightning-AI/lightning/pull/7026))
- Default `seed_everything(workers=True)` in the `LightningCLI` ([#7504](https://github.com/Lightning-AI/lightning/pull/7504))
- Changed `model.state_dict()` in `CheckpointConnector` to allow `training_type_plugin` to customize the model's `state_dict()` ([#7474](https://github.com/Lightning-AI/lightning/pull/7474))
- `MLflowLogger` now uses the env variable `MLFLOW_TRACKING_URI` as default tracking URI ([#7457](https://github.com/Lightning-AI/lightning/pull/7457))
- Changed `Trainer` arg and functionality from `reload_dataloaders_every_epoch` to `reload_dataloaders_every_n_epochs` ([#5043](https://github.com/Lightning-AI/lightning/pull/5043))
- Changed `WandbLogger(log_model={True/'all'})` to log models as artifacts ([#6231](https://github.com/Lightning-AI/lightning/pull/6231))
- MLFlowLogger now accepts `run_name` as an constructor argument ([#7622](https://github.com/Lightning-AI/lightning/pull/7622))
- Changed `teardown()` in `Accelerator` to allow `training_type_plugin` to customize `teardown` logic ([#7579](https://github.com/Lightning-AI/lightning/pull/7579))
- `Trainer.fit` now raises an error when using manual optimization with unsupported features such as `gradient_clip_val` or `accumulate_grad_batches` ([#7788](https://github.com/Lightning-AI/lightning/pull/7788))
- Accelerator hooks are called regardless if `LightningModule` overrides the same hooks ([#7826](https://github.com/Lightning-AI/lightning/pull/7826))
- Moved profilers to their own file ([#7822](https://github.com/Lightning-AI/lightning/pull/7822))
- The `on_after_backward` hook is now called on accumulating iterations. Use the `on_before_optimizer_step` hook to mimic the old behaviour ([#8328](https://github.com/Lightning-AI/lightning/pull/8328))
- The mixed precision loss is no longer unscaled before the `on_after_backward` hook. Use the `on_before_optimizer_step` hook to mimic the old behaviour  ([#8328](https://github.com/Lightning-AI/lightning/pull/8328))
- The `TrainingTypePlugin.{pre,post}_backward` hooks no longer take the `optimizer, opt_idx, should_accumulate` arguments ([#8328](https://github.com/Lightning-AI/lightning/pull/8328))
- The `PrecisionPlugin.backward` hooks no longer returns a value ([#8328](https://github.com/Lightning-AI/lightning/pull/8328))
- The `PrecisionPlugin.backward` hooks no longer takes a `should_accumulate` argument ([#8328](https://github.com/Lightning-AI/lightning/pull/8328))
- Added the `on_before_backward` hook ([#7865](https://github.com/Lightning-AI/lightning/pull/7865))
- `LightningCLI` now aborts with a clearer message if config already exists and disables save config during `fast_dev_run`([#7963](https://github.com/Lightning-AI/lightning/pull/7963))
- Saved the `LightningCLI` config on `setup` and only on the main process ([#8017](https://github.com/Lightning-AI/lightning/pull/8017))
- Dropped the `LightningCLI` `ArgumentParser` when pickling ([#8017](https://github.com/Lightning-AI/lightning/pull/8017))
- Skip `broadcast` if distributed not initialized for the spawn plugins ([#8017](https://github.com/Lightning-AI/lightning/pull/8017))
- `Trainer(resume_from_checkpoint=...)` now restores the model directly after `LightningModule.setup()`, which is before `LightningModule.configure_sharded_model()` ([#7652](https://github.com/Lightning-AI/lightning/pull/7652))
- Moved `torch.cuda.set_device()` to enable collective calls earlier in setup ([#8312](https://github.com/Lightning-AI/lightning/pull/8312))
- Used XLA utility API to move data to CPU (Single TPU core) ([#8078](https://github.com/Lightning-AI/lightning/pull/8078))
- Improved error messages in `replace_sampler` when the `DataLoader` attributes are not included in the signature or the signature is missing optional arguments ([#8519](https://github.com/Lightning-AI/lightning/pull/8519))
- Moved `DeviceDtypeModuleMixin` and `HyperparametersMixin` mixin to `core` ([#8396](https://github.com/Lightning-AI/lightning/pull/8396))
- Return the `default_root_dir` as the `log_dir` when the logger is a `LoggerCollection` ([#8187](https://github.com/Lightning-AI/lightning/pull/8187))

### Deprecated

- Deprecated `LightningModule.loaded_optimizer_states_dict` ([#8229](https://github.com/Lightning-AI/lightning/pull/8229))
- Standardized the dataloaders arguments of `trainer.{fit,valdiate,test,tune}` ([#7431](https://github.com/Lightning-AI/lightning/pull/7431))
- Deprecated `DataModule` properties: `has_prepared_data`, `has_setup_fit`, `has_setup_validate`, `has_setup_test`, `has_setup_predict`, `has_teardown_fit`, `has_teardown_validate`, `has_teardown_test`, `has_teardown_predict` ([#7657](https://github.com/Lightning-AI/lightning/pull/7657/))
- Deprecated `TrainerModelHooksMixin` in favor of `pl.utilities.signature_utils` ([#7422](https://github.com/Lightning-AI/lightning/pull/7422))
- Deprecated `num_nodes` and `sync_batchnorm` arguments in `DDPPlugin` and `DDPSpawnPlugin` ([#7026](https://github.com/Lightning-AI/lightning/pull/7026))
- Deprecated `self.log(sync_dist_op)` in favor of `self.log(reduce_fx)`. ([#7891](https://github.com/Lightning-AI/lightning/pull/7891))
- Deprecated `is_overridden(model=...)` in favor of `is_overridden(instance=...)` ([#7918](https://github.com/Lightning-AI/lightning/pull/7918))
- Deprecated automatically detaching returned extras with grads ([#7994](https://github.com/Lightning-AI/lightning/pull/7994))
- Deprecated default value of `monitor` argument in EarlyStopping callback to enforce `monitor` as a required argument ([#7907](https://github.com/Lightning-AI/lightning/pull/7907))
- Deprecated importing `rank_zero_{warn,deprecation}` directly from `pl.utilities.distributed` ([#8085](https://github.com/Lightning-AI/lightning/pull/8085))
- Deprecated the use of `CheckpointConnector.hpc_load()` in favor of `CheckpointConnector.restore()` ([#7652](https://github.com/Lightning-AI/lightning/pull/7652))
- Deprecated `ModelCheckpoint(every_n_val_epochs)` in favor of `ModelCheckpoint(every_n_epochs)` ([#8383](https://github.com/Lightning-AI/lightning/pull/8383))
- Deprecated `DDPPlugin.task_idx` in favor of `DDPPlugin.local_rank` ([#8203](https://github.com/Lightning-AI/lightning/pull/8203))
- Deprecated the `Trainer.train_loop` property in favor of `Trainer.fit_loop` ([#8025](https://github.com/Lightning-AI/lightning/pull/8025))
- Deprecated the `Trainer.disable_validation` property in favor of `not Trainer.enable_validation` ([#8291](https://github.com/Lightning-AI/lightning/pull/8291))
- Deprecated `mode` parameter in `ModelSummary` in favor of `max_depth` ([#8062](https://github.com/Lightning-AI/lightning/pull/8062))
- Deprecated `reload_dataloaders_every_epoch` argument of `Trainer` in favor of `reload_dataloaders_every_n_epochs` ([#5043](https://github.com/Lightning-AI/lightning/pull/5043))
- Deprecated `distributed_backend` argument for `Trainer` ([#8575](https://github.com/Lightning-AI/lightning/pull/8575))

### Removed

- Dropped official support/testing for PyTorch <1.6 ([#8288](https://github.com/Lightning-AI/lightning/pull/8288))
- Removed `ProfilerConnector` ([#7654](https://github.com/Lightning-AI/lightning/pull/7654))
- Pruned deprecated classif. metrics from `pl.metrics.functional.classification` ([#7499](https://github.com/Lightning-AI/lightning/pull/7499))
- Removed deprecated data parallel classes `LightningDataParallel` and `LightningDistributedDataParallel` from `pl.overrides.data_parallel` ([#7510](https://github.com/Lightning-AI/lightning/pull/7510))
- Removed deprecated trainer attributes - `get_model` and `accelerator_backend` ([#7502](https://github.com/Lightning-AI/lightning/pull/7502))
- Removed support for automatically monitoring the `val_loss` key with `ModelCheckpoint`. Pass your `monitor` of choice to the `ModelCheckpoint` instance instead ([#8293](https://github.com/Lightning-AI/lightning/pull/8293))
- Removed support for `self.log(tbptt_reduce_fx)` and `self.log(tbptt_pad_token)`. Please, open a discussion explaining your use-case if you relied on these. ([#7644](https://github.com/Lightning-AI/lightning/pull/7644))
- Removed deprecated utils modules `model_utils`, `warning_utils`, `xla_device_utils` and partially `argparse_utils` ([#7503](https://github.com/Lightning-AI/lightning/pull/7503))
- Removed `RPCPlugin` and `RPCSequentialPlugin`. If you were successfully using these plugins, please open a GitHub discussion about your use case ([#8101](https://github.com/Lightning-AI/lightning/pull/8101))
- Removed deprecated trainer attributes - `on_cpu`, `on_tpu`, `use_tpu`, `on_gpu`, `use_dp`, `use_ddp`, `use_ddp2`, `use_horovod`, `use_single_gpu` ([#7501](https://github.com/Lightning-AI/lightning/pull/7501))
- Removed deprecated `optimizer` argument in `LightningModule.manual_backward()`; Toggling optimizers in manual optimization should be done using `LightningModule.{un}toggle_optimizer()` ([#8287](https://github.com/Lightning-AI/lightning/pull/8287))
- Removed DeepSpeed FP16 Exception as FP32 is now supported ([#8462](https://github.com/Lightning-AI/lightning/pull/8462))
- Removed environment variable `PL_EXP_VERSION` from DDP subprocesses ([7403](https://github.com/Lightning-AI/lightning/pull/7403))

### Fixed

- Fixed the `GPUStatsMonitor` callbacks to use the correct GPU IDs if `CUDA_VISIBLE_DEVICES` set ([#8260](https://github.com/Lightning-AI/lightning/pull/8260))
- Fixed `lr_scheduler` checkpointed state by calling `update_lr_schedulers` before saving checkpoints ([#7877](https://github.com/Lightning-AI/lightning/pull/7877))
- Fixed ambiguous warning when both overfit and train dataloader shuffling are enabled ([#7685](https://github.com/Lightning-AI/lightning/pull/7685))
- Fixed dev debugger memory growing due to tracking events even when disabled ([#7875](https://github.com/Lightning-AI/lightning/pull/7875))
- Fixed `None` loss keys getting added in `training_epoch_end` when using manual optimization and not returning a loss ([#7772](https://github.com/Lightning-AI/lightning/pull/7772))
- Fixed a bug where `precision=64` with `accelerator='ddp_spawn'` would throw a pickle error ([#6924](https://github.com/Lightning-AI/lightning/pull/6924))
- Do not override the existing `epoch` value in `logged_metrics` when already logged by the user ([#7982](https://github.com/Lightning-AI/lightning/pull/7982))
- Support for manual optimization with DeepSpeed ([#7970](https://github.com/Lightning-AI/lightning/pull/7970))
- Fixed `dataloader_idx` argument value when predicting with only one `DataLoader` ([#7941](https://github.com/Lightning-AI/lightning/pull/7941))
- Fixed passing the `stage` argument of `Callback.{setup,teardown}` as a keyword ([#7973](https://github.com/Lightning-AI/lightning/pull/7973))
- Fixed metrics generated during `validation sanity checking` are cleaned on end ([#8171](https://github.com/Lightning-AI/lightning/pull/8171))
- Fixed `log_gpu_memory` metrics not being added to `logging` when nothing else is logged ([#8174](https://github.com/Lightning-AI/lightning/pull/8174))
- Fixed a bug where calling `log` with a `Metric` instance would raise an error if it was a nested attribute of the model ([#8181](https://github.com/Lightning-AI/lightning/pull/8181))
- Fixed a bug where using `precision=64` would cause buffers with complex dtype to be cast to real ([#8208](https://github.com/Lightning-AI/lightning/pull/8208))
- Fixed `is_overridden` returning true for wrapped functions with no changes ([#8296](https://github.com/Lightning-AI/lightning/pull/8296))
- Fixed a bug where `truncated_bptt_steps` would throw an AttributeError when the target RNN has multiple hidden states ([#8145](https://github.com/Lightning-AI/lightning/pull/8145))
- Fixed `self.optimizers()` not returning a single optimizer if it had been wrapped ([#8326](https://github.com/Lightning-AI/lightning/pull/8326))
- Fixed the `on_after_backward` hook not getting called when using manual optimization and no plugins ([#8328](https://github.com/Lightning-AI/lightning/pull/8328))
- Fixed the `LightningModule.backward` hook only getting called with the `apex` plugin when using manual optimization ([#8328](https://github.com/Lightning-AI/lightning/pull/8328))
- Fixed moving batch to device before sending it to the `on_*_batch_start`/`on_*_batch_end` callbacks and model hooks ([#7378](https://github.com/Lightning-AI/lightning/pull/7378))
- Fixed passing a custom `DDPPlugin` when choosing `accelerator="ddp_cpu"` for the accelerator ([#6208](https://github.com/Lightning-AI/lightning/pull/6208))
- Fixed missing call to `LightningModule.untoggle_optimizer` in training loop when running gradient accumulation with multiple optimizers ([#8284](https://github.com/Lightning-AI/lightning/pull/8284))
- Fixed hash of LightningEnum to work with value instead of name ([#8421](https://github.com/Lightning-AI/lightning/pull/8421)).
- Fixed a bug where an extra checkpoint was saved at the end of training if the `val_check_interval` did not align with the number of training batches ([#7724](https://github.com/Lightning-AI/lightning/pull/7724))
- Fixed hash of LightningEnum to work with value instead of name([#8421](https://github.com/Lightning-AI/lightning/pull/8421)).
- Fixed `move_data_to_device` to return the batch if the object `to` function didn't return `self` ([#8433](https://github.com/Lightning-AI/lightning/pull/8433))
- Fixed progress bar updates for Pod Training ([#8258](https://github.com/Lightning-AI/lightning/pull/8258))
- Fixed clearing dataloader references before attaching new dataloaders in consecutive `Trainer.{fit,validate,test,predict}´ runs ([#8442](https://github.com/Lightning-AI/lightning/pull/8442))
- Fixed memory leaks on GPU by moving `optimizer_states`, `ResultCollection.extra`, `ResultMetric` attributes, and `LoggerConnector` metrics to `cpu`. Also, delete the DDP wrapper on `teardown` ([#8490](https://github.com/Lightning-AI/lightning/pull/8490))
- Fixed `SWA` callback using LightningModule `prevent_trainer_and_dataloaders_deepcopy` to avoid OOM ([#8472](https://github.com/Lightning-AI/lightning/pull/8472))
- Fixed `ModelPruning` callback `on_save_checkpoint` to avoid making a `deepcopy` potentially leading to OOM ([#8472](https://github.com/Lightning-AI/lightning/pull/8472))
- Fixed the sampler replacement logic for `DataLoader`s which do not define all `DataLoader` attributes as `__init__` parameters ([#8519](https://github.com/Lightning-AI/lightning/pull/8519))
- Fixed DeepSpeed Windows support ([#8488](https://github.com/Lightning-AI/lightning/pull/8488))
- Fixed DeepSpeed not properly setting the trainer `lr_schedulers` attribute ([#8527](https://github.com/Lightning-AI/lightning/pull/8527))
- Fixed experiment version and log-dir divergence in DDP when using multiple `Trainer` instances in sequence ([7403](https://github.com/Lightning-AI/lightning/pull/7403))
- Enabled manual optimization for TPUs ([#8458](https://github.com/Lightning-AI/lightning/pull/8458))
- Fixed `accumulate_grad_batches` not been recomputed during model reload ([#5334](https://github.com/Lightning-AI/lightning/pull/5334))
- Fixed a `TypeError` when wrapping optimizers in the `HorovodPlugin` and running `Trainer.test` ([#7840](https://github.com/Lightning-AI/lightning/pull/7840))
- Fixed `BackboneFinetuning` restoration ([#8501](https://github.com/Lightning-AI/lightning/pull/8501))
- Fixed `lr_scheduler` with metric (e.g. `torch.optim.lr_scheduler.ReduceLROnPlateau`) when using `automatic_optimization = False` ([#7643](https://github.com/Lightning-AI/lightning/pull/7643))
- Fixed `DeepSpeed` breaking with no schedulers ([#8580](https://github.com/Lightning-AI/lightning/pull/8580))


## [1.3.8] - 2021-07-01

### Fixed

- Fixed a sync deadlock when checkpointing a `LightningModule` that uses a torchmetrics 0.4 `Metric` ([#8218](https://github.com/Lightning-AI/lightning/pull/8218))
- Fixed compatibility TorchMetrics v0.4 ([#8206](https://github.com/Lightning-AI/lightning/pull/8206))
- Added torchelastic check when sanitizing GPUs ([#8095](https://github.com/Lightning-AI/lightning/pull/8095))
- Fixed a DDP info message that was never shown ([#8111](https://github.com/Lightning-AI/lightning/pull/8111))
- Fixed metrics deprecation message at module import level ([#8163](https://github.com/Lightning-AI/lightning/pull/8163))
- Fixed a bug where an infinite recursion would be triggered when using the `BaseFinetuning` callback on a model that contains a `ModuleDict` ([#8170](https://github.com/Lightning-AI/lightning/pull/8170))
- Added a mechanism to detect `deadlock` for `DDP` when only 1 process trigger an `Exception`. The mechanism will `kill the processes` when it happens ([#8167](https://github.com/Lightning-AI/lightning/pull/8167))
- Fixed NCCL error when selecting non-consecutive device ids ([#8165](https://github.com/Lightning-AI/lightning/pull/8165))
- Fixed SWA to also work with `IterableDataset` ([#8172](https://github.com/Lightning-AI/lightning/pull/8172))


## [1.3.7] - 2021-06-22

### Fixed

- Fixed a bug where skipping an optimizer while using amp causes amp to trigger an assertion error ([#7975](https://github.com/Lightning-AI/lightning/pull/7975))
- Fixed deprecation messages not showing due to incorrect stacklevel ([#8002](https://github.com/Lightning-AI/lightning/pull/8002), [#8005](https://github.com/Lightning-AI/lightning/pull/8005))
- Fixed setting a `DistributedSampler` when using a distributed plugin in a custom accelerator ([#7814](https://github.com/Lightning-AI/lightning/pull/7814))
- Improved `PyTorchProfiler` chrome traces names ([#8009](https://github.com/Lightning-AI/lightning/pull/8009))
- Fixed moving the best score to device in `EarlyStopping` callback for TPU devices ([#7959](https://github.com/Lightning-AI/lightning/pull/7959))
- Fixes access to `callback_metrics` in ddp_spawn ([#7916](https://github.com/Lightning-AI/lightning/pull/7916))


## [1.3.6] - 2021-06-15

### Fixed

- Fixed logs overwriting issue for remote filesystems ([#7889](https://github.com/Lightning-AI/lightning/pull/7889))
- Fixed `DataModule.prepare_data` could only be called on the global rank 0 process ([#7945](https://github.com/Lightning-AI/lightning/pull/7945))
- Fixed setting `worker_init_fn` to seed dataloaders correctly when using DDP ([#7942](https://github.com/Lightning-AI/lightning/pull/7942))
- Fixed `BaseFinetuning` callback to properly handle parent modules w/ parameters ([#7931](https://github.com/Lightning-AI/lightning/pull/7931))


## [1.3.5] - 2021-06-08

### Added

- Added warning to Training Step output ([#7779](https://github.com/Lightning-AI/lightning/pull/7779))

### Fixed

- Fixed `LearningRateMonitor` and `BackboneFinetuning` ([#7835](https://github.com/Lightning-AI/lightning/pull/7835))
- Minor improvements to `apply_to_collection` and type signature of `log_dict` ([#7851](https://github.com/Lightning-AI/lightning/pull/7851))
- Fixed docker versions ([#7834](https://github.com/Lightning-AI/lightning/pull/7834))
- Fixed sharded training check for fp16 precision ([#7825](https://github.com/Lightning-AI/lightning/pull/7825))
- Fixed support for torch Module type hints in LightningCLI ([#7807](https://github.com/Lightning-AI/lightning/pull/7807))

### Changed

- Move `training_output` validation to after `train_step_end` ([#7868](https://github.com/Lightning-AI/lightning/pull/7868))


## [1.3.4] - 2021-06-01

### Fixed

- Fixed info message when max training time reached ([#7780](https://github.com/Lightning-AI/lightning/pull/7780))
- Fixed missing `__len__` method to `IndexBatchSamplerWrapper` ([#7681](https://github.com/Lightning-AI/lightning/pull/7681))


## [1.3.3] - 2021-05-27

### Changed

- Changed calling of `untoggle_optimizer(opt_idx)` out of the closure function ([#7563](https://github.com/Lightning-AI/lightning/pull/7563))

### Fixed

- Fixed `ProgressBar` pickling after calling `trainer.predict` ([#7608](https://github.com/Lightning-AI/lightning/pull/7608))
- Fixed broadcasting in multi-node, multi-gpu DDP using torch 1.7 ([#7592](https://github.com/Lightning-AI/lightning/pull/7592))
- Fixed dataloaders are not reset when tuning the model ([#7566](https://github.com/Lightning-AI/lightning/pull/7566))
- Fixed print errors in `ProgressBar` when `trainer.fit` is not called ([#7674](https://github.com/Lightning-AI/lightning/pull/7674))
- Fixed global step update when the epoch is skipped ([#7677](https://github.com/Lightning-AI/lightning/pull/7677))
- Fixed training loop total batch counter when accumulate grad batches was enabled ([#7692](https://github.com/Lightning-AI/lightning/pull/7692))


## [1.3.2] - 2021-05-18

### Changed

- `DataModule`s now avoid duplicate `{setup,teardown,prepare_data}` calls for the same stage ([#7238](https://github.com/Lightning-AI/lightning/pull/7238))

### Fixed

- Fixed parsing of multiple training dataloaders ([#7433](https://github.com/Lightning-AI/lightning/pull/7433))
- Fixed recursive passing of `wrong_type` keyword argument in `pl.utilities.apply_to_collection` ([#7433](https://github.com/Lightning-AI/lightning/pull/7433))
- Fixed setting correct `DistribType` for `ddp_cpu` (spawn) backend ([#7492](https://github.com/Lightning-AI/lightning/pull/7492))
- Fixed incorrect number of calls to LR scheduler when `check_val_every_n_epoch > 1` ([#7032](https://github.com/Lightning-AI/lightning/pull/7032))


## [1.3.1] - 2021-05-11

### Fixed

- Fixed DeepSpeed with IterableDatasets ([#7362](https://github.com/Lightning-AI/lightning/pull/7362))
- Fixed `Trainer.current_epoch` not getting restored after tuning ([#7434](https://github.com/Lightning-AI/lightning/pull/7434))
- Fixed local rank displayed in console log ([#7395](https://github.com/Lightning-AI/lightning/pull/7395))


## [1.3.0] - 2021-05-06

### Added

- Added support for the `EarlyStopping` callback to run at the end of the training epoch ([#6944](https://github.com/Lightning-AI/lightning/pull/6944))
- Added synchronization points before and after `setup` hooks are run ([#7202](https://github.com/Lightning-AI/lightning/pull/7202))
- Added a `teardown` hook to `ClusterEnvironment` ([#6942](https://github.com/Lightning-AI/lightning/pull/6942))
- Added utils for metrics to scalar conversions ([#7180](https://github.com/Lightning-AI/lightning/pull/7180))
- Added utils for NaN/Inf detection for gradients and parameters ([#6834](https://github.com/Lightning-AI/lightning/pull/6834))
- Added more explicit exception message when trying to execute `trainer.test()` or `trainer.validate()` with `fast_dev_run=True` ([#6667](https://github.com/Lightning-AI/lightning/pull/6667))
- Added `LightningCLI` class to provide simple reproducibility with minimum boilerplate training CLI (
    [#4492](https://github.com/Lightning-AI/lightning/pull/4492),
    [#6862](https://github.com/Lightning-AI/lightning/pull/6862),
    [#7156](https://github.com/Lightning-AI/lightning/pull/7156),
    [#7299](https://github.com/Lightning-AI/lightning/pull/7299))
- Added `gradient_clip_algorithm` argument to Trainer for gradient clipping by value ([#6123](https://github.com/Lightning-AI/lightning/pull/6123)).
- Added a way to print to terminal without breaking up the progress bar ([#5470](https://github.com/Lightning-AI/lightning/pull/5470))
- Added support to checkpoint after training steps in `ModelCheckpoint` callback ([#6146](https://github.com/Lightning-AI/lightning/pull/6146))
- Added `TrainerStatus.{INITIALIZING,RUNNING,FINISHED,INTERRUPTED}` ([#7173](https://github.com/Lightning-AI/lightning/pull/7173))
- Added `Trainer.validate()` method to perform one evaluation epoch over the validation set ([#4948](https://github.com/Lightning-AI/lightning/pull/4948))
- Added `LightningEnvironment` for Lightning-specific DDP ([#5915](https://github.com/Lightning-AI/lightning/pull/5915))
- Added `teardown()` hook to LightningDataModule ([#4673](https://github.com/Lightning-AI/lightning/pull/4673))
- Added `auto_insert_metric_name` parameter to `ModelCheckpoint` ([#6277](https://github.com/Lightning-AI/lightning/pull/6277))
- Added arg to `self.log` that enables users to give custom names when dealing with multiple dataloaders ([#6274](https://github.com/Lightning-AI/lightning/pull/6274))
- Added `teardown` method to `BaseProfiler` to enable subclasses defining post-profiling steps outside of `__del__` ([#6370](https://github.com/Lightning-AI/lightning/pull/6370))
- Added `setup` method to `BaseProfiler` to enable subclasses defining pre-profiling steps for every process ([#6633](https://github.com/Lightning-AI/lightning/pull/6633))
- Added no return warning to predict ([#6139](https://github.com/Lightning-AI/lightning/pull/6139))
- Added `Trainer.predict` config validation ([#6543](https://github.com/Lightning-AI/lightning/pull/6543))
- Added `AbstractProfiler` interface ([#6621](https://github.com/Lightning-AI/lightning/pull/6621))
- Added support for including module names for forward in the autograd trace of `PyTorchProfiler` ([#6349](https://github.com/Lightning-AI/lightning/pull/6349))
- Added support for the PyTorch 1.8.1 autograd profiler ([#6618](https://github.com/Lightning-AI/lightning/pull/6618))
- Added `outputs` parameter to callback's `on_validation_epoch_end` & `on_test_epoch_end` hooks ([#6120](https://github.com/Lightning-AI/lightning/pull/6120))
- Added `configure_sharded_model` hook ([#6679](https://github.com/Lightning-AI/lightning/pull/6679))
- Added support for `precision=64`, enabling training with double precision ([#6595](https://github.com/Lightning-AI/lightning/pull/6595))
- Added support for DDP communication hooks ([#6736](https://github.com/Lightning-AI/lightning/pull/6736))
- Added `artifact_location` argument to `MLFlowLogger` which will be passed to the `MlflowClient.create_experiment` call ([#6677](https://github.com/Lightning-AI/lightning/pull/6677))
- Added `model` parameter to precision plugins' `clip_gradients` signature (
    [#6764](https://github.com/Lightning-AI/lightning/pull/6764),
    [#7231](https://github.com/Lightning-AI/lightning/pull/7231))
- Added `is_last_batch` attribute to `Trainer` ([#6825](https://github.com/Lightning-AI/lightning/pull/6825))
- Added `LightningModule.lr_schedulers()` for manual optimization  ([#6567](https://github.com/Lightning-AI/lightning/pull/6567))
- Added `MpModelWrapper` in TPU Spawn ([#7045](https://github.com/Lightning-AI/lightning/pull/7045))
- Added `max_time` Trainer argument to limit training time ([#6823](https://github.com/Lightning-AI/lightning/pull/6823))
- Added `on_predict_{batch,epoch}_{start,end}` hooks ([#7141](https://github.com/Lightning-AI/lightning/pull/7141))
- Added new `EarlyStopping` parameters `stopping_threshold` and `divergence_threshold` ([#6868](https://github.com/Lightning-AI/lightning/pull/6868))
- Added `debug` flag to TPU Training Plugins (PT_XLA_DEBUG) ([#7219](https://github.com/Lightning-AI/lightning/pull/7219))
- Added new `UnrepeatedDistributedSampler` and `IndexBatchSamplerWrapper` for tracking distributed predictions ([#7215](https://github.com/Lightning-AI/lightning/pull/7215))
- Added `trainer.predict(return_predictions=None|False|True)` ([#7215](https://github.com/Lightning-AI/lightning/pull/7215))
- Added `BasePredictionWriter` callback to implement prediction saving ([#7127](https://github.com/Lightning-AI/lightning/pull/7127))
- Added `trainer.tune(scale_batch_size_kwargs, lr_find_kwargs)` arguments to configure the tuning algorithms ([#7258](https://github.com/Lightning-AI/lightning/pull/7258))
- Added `tpu_distributed` check for TPU Spawn barrier ([#7241](https://github.com/Lightning-AI/lightning/pull/7241))
- Added device updates to TPU Spawn for Pod training ([#7243](https://github.com/Lightning-AI/lightning/pull/7243))
- Added warning when missing `Callback` and using `resume_from_checkpoint` ([#7254](https://github.com/Lightning-AI/lightning/pull/7254))
- DeepSpeed single file saving ([#6900](https://github.com/Lightning-AI/lightning/pull/6900))
- Added Training type Plugins Registry (
    [#6982](https://github.com/Lightning-AI/lightning/pull/6982),
    [#7063](https://github.com/Lightning-AI/lightning/pull/7063),
    [#7214](https://github.com/Lightning-AI/lightning/pull/7214),
    [#7224](https://github.com/Lightning-AI/lightning/pull/7224)
)
- Add `ignore` param to `save_hyperparameters` ([#6056](https://github.com/Lightning-AI/lightning/pull/6056))

### Changed

- Changed `LightningModule.truncated_bptt_steps` to be property ([#7323](https://github.com/Lightning-AI/lightning/pull/7323))
- Changed `EarlyStopping` callback from by default running `EarlyStopping.on_validation_end` if only training is run. Set `check_on_train_epoch_end` to run the callback at the end of the train epoch instead of at the end of the validation epoch ([#7069](https://github.com/Lightning-AI/lightning/pull/7069))
- Renamed `pl.callbacks.swa` to `pl.callbacks.stochastic_weight_avg` ([#6259](https://github.com/Lightning-AI/lightning/pull/6259))
- Refactor `RunningStage` and `TrainerState` usage (
    [#4945](https://github.com/Lightning-AI/lightning/pull/4945),
    [#7173](https://github.com/Lightning-AI/lightning/pull/7173))
    * Added `RunningStage.SANITY_CHECKING`
    * Added `TrainerFn.{FITTING,VALIDATING,TESTING,PREDICTING,TUNING}`
    * Changed `trainer.evaluating` to return `True` if validating or testing
- Changed `setup()` and `teardown()` stage argument to take any of `{fit,validate,test,predict}` ([#6386](https://github.com/Lightning-AI/lightning/pull/6386))
- Changed profilers to save separate report files per state and rank ([#6621](https://github.com/Lightning-AI/lightning/pull/6621))
- The trainer no longer tries to save a checkpoint on exception or run callback's `on_train_end` functions ([#6864](https://github.com/Lightning-AI/lightning/pull/6864))
- Changed `PyTorchProfiler` to use `torch.autograd.profiler.record_function` to record functions ([#6349](https://github.com/Lightning-AI/lightning/pull/6349))
- Disabled `lr_scheduler.step()` in manual optimization  ([#6825](https://github.com/Lightning-AI/lightning/pull/6825))
- Changed warnings and recommendations for dataloaders in `ddp_spawn` ([#6762](https://github.com/Lightning-AI/lightning/pull/6762))
- `pl.seed_everything` will now also set the seed on the `DistributedSampler` ([#7024](https://github.com/Lightning-AI/lightning/pull/7024))
- Changed default setting for communication of multi-node training using `DDPShardedPlugin` ([#6937](https://github.com/Lightning-AI/lightning/pull/6937))
- `trainer.tune()` now returns the tuning result ([#7258](https://github.com/Lightning-AI/lightning/pull/7258))
- `LightningModule.from_datasets()` now accepts `IterableDataset` instances as training datasets. ([#7503](https://github.com/Lightning-AI/lightning/pull/7503))
- Changed `resume_from_checkpoint` warning to an error when the checkpoint file does not exist ([#7075](https://github.com/Lightning-AI/lightning/pull/7075))
- Automatically set `sync_batchnorm` for `training_type_plugin` ([#6536](https://github.com/Lightning-AI/lightning/pull/6536))
- Allowed training type plugin to delay optimizer creation ([#6331](https://github.com/Lightning-AI/lightning/pull/6331))
- Removed ModelSummary validation from train loop on_trainer_init ([#6610](https://github.com/Lightning-AI/lightning/pull/6610))
- Moved `save_function` to accelerator ([#6689](https://github.com/Lightning-AI/lightning/pull/6689))
- Updated DeepSpeed ZeRO ([#6546](https://github.com/Lightning-AI/lightning/pull/6546),
    [#6752](https://github.com/Lightning-AI/lightning/pull/6752),
    [#6142](https://github.com/Lightning-AI/lightning/pull/6142),
    [#6321](https://github.com/Lightning-AI/lightning/pull/6321))
- Improved verbose logging for `EarlyStopping` callback ([#6811](https://github.com/Lightning-AI/lightning/pull/6811))
- Run ddp_spawn dataloader checks on Windows ([#6930](https://github.com/Lightning-AI/lightning/pull/6930))
- Updated mlflow with using `resolve_tags` ([#6746](https://github.com/Lightning-AI/lightning/pull/6746))
- Moved `save_hyperparameters` to its own function ([#7119](https://github.com/Lightning-AI/lightning/pull/7119))
- Replaced `_DataModuleWrapper` with `__new__` ([#7289](https://github.com/Lightning-AI/lightning/pull/7289))
- Reset `current_fx` properties on lightning module in teardown ([#7247](https://github.com/Lightning-AI/lightning/pull/7247))
- Auto-set `DataLoader.worker_init_fn` with `seed_everything` ([#6960](https://github.com/Lightning-AI/lightning/pull/6960))
- Remove `model.trainer` call inside of dataloading mixin ([#7317](https://github.com/Lightning-AI/lightning/pull/7317))
- Split profilers module ([#6261](https://github.com/Lightning-AI/lightning/pull/6261))
- Ensure accelerator is valid if running interactively ([#5970](https://github.com/Lightning-AI/lightning/pull/5970))
- Disabled batch transfer in DP mode ([#6098](https://github.com/Lightning-AI/lightning/pull/6098))

### Deprecated

- Deprecated `outputs` in both `LightningModule.on_train_epoch_end` and `Callback.on_train_epoch_end` hooks ([#7339](https://github.com/Lightning-AI/lightning/pull/7339))
- Deprecated `Trainer.truncated_bptt_steps` in favor of `LightningModule.truncated_bptt_steps` ([#7323](https://github.com/Lightning-AI/lightning/pull/7323))
- Deprecated `outputs` in both `LightningModule.on_train_epoch_end` and `Callback.on_train_epoch_end` hooks ([#7339](https://github.com/Lightning-AI/lightning/pull/7339))
- Deprecated `LightningModule.grad_norm` in favor of `pl.utilities.grads.grad_norm` ([#7292](https://github.com/Lightning-AI/lightning/pull/7292))
- Deprecated the `save_function` property from the `ModelCheckpoint` callback ([#7201](https://github.com/Lightning-AI/lightning/pull/7201))
- Deprecated `LightningModule.write_predictions` and `LightningModule.write_predictions_dict` ([#7066](https://github.com/Lightning-AI/lightning/pull/7066))
- Deprecated `TrainerLoggingMixin` in favor of a separate utilities module for metric handling ([#7180](https://github.com/Lightning-AI/lightning/pull/7180))
- Deprecated `TrainerTrainingTricksMixin` in favor of a separate utilities module for NaN/Inf detection for gradients and parameters ([#6834](https://github.com/Lightning-AI/lightning/pull/6834))
- `period` has been deprecated in favor of `every_n_val_epochs` in the `ModelCheckpoint` callback ([#6146](https://github.com/Lightning-AI/lightning/pull/6146))
- Deprecated `trainer.running_sanity_check` in favor of `trainer.sanity_checking` ([#4945](https://github.com/Lightning-AI/lightning/pull/4945))
- Deprecated `Profiler(output_filename)` in favor of `dirpath` and `filename` ([#6621](https://github.com/Lightning-AI/lightning/pull/6621))
- Deprecated `PyTorchProfiler(profiled_functions)` in favor of `record_functions` ([#6349](https://github.com/Lightning-AI/lightning/pull/6349))
- Deprecated `@auto_move_data` in favor of `trainer.predict` ([#6993](https://github.com/Lightning-AI/lightning/pull/6993))
- Deprecated `Callback.on_load_checkpoint(checkpoint)` in favor of `Callback.on_load_checkpoint(trainer, pl_module, checkpoint)` ([#7253](https://github.com/Lightning-AI/lightning/pull/7253))
- Deprecated metrics in favor of `torchmetrics` (
    [#6505](https://github.com/Lightning-AI/lightning/pull/6505),
    [#6530](https://github.com/Lightning-AI/lightning/pull/6530),
    [#6540](https://github.com/Lightning-AI/lightning/pull/6540),
    [#6547](https://github.com/Lightning-AI/lightning/pull/6547),
    [#6515](https://github.com/Lightning-AI/lightning/pull/6515),
    [#6572](https://github.com/Lightning-AI/lightning/pull/6572),
    [#6573](https://github.com/Lightning-AI/lightning/pull/6573),
    [#6584](https://github.com/Lightning-AI/lightning/pull/6584),
    [#6636](https://github.com/Lightning-AI/lightning/pull/6636),
    [#6637](https://github.com/Lightning-AI/lightning/pull/6637),
    [#6649](https://github.com/Lightning-AI/lightning/pull/6649),
    [#6659](https://github.com/Lightning-AI/lightning/pull/6659),
    [#7131](https://github.com/Lightning-AI/lightning/pull/7131),
)
- Deprecated the `LightningModule.datamodule` getter and setter methods; access them through `Trainer.datamodule` instead ([#7168](https://github.com/Lightning-AI/lightning/pull/7168))
- Deprecated the use of `Trainer(gpus="i")` (string) for selecting the i-th GPU; from v1.5 this will set the number of GPUs instead of the index ([#6388](https://github.com/Lightning-AI/lightning/pull/6388))

### Removed

- Removed the `exp_save_path` property from the `LightningModule` ([#7266](https://github.com/Lightning-AI/lightning/pull/7266))
- Removed training loop explicitly calling `EarlyStopping.on_validation_end` if no validation is run ([#7069](https://github.com/Lightning-AI/lightning/pull/7069))
- Removed `automatic_optimization` as a property from the training loop in favor of `LightningModule.automatic_optimization` ([#7130](https://github.com/Lightning-AI/lightning/pull/7130))
- Removed evaluation loop legacy returns for `*_epoch_end` hooks ([#6973](https://github.com/Lightning-AI/lightning/pull/6973))
- Removed support for passing a bool value to `profiler` argument of Trainer ([#6164](https://github.com/Lightning-AI/lightning/pull/6164))
- Removed no return warning from val/test step ([#6139](https://github.com/Lightning-AI/lightning/pull/6139))
- Removed passing a `ModelCheckpoint` instance to `Trainer(checkpoint_callback)` ([#6166](https://github.com/Lightning-AI/lightning/pull/6166))
- Removed deprecated Trainer argument `enable_pl_optimizer` and `automatic_optimization` ([#6163](https://github.com/Lightning-AI/lightning/pull/6163))
- Removed deprecated metrics ([#6161](https://github.com/Lightning-AI/lightning/pull/6161))
    * from `pl.metrics.functional.classification` removed `to_onehot`, `to_categorical`, `get_num_classes`, `roc`, `multiclass_roc`, `average_precision`, `precision_recall_curve`, `multiclass_precision_recall_curve`
    * from `pl.metrics.functional.reduction` removed `reduce`, `class_reduce`
- Removed deprecated `ModelCheckpoint` arguments `prefix`, `mode="auto"` ([#6162](https://github.com/Lightning-AI/lightning/pull/6162))
- Removed `mode='auto'` from `EarlyStopping` ([#6167](https://github.com/Lightning-AI/lightning/pull/6167))
- Removed `epoch` and `step` arguments from `ModelCheckpoint.format_checkpoint_name()`, these are now included in the `metrics` argument ([#7344](https://github.com/Lightning-AI/lightning/pull/7344))
- Removed legacy references for magic keys in the `Result` object ([#6016](https://github.com/Lightning-AI/lightning/pull/6016))
- Removed deprecated `LightningModule` `hparams` setter ([#6207](https://github.com/Lightning-AI/lightning/pull/6207))
- Removed legacy code to log or include metrics in the progress bar by returning them in a dict with the `"log"/"progress_bar"` magic keys. Use `self.log` instead ([#6734](https://github.com/Lightning-AI/lightning/pull/6734))
- Removed `trainer.fit()` return value of `1`. It has no return now ([#7237](https://github.com/Lightning-AI/lightning/pull/7237))
- Removed `logger_connector` legacy code ([#6733](https://github.com/Lightning-AI/lightning/pull/6733))
- Removed unused mixin attributes ([#6487](https://github.com/Lightning-AI/lightning/pull/6487))

### Fixed

- Fixed NaN errors in progress bars when training with iterable datasets with no length defined ([#7306](https://github.com/Lightning-AI/lightning/pull/7306))
- Fixed attaching train and validation dataloaders when `reload_dataloaders_every_epoch=True` and `num_sanity_val_steps=0` ([#7207](https://github.com/Lightning-AI/lightning/pull/7207))
- Added a barrier in the accelerator `teardown` to synchronize processes before execution finishes ([#6814](https://github.com/Lightning-AI/lightning/pull/6814))
- Fixed multi-node DDP sub-process launch by using `local_rank` instead of `global_rank` for main process assertion ([#7061](https://github.com/Lightning-AI/lightning/pull/7061))
- Fixed incorrect removal of `WORLD_SIZE` environment variable in DDP training when launching with torch distributed/torchelastic ([#6942](https://github.com/Lightning-AI/lightning/pull/6942))
- Made the `Plugin.reduce` method more consistent across all Plugins to reflect a mean-reduction by default ([#6011](https://github.com/Lightning-AI/lightning/pull/6011))
- Move lightning module to correct device type when using LightningDistributedWrapper ([#6070](https://github.com/Lightning-AI/lightning/pull/6070))
- Do not print top-k verbose log with `ModelCheckpoint(monitor=None)` ([#6109](https://github.com/Lightning-AI/lightning/pull/6109))
- Fixed `ModelCheckpoint(save_top_k=0, save_last=True)` not saving the `last` checkpoint ([#6136](https://github.com/Lightning-AI/lightning/pull/6136))
- Fixed `.teardown(stage='fit')` and `.on_fit_{start,end}()` getting called during `trainer.test` ([#6386](https://github.com/Lightning-AI/lightning/pull/6386))
- Fixed LightningModule `all_gather` on cpu tensors ([#6416](https://github.com/Lightning-AI/lightning/pull/6416))
- Fixed torch distributed not available in setup hook for DDP ([#6506](https://github.com/Lightning-AI/lightning/pull/6506))
- Fixed `trainer.tuner.{lr_find,scale_batch_size}` not setting the `Trainer` state properly ([#7258](https://github.com/Lightning-AI/lightning/pull/7258))
- Fixed bug where the learning rate schedulers did not follow the optimizer frequencies ([#4868](https://github.com/Lightning-AI/lightning/pull/4868))
- Fixed pickle error checker to now check for `pickle.PickleError` to catch all pickle errors ([#6917](https://github.com/Lightning-AI/lightning/pull/6917))
- Fixed a bug where the outputs object passed to `LightningModule.training_epoch_end` was different from the object passed to the `on_train_end_epoch` hook ([#6969](https://github.com/Lightning-AI/lightning/pull/6969))
- Fixed a bug where the outputs passed to `train_batch_end` would be lists even when using a single optimizer and no truncated backprop through time steps ([#6969](https://github.com/Lightning-AI/lightning/pull/6969))
- Fixed bug for trainer error handling which would cause hang for distributed training ([#6864](https://github.com/Lightning-AI/lightning/pull/6864))
- Fixed `self.device` not returning the correct device in replicas of data-parallel ([#6414](https://github.com/Lightning-AI/lightning/pull/6414))
- Fixed `lr_find` trying beyond `num_training` steps and suggesting a too high learning rate ([#7076](https://github.com/Lightning-AI/lightning/pull/7076))
- Fixed logger creating incorrect version folder in DDP with repeated `Trainer.fit` calls ([#7077](https://github.com/Lightning-AI/lightning/pull/7077))
- Fixed metric objects passed directly to `self.log` not being reset correctly ([#7055](https://github.com/Lightning-AI/lightning/pull/7055))
- Fixed `CombinedLoader` in distributed settings for validation / testing ([#7102](https://github.com/Lightning-AI/lightning/pull/7102))
- Fixed the save_dir in `WandbLogger` when the run was initiated externally ([#7106](https://github.com/Lightning-AI/lightning/pull/7106))
- Fixed `num_sanity_val_steps` affecting reproducibility of training data shuffling ([#7014](https://github.com/Lightning-AI/lightning/pull/7014))
- Fixed resetting device after `fitting/evaluating/predicting` ([#7188](https://github.com/Lightning-AI/lightning/pull/7188))
- Fixed bug where `trainer.tuner.scale_batch_size(max_trials=0)` would not return the correct batch size result ([#7262](https://github.com/Lightning-AI/lightning/pull/7262))
- Fixed metrics not being properly logged with `precision=16` and `manual_optimization` ([#7228](https://github.com/Lightning-AI/lightning/pull/7228))
- Fixed `BaseFinetuning` properly reloading `optimizer_states` when using `resume_from_checkpoint` ([#6891](https://github.com/Lightning-AI/lightning/pull/6891))
- Fixed `parameters_to_ignore` not properly set to DDPWrapper ([#7239](https://github.com/Lightning-AI/lightning/pull/7239))
- Fixed parsing of `fast_dev_run=True` with the built-in `ArgumentParser` ([#7240](https://github.com/Lightning-AI/lightning/pull/7240))
- Fixed handling an `IterableDataset` that fails to produce a batch at the beginning of an epoch ([#7294](https://github.com/Lightning-AI/lightning/pull/7294))
- Fixed `LightningModule.save_hyperparameters()` when attempting to save an empty container ([#7268](https://github.com/Lightning-AI/lightning/pull/7268))
- Fixed `apex` not properly instantiated when running with `ddp` ([#7274](https://github.com/Lightning-AI/lightning/pull/7274))
- Fixed optimizer `state` not moved to `GPU` ([#7277](https://github.com/Lightning-AI/lightning/pull/7277))
- Fixed custom init args for `WandbLogger` ([#6989](https://github.com/Lightning-AI/lightning/pull/6989))
- Fixed a bug where an error would be raised if the train dataloader sometimes produced None for a batch ([#7342](https://github.com/Lightning-AI/lightning/pull/7342))
- Fixed examples (
    [#6600](https://github.com/Lightning-AI/lightning/pull/6600),
    [#6638](https://github.com/Lightning-AI/lightning/pull/6638),
    [#7096](https://github.com/Lightning-AI/lightning/pull/7096),
    [#7246](https://github.com/Lightning-AI/lightning/pull/7246),
    [#6357](https://github.com/Lightning-AI/lightning/pull/6357),
    [#6476](https://github.com/Lightning-AI/lightning/pull/6476),
    [#6294](https://github.com/Lightning-AI/lightning/pull/6294),
    [#6373](https://github.com/Lightning-AI/lightning/pull/6373),
    [#6088](https://github.com/Lightning-AI/lightning/pull/6088),
    [#7398](https://github.com/Lightning-AI/lightning/pull/7398)
)
- Resolved schedule step bug for PyTorch Profiler ([#6674](https://github.com/Lightning-AI/lightning/pull/6674),
    [#6681](https://github.com/Lightning-AI/lightning/pull/6681))
- Updated logic for checking TPUs availability ([#6767](https://github.com/Lightning-AI/lightning/pull/6767))
- Resolve TPU miss rendezvous ([#6781](https://github.com/Lightning-AI/lightning/pull/6781))
- Fixed auto-scaling mode when calling tune method on trainer ([#7321](https://github.com/Lightning-AI/lightning/pull/7321))
- Fixed finetuning complex models correctly unfreezes ([#6880](https://github.com/Lightning-AI/lightning/pull/6880))
- Ensure we set the eval/train flag correctly on accelerator model ([#6877](https://github.com/Lightning-AI/lightning/pull/6877))
- Set better defaults for `rank_zero_only.rank` when training is launched with SLURM and torchelastic ([#6802](https://github.com/Lightning-AI/lightning/pull/6802))
- Fixed matching the number of outputs of backward with forward for AllGatherGrad ([#6625](https://github.com/Lightning-AI/lightning/pull/6625))
- Fixed the `gradient_clip_algorithm` has no effect ([#6928](https://github.com/Lightning-AI/lightning/pull/6928))
- Fixed CUDA OOM detection and handling ([#6934](https://github.com/Lightning-AI/lightning/pull/6934))
- Fixed `unfreeze_and_add_param_group` expects `modules` rather than `module` ([#6822](https://github.com/Lightning-AI/lightning/pull/6822))
- Fixed DPP + SyncBN when move on device ([#6838](https://github.com/Lightning-AI/lightning/pull/6838))
- Fixed missing arguments in `lr_find` call ([#6784](https://github.com/Lightning-AI/lightning/pull/6784))
- Fixed `set_default_tensor_type` to `torch.DoubleTensor` with precision=64 ([#7108](https://github.com/Lightning-AI/lightning/pull/7108))
- Fixed `NeptuneLogger.log_text(step=None)` ([#7194](https://github.com/Lightning-AI/lightning/pull/7194))
- Fixed importing torchtext batch ([#6365](https://github.com/Lightning-AI/lightning/pull/6365),
    [#6323](https://github.com/Lightning-AI/lightning/pull/6323),
    [#6211](https://github.com/Lightning-AI/lightning/pull/6211))


## [1.2.9] - 2021-04-20

### Fixed

- Fixed the order to call for world ranks & the `root_device` property in `TPUSpawnPlugin` ([#7074](https://github.com/Lightning-AI/lightning/pull/7074))
- Fixed multi-gpu join for Horovod ([#6954](https://github.com/Lightning-AI/lightning/pull/6954))
- Fixed parsing for pre-release package versions ([#6999](https://github.com/Lightning-AI/lightning/pull/6999))


## [1.2.8] - 2021-04-14

### Added

- Added TPUSpawn + IterableDataset error message ([#6875](https://github.com/Lightning-AI/lightning/pull/6875))

### Fixed

- Fixed process rank not being available right away after `Trainer` instantiation ([#6941](https://github.com/Lightning-AI/lightning/pull/6941))
- Fixed `sync_dist` for tpus ([#6950](https://github.com/Lightning-AI/lightning/pull/6950))
- Fixed `AttributeError` for `require_backward_grad_sync` when running manual optimization with sharded plugin ([#6915](https://github.com/Lightning-AI/lightning/pull/6915))
- Fixed `--gpus` default for parser returned by `Trainer.add_argparse_args` ([#6898](https://github.com/Lightning-AI/lightning/pull/6898))
- Fixed TPU Spawn all gather ([#6896](https://github.com/Lightning-AI/lightning/pull/6896))
- Fixed `EarlyStopping` logic when `min_epochs` or `min_steps` requirement is not met ([#6705](https://github.com/Lightning-AI/lightning/pull/6705))
- Fixed csv extension check ([#6436](https://github.com/Lightning-AI/lightning/pull/6436))
- Fixed checkpoint issue when using Horovod distributed backend ([#6958](https://github.com/Lightning-AI/lightning/pull/6958))
- Fixed tensorboard exception raising ([#6901](https://github.com/Lightning-AI/lightning/pull/6901))
- Fixed setting the eval/train flag correctly on accelerator model ([#6983](https://github.com/Lightning-AI/lightning/pull/6983))
- Fixed DDP_SPAWN compatibility with bug_report_model.py ([#6892](https://github.com/Lightning-AI/lightning/pull/6892))
- Fixed bug where `BaseFinetuning.flatten_modules()` was duplicating leaf node parameters ([#6879](https://github.com/Lightning-AI/lightning/pull/6879))
- Set better defaults for `rank_zero_only.rank` when training is launched with SLURM and torchelastic:
    * Support SLURM and torchelastic global rank environment variables ([#5715](https://github.com/Lightning-AI/lightning/pull/5715))
    * Remove hardcoding of local rank in accelerator connector ([#6878](https://github.com/Lightning-AI/lightning/pull/6878))


## [1.2.7] - 2021-04-06

### Fixed

- Fixed resolve a bug with omegaconf and xm.save ([#6741](https://github.com/Lightning-AI/lightning/pull/6741))
- Fixed an issue with IterableDataset when __len__ is not defined ([#6828](https://github.com/Lightning-AI/lightning/pull/6828))
- Sanitize None params during pruning ([#6836](https://github.com/Lightning-AI/lightning/pull/6836))
- Enforce an epoch scheduler interval when using SWA ([#6588](https://github.com/Lightning-AI/lightning/pull/6588))
- Fixed TPU Colab hang issue, post training ([#6816](https://github.com/Lightning-AI/lightning/pull/6816))
- Fixed a bug where `TensorBoardLogger` would give a warning and not log correctly to a symbolic link `save_dir` ([#6730](https://github.com/Lightning-AI/lightning/pull/6730))
- Fixed bug where `predict` could not be used when `progress_bar_refresh_rate=0` ([#6884](https://github.com/Lightning-AI/lightning/pull/6884))


## [1.2.6] - 2021-03-30

### Changed

- Changed the behavior of `on_epoch_start` to run at the beginning of validation & test epoch ([#6498](https://github.com/Lightning-AI/lightning/pull/6498))

### Removed

- Removed legacy code to include `step` dictionary returns in `callback_metrics`. Use `self.log_dict` instead. ([#6682](https://github.com/Lightning-AI/lightning/pull/6682))

### Fixed

- Fixed `DummyLogger.log_hyperparams` raising a `TypeError` when running with `fast_dev_run=True` ([#6398](https://github.com/Lightning-AI/lightning/pull/6398))
- Fixed error on TPUs when there was no `ModelCheckpoint` ([#6654](https://github.com/Lightning-AI/lightning/pull/6654))
- Fixed `trainer.test` freeze on TPUs ([#6654](https://github.com/Lightning-AI/lightning/pull/6654))
- Fixed a bug where gradients were disabled after calling `Trainer.predict` ([#6657](https://github.com/Lightning-AI/lightning/pull/6657))
- Fixed bug where no TPUs were detected in a TPU pod env ([#6719](https://github.com/Lightning-AI/lightning/pull/6719))


## [1.2.5] - 2021-03-23

### Changed

- Update Gradient Clipping for the TPU Accelerator ([#6576](https://github.com/Lightning-AI/lightning/pull/6576))
- Refactored setup for typing friendly ([#6590](https://github.com/Lightning-AI/lightning/pull/6590))

### Fixed

- Fixed a bug where `all_gather` would not work correctly with `tpu_cores=8` ([#6587](https://github.com/Lightning-AI/lightning/pull/6587))
- Fixed comparing required versions ([#6434](https://github.com/Lightning-AI/lightning/pull/6434))
- Fixed duplicate logs appearing in console when using the python logging module ([#6275](https://github.com/Lightning-AI/lightning/pull/6275))
- Added Autocast in validation, test and predict modes for Native AMP ([#6565](https://github.com/Lightning-AI/lightning/pull/6565))


## [1.2.4] - 2021-03-16

### Changed

- Changed the default of `find_unused_parameters` back to `True` in DDP and DDP Spawn ([#6438](https://github.com/Lightning-AI/lightning/pull/6438))

### Fixed

- Expose DeepSpeed loss parameters to allow users to fix loss instability ([#6115](https://github.com/Lightning-AI/lightning/pull/6115))
- Fixed DP reduction with collection ([#6324](https://github.com/Lightning-AI/lightning/pull/6324))
- Fixed an issue where the tuner would not tune the learning rate if also tuning the batch size ([#4688](https://github.com/Lightning-AI/lightning/pull/4688))
- Fixed broadcast to use PyTorch `broadcast_object_list` and add `reduce_decision` ([#6410](https://github.com/Lightning-AI/lightning/pull/6410))
- Fixed logger creating directory structure too early in DDP ([#6380](https://github.com/Lightning-AI/lightning/pull/6380))
- Fixed DeepSpeed additional memory use on rank 0 when default device not set early enough ([#6460](https://github.com/Lightning-AI/lightning/pull/6460))
- Fixed an issue with `Tuner.scale_batch_size` not finding the batch size attribute in the datamodule ([#5968](https://github.com/Lightning-AI/lightning/pull/5968))
- Fixed an exception in the layer summary when the model contains torch.jit scripted submodules ([#6511](https://github.com/Lightning-AI/lightning/pull/6511))
- Fixed when Train loop config was run during `Trainer.predict` ([#6541](https://github.com/Lightning-AI/lightning/pull/6541))


## [1.2.3] - 2021-03-09

### Fixed

- Fixed `ModelPruning(make_pruning_permanent=True)` pruning buffers getting removed when saved during training ([#6073](https://github.com/Lightning-AI/lightning/pull/6073))
- Fixed when `_stable_1d_sort` to work when `n >= N` ([#6177](https://github.com/Lightning-AI/lightning/pull/6177))
- Fixed `AttributeError` when `logger=None` on TPU ([#6221](https://github.com/Lightning-AI/lightning/pull/6221))
- Fixed PyTorch Profiler with `emit_nvtx` ([#6260](https://github.com/Lightning-AI/lightning/pull/6260))
- Fixed `trainer.test` from `best_path` hangs after calling `trainer.fit`  ([#6272](https://github.com/Lightning-AI/lightning/pull/6272))
- Fixed `SingleTPU` calling `all_gather` ([#6296](https://github.com/Lightning-AI/lightning/pull/6296))
- Ensure we check DeepSpeed/Sharded in multi-node DDP ([#6297](https://github.com/Lightning-AI/lightning/pull/6297)
- Check `LightningOptimizer` doesn't delete optimizer hooks ([#6305](https://github.com/Lightning-AI/lightning/pull/6305)
- Resolve memory leak for evaluation ([#6326](https://github.com/Lightning-AI/lightning/pull/6326)
- Ensure that clip gradients is only called if the value is greater than 0 ([#6330](https://github.com/Lightning-AI/lightning/pull/6330)
- Fixed `Trainer` not resetting `lightning_optimizers` when calling `Trainer.fit()` multiple times ([#6372](https://github.com/Lightning-AI/lightning/pull/6372))


## [1.2.2] - 2021-03-02

### Added

- Added `checkpoint` parameter to callback's `on_save_checkpoint` hook ([#6072](https://github.com/Lightning-AI/lightning/pull/6072))

### Changed

- Changed the order of `backward`, `step`, `zero_grad` to `zero_grad`, `backward`, `step` ([#6147](https://github.com/Lightning-AI/lightning/pull/6147))
- Changed default for DeepSpeed CPU Offload to False, due to prohibitively slow speeds at smaller scale ([#6262](https://github.com/Lightning-AI/lightning/pull/6262))

### Fixed

- Fixed epoch level schedulers not being called when `val_check_interval < 1.0` ([#6075](https://github.com/Lightning-AI/lightning/pull/6075))
- Fixed multiple early stopping callbacks ([#6197](https://github.com/Lightning-AI/lightning/pull/6197))
- Fixed incorrect usage of `detach()`, `cpu()`, `to()` ([#6216](https://github.com/Lightning-AI/lightning/pull/6216))
- Fixed LBFGS optimizer support which didn't converge in automatic optimization ([#6147](https://github.com/Lightning-AI/lightning/pull/6147))
- Prevent `WandbLogger` from dropping values ([#5931](https://github.com/Lightning-AI/lightning/pull/5931))
- Fixed error thrown when using valid distributed mode in multi node ([#6297](https://github.com/Lightning-AI/lightning/pull/6297)


## [1.2.1] - 2021-02-23

### Fixed

- Fixed incorrect yield logic for the amp autocast context manager ([#6080](https://github.com/Lightning-AI/lightning/pull/6080))
- Fixed priority of plugin/accelerator when setting distributed mode ([#6089](https://github.com/Lightning-AI/lightning/pull/6089))
- Fixed error message for AMP + CPU incompatibility ([#6107](https://github.com/Lightning-AI/lightning/pull/6107))
- Disabled batch transfer in DP mode ([#6093](https://github.com/Lightning-AI/lightning/pull/6093))


## [1.2.0] - 2021-02-18

### Added

- Added `DataType`, `AverageMethod` and `MDMCAverageMethod` enum in metrics ([#5657](https://github.com/Lightning-AI/lightning/pull/5689))
- Added support for summarized model total params size in megabytes ([#5590](https://github.com/Lightning-AI/lightning/pull/5590))
- Added support for multiple train loaders ([#1959](https://github.com/Lightning-AI/lightning/pull/1959))
- Added `Accuracy` metric now generalizes to Top-k accuracy for (multi-dimensional) multi-class inputs using the `top_k` parameter ([#4838](https://github.com/Lightning-AI/lightning/pull/4838))
- Added `Accuracy` metric now enables the computation of subset accuracy for multi-label or multi-dimensional multi-class inputs with the `subset_accuracy` parameter ([#4838](https://github.com/Lightning-AI/lightning/pull/4838))
- Added `HammingDistance` metric to compute the hamming distance (loss) ([#4838](https://github.com/Lightning-AI/lightning/pull/4838))
- Added `max_fpr` parameter to `auroc` metric for computing partial auroc metric ([#3790](https://github.com/Lightning-AI/lightning/pull/3790))
- Added `StatScores` metric to compute the number of true positives, false positives, true negatives and false negatives ([#4839](https://github.com/Lightning-AI/lightning/pull/4839))
- Added `R2Score` metric ([#5241](https://github.com/Lightning-AI/lightning/pull/5241))
- Added `LambdaCallback` ([#5347](https://github.com/Lightning-AI/lightning/pull/5347))
- Added `BackboneLambdaFinetuningCallback` ([#5377](https://github.com/Lightning-AI/lightning/pull/5377))
- Accelerator `all_gather` supports collection ([#5221](https://github.com/Lightning-AI/lightning/pull/5221))
- Added `image_gradients` functional metric to compute the image gradients of a given input image. ([#5056](https://github.com/Lightning-AI/lightning/pull/5056))
- Added `MetricCollection` ([#4318](https://github.com/Lightning-AI/lightning/pull/4318))
- Added `.clone()` method to metrics ([#4318](https://github.com/Lightning-AI/lightning/pull/4318))
- Added `IoU` class interface ([#4704](https://github.com/Lightning-AI/lightning/pull/4704))
- Support to tie weights after moving model to TPU via `on_post_move_to_device` hook
- Added missing val/test hooks in `LightningModule` ([#5467](https://github.com/Lightning-AI/lightning/pull/5467))
- The `Recall` and `Precision` metrics (and their functional counterparts `recall` and `precision`) can now be generalized to Recall@K and Precision@K with the use of `top_k` parameter ([#4842](https://github.com/Lightning-AI/lightning/pull/4842))
- Added `ModelPruning` Callback ([#5618](https://github.com/Lightning-AI/lightning/pull/5618),
    [#5825](https://github.com/Lightning-AI/lightning/pull/5825),
    [#6045](https://github.com/Lightning-AI/lightning/pull/6045))
- Added `PyTorchProfiler` ([#5560](https://github.com/Lightning-AI/lightning/pull/5560))
- Added compositional metrics ([#5464](https://github.com/Lightning-AI/lightning/pull/5464))
- Added Trainer method `predict(...)` for high performance predictions ([#5579](https://github.com/Lightning-AI/lightning/pull/5579))
- Added `on_before_batch_transfer` and `on_after_batch_transfer` data hooks ([#3671](https://github.com/Lightning-AI/lightning/pull/3671))
- Added AUC/AUROC class interface ([#5479](https://github.com/Lightning-AI/lightning/pull/5479))
- Added `PredictLoop` object ([#5752](https://github.com/Lightning-AI/lightning/pull/5752))
- Added `QuantizationAwareTraining` callback ([#5706](https://github.com/Lightning-AI/lightning/pull/5706),
    [#6040](https://github.com/Lightning-AI/lightning/pull/6040))
- Added `LightningModule.configure_callbacks` to enable the definition of model-specific callbacks ([#5621](https://github.com/Lightning-AI/lightning/pull/5621))
- Added `dim` to `PSNR` metric for mean-squared-error reduction ([#5957](https://github.com/Lightning-AI/lightning/pull/5957))
- Added promxial policy optimization template to pl_examples ([#5394](https://github.com/Lightning-AI/lightning/pull/5394))
- Added `log_graph` to `CometLogger` ([#5295](https://github.com/Lightning-AI/lightning/pull/5295))
- Added possibility for nested loaders ([#5404](https://github.com/Lightning-AI/lightning/pull/5404))
- Added `sync_step` to Wandb logger ([#5351](https://github.com/Lightning-AI/lightning/pull/5351))
- Added `StochasticWeightAveraging` callback ([#5640](https://github.com/Lightning-AI/lightning/pull/5640))
- Added `LightningDataModule.from_datasets(...)` ([#5133](https://github.com/Lightning-AI/lightning/pull/5133))
- Added `PL_TORCH_DISTRIBUTED_BACKEND` env variable to select backend ([#5981](https://github.com/Lightning-AI/lightning/pull/5981))
- Added `Trainer` flag to activate Stochastic Weight Averaging (SWA) `Trainer(stochastic_weight_avg=True)` ([#6038](https://github.com/Lightning-AI/lightning/pull/6038))
- Added DeepSpeed integration ([#5954](https://github.com/Lightning-AI/lightning/pull/5954),
    [#6042](https://github.com/Lightning-AI/lightning/pull/6042))

### Changed

- Changed `stat_scores` metric now calculates stat scores over all classes and gains new parameters, in line with the new `StatScores` metric ([#4839](https://github.com/Lightning-AI/lightning/pull/4839))
- Changed `computer_vision_fine_tunning` example to use `BackboneLambdaFinetuningCallback` ([#5377](https://github.com/Lightning-AI/lightning/pull/5377))
- Changed `automatic casting` for LoggerConnector `metrics` ([#5218](https://github.com/Lightning-AI/lightning/pull/5218))
- Changed `iou` [func] to allow float input ([#4704](https://github.com/Lightning-AI/lightning/pull/4704))
- Metric `compute()` method will no longer automatically call `reset()` ([#5409](https://github.com/Lightning-AI/lightning/pull/5409))
- Set PyTorch 1.4 as min requirements, also for testing and examples `torchvision>=0.5` and `torchtext>=0.5` ([#5418](https://github.com/Lightning-AI/lightning/pull/5418))
- Changed `callbacks` argument in `Trainer` to allow `Callback` input ([#5446](https://github.com/Lightning-AI/lightning/pull/5446))
- Changed the default of `find_unused_parameters` to `False` in DDP ([#5185](https://github.com/Lightning-AI/lightning/pull/5185))
- Changed `ModelCheckpoint` version suffixes to start at 1 ([#5008](https://github.com/Lightning-AI/lightning/pull/5008))
- Progress bar metrics tensors are now converted to float ([#5692](https://github.com/Lightning-AI/lightning/pull/5692))
- Changed the default value for the `progress_bar_refresh_rate` Trainer argument in Google COLAB notebooks to 20 ([#5516](https://github.com/Lightning-AI/lightning/pull/5516))
- Extended support for purely iteration-based training ([#5726](https://github.com/Lightning-AI/lightning/pull/5726))
- Made `LightningModule.global_rank`, `LightningModule.local_rank` and `LightningModule.logger` read-only properties ([#5730](https://github.com/Lightning-AI/lightning/pull/5730))
- Forced `ModelCheckpoint` callbacks to run after all others to guarantee all states are saved to the checkpoint ([#5731](https://github.com/Lightning-AI/lightning/pull/5731))
- Refactored Accelerators and Plugins:
    * Added base classes for plugins ([#5715](https://github.com/Lightning-AI/lightning/pull/5715))
    * Added parallel plugins for DP, DDP, DDPSpawn, DDP2 and Horovod ([#5714](https://github.com/Lightning-AI/lightning/pull/5714))
    * Precision Plugins ([#5718](https://github.com/Lightning-AI/lightning/pull/5718))
    * Added new Accelerators for CPU, GPU and TPU ([#5719](https://github.com/Lightning-AI/lightning/pull/5719))
    * Added RPC and Sharded plugins ([#5732](https://github.com/Lightning-AI/lightning/pull/5732))
    * Added missing `LightningModule`-wrapper logic to new plugins and accelerator ([#5734](https://github.com/Lightning-AI/lightning/pull/5734))
    * Moved device-specific teardown logic from training loop to accelerator ([#5973](https://github.com/Lightning-AI/lightning/pull/5973))
    * Moved accelerator_connector.py to the connectors subfolder ([#6033](https://github.com/Lightning-AI/lightning/pull/6033))
    * Trainer only references accelerator ([#6039](https://github.com/Lightning-AI/lightning/pull/6039))
    * Made parallel devices optional across all plugins ([#6051](https://github.com/Lightning-AI/lightning/pull/6051))
    * Cleaning ([#5948](https://github.com/Lightning-AI/lightning/pull/5948),
        [#5949](https://github.com/Lightning-AI/lightning/pull/5949),
        [#5950](https://github.com/Lightning-AI/lightning/pull/5950))
- Enabled `self.log` in callbacks ([#5094](https://github.com/Lightning-AI/lightning/pull/5094))
- Renamed xxx_AVAILABLE as protected ([#5082](https://github.com/Lightning-AI/lightning/pull/5082))
- Unified module names in Utils ([#5199](https://github.com/Lightning-AI/lightning/pull/5199))
- Separated utils: imports & enums ([#5256](https://github.com/Lightning-AI/lightning/pull/5256)
    [#5874](https://github.com/Lightning-AI/lightning/pull/5874))
- Refactor: clean trainer device & distributed getters ([#5300](https://github.com/Lightning-AI/lightning/pull/5300))
- Simplified training phase as LightningEnum ([#5419](https://github.com/Lightning-AI/lightning/pull/5419))
- Updated metrics to use LightningEnum ([#5689](https://github.com/Lightning-AI/lightning/pull/5689))
- Changed the seq of `on_train_batch_end`, `on_batch_end` & `on_train_epoch_end`, `on_epoch_end hooks` ([#5688](https://github.com/Lightning-AI/lightning/pull/5688))
- Refactored `setup_training` and remove `test_mode` ([#5388](https://github.com/Lightning-AI/lightning/pull/5388))
- Disabled training with zero `num_training_batches` when insufficient `limit_train_batches` ([#5703](https://github.com/Lightning-AI/lightning/pull/5703))
- Refactored `EpochResultStore` ([#5522](https://github.com/Lightning-AI/lightning/pull/5522))
- Update `lr_finder` to check for attribute if not running `fast_dev_run` ([#5990](https://github.com/Lightning-AI/lightning/pull/5990))
- LightningOptimizer manual optimizer is more flexible and expose `toggle_model` ([#5771](https://github.com/Lightning-AI/lightning/pull/5771))
- `MlflowLogger` limit parameter value length to 250 char ([#5893](https://github.com/Lightning-AI/lightning/pull/5893))
- Re-introduced fix for Hydra directory sync with multiple process ([#5993](https://github.com/Lightning-AI/lightning/pull/5993))

### Deprecated

- Function `stat_scores_multiple_classes` is deprecated in favor of `stat_scores` ([#4839](https://github.com/Lightning-AI/lightning/pull/4839))
- Moved accelerators and plugins to its `legacy` pkg ([#5645](https://github.com/Lightning-AI/lightning/pull/5645))
- Deprecated `LightningDistributedDataParallel` in favor of new wrapper module `LightningDistributedModule` ([#5185](https://github.com/Lightning-AI/lightning/pull/5185))
- Deprecated `LightningDataParallel` in favor of new wrapper module `LightningParallelModule` ([#5670](https://github.com/Lightning-AI/lightning/pull/5670))
- Renamed utils modules ([#5199](https://github.com/Lightning-AI/lightning/pull/5199))
    * `argparse_utils` >> `argparse`
    * `model_utils` >> `model_helpers`
    * `warning_utils` >> `warnings`
    * `xla_device_utils` >> `xla_device`
- Deprecated using `'val_loss'` to set the `ModelCheckpoint` monitor ([#6012](https://github.com/Lightning-AI/lightning/pull/6012))
- Deprecated `.get_model()` with explicit `.lightning_module` property ([#6035](https://github.com/Lightning-AI/lightning/pull/6035))
- Deprecated Trainer attribute `accelerator_backend` in favor of `accelerator` ([#6034](https://github.com/Lightning-AI/lightning/pull/6034))

### Removed

- Removed deprecated checkpoint argument `filepath` ([#5321](https://github.com/Lightning-AI/lightning/pull/5321))
- Removed deprecated `Fbeta`, `f1_score` and `fbeta_score` metrics ([#5322](https://github.com/Lightning-AI/lightning/pull/5322))
- Removed deprecated `TrainResult` ([#5323](https://github.com/Lightning-AI/lightning/pull/5323))
- Removed deprecated `EvalResult` ([#5633](https://github.com/Lightning-AI/lightning/pull/5633))
- Removed `LoggerStages` ([#5673](https://github.com/Lightning-AI/lightning/pull/5673))

### Fixed

- Fixed distributed setting and `ddp_cpu` only with `num_processes>1` ([#5297](https://github.com/Lightning-AI/lightning/pull/5297))
- Fixed `num_workers` for Windows example ([#5375](https://github.com/Lightning-AI/lightning/pull/5375))
- Fixed loading yaml ([#5619](https://github.com/Lightning-AI/lightning/pull/5619))
- Fixed support custom DataLoader with DDP if they can be re-instantiated ([#5745](https://github.com/Lightning-AI/lightning/pull/5745))
- Fixed repeated `.fit()` calls ignore max_steps iteration bound ([#5936](https://github.com/Lightning-AI/lightning/pull/5936))
- Fixed throwing `MisconfigurationError` on unknown mode ([#5255](https://github.com/Lightning-AI/lightning/pull/5255))
- Resolve bug with Finetuning ([#5744](https://github.com/Lightning-AI/lightning/pull/5744))
- Fixed `ModelCheckpoint` race condition in file existence check ([#5155](https://github.com/Lightning-AI/lightning/pull/5155))
- Fixed some compatibility with PyTorch 1.8 ([#5864](https://github.com/Lightning-AI/lightning/pull/5864))
- Fixed forward cache ([#5895](https://github.com/Lightning-AI/lightning/pull/5895))
- Fixed recursive detach of tensors to CPU ([#6007](https://github.com/Lightning-AI/lightning/pull/6007))
- Fixed passing wrong strings for scheduler interval doesn't throw an error ([#5923](https://github.com/Lightning-AI/lightning/pull/5923))
- Fixed wrong `requires_grad` state after `return None` with multiple optimizers ([#5738](https://github.com/Lightning-AI/lightning/pull/5638))
- Fixed add `on_epoch_end` hook at the end of `validation`, `test` epoch ([#5986](https://github.com/Lightning-AI/lightning/pull/5986))
- Fixed missing `process_dataloader` call for `TPUSpawn` when in distributed mode ([#6015](https://github.com/Lightning-AI/lightning/pull/6015))
- Fixed progress bar flickering by appending 0 to floats/strings ([#6009](https://github.com/Lightning-AI/lightning/pull/6009))
- Fixed synchronization issues with TPU training ([#6027](https://github.com/Lightning-AI/lightning/pull/6027))
- Fixed `hparams.yaml` saved twice when using `TensorBoardLogger` ([#5953](https://github.com/Lightning-AI/lightning/pull/5953))
- Fixed basic examples ([#5912](https://github.com/Lightning-AI/lightning/pull/5912),
    [#5985](https://github.com/Lightning-AI/lightning/pull/5985))
- Fixed `fairscale` compatible with PT 1.8 ([#5996](https://github.com/Lightning-AI/lightning/pull/5996))
- Ensured `process_dataloader` is called when `tpu_cores > 1` to use Parallel DataLoader ([#6015](https://github.com/Lightning-AI/lightning/pull/6015))
- Attempted SLURM auto resume call when non-shell call fails ([#6002](https://github.com/Lightning-AI/lightning/pull/6002))
- Fixed wrapping optimizers upon assignment ([#6006](https://github.com/Lightning-AI/lightning/pull/6006))
- Fixed allowing hashing of metrics with lists in their state ([#5939](https://github.com/Lightning-AI/lightning/pull/5939))


## [1.1.8] - 2021-02-08

### Fixed

- Separate epoch validation from step validation ([#5208](https://github.com/Lightning-AI/lightning/pull/5208))
- Fixed `toggle_optimizers` not handling all optimizer parameters ([#5775](https://github.com/Lightning-AI/lightning/pull/5775))


## [1.1.7] - 2021-02-03

### Fixed

- Fixed `TensorBoardLogger` not closing `SummaryWriter` on `finalize` ([#5696](https://github.com/Lightning-AI/lightning/pull/5696))
- Fixed filtering of pytorch  "unsqueeze" warning when using DP ([#5622](https://github.com/Lightning-AI/lightning/pull/5622))
- Fixed `num_classes` argument in F1 metric ([#5663](https://github.com/Lightning-AI/lightning/pull/5663))
- Fixed `log_dir` property ([#5537](https://github.com/Lightning-AI/lightning/pull/5537))
- Fixed a race condition in `ModelCheckpoint` when checking if a checkpoint file exists ([#5144](https://github.com/Lightning-AI/lightning/pull/5144))
- Remove unnecessary intermediate layers in Dockerfiles ([#5697](https://github.com/Lightning-AI/lightning/pull/5697))
- Fixed auto learning rate ordering ([#5638](https://github.com/Lightning-AI/lightning/pull/5638))


## [1.1.6] - 2021-01-26

### Changed

- Increased TPU check timeout from 20s to 100s ([#5598](https://github.com/Lightning-AI/lightning/pull/5598))
- Ignored `step` param in Neptune logger's log_metric method ([#5510](https://github.com/Lightning-AI/lightning/pull/5510))
- Pass batch outputs to `on_train_batch_end` instead of `epoch_end` outputs ([#4369](https://github.com/Lightning-AI/lightning/pull/4369))

### Fixed

- Fixed `toggle_optimizer` to reset `requires_grad` state  ([#5574](https://github.com/Lightning-AI/lightning/pull/5574))
- Fixed FileNotFoundError for best checkpoint when using DDP with Hydra ([#5629](https://github.com/Lightning-AI/lightning/pull/5629))
- Fixed an error when logging a progress bar metric with a reserved name ([#5620](https://github.com/Lightning-AI/lightning/pull/5620))
- Fixed `Metric`'s `state_dict` not included when child modules ([#5614](https://github.com/Lightning-AI/lightning/pull/5614))
- Fixed Neptune logger creating multiple experiments when GPUs > 1 ([#3256](https://github.com/Lightning-AI/lightning/pull/3256))
- Fixed duplicate logs appearing in console when using the python logging module ([#5509](https://github.com/Lightning-AI/lightning/pull/5509))
- Fixed tensor printing in `trainer.test()` ([#5138](https://github.com/Lightning-AI/lightning/pull/5138))
- Fixed not using dataloader when `hparams` present ([#4559](https://github.com/Lightning-AI/lightning/pull/4559))


## [1.1.5] - 2021-01-19

### Fixed

- Fixed a visual bug in the progress bar display initialization ([#4579](https://github.com/Lightning-AI/lightning/pull/4579))
- Fixed logging `on_train_batch_end` in a callback with multiple optimizers ([#5521](https://github.com/Lightning-AI/lightning/pull/5521))
- Fixed `reinit_scheduler_properties` with correct optimizer ([#5519](https://github.com/Lightning-AI/lightning/pull/5519))
- Fixed `val_check_interval` with `fast_dev_run` ([#5540](https://github.com/Lightning-AI/lightning/pull/5540))


## [1.1.4] - 2021-01-12

### Added

- Add automatic optimization property setter to lightning module ([#5169](https://github.com/Lightning-AI/lightning/pull/5169))

### Changed

- Changed deprecated `enable_pl_optimizer=True` ([#5244](https://github.com/Lightning-AI/lightning/pull/5244))

### Fixed

- Fixed `transfer_batch_to_device` for DDP with `len(devices_ids) == 1` ([#5195](https://github.com/Lightning-AI/lightning/pull/5195))
- Logging only on `not should_accumulate()` during training ([#5417](https://github.com/Lightning-AI/lightning/pull/5417))
- Resolve interpolation bug with Hydra ([#5406](https://github.com/Lightning-AI/lightning/pull/5406))
- Check environ before selecting a seed to prevent warning message ([#4743](https://github.com/Lightning-AI/lightning/pull/4743))
- Fixed signature mismatch in `model_to_device` of `DDPCPUHPCAccelerator` ([#5505](https://github.com/Lightning-AI/lightning/pull/5505))

## [1.1.3] - 2021-01-05

### Added

- Added a check for optimizer attached to `lr_scheduler` ([#5338](https://github.com/Lightning-AI/lightning/pull/5338))
- Added support for passing non-existing filepaths to `resume_from_checkpoint` ([#4402](https://github.com/Lightning-AI/lightning/pull/4402))

### Changed

- Skip restore from `resume_from_checkpoint` while `testing` ([#5161](https://github.com/Lightning-AI/lightning/pull/5161))
- Allowed `log_momentum` for adaptive optimizers in `LearningRateMonitor` ([#5333](https://github.com/Lightning-AI/lightning/pull/5333))
- Disabled checkpointing, earlystopping and logging with `fast_dev_run` ([#5277](https://github.com/Lightning-AI/lightning/pull/5277))
- Distributed group defaults to `WORLD` if `None` ([#5125](https://github.com/Lightning-AI/lightning/pull/5125))

### Fixed

- Fixed `trainer.test` returning non-test metrics ([#5214](https://github.com/Lightning-AI/lightning/pull/5214))
- Fixed metric state reset ([#5273](https://github.com/Lightning-AI/lightning/pull/5273))
- Fixed `--num-nodes` on `DDPSequentialPlugin` ([#5327](https://github.com/Lightning-AI/lightning/pull/5327))
- Fixed invalid value for `weights_summary` ([#5296](https://github.com/Lightning-AI/lightning/pull/5296))
- Fixed `Trainer.test` not using the latest `best_model_path` ([#5161](https://github.com/Lightning-AI/lightning/pull/5161))
- Fixed existence check for hparams not using underlying filesystem ([#5250](https://github.com/Lightning-AI/lightning/pull/5250))
- Fixed `LightningOptimizer` AMP bug ([#5191](https://github.com/Lightning-AI/lightning/pull/5191))
- Fixed casted key to string in `_flatten_dict` ([#5354](https://github.com/Lightning-AI/lightning/pull/5354))


## [1.1.2] - 2020-12-23

### Added

- Support number for logging with `sync_dist=True` ([#5080](https://github.com/Lightning-AI/lightning/pull/5080))
- Added offset logging step when resuming for Wandb logger ([#5050](https://github.com/Lightning-AI/lightning/pull/5050))

### Removed

- `enable_pl_optimizer=False` by default to temporarily fix AMP issues ([#5163](https://github.com/Lightning-AI/lightning/pull/5163))

### Fixed

- Metric reduction with Logging ([#5150](https://github.com/Lightning-AI/lightning/pull/5150))
- Remove nan loss in manual optimization ([#5121](https://github.com/Lightning-AI/lightning/pull/5121))
- Un-balanced logging properly supported ([#5119](https://github.com/Lightning-AI/lightning/pull/5119))
- Fix hanging in DDP HPC accelerators ([#5157](https://github.com/Lightning-AI/lightning/pull/5157))
- Fix reset `TensorRunningAccum` ([#5106](https://github.com/Lightning-AI/lightning/pull/5106))
- Updated `DALIClassificationLoader` to not use deprecated arguments ([#4925](https://github.com/Lightning-AI/lightning/pull/4925))
- Corrected call to `torch.no_grad` ([#5124](https://github.com/Lightning-AI/lightning/pull/5124))


## [1.1.1] - 2020-12-15

### Added

- Add a notebook example to reach a quick baseline of ~94% accuracy on CIFAR10 using Resnet in Lightning ([#4818](https://github.com/Lightning-AI/lightning/pull/4818))

### Changed

- Simplify accelerator steps ([#5015](https://github.com/Lightning-AI/lightning/pull/5015))
- Refactor load in checkpoint connector ([#4593](https://github.com/Lightning-AI/lightning/pull/4593))
- Fixed the saved filename in `ModelCheckpoint` when it already exists ([#4861](https://github.com/Lightning-AI/lightning/pull/4861))

### Removed

- Drop duplicate metrics ([#5014](https://github.com/Lightning-AI/lightning/pull/5014))
- Remove beta arg from F1 class and functional ([#5076](https://github.com/Lightning-AI/lightning/pull/5076))

### Fixed

- Fixed trainer by default `None` in `DDPAccelerator` ([#4915](https://github.com/Lightning-AI/lightning/pull/4915))
- Fixed `LightningOptimizer` to expose optimizer attributes ([#5095](https://github.com/Lightning-AI/lightning/pull/5095))
- Do not warn when the `name` key is used in the `lr_scheduler` dict ([#5057](https://github.com/Lightning-AI/lightning/pull/5057))
- Check if optimizer supports closure ([#4981](https://github.com/Lightning-AI/lightning/pull/4981))
- Add deprecated metric utility functions back to functional (
    [#5067](https://github.com/Lightning-AI/lightning/pull/5067),
    [#5068](https://github.com/Lightning-AI/lightning/pull/5068))
- Allow any input in `to_onnx` and `to_torchscript` ([#4378](https://github.com/Lightning-AI/lightning/pull/4378))
- Fixed `DDPHPCAccelerator` hangs in DDP construction by calling `init_device` ([#5157](https://github.com/Lightning-AI/lightning/pull/5157))


## [1.1.0] - 2020-12-09

### Added

- Added "monitor" key to saved `ModelCheckpoints` ([#4383](https://github.com/Lightning-AI/lightning/pull/4383))
- Added `ConfusionMatrix` class interface ([#4348](https://github.com/Lightning-AI/lightning/pull/4348))
- Added multiclass AUROC metric ([#4236](https://github.com/Lightning-AI/lightning/pull/4236))
- Added global step indexing to the checkpoint name for a better sub-epoch checkpointing experience ([#3807](https://github.com/Lightning-AI/lightning/pull/3807))
- Added optimizer hooks in callbacks ([#4379](https://github.com/Lightning-AI/lightning/pull/4379))
- Added option to log momentum ([#4384](https://github.com/Lightning-AI/lightning/pull/4384))
- Added `current_score` to `ModelCheckpoint.on_save_checkpoint` ([#4721](https://github.com/Lightning-AI/lightning/pull/4721))
- Added logging using `self.log` in train and evaluation for epoch end hooks (
    [#4552](https://github.com/Lightning-AI/lightning/pull/4552),
    [#4495](https://github.com/Lightning-AI/lightning/pull/4495),
    [#4439](https://github.com/Lightning-AI/lightning/pull/4439),
    [#4684](https://github.com/Lightning-AI/lightning/pull/4684),
    [#4913](https://github.com/Lightning-AI/lightning/pull/4913))
- Added ability for DDP plugin to modify optimizer state saving ([#4675](https://github.com/Lightning-AI/lightning/pull/4675))
- Added `prefix` argument in loggers ([#4557](https://github.com/Lightning-AI/lightning/pull/4557))
- Added printing of total num of params, trainable and non-trainable params in ModelSummary ([#4521](https://github.com/Lightning-AI/lightning/pull/4521))
- Added `PrecisionRecallCurve, ROC, AveragePrecision` class metric ([#4549](https://github.com/Lightning-AI/lightning/pull/4549))
- Added custom `Apex` and `NativeAMP` as `Precision plugins` ([#4355](https://github.com/Lightning-AI/lightning/pull/4355))
- Added `DALI MNIST` example ([#3721](https://github.com/Lightning-AI/lightning/pull/3721))
- Added `sharded plugin` for DDP for multi-gpu training memory optimizations (
    [#4639](https://github.com/Lightning-AI/lightning/pull/4639),
    [#4686](https://github.com/Lightning-AI/lightning/pull/4686),
    [#4737](https://github.com/Lightning-AI/lightning/pull/4737),
    [#4773](https://github.com/Lightning-AI/lightning/pull/4773))
- Added `experiment_id` to the NeptuneLogger ([#3462](https://github.com/Lightning-AI/lightning/pull/3462))
- Added `PyTorch Geometric` integration example with Lightning ([#4568](https://github.com/Lightning-AI/lightning/pull/4568))
- Added `all_gather` method to `LightningModule` which allows gradient based tensor synchronizations for use-cases such as negative sampling. ([#5012](https://github.com/Lightning-AI/lightning/pull/5012))
- Enabled `self.log` in most functions ([#4969](https://github.com/Lightning-AI/lightning/pull/4969))
- Added changeable extension variable for `ModelCheckpoint` ([#4977](https://github.com/Lightning-AI/lightning/pull/4977))


### Changed

- Tuner algorithms will be skipped if `fast_dev_run=True` ([#3903](https://github.com/Lightning-AI/lightning/pull/3903))
- `WandbLogger` does not force wandb `reinit` arg to True anymore and creates a run only when needed ([#4648](https://github.com/Lightning-AI/lightning/pull/4648))
- Changed `automatic_optimization` to be a model attribute ([#4602](https://github.com/Lightning-AI/lightning/pull/4602))
- Changed `Simple Profiler` report to order by percentage time spent + num calls ([#4880](https://github.com/Lightning-AI/lightning/pull/4880))
- Simplify optimization Logic ([#4984](https://github.com/Lightning-AI/lightning/pull/4984))
- Classification metrics overhaul ([#4837](https://github.com/Lightning-AI/lightning/pull/4837))
- Updated `fast_dev_run` to accept integer representing num_batches ([#4629](https://github.com/Lightning-AI/lightning/pull/4629))
- Refactored optimizer ([#4658](https://github.com/Lightning-AI/lightning/pull/4658))


### Deprecated

- Deprecated `prefix` argument in `ModelCheckpoint` ([#4765](https://github.com/Lightning-AI/lightning/pull/4765))
- Deprecated the old way of assigning hyper-parameters through `self.hparams = ...` ([#4813](https://github.com/Lightning-AI/lightning/pull/4813))
- Deprecated `mode='auto'` from `ModelCheckpoint` and `EarlyStopping` ([#4695](https://github.com/Lightning-AI/lightning/pull/4695))

### Removed

- Removed `reorder` parameter of the `auc` metric ([#5004](https://github.com/Lightning-AI/lightning/pull/5004))
- Removed `multiclass_roc` and `multiclass_precision_recall_curve`, use `roc` and `precision_recall_curve` instead ([#4549](https://github.com/Lightning-AI/lightning/pull/4549))

### Fixed

- Added feature to move tensors to CPU before saving ([#4309](https://github.com/Lightning-AI/lightning/pull/4309))
- Fixed `LoggerConnector` to have logged metrics on root device in DP ([#4138](https://github.com/Lightning-AI/lightning/pull/4138))
- Auto convert tensors to contiguous format when `gather_all` ([#4907](https://github.com/Lightning-AI/lightning/pull/4907))
- Fixed `PYTHONPATH` for ddp test model ([#4528](https://github.com/Lightning-AI/lightning/pull/4528))
- Fixed allowing logger to support indexing ([#4595](https://github.com/Lightning-AI/lightning/pull/4595))
- Fixed DDP and manual_optimization ([#4976](https://github.com/Lightning-AI/lightning/pull/4976))


## [1.0.8] - 2020-11-24

### Added

- Added casting to python types for numpy scalars when logging `hparams` ([#4647](https://github.com/Lightning-AI/lightning/pull/4647))
- Added warning when progress bar refresh rate is less than 20 on Google Colab to prevent crashing ([#4654](https://github.com/Lightning-AI/lightning/pull/4654))
- Added `F1` class metric ([#4656](https://github.com/Lightning-AI/lightning/pull/4656))

### Changed

- Consistently use `step=trainer.global_step` in `LearningRateMonitor` independently of `logging_interval` ([#4376](https://github.com/Lightning-AI/lightning/pull/4376))
- Metric states are no longer as default added to `state_dict` ([#4685](https://github.com/Lightning-AI/lightning/pull/4685))
- Renamed class metric `Fbeta` >> `FBeta` ([#4656](https://github.com/Lightning-AI/lightning/pull/4656))
- Model summary: add 1 decimal place ([#4745](https://github.com/Lightning-AI/lightning/pull/4745))
- Do not override `PYTHONWARNINGS` ([#4700](https://github.com/Lightning-AI/lightning/pull/4700))
- Changed `init_ddp_connection` moved from `DDP` to `DDPPlugin` ([#4407](https://github.com/Lightning-AI/lightning/pull/4407))


### Fixed

- Fixed checkpoint `hparams` dict casting when `omegaconf` is available ([#4770](https://github.com/Lightning-AI/lightning/pull/4770))
- Fixed incomplete progress bars when total batches not divisible by refresh rate ([#4577](https://github.com/Lightning-AI/lightning/pull/4577))
- Updated SSIM metric ([#4566](https://github.com/Lightning-AI/lightning/pull/4566))
- Fixed batch_arg_name - add `batch_arg_name` to all calls to `_adjust_batch_size`bug ([#4812](https://github.com/Lightning-AI/lightning/pull/4812))
- Fixed `torchtext` data to GPU ([#4785](https://github.com/Lightning-AI/lightning/pull/4785))
- Fixed a crash bug in MLFlow logger ([#4716](https://github.com/Lightning-AI/lightning/pull/4716))

## [1.0.7] - 2020-11-17

### Added

- Added lambda closure to `manual_optimizer_step` ([#4618](https://github.com/Lightning-AI/lightning/pull/4618))

### Changed

- Change Metrics `persistent` default mode to `False` ([#4685](https://github.com/Lightning-AI/lightning/pull/4685))
- LoggerConnector log_metrics will use `total_batch_idx` instead of `global_step` when logging on `training step` ([#4738](https://github.com/Lightning-AI/lightning/pull/4738))


### Fixed

- Prevent crash if `sync_dist=True` on CPU ([#4626](https://github.com/Lightning-AI/lightning/pull/4626))
- Fixed average pbar Metrics ([#4534](https://github.com/Lightning-AI/lightning/pull/4534))
- Fixed `setup` callback hook to correctly pass the LightningModule through ([#4608](https://github.com/Lightning-AI/lightning/pull/4608))
- Allowing decorate model init with saving `hparams` inside ([#4662](https://github.com/Lightning-AI/lightning/pull/4662))
- Fixed `split_idx` set by `LoggerConnector` in `on_trainer_init` to `Trainer`  ([#4697](https://github.com/Lightning-AI/lightning/pull/4697))


## [1.0.6] - 2020-11-11

### Added

- Added metrics aggregation in Horovod and fixed early stopping ([#3775](https://github.com/Lightning-AI/lightning/pull/3775))
- Added `manual_optimizer_step` which work with `AMP Native` and `accumulated_grad_batches` ([#4485](https://github.com/Lightning-AI/lightning/pull/4485))
- Added `persistent(mode)` method to metrics, to enable and disable metric states being added to `state_dict` ([#4482](https://github.com/Lightning-AI/lightning/pull/4482))
- Added congratulations at the end of our notebooks ([#4555](https://github.com/Lightning-AI/lightning/pull/4555))
- Added parameters `move_metrics_to_cpu` in Trainer to disable gpu leak ([#4592](https://github.com/Lightning-AI/lightning/pull/4592))


### Changed

- Changed `fsspec` to tuner ([#4458](https://github.com/Lightning-AI/lightning/pull/4458))
- Unify SLURM/TorchElastic under backend plugin ([#4578](https://github.com/Lightning-AI/lightning/pull/4578),
        [#4580](https://github.com/Lightning-AI/lightning/pull/4580),
        [#4581](https://github.com/Lightning-AI/lightning/pull/4581),
        [#4582](https://github.com/Lightning-AI/lightning/pull/4582),
        [#4583](https://github.com/Lightning-AI/lightning/pull/4583))

### Fixed

- Fixed feature-lack in `hpc_load` ([#4526](https://github.com/Lightning-AI/lightning/pull/4526))
- Fixed metrics states being overridden in DDP mode ([#4482](https://github.com/Lightning-AI/lightning/pull/4482))
- Fixed `lightning_getattr`, `lightning_hasattr` not finding the correct attributes in datamodule ([#4347](https://github.com/Lightning-AI/lightning/pull/4347))
- Fixed automatic optimization AMP by `manual_optimization_step` ([#4485](https://github.com/Lightning-AI/lightning/pull/4485))
- Replace `MisconfigurationException` with warning in `ModelCheckpoint` Callback ([#4560](https://github.com/Lightning-AI/lightning/pull/4560))
- Fixed logged keys in mlflow logger ([#4412](https://github.com/Lightning-AI/lightning/pull/4412))
- Fixed `is_picklable` by catching `AttributeError` ([#4508](https://github.com/Lightning-AI/lightning/pull/4508))
- Fixed multi test dataloaders dict `AttributeError` error ([#4480](https://github.com/Lightning-AI/lightning/pull/4480))
- Fixed show progress bar only for `progress_rank 0` on `DDP_SLURM` ([#4437](https://github.com/Lightning-AI/lightning/pull/4437))

## [1.0.5] - 2020-11-03

### Added

- Added PyTorch 1.7 Stable support ([#3821](https://github.com/Lightning-AI/lightning/pull/3821))
- Added timeout for `tpu_device_exists` to ensure process does not hang indefinitely ([#4340](https://github.com/Lightning-AI/lightning/pull/4340))

### Changed

- W&B log in sync with `Trainer` step ([#4405](https://github.com/Lightning-AI/lightning/pull/4405))
- Hook `on_after_backward` is called only when `optimizer_step` is being called ([#4439](https://github.com/Lightning-AI/lightning/pull/4439))
- Moved `track_and_norm_grad` into `training loop` and called only when `optimizer_step` is being called ([#4439](https://github.com/Lightning-AI/lightning/pull/4439))
- Changed type checker with explicit cast of `ref_model` object ([#4457](https://github.com/Lightning-AI/lightning/pull/4457))
- Changed `distributed_backend` -> `accelerator` ([#4429](https://github.com/Lightning-AI/lightning/pull/4429))

### Deprecated

- Deprecated passing `ModelCheckpoint` instance to `checkpoint_callback` Trainer argument ([#4336](https://github.com/Lightning-AI/lightning/pull/4336))

### Fixed

- Disable saving checkpoints if not trained ([#4372](https://github.com/Lightning-AI/lightning/pull/4372))
- Fixed error using `auto_select_gpus=True` with `gpus=-1` ([#4209](https://github.com/Lightning-AI/lightning/pull/4209))
- Disabled training when `limit_train_batches=0` ([#4371](https://github.com/Lightning-AI/lightning/pull/4371))
- Fixed that metrics do not store computational graph for all seen data ([#4313](https://github.com/Lightning-AI/lightning/pull/4313))
- Fixed AMP unscale for `on_after_backward` ([#4439](https://github.com/Lightning-AI/lightning/pull/4439))
- Fixed TorchScript export when module includes Metrics ([#4428](https://github.com/Lightning-AI/lightning/pull/4428))
- Fixed TorchScript trace method's data to device and docstring ([#4360](https://github.com/Lightning-AI/lightning/pull/4360))
- Fixed CSV logger warning ([#4419](https://github.com/Lightning-AI/lightning/pull/4419))
- Fixed skip DDP parameter sync ([#4301](https://github.com/Lightning-AI/lightning/pull/4301))
- Fixed `WandbLogger` _sanitize_callable function ([#4422](https://github.com/Lightning-AI/lightning/pull/4422))
- Fixed `AMP Native` `_unscale` gradient ([#4441](https://github.com/Lightning-AI/lightning/pull/4441))


## [1.0.4] - 2020-10-27

### Added

- Added `dirpath` and `filename` parameter in `ModelCheckpoint` ([#4213](https://github.com/Lightning-AI/lightning/pull/4213))
- Added plugins docs and DDPPlugin to customize ddp across all accelerators ([#4258](https://github.com/Lightning-AI/lightning/pull/4285))
- Added `strict` option to the scheduler dictionary ([#3586](https://github.com/Lightning-AI/lightning/pull/3586))
- Added `fsspec` support for profilers ([#4162](https://github.com/Lightning-AI/lightning/pull/4162))
- Added autogenerated helptext to `Trainer.add_argparse_args` ([#4344](https://github.com/Lightning-AI/lightning/pull/4344))
- Added support for string values in `Trainer`'s `profiler` parameter ([#3656](https://github.com/Lightning-AI/lightning/pull/3656))
- Added `optimizer_closure` to `optimizer.step` when supported ([#4190](https://github.com/Lightning-AI/lightning/pull/4190))
- Added unification of regression metrics ([#4166](https://github.com/Lightning-AI/lightning/pull/4166))
- Added checkpoint load from Bytes ([#4314](https://github.com/Lightning-AI/lightning/pull/4314))

### Changed

- Improved error messages for invalid `configure_optimizers` returns ([#3587](https://github.com/Lightning-AI/lightning/pull/3587))
- Allow changing the logged step value in `validation_step` ([#4130](https://github.com/Lightning-AI/lightning/pull/4130))
- Allow setting `replace_sampler_ddp=True` with a distributed sampler already added ([#4273](https://github.com/Lightning-AI/lightning/pull/4273))
- Fixed sanitized parameters for `WandbLogger.log_hyperparams` ([#4320](https://github.com/Lightning-AI/lightning/pull/4320))

### Deprecated

- Deprecated `filepath` in `ModelCheckpoint` ([#4213](https://github.com/Lightning-AI/lightning/pull/4213))
- Deprecated `reorder` parameter of the `auc` metric ([#4237](https://github.com/Lightning-AI/lightning/pull/4237))
- Deprecated bool values in `Trainer`'s `profiler` parameter ([#3656](https://github.com/Lightning-AI/lightning/pull/3656))

### Fixed

- Fixed setting device ids in DDP ([#4297](https://github.com/Lightning-AI/lightning/pull/4297))
- Fixed synchronization of best model path in `ddp_accelerator` ([#4323](https://github.com/Lightning-AI/lightning/pull/4323))
- Fixed `WandbLogger` not uploading checkpoint artifacts at the end of training ([#4341](https://github.com/Lightning-AI/lightning/pull/4341))
- Fixed `FBeta` computation ([#4183](https://github.com/Lightning-AI/lightning/pull/4183))
- Fixed `accumulation across batches` has completed `before breaking training loop` ([#4278](https://github.com/Lightning-AI/lightning/pull/4278))
- Fixed `ModelCheckpoint` don't increase current_epoch and global_step when not training ([#4291](https://github.com/Lightning-AI/lightning/pull/4291))
- Fixed `COMET_EXPERIMENT_KEY` environment variable usage in comet logger ([#4230](https://github.com/Lightning-AI/lightning/pull/4230))

## [1.0.3] - 2020-10-20

### Added

- Added persistent flag to `Metric.add_state` ([#4195](https://github.com/Lightning-AI/lightning/pull/4195))

### Changed

- Used `checkpoint_connector.hpc_save` in SLURM ([#4217](https://github.com/Lightning-AI/lightning/pull/4217))
- Moved base req. to root ([#4219](https://github.com/Lightning-AI/lightning/pull/4219))

### Fixed

- Fixed `hparams` assign in init ([#4189](https://github.com/Lightning-AI/lightning/pull/4189))
- Fixed overwrite check for model hooks ([#4010](https://github.com/Lightning-AI/lightning/pull/4010))


## [1.0.2] - 2020-10-15

### Added

- Added trace functionality to the function `to_torchscript` ([#4142](https://github.com/Lightning-AI/lightning/pull/4142))

### Changed

- Called `on_load_checkpoint` before loading `state_dict` ([#4057](https://github.com/Lightning-AI/lightning/pull/4057))

### Removed

- Removed duplicate metric vs step log for train loop ([#4173](https://github.com/Lightning-AI/lightning/pull/4173))

### Fixed

- Fixed the `self.log` problem in `validation_step()` ([#4169](https://github.com/Lightning-AI/lightning/pull/4169))
- Fixed `hparams` saving - save the state when `save_hyperparameters()` is called [in `__init__`] ([#4163](https://github.com/Lightning-AI/lightning/pull/4163))
- Fixed runtime failure while exporting `hparams` to yaml ([#4158](https://github.com/Lightning-AI/lightning/pull/4158))


## [1.0.1] - 2020-10-14

### Added

- Added getstate/setstate method for torch.save serialization ([#4127](https://github.com/Lightning-AI/lightning/pull/4127))


## [1.0.0] - 2020-10-13

### Added

- Added Explained Variance Metric + metric fix ([#4013](https://github.com/Lightning-AI/lightning/pull/4013))
- Added Metric <-> Lightning Module integration tests ([#4008](https://github.com/Lightning-AI/lightning/pull/4008))
- Added parsing OS env vars in `Trainer` ([#4022](https://github.com/Lightning-AI/lightning/pull/4022))
- Added classification metrics ([#4043](https://github.com/Lightning-AI/lightning/pull/4043))
- Updated explained variance metric ([#4024](https://github.com/Lightning-AI/lightning/pull/4024))
- Enabled plugins ([#4041](https://github.com/Lightning-AI/lightning/pull/4041))
- Enabled custom clusters ([#4048](https://github.com/Lightning-AI/lightning/pull/4048))
- Enabled passing in custom accelerators ([#4050](https://github.com/Lightning-AI/lightning/pull/4050))
- Added `LightningModule.toggle_optimizer` ([#4058](https://github.com/Lightning-AI/lightning/pull/4058))
- Added `LightningModule.manual_backward` ([#4063](https://github.com/Lightning-AI/lightning/pull/4063))
- Added `output` argument to `*_batch_end` hooks ([#3965](https://github.com/Lightning-AI/lightning/pull/3965),
    [#3966](https://github.com/Lightning-AI/lightning/pull/3966))
- Added `output` argument to `*_epoch_end` hooks ([#3967](https://github.com/Lightning-AI/lightning/pull/3967))

### Changed

- Integrated metrics API with self.log ([#3961](https://github.com/Lightning-AI/lightning/pull/3961))
- Decoupled Apex ([#4052](https://github.com/Lightning-AI/lightning/pull/4052),
        [#4054](https://github.com/Lightning-AI/lightning/pull/4054),
        [#4055](https://github.com/Lightning-AI/lightning/pull/4055),
        [#4056](https://github.com/Lightning-AI/lightning/pull/4056),
        [#4058](https://github.com/Lightning-AI/lightning/pull/4058),
        [#4060](https://github.com/Lightning-AI/lightning/pull/4060),
        [#4061](https://github.com/Lightning-AI/lightning/pull/4061),
        [#4062](https://github.com/Lightning-AI/lightning/pull/4062),
        [#4063](https://github.com/Lightning-AI/lightning/pull/4063),
        [#4064](https://github.com/Lightning-AI/lightning/pull/4064),
        [#4065](https://github.com/Lightning-AI/lightning/pull/4065))
- Renamed all backends to `Accelerator` ([#4066](https://github.com/Lightning-AI/lightning/pull/4066))
- Enabled manual returns ([#4089](https://github.com/Lightning-AI/lightning/pull/4089))

### Removed

- Removed support for EvalResult and TrainResult ([#3968](https://github.com/Lightning-AI/lightning/pull/3968))
- Removed deprecated trainer flags: `overfit_pct`, `log_save_interval`, `row_log_interval` ([#3969](https://github.com/Lightning-AI/lightning/pull/3969))
- Removed deprecated early_stop_callback ([#3982](https://github.com/Lightning-AI/lightning/pull/3982))
- Removed deprecated model hooks ([#3980](https://github.com/Lightning-AI/lightning/pull/3980))
- Removed deprecated callbacks ([#3979](https://github.com/Lightning-AI/lightning/pull/3979))
- Removed `trainer` argument in `LightningModule.backward` [#4056](https://github.com/Lightning-AI/lightning/pull/4056))

### Fixed

- Fixed `current_epoch` property update to reflect true epoch number inside `LightningDataModule`, when `reload_dataloaders_every_epoch=True`. ([#3974](https://github.com/Lightning-AI/lightning/pull/3974))
- Fixed to print scaler value in progress bar ([#4053](https://github.com/Lightning-AI/lightning/pull/4053))
- Fixed mismatch between docstring and code regarding when `on_load_checkpoint` hook is called ([#3996](https://github.com/Lightning-AI/lightning/pull/3996))


## [0.10.0] - 2020-10-07

### Added

- Added new Metrics API. ([#3868](https://github.com/Lightning-AI/lightning/pull/3868), [#3921](https://github.com/Lightning-AI/lightning/pull/3921))
- Enable PyTorch 1.7 compatibility ([#3541](https://github.com/Lightning-AI/lightning/pull/3541))
- Added `LightningModule.to_torchscript` to support exporting as `ScriptModule` ([#3258](https://github.com/Lightning-AI/lightning/pull/3258))
- Added warning when dropping unpicklable `hparams` ([#2874](https://github.com/Lightning-AI/lightning/pull/2874))
- Added EMB similarity ([#3349](https://github.com/Lightning-AI/lightning/pull/3349))
- Added `ModelCheckpoint.to_yaml` method ([#3048](https://github.com/Lightning-AI/lightning/pull/3048))
- Allow `ModelCheckpoint` monitor to be `None`, meaning it will always save ([#3630](https://github.com/Lightning-AI/lightning/pull/3630))
- Disabled optimizers setup during testing ([#3059](https://github.com/Lightning-AI/lightning/pull/3059))
- Added support for datamodules to save and load checkpoints when training ([#3563](https://github.com/Lightning-AI/lightning/pull/3563))
- Added support for datamodule in learning rate finder ([#3425](https://github.com/Lightning-AI/lightning/pull/3425))
- Added gradient clip test for native AMP ([#3754](https://github.com/Lightning-AI/lightning/pull/3754))
- Added dist lib to enable syncing anything across devices ([#3762](https://github.com/Lightning-AI/lightning/pull/3762))
- Added `broadcast` to `TPUBackend` ([#3814](https://github.com/Lightning-AI/lightning/pull/3814))
- Added `XLADeviceUtils` class to check XLA device type ([#3274](https://github.com/Lightning-AI/lightning/pull/3274))

### Changed

- Refactored accelerator backends:
   * moved TPU `xxx_step` to backend ([#3118](https://github.com/Lightning-AI/lightning/pull/3118))
   * refactored DDP backend `forward` ([#3119](https://github.com/Lightning-AI/lightning/pull/3119))
   * refactored GPU backend `__step` ([#3120](https://github.com/Lightning-AI/lightning/pull/3120))
   * refactored Horovod backend ([#3121](https://github.com/Lightning-AI/lightning/pull/3121),
        [#3122](https://github.com/Lightning-AI/lightning/pull/3122))
   * remove obscure forward call in eval + CPU backend `___step` ([#3123](https://github.com/Lightning-AI/lightning/pull/3123))
   * reduced all simplified forward ([#3126](https://github.com/Lightning-AI/lightning/pull/3126))
   * added hook base method ([#3127](https://github.com/Lightning-AI/lightning/pull/3127))
   * refactor eval loop to use hooks - use `test_mode` for if so we can split later ([#3129](https://github.com/Lightning-AI/lightning/pull/3129))
   * moved `___step_end` hooks ([#3130](https://github.com/Lightning-AI/lightning/pull/3130))
   * training forward refactor ([#3134](https://github.com/Lightning-AI/lightning/pull/3134))
   * training AMP scaling refactor ([#3135](https://github.com/Lightning-AI/lightning/pull/3135))
   * eval step scaling factor ([#3136](https://github.com/Lightning-AI/lightning/pull/3136))
   * add eval loop object to streamline eval loop ([#3138](https://github.com/Lightning-AI/lightning/pull/3138))
   * refactored dataloader process hook ([#3139](https://github.com/Lightning-AI/lightning/pull/3139))
   * refactored inner eval loop ([#3141](https://github.com/Lightning-AI/lightning/pull/3141))
   * final inner eval loop hooks ([#3154](https://github.com/Lightning-AI/lightning/pull/3154))
   * clean up hooks in `run_evaluation` ([#3156](https://github.com/Lightning-AI/lightning/pull/3156))
   * clean up data reset ([#3161](https://github.com/Lightning-AI/lightning/pull/3161))
   * expand eval loop out ([#3165](https://github.com/Lightning-AI/lightning/pull/3165))
   * moved hooks around in eval loop ([#3195](https://github.com/Lightning-AI/lightning/pull/3195))
   * remove `_evaluate` fx ([#3197](https://github.com/Lightning-AI/lightning/pull/3197))
   * `Trainer.fit` hook clean up ([#3198](https://github.com/Lightning-AI/lightning/pull/3198))
   * DDPs train hooks ([#3203](https://github.com/Lightning-AI/lightning/pull/3203))
   * refactor DDP backend ([#3204](https://github.com/Lightning-AI/lightning/pull/3204),
        [#3207](https://github.com/Lightning-AI/lightning/pull/3207),
        [#3208](https://github.com/Lightning-AI/lightning/pull/3208),
        [#3209](https://github.com/Lightning-AI/lightning/pull/3209),
        [#3210](https://github.com/Lightning-AI/lightning/pull/3210))
   * reduced accelerator selection ([#3211](https://github.com/Lightning-AI/lightning/pull/3211))
   * group prepare data hook ([#3212](https://github.com/Lightning-AI/lightning/pull/3212))
   * added data connector ([#3285](https://github.com/Lightning-AI/lightning/pull/3285))
   * modular is_overridden ([#3290](https://github.com/Lightning-AI/lightning/pull/3290))
   * adding `Trainer.tune()` ([#3293](https://github.com/Lightning-AI/lightning/pull/3293))
   * move `run_pretrain_routine` -> `setup_training` ([#3294](https://github.com/Lightning-AI/lightning/pull/3294))
   * move train outside of setup training ([#3297](https://github.com/Lightning-AI/lightning/pull/3297))
   * move `prepare_data` to data connector ([#3307](https://github.com/Lightning-AI/lightning/pull/3307))
   * moved accelerator router ([#3309](https://github.com/Lightning-AI/lightning/pull/3309))
   * train loop refactor - moving train loop to own object ([#3310](https://github.com/Lightning-AI/lightning/pull/3310),
        [#3312](https://github.com/Lightning-AI/lightning/pull/3312),
        [#3313](https://github.com/Lightning-AI/lightning/pull/3313),
        [#3314](https://github.com/Lightning-AI/lightning/pull/3314))
   * duplicate data interface definition up into DataHooks class ([#3344](https://github.com/Lightning-AI/lightning/pull/3344))
   * inner train loop ([#3359](https://github.com/Lightning-AI/lightning/pull/3359),
        [#3361](https://github.com/Lightning-AI/lightning/pull/3361),
        [#3362](https://github.com/Lightning-AI/lightning/pull/3362),
        [#3363](https://github.com/Lightning-AI/lightning/pull/3363),
        [#3365](https://github.com/Lightning-AI/lightning/pull/3365),
        [#3366](https://github.com/Lightning-AI/lightning/pull/3366),
        [#3367](https://github.com/Lightning-AI/lightning/pull/3367),
        [#3368](https://github.com/Lightning-AI/lightning/pull/3368),
        [#3369](https://github.com/Lightning-AI/lightning/pull/3369),
        [#3370](https://github.com/Lightning-AI/lightning/pull/3370),
        [#3371](https://github.com/Lightning-AI/lightning/pull/3371),
        [#3372](https://github.com/Lightning-AI/lightning/pull/3372),
        [#3373](https://github.com/Lightning-AI/lightning/pull/3373),
        [#3374](https://github.com/Lightning-AI/lightning/pull/3374),
        [#3375](https://github.com/Lightning-AI/lightning/pull/3375),
        [#3376](https://github.com/Lightning-AI/lightning/pull/3376),
        [#3385](https://github.com/Lightning-AI/lightning/pull/3385),
        [#3388](https://github.com/Lightning-AI/lightning/pull/3388),
        [#3397](https://github.com/Lightning-AI/lightning/pull/3397))
   * all logging related calls in a connector ([#3395](https://github.com/Lightning-AI/lightning/pull/3395))
   * device parser ([#3400](https://github.com/Lightning-AI/lightning/pull/3400),
        [#3405](https://github.com/Lightning-AI/lightning/pull/3405))
   * added model connector ([#3407](https://github.com/Lightning-AI/lightning/pull/3407))
   * moved eval loop logging to loggers ([#3408](https://github.com/Lightning-AI/lightning/pull/3408))
   * moved eval loop (#3412[#3408](https://github.com/Lightning-AI/lightning/pull/3408))
   * trainer/separate argparse ([#3421](https://github.com/Lightning-AI/lightning/pull/3421),
        [#3428](https://github.com/Lightning-AI/lightning/pull/3428),
        [#3432](https://github.com/Lightning-AI/lightning/pull/3432))
   * move `lr_finder` ([#3434](https://github.com/Lightning-AI/lightning/pull/3434))
   * organize args (#[#3435](https://github.com/Lightning-AI/lightning/pull/3435),
        [#3442](https://github.com/Lightning-AI/lightning/pull/3442),
        [#3447](https://github.com/Lightning-AI/lightning/pull/3447),
        [#3448](https://github.com/Lightning-AI/lightning/pull/3448),
        [#3449](https://github.com/Lightning-AI/lightning/pull/3449),
        [#3456](https://github.com/Lightning-AI/lightning/pull/3456))
   * move specific accelerator code ([#3457](https://github.com/Lightning-AI/lightning/pull/3457))
   * group connectors ([#3472](https://github.com/Lightning-AI/lightning/pull/3472))
   * accelerator connector methods x/n ([#3469](https://github.com/Lightning-AI/lightning/pull/3469),
        [#3470](https://github.com/Lightning-AI/lightning/pull/3470),
        [#3474](https://github.com/Lightning-AI/lightning/pull/3474))
   * merge backends x/n ([#3476](https://github.com/Lightning-AI/lightning/pull/3476),
        [#3477](https://github.com/Lightning-AI/lightning/pull/3477),
        [#3478](https://github.com/Lightning-AI/lightning/pull/3478),
        [#3480](https://github.com/Lightning-AI/lightning/pull/3480),
        [#3482](https://github.com/Lightning-AI/lightning/pull/3482))
   * apex plugin ([#3502](https://github.com/Lightning-AI/lightning/pull/3502))
   * precision plugins ([#3504](https://github.com/Lightning-AI/lightning/pull/3504))
   * Result - make monitor default to `checkpoint_on` to simplify ([#3571](https://github.com/Lightning-AI/lightning/pull/3571))
   * reference to the Trainer on the `LightningDataModule` ([#3684](https://github.com/Lightning-AI/lightning/pull/3684))
   * add `.log` to lightning module ([#3686](https://github.com/Lightning-AI/lightning/pull/3686),
        [#3699](https://github.com/Lightning-AI/lightning/pull/3699),
        [#3701](https://github.com/Lightning-AI/lightning/pull/3701),
        [#3704](https://github.com/Lightning-AI/lightning/pull/3704),
        [#3715](https://github.com/Lightning-AI/lightning/pull/3715))
   * enable tracking original metric when step and epoch are both true ([#3685](https://github.com/Lightning-AI/lightning/pull/3685))
   * deprecated results obj, added support for simpler comms ([#3681](https://github.com/Lightning-AI/lightning/pull/3681))
   * move backends back to individual files ([#3712](https://github.com/Lightning-AI/lightning/pull/3712))
   * fixes logging for eval steps ([#3763](https://github.com/Lightning-AI/lightning/pull/3763))
   * decoupled DDP, DDP spawn ([#3733](https://github.com/Lightning-AI/lightning/pull/3733),
        [#3766](https://github.com/Lightning-AI/lightning/pull/3766),
        [#3767](https://github.com/Lightning-AI/lightning/pull/3767),
        [#3774](https://github.com/Lightning-AI/lightning/pull/3774),
        [#3802](https://github.com/Lightning-AI/lightning/pull/3802),
        [#3806](https://github.com/Lightning-AI/lightning/pull/3806),
        [#3817](https://github.com/Lightning-AI/lightning/pull/3817),
        [#3819](https://github.com/Lightning-AI/lightning/pull/3819),
        [#3927](https://github.com/Lightning-AI/lightning/pull/3927))
   * remove weight loading hack for ddp_cpu ([#3808](https://github.com/Lightning-AI/lightning/pull/3808))
   * separate `torchelastic` from DDP ([#3810](https://github.com/Lightning-AI/lightning/pull/3810))
   * separate SLURM from DDP ([#3809](https://github.com/Lightning-AI/lightning/pull/3809))
   * decoupled DDP2 ([#3816](https://github.com/Lightning-AI/lightning/pull/3816))
   * bug fix with logging val epoch end + monitor ([#3812](https://github.com/Lightning-AI/lightning/pull/3812))
   * callback system and init DDP ([#3836](https://github.com/Lightning-AI/lightning/pull/3836))
   * adding compute environments ([#3837](https://github.com/Lightning-AI/lightning/pull/3837), [#3842](https://github.com/Lightning-AI/lightning/pull/3842))
   * epoch can now log independently ([#3843](https://github.com/Lightning-AI/lightning/pull/3843))
   * test selecting the correct backend. temp backends while slurm and TorchElastic are decoupled ([#3848](https://github.com/Lightning-AI/lightning/pull/3848))
   * fixed `init_slurm_connection` causing hostname errors ([#3856](https://github.com/Lightning-AI/lightning/pull/3856))
   * moves init apex from LM to apex connector ([#3923](https://github.com/Lightning-AI/lightning/pull/3923))
   * moves sync bn to each backend ([#3925](https://github.com/Lightning-AI/lightning/pull/3925))
   * moves configure ddp to each backend ([#3924](https://github.com/Lightning-AI/lightning/pull/3924))
- Deprecation warning ([#3844](https://github.com/Lightning-AI/lightning/pull/3844))
- Changed `LearningRateLogger` to `LearningRateMonitor` ([#3251](https://github.com/Lightning-AI/lightning/pull/3251))
- Used `fsspec` instead of `gfile` for all IO ([#3320](https://github.com/Lightning-AI/lightning/pull/3320))
    * Swapped `torch.load` for `fsspec` load in DDP spawn backend ([#3787](https://github.com/Lightning-AI/lightning/pull/3787))
    * Swapped `torch.load` for `fsspec` load in cloud_io loading ([#3692](https://github.com/Lightning-AI/lightning/pull/3692))
    * Added support for `to_disk()` to use remote filepaths with `fsspec` ([#3930](https://github.com/Lightning-AI/lightning/pull/3930))
    * Updated model_checkpoint's to_yaml to use `fsspec` open ([#3801](https://github.com/Lightning-AI/lightning/pull/3801))
    * Fixed `fsspec` is inconsistent when doing `fs.ls` ([#3805](https://github.com/Lightning-AI/lightning/pull/3805))
- Refactor `GPUStatsMonitor` to improve training speed ([#3257](https://github.com/Lightning-AI/lightning/pull/3257))
- Changed IoU score behavior for classes absent in target and pred ([#3098](https://github.com/Lightning-AI/lightning/pull/3098))
- Changed IoU `remove_bg` bool to `ignore_index` optional int ([#3098](https://github.com/Lightning-AI/lightning/pull/3098))
- Changed defaults of `save_top_k` and `save_last` to `None` in ModelCheckpoint ([#3680](https://github.com/Lightning-AI/lightning/pull/3680))
- `row_log_interval` and `log_save_interval` are now based on training loop's `global_step` instead of epoch-internal batch index ([#3667](https://github.com/Lightning-AI/lightning/pull/3667))
- Silenced some warnings. verified ddp refactors ([#3483](https://github.com/Lightning-AI/lightning/pull/3483))
- Cleaning up stale logger tests ([#3490](https://github.com/Lightning-AI/lightning/pull/3490))
- Allow `ModelCheckpoint` monitor to be `None` ([#3633](https://github.com/Lightning-AI/lightning/pull/3633))
- Enable `None` model checkpoint default ([#3669](https://github.com/Lightning-AI/lightning/pull/3669))
- Skipped `best_model_path` if `checkpoint_callback` is `None` ([#2962](https://github.com/Lightning-AI/lightning/pull/2962))
- Used `raise .. from ..` to explicitly chain exceptions ([#3750](https://github.com/Lightning-AI/lightning/pull/3750))
-  Mocking loggers ([#3596](https://github.com/Lightning-AI/lightning/pull/3596),
    [#3617](https://github.com/Lightning-AI/lightning/pull/3617),
    [#3851](https://github.com/Lightning-AI/lightning/pull/3851),
    [#3859](https://github.com/Lightning-AI/lightning/pull/3859),
    [#3884](https://github.com/Lightning-AI/lightning/pull/3884),
    [#3853](https://github.com/Lightning-AI/lightning/pull/3853),
    [#3910](https://github.com/Lightning-AI/lightning/pull/3910),
    [#3889](https://github.com/Lightning-AI/lightning/pull/3889),
    [#3926](https://github.com/Lightning-AI/lightning/pull/3926))
- Write predictions in LightningModule instead of EvalResult [#3882](https://github.com/Lightning-AI/lightning/pull/3882)

### Deprecated

- Deprecated `TrainResult` and `EvalResult`, use `self.log` and `self.write` from the `LightningModule` to log metrics and write predictions. `training_step` can now only return a scalar (for the loss) or a dictionary with anything you want. ([#3681](https://github.com/Lightning-AI/lightning/pull/3681))
- Deprecate `early_stop_callback` Trainer argument ([#3845](https://github.com/Lightning-AI/lightning/pull/3845))
- Rename Trainer arguments `row_log_interval` >> `log_every_n_steps` and `log_save_interval` >> `flush_logs_every_n_steps` ([#3748](https://github.com/Lightning-AI/lightning/pull/3748))

### Removed

- Removed experimental Metric API ([#3943](https://github.com/Lightning-AI/lightning/pull/3943),
        [#3949](https://github.com/Lightning-AI/lightning/pull/3949),
        [#3946](https://github.com/Lightning-AI/lightning/pull/3946)), listed changes before final removal:
    * Added `EmbeddingSimilarity` metric ([#3349](https://github.com/Lightning-AI/lightning/pull/3349), [#3358](https://github.com/Lightning-AI/lightning/pull/3358))
    * Added hooks to metric module interface ([#2528](https://github.com/Lightning-AI/lightning/pull/2528))
    * Added error when AUROC metric is used for multiclass problems ([#3350](https://github.com/Lightning-AI/lightning/pull/3350))
    * Fixed `ModelCheckpoint` with `save_top_k=-1` option not tracking the best models when a monitor metric is available ([#3735](https://github.com/Lightning-AI/lightning/pull/3735))
    * Fixed counter-intuitive error being thrown in `Accuracy` metric for zero target tensor ([#3764](https://github.com/Lightning-AI/lightning/pull/3764))
    * Fixed aggregation of metrics ([#3517](https://github.com/Lightning-AI/lightning/pull/3517))
    * Fixed Metric aggregation ([#3321](https://github.com/Lightning-AI/lightning/pull/3321))
    * Fixed RMSLE metric ([#3188](https://github.com/Lightning-AI/lightning/pull/3188))
    * Renamed `reduction` to `class_reduction` in classification metrics ([#3322](https://github.com/Lightning-AI/lightning/pull/3322))
    * Changed `class_reduction` similar to sklearn for classification metrics ([#3322](https://github.com/Lightning-AI/lightning/pull/3322))
    * Renaming of precision recall metric ([#3308](https://github.com/Lightning-AI/lightning/pull/3308))

### Fixed

- Fixed `on_train_batch_start` hook to end epoch early ([#3700](https://github.com/Lightning-AI/lightning/pull/3700))
- Fixed `num_sanity_val_steps` is clipped to `limit_val_batches` ([#2917](https://github.com/Lightning-AI/lightning/pull/2917))
- Fixed ONNX model save on GPU ([#3145](https://github.com/Lightning-AI/lightning/pull/3145))
- Fixed `GpuUsageLogger` to work on different platforms ([#3008](https://github.com/Lightning-AI/lightning/pull/3008))
- Fixed auto-scale batch size not dumping `auto_lr_find` parameter ([#3151](https://github.com/Lightning-AI/lightning/pull/3151))
- Fixed `batch_outputs` with optimizer frequencies ([#3229](https://github.com/Lightning-AI/lightning/pull/3229))
- Fixed setting batch size in `LightningModule.datamodule` when using `auto_scale_batch_size` ([#3266](https://github.com/Lightning-AI/lightning/pull/3266))
- Fixed Horovod distributed backend compatibility with native AMP ([#3404](https://github.com/Lightning-AI/lightning/pull/3404))
- Fixed batch size auto scaling exceeding the size of the dataset ([#3271](https://github.com/Lightning-AI/lightning/pull/3271))
- Fixed getting `experiment_id` from MLFlow only once instead of each training loop ([#3394](https://github.com/Lightning-AI/lightning/pull/3394))
- Fixed `overfit_batches` which now correctly disables shuffling for the training loader. ([#3501](https://github.com/Lightning-AI/lightning/pull/3501))
- Fixed gradient norm tracking for `row_log_interval > 1` ([#3489](https://github.com/Lightning-AI/lightning/pull/3489))
- Fixed `ModelCheckpoint` name formatting ([#3164](https://github.com/Lightning-AI/lightning/pull/3163))
- Fixed example implementation of AutoEncoder ([#3190](https://github.com/Lightning-AI/lightning/pull/3190))
- Fixed invalid paths when remote logging with TensorBoard ([#3236](https://github.com/Lightning-AI/lightning/pull/3236))
- Fixed change `t()` to `transpose()` as XLA devices do not support `.t()` on 1-dim tensor ([#3252](https://github.com/Lightning-AI/lightning/pull/3252))
- Fixed (weights only) checkpoints loading without PL ([#3287](https://github.com/Lightning-AI/lightning/pull/3287))
- Fixed `gather_all_tensors` cross GPUs in DDP ([#3319](https://github.com/Lightning-AI/lightning/pull/3319))
- Fixed CometML save dir ([#3419](https://github.com/Lightning-AI/lightning/pull/3419))
- Fixed forward key metrics ([#3467](https://github.com/Lightning-AI/lightning/pull/3467))
- Fixed normalize mode at confusion matrix (replace NaNs with zeros) ([#3465](https://github.com/Lightning-AI/lightning/pull/3465))
- Fixed global step increment in training loop when `training_epoch_end` hook is used ([#3673](https://github.com/Lightning-AI/lightning/pull/3673))
- Fixed dataloader shuffling not getting turned off with `overfit_batches > 0` and `distributed_backend = "ddp"` ([#3534](https://github.com/Lightning-AI/lightning/pull/3534))
- Fixed determinism in `DDPSpawnBackend` when using `seed_everything` in main process ([#3335](https://github.com/Lightning-AI/lightning/pull/3335))
- Fixed `ModelCheckpoint` `period` to actually save every `period` epochs ([#3630](https://github.com/Lightning-AI/lightning/pull/3630))
- Fixed `val_progress_bar` total with `num_sanity_val_steps` ([#3751](https://github.com/Lightning-AI/lightning/pull/3751))
- Fixed Tuner dump: add `current_epoch` to dumped_params ([#3261](https://github.com/Lightning-AI/lightning/pull/3261))
- Fixed `current_epoch` and `global_step` properties mismatch between `Trainer` and `LightningModule` ([#3785](https://github.com/Lightning-AI/lightning/pull/3785))
- Fixed learning rate scheduler for optimizers with internal state ([#3897](https://github.com/Lightning-AI/lightning/pull/3897))
- Fixed `tbptt_reduce_fx` when non-floating tensors are logged ([#3796](https://github.com/Lightning-AI/lightning/pull/3796))
- Fixed model checkpoint frequency ([#3852](https://github.com/Lightning-AI/lightning/pull/3852))
- Fixed logging non-tensor scalar with result breaks subsequent epoch aggregation ([#3855](https://github.com/Lightning-AI/lightning/pull/3855))
- Fixed `TrainerEvaluationLoopMixin` activates `model.train()` at the end ([#3858](https://github.com/Lightning-AI/lightning/pull/3858))
- Fixed `overfit_batches` when using with multiple val/test_dataloaders ([#3857](https://github.com/Lightning-AI/lightning/pull/3857))
- Fixed enables `training_step` to return `None` ([#3862](https://github.com/Lightning-AI/lightning/pull/3862))
- Fixed init nan for checkpointing ([#3863](https://github.com/Lightning-AI/lightning/pull/3863))
- Fixed for `load_from_checkpoint` ([#2776](https://github.com/Lightning-AI/lightning/pull/2776))
- Fixes incorrect `batch_sizes` when Dataloader returns a dict with multiple tensors ([#3668](https://github.com/Lightning-AI/lightning/pull/3668))
- Fixed unexpected signature for `validation_step` ([#3947](https://github.com/Lightning-AI/lightning/pull/3947))

## [0.9.0] - 2020-08-20

### Added

- Added SyncBN for DDP ([#2801](https://github.com/Lightning-AI/lightning/pull/2801),
     [#2838](https://github.com/Lightning-AI/lightning/pull/2838))
- Added basic `CSVLogger` ([#2721](https://github.com/Lightning-AI/lightning/pull/2721))
- Added SSIM metrics ([#2671](https://github.com/Lightning-AI/lightning/pull/2671))
- Added BLEU metrics ([#2535](https://github.com/Lightning-AI/lightning/pull/2535))
- Added support to export a model to ONNX format ([#2596](https://github.com/Lightning-AI/lightning/pull/2596))
- Added support for `Trainer(num_sanity_val_steps=-1)` to check all validation data before training ([#2246](https://github.com/Lightning-AI/lightning/pull/2246))
- Added struct. output:
  * tests for val loop flow ([#2605](https://github.com/Lightning-AI/lightning/pull/2605))
  * `EvalResult` support for train and val. loop ([#2615](https://github.com/Lightning-AI/lightning/pull/2615),
       [#2651](https://github.com/Lightning-AI/lightning/pull/2651))
  * weighted average in results obj ([#2930](https://github.com/Lightning-AI/lightning/pull/2930))
  * fix result obj DP auto reduce ([#3013](https://github.com/Lightning-AI/lightning/pull/3013))
- Added class `LightningDataModule` ([#2668](https://github.com/Lightning-AI/lightning/pull/2668))
- Added support for PyTorch 1.6 ([#2745](https://github.com/Lightning-AI/lightning/pull/2745))
- Added call DataModule hooks implicitly in trainer ([#2755](https://github.com/Lightning-AI/lightning/pull/2755))
- Added support for Mean in DDP Sync ([#2568](https://github.com/Lightning-AI/lightning/pull/2568))
- Added remaining `sklearn` metrics: `AveragePrecision`, `BalancedAccuracy`, `CohenKappaScore`, `DCG`, `Hamming`, `Hinge`, `Jaccard`, `MeanAbsoluteError`, `MeanSquaredError`, `MeanSquaredLogError`, `MedianAbsoluteError`, `R2Score`, `MeanPoissonDeviance`, `MeanGammaDeviance`, `MeanTweedieDeviance`, `ExplainedVariance` ([#2562](https://github.com/Lightning-AI/lightning/pull/2562))
- Added support for `limit_{mode}_batches (int)` to work with infinite dataloader (IterableDataset) ([#2840](https://github.com/Lightning-AI/lightning/pull/2840))
- Added support returning python scalars in DP ([#1935](https://github.com/Lightning-AI/lightning/pull/1935))
- Added support to Tensorboard logger for OmegaConf `hparams` ([#2846](https://github.com/Lightning-AI/lightning/pull/2846))
- Added tracking of basic states in `Trainer` ([#2541](https://github.com/Lightning-AI/lightning/pull/2541))
- Tracks all outputs including TBPTT and multiple optimizers ([#2890](https://github.com/Lightning-AI/lightning/pull/2890))
- Added GPU Usage Logger ([#2932](https://github.com/Lightning-AI/lightning/pull/2932))
- Added `strict=False` for `load_from_checkpoint` ([#2819](https://github.com/Lightning-AI/lightning/pull/2819))
- Added saving test predictions on multiple GPUs ([#2926](https://github.com/Lightning-AI/lightning/pull/2926))
- Auto log the computational graph for loggers that support this ([#3003](https://github.com/Lightning-AI/lightning/pull/3003))
- Added warning when changing monitor and using results obj ([#3014](https://github.com/Lightning-AI/lightning/pull/3014))
- Added a hook `transfer_batch_to_device` to the `LightningDataModule` ([#3038](https://github.com/Lightning-AI/lightning/pull/3038))

### Changed

- Truncated long version numbers in progress bar ([#2594](https://github.com/Lightning-AI/lightning/pull/2594))
- Enabling val/test loop disabling ([#2692](https://github.com/Lightning-AI/lightning/pull/2692))
- Refactored into `accelerator` module:
    * GPU training ([#2704](https://github.com/Lightning-AI/lightning/pull/2704))
    * TPU training ([#2708](https://github.com/Lightning-AI/lightning/pull/2708))
    * DDP(2) backend ([#2796](https://github.com/Lightning-AI/lightning/pull/2796))
    * Retrieve last logged val from result by key ([#3049](https://github.com/Lightning-AI/lightning/pull/3049))
- Using `.comet.config` file for `CometLogger` ([#1913](https://github.com/Lightning-AI/lightning/pull/1913))
- Updated hooks arguments - breaking for `setup` and `teardown` ([#2850](https://github.com/Lightning-AI/lightning/pull/2850))
- Using `gfile` to support remote directories ([#2164](https://github.com/Lightning-AI/lightning/pull/2164))
- Moved optimizer creation after device placement for DDP backends ([#2904](https://github.com/Lightning-AI/lightning/pull/2904))
- Support `**DictConfig` for `hparam` serialization ([#2519](https://github.com/Lightning-AI/lightning/pull/2519))
- Removed callback metrics from test results obj ([#2994](https://github.com/Lightning-AI/lightning/pull/2994))
- Re-enabled naming metrics in ckpt name ([#3060](https://github.com/Lightning-AI/lightning/pull/3060))
- Changed progress bar epoch counting to start from 0 ([#3061](https://github.com/Lightning-AI/lightning/pull/3061))

### Deprecated

- Deprecated Trainer attribute `ckpt_path`, which will now be set by `weights_save_path` ([#2681](https://github.com/Lightning-AI/lightning/pull/2681))

### Removed

- Removed deprecated: ([#2760](https://github.com/Lightning-AI/lightning/pull/2760))
    * core decorator `data_loader`
    * Module hook `on_sanity_check_start` and loading `load_from_metrics`
    * package `pl.logging`
    * Trainer arguments: `show_progress_bar`, `num_tpu_cores`, `use_amp`, `print_nan_grads`
    * LR Finder argument `num_accumulation_steps`

### Fixed

- Fixed `accumulate_grad_batches` for last batch ([#2853](https://github.com/Lightning-AI/lightning/pull/2853))
- Fixed setup call while testing ([#2624](https://github.com/Lightning-AI/lightning/pull/2624))
- Fixed local rank zero casting ([#2640](https://github.com/Lightning-AI/lightning/pull/2640))
- Fixed single scalar return from training ([#2587](https://github.com/Lightning-AI/lightning/pull/2587))
- Fixed Horovod backend to scale LR schedlers with the optimizer ([#2626](https://github.com/Lightning-AI/lightning/pull/2626))
- Fixed `dtype` and `device` properties not getting updated in submodules ([#2657](https://github.com/Lightning-AI/lightning/pull/2657))
- Fixed `fast_dev_run` to run for all dataloaders ([#2581](https://github.com/Lightning-AI/lightning/pull/2581))
- Fixed `save_dir` in loggers getting ignored by default value of `weights_save_path` when user did not specify `weights_save_path` ([#2681](https://github.com/Lightning-AI/lightning/pull/2681))
- Fixed `weights_save_path` getting ignored when `logger=False` is passed to Trainer ([#2681](https://github.com/Lightning-AI/lightning/pull/2681))
- Fixed TPU multi-core and Float16 ([#2632](https://github.com/Lightning-AI/lightning/pull/2632))
- Fixed test metrics not being logged with `LoggerCollection` ([#2723](https://github.com/Lightning-AI/lightning/pull/2723))
- Fixed data transfer to device when using `torchtext.data.Field` and `include_lengths is True` ([#2689](https://github.com/Lightning-AI/lightning/pull/2689))
- Fixed shuffle argument for distributed sampler ([#2789](https://github.com/Lightning-AI/lightning/pull/2789))
- Fixed logging interval ([#2694](https://github.com/Lightning-AI/lightning/pull/2694))
- Fixed loss value in the progress bar is wrong when `accumulate_grad_batches > 1` ([#2738](https://github.com/Lightning-AI/lightning/pull/2738))
- Fixed correct CWD for ddp sub-processes when using Hydra ([#2719](https://github.com/Lightning-AI/lightning/pull/2719))
- Fixed selecting GPUs using `CUDA_VISIBLE_DEVICES` ([#2739](https://github.com/Lightning-AI/lightning/pull/2739))
- Fixed false `num_classes` warning in metrics ([#2781](https://github.com/Lightning-AI/lightning/pull/2781))
- Fixed shell injection vulnerability in subprocess call ([#2786](https://github.com/Lightning-AI/lightning/pull/2786))
- Fixed LR finder and `hparams` compatibility ([#2821](https://github.com/Lightning-AI/lightning/pull/2821))
- Fixed `ModelCheckpoint` not saving the latest information when `save_last=True` ([#2881](https://github.com/Lightning-AI/lightning/pull/2881))
- Fixed ImageNet example: learning rate scheduler, number of workers and batch size when using DDP ([#2889](https://github.com/Lightning-AI/lightning/pull/2889))
- Fixed apex gradient clipping ([#2829](https://github.com/Lightning-AI/lightning/pull/2829))
- Fixed save apex scaler states ([#2828](https://github.com/Lightning-AI/lightning/pull/2828))
- Fixed a model loading issue with inheritance and variable positional arguments ([#2911](https://github.com/Lightning-AI/lightning/pull/2911))
- Fixed passing `non_blocking=True` when transferring a batch object that does not support it ([#2910](https://github.com/Lightning-AI/lightning/pull/2910))
- Fixed checkpointing to remote file paths ([#2925](https://github.com/Lightning-AI/lightning/pull/2925))
- Fixed adding val step argument to metrics ([#2986](https://github.com/Lightning-AI/lightning/pull/2986))
- Fixed an issue that caused `Trainer.test()` to stall in ddp mode ([#2997](https://github.com/Lightning-AI/lightning/pull/2997))
- Fixed gathering of results with tensors of varying shape ([#3020](https://github.com/Lightning-AI/lightning/pull/3020))
- Fixed batch size auto-scaling feature to set the new value on the correct model attribute ([#3043](https://github.com/Lightning-AI/lightning/pull/3043))
- Fixed automatic batch scaling not working with half precision ([#3045](https://github.com/Lightning-AI/lightning/pull/3045))
- Fixed setting device to root gpu ([#3042](https://github.com/Lightning-AI/lightning/pull/3042))

## [0.8.5] - 2020-07-09

### Added

- Added a PSNR metric: peak signal-to-noise ratio ([#2483](https://github.com/Lightning-AI/lightning/pull/2483))
- Added functional regression metrics ([#2492](https://github.com/Lightning-AI/lightning/pull/2492))

### Removed

- Removed auto val reduce ([#2462](https://github.com/Lightning-AI/lightning/pull/2462))

### Fixed

- Flattening Wandb Hyperparameters ([#2459](https://github.com/Lightning-AI/lightning/pull/2459))
- Fixed using the same DDP python interpreter and actually running ([#2482](https://github.com/Lightning-AI/lightning/pull/2482))
- Fixed model summary input type conversion for models that have input dtype different from model parameters ([#2510](https://github.com/Lightning-AI/lightning/pull/2510))
- Made `TensorBoardLogger` and `CometLogger` pickleable ([#2518](https://github.com/Lightning-AI/lightning/pull/2518))
- Fixed a problem with `MLflowLogger` creating multiple run folders ([#2502](https://github.com/Lightning-AI/lightning/pull/2502))
- Fixed global_step increment ([#2455](https://github.com/Lightning-AI/lightning/pull/2455))
- Fixed TPU hanging example ([#2488](https://github.com/Lightning-AI/lightning/pull/2488))
- Fixed `argparse` default value bug ([#2526](https://github.com/Lightning-AI/lightning/pull/2526))
- Fixed Dice and IoU to avoid NaN by adding small eps ([#2545](https://github.com/Lightning-AI/lightning/pull/2545))
- Fixed accumulate gradients schedule at epoch 0 (continued) ([#2513](https://github.com/Lightning-AI/lightning/pull/2513))
- Fixed Trainer `.fit()` returning last not best weights in "ddp_spawn" ([#2565](https://github.com/Lightning-AI/lightning/pull/2565))
- Fixed passing (do not pass) TPU weights back on test ([#2566](https://github.com/Lightning-AI/lightning/pull/2566))
- Fixed DDP tests and `.test()` ([#2512](https://github.com/Lightning-AI/lightning/pull/2512),
     [#2570](https://github.com/Lightning-AI/lightning/pull/2570))

## [0.8.4] - 2020-07-01

### Added

- Added reduce ddp results on eval ([#2434](https://github.com/Lightning-AI/lightning/pull/2434))
- Added a warning when an `IterableDataset` has `__len__` defined ([#2437](https://github.com/Lightning-AI/lightning/pull/2437))

### Changed

- Enabled no returns from eval ([#2446](https://github.com/Lightning-AI/lightning/pull/2446))

### Fixed

- Fixes train outputs ([#2428](https://github.com/Lightning-AI/lightning/pull/2428))
- Fixes Conda dependencies ([#2412](https://github.com/Lightning-AI/lightning/pull/2412))
- Fixed Apex scaling with decoupled backward ([#2433](https://github.com/Lightning-AI/lightning/pull/2433))
- Fixed crashing or wrong displaying progressbar because of missing ipywidgets ([#2417](https://github.com/Lightning-AI/lightning/pull/2417))
- Fixed TPU saving dir ([fc26078e](https://github.com/Lightning-AI/lightning/commit/fc26078e395f8a001f4c6dd7b3fe7ca202f914a3), [04e68f02](https://github.com/Lightning-AI/lightning/commit/04e68f022fc03dd5f1555ee86dea997d42a448ad))
- Fixed logging on rank 0 only ([#2425](https://github.com/Lightning-AI/lightning/pull/2425))


## [0.8.3] - 2020-06-29

### Fixed

- Fixed AMP wrong call ([593837e](https://github.com/Lightning-AI/lightning/commit/593837e1da24ff6c942b24ed803fc1496a304609))
- Fixed batch typo ([92d1e75](https://github.com/Lightning-AI/lightning/commit/92d1e75b2638a493d9d21ed5fe00a22093888285))

## [0.8.2] - 2020-06-28

### Added

- Added TorchText support for moving data to GPU ([#2379](https://github.com/Lightning-AI/lightning/pull/2379))

### Changed

- Changed epoch indexing from 0 instead of 1 ([#2289](https://github.com/Lightning-AI/lightning/pull/2289))
- Refactor Model `backward` ([#2276](https://github.com/Lightning-AI/lightning/pull/2276))
- Refactored `training_batch` + tests to verify correctness ([#2327](https://github.com/Lightning-AI/lightning/pull/2327),
     [#2328](https://github.com/Lightning-AI/lightning/pull/2328))
- Refactored training loop ([#2336](https://github.com/Lightning-AI/lightning/pull/2336))
- Made optimization steps for hooks ([#2363](https://github.com/Lightning-AI/lightning/pull/2363))
- Changed default apex level to 'O2' ([#2362](https://github.com/Lightning-AI/lightning/pull/2362))

### Removed

- Moved `TrainsLogger` to Bolts ([#2384](https://github.com/Lightning-AI/lightning/pull/2384))

### Fixed

- Fixed parsing TPU arguments and TPU tests ([#2094](https://github.com/Lightning-AI/lightning/pull/2094))
- Fixed number batches in case of multiple dataloaders and `limit_{*}_batches` ([#1920](https://github.com/Lightning-AI/lightning/pull/1920),
     [#2226](https://github.com/Lightning-AI/lightning/pull/2226))
- Fixed an issue with forward hooks not being removed after model summary ([#2298](https://github.com/Lightning-AI/lightning/pull/2298))
- Fix for `load_from_checkpoint()` not working with absolute path on Windows ([#2294](https://github.com/Lightning-AI/lightning/pull/2294))
- Fixed an issue how _has_len handles `NotImplementedError` e.g. raised by `torchtext.data.Iterator` ([#2293](https://github.com/Lightning-AI/lightning/pull/2293)), ([#2307](https://github.com/Lightning-AI/lightning/pull/2307))
- Fixed `average_precision` metric ([#2319](https://github.com/Lightning-AI/lightning/pull/2319))
- Fixed ROC metric for CUDA tensors ([#2304](https://github.com/Lightning-AI/lightning/pull/2304))
- Fixed lost compatibility with custom datatypes implementing `.to` ([#2335](https://github.com/Lightning-AI/lightning/pull/2335))
- Fixed loading model with kwargs ([#2387](https://github.com/Lightning-AI/lightning/pull/2387))
- Fixed sum(0) for `trainer.num_val_batches` ([#2268](https://github.com/Lightning-AI/lightning/pull/2268))
- Fixed checking if the parameters are a `DictConfig` Object ([#2216](https://github.com/Lightning-AI/lightning/pull/2216))
- Fixed SLURM weights saving ([#2341](https://github.com/Lightning-AI/lightning/pull/2341))
- Fixed swaps LR scheduler order ([#2356](https://github.com/Lightning-AI/lightning/pull/2356))
- Fixed adding tensorboard `hparams` logging test ([#2342](https://github.com/Lightning-AI/lightning/pull/2342))
- Fixed use model ref for tear down ([#2360](https://github.com/Lightning-AI/lightning/pull/2360))
- Fixed logger crash on DDP ([#2388](https://github.com/Lightning-AI/lightning/pull/2388))
- Fixed several issues with early stopping and checkpoint callbacks ([#1504](https://github.com/Lightning-AI/lightning/pull/1504),
     [#2391](https://github.com/Lightning-AI/lightning/pull/2391))
- Fixed loading past checkpoints from v0.7.x ([#2405](https://github.com/Lightning-AI/lightning/pull/2405))
- Fixed loading model without arguments ([#2403](https://github.com/Lightning-AI/lightning/pull/2403))
- Fixed Windows compatibility issue ([#2358](https://github.com/Lightning-AI/lightning/pull/2358))

## [0.8.1] - 2020-06-19

### Fixed

- Fixed the `load_from_checkpoint` path detected as URL bug ([#2244](https://github.com/Lightning-AI/lightning/pull/2244))
- Fixed hooks - added barrier ([#2245](https://github.com/Lightning-AI/lightning/pull/2245),
     [#2257](https://github.com/Lightning-AI/lightning/pull/2257),
     [#2260](https://github.com/Lightning-AI/lightning/pull/220))
- Fixed `hparams` - remove frame inspection on `self.hparams` ([#2253](https://github.com/Lightning-AI/lightning/pull/2253))
- Fixed setup and on fit calls ([#2252](https://github.com/Lightning-AI/lightning/pull/2252))
- Fixed GPU template ([#2255](https://github.com/Lightning-AI/lightning/pull/2255))

## [0.8.0] - 2020-06-18

### Added

- Added `overfit_batches`, `limit_{val|test}_batches` flags (overfit now uses training set for all three) ([#2213](https://github.com/Lightning-AI/lightning/pull/2213))
- Added metrics
  * Base classes ([#1326](https://github.com/Lightning-AI/lightning/pull/1326),
       [#1877](https://github.com/Lightning-AI/lightning/pull/1877))
  * Sklearn metrics classes ([#1327](https://github.com/Lightning-AI/lightning/pull/1327))
  * Native torch metrics ([#1488](https://github.com/Lightning-AI/lightning/pull/1488),
       [#2062](https://github.com/Lightning-AI/lightning/pull/2062))
  * docs for all Metrics ([#2184](https://github.com/Lightning-AI/lightning/pull/2184),
       [#2209](https://github.com/Lightning-AI/lightning/pull/2209))
  * Regression metrics ([#2221](https://github.com/Lightning-AI/lightning/pull/2221))
- Allow dataloaders without sampler field present ([#1907](https://github.com/Lightning-AI/lightning/pull/1907))
- Added option `save_last` to save the model at the end of every epoch in `ModelCheckpoint` ([#1908](https://github.com/Lightning-AI/lightning/pull/1908))
- Early stopping checks `on_validation_end` ([#1458](https://github.com/Lightning-AI/lightning/pull/1458))
- Speed up single-core TPU training by loading data using `ParallelLoader` ([#2033](https://github.com/Lightning-AI/lightning/pull/2033))
- Added a model hook `transfer_batch_to_device` that enables moving custom data structures to the target device ([#1756](https://github.com/Lightning-AI/lightning/pull/1756))
- Added [black](https://black.readthedocs.io/en/stable/) formatter for the code with code-checker on pull ([#1610](https://github.com/Lightning-AI/lightning/pull/1610))
- Added back the slow spawn ddp implementation as `ddp_spawn` ([#2115](https://github.com/Lightning-AI/lightning/pull/2115))
- Added loading checkpoints from URLs ([#1667](https://github.com/Lightning-AI/lightning/pull/1667))
- Added a callback method `on_keyboard_interrupt` for handling KeyboardInterrupt events during training ([#2134](https://github.com/Lightning-AI/lightning/pull/2134))
- Added a decorator `auto_move_data` that moves data to the correct device when using the LightningModule for inference ([#1905](https://github.com/Lightning-AI/lightning/pull/1905))
- Added `ckpt_path` option to `LightningModule.test(...)` to load particular checkpoint ([#2190](https://github.com/Lightning-AI/lightning/pull/2190))
- Added `setup` and `teardown` hooks for model ([#2229](https://github.com/Lightning-AI/lightning/pull/2229))

### Changed

- Allow user to select individual TPU core to train on ([#1729](https://github.com/Lightning-AI/lightning/pull/1729))
- Removed non-finite values from loss in `LRFinder` ([#1862](https://github.com/Lightning-AI/lightning/pull/1862))
- Allow passing model hyperparameters as complete kwarg list ([#1896](https://github.com/Lightning-AI/lightning/pull/1896))
- Renamed `ModelCheckpoint`'s attributes `best` to `best_model_score` and `kth_best_model` to `kth_best_model_path` ([#1799](https://github.com/Lightning-AI/lightning/pull/1799))
- Re-Enable Logger's `ImportError`s ([#1938](https://github.com/Lightning-AI/lightning/pull/1938))
- Changed the default value of the Trainer argument `weights_summary` from `full` to `top` ([#2029](https://github.com/Lightning-AI/lightning/pull/2029))
- Raise an error when lightning replaces an existing sampler ([#2020](https://github.com/Lightning-AI/lightning/pull/2020))
- Enabled `prepare_data` from correct processes - clarify local vs global rank ([#2166](https://github.com/Lightning-AI/lightning/pull/2166))
- Remove explicit flush from tensorboard logger ([#2126](https://github.com/Lightning-AI/lightning/pull/2126))
- Changed epoch indexing from 1 instead of 0 ([#2206](https://github.com/Lightning-AI/lightning/pull/2206))

### Deprecated

- Deprecated flags: ([#2213](https://github.com/Lightning-AI/lightning/pull/2213))
  * `overfit_pct` in favour of `overfit_batches`
  * `val_percent_check` in favour of `limit_val_batches`
  * `test_percent_check` in favour of `limit_test_batches`
- Deprecated `ModelCheckpoint`'s attributes `best` and `kth_best_model` ([#1799](https://github.com/Lightning-AI/lightning/pull/1799))
- Dropped official support/testing for older PyTorch versions <1.3 ([#1917](https://github.com/Lightning-AI/lightning/pull/1917))
- Deprecated Trainer `proc_rank` in favour of `global_rank` ([#2166](https://github.com/Lightning-AI/lightning/pull/2166),
     [#2269](https://github.com/Lightning-AI/lightning/pull/2269))

### Removed

- Removed unintended Trainer argument `progress_bar_callback`, the callback should be passed in by `Trainer(callbacks=[...])` instead ([#1855](https://github.com/Lightning-AI/lightning/pull/1855))
- Removed obsolete `self._device` in Trainer ([#1849](https://github.com/Lightning-AI/lightning/pull/1849))
- Removed deprecated API ([#2073](https://github.com/Lightning-AI/lightning/pull/2073))
   * Packages: `pl.pt_overrides`, `pl.root_module`
   * Modules: `pl.logging.comet_logger`, `pl.logging.mlflow_logger`, `pl.logging.test_tube_logger`, `pl.overrides.override_data_parallel`, `pl.core.model_saving`, `pl.core.root_module`
   * Trainer arguments: `add_row_log_interval`, `default_save_path`, `gradient_clip`, `nb_gpu_nodes`, `max_nb_epochs`, `min_nb_epochs`, `nb_sanity_val_steps`
   * Trainer attributes: `nb_gpu_nodes`, `num_gpu_nodes`, `gradient_clip`, `max_nb_epochs`, `min_nb_epochs`, `nb_sanity_val_steps`, `default_save_path`, `tng_tqdm_dic`

### Fixed

- Run graceful training teardown on interpreter exit ([#1631](https://github.com/Lightning-AI/lightning/pull/1631))
- Fixed user warning when apex was used together with learning rate schedulers ([#1873](https://github.com/Lightning-AI/lightning/pull/1873))
- Fixed multiple calls of `EarlyStopping` callback ([#1863](https://github.com/Lightning-AI/lightning/pull/1863))
- Fixed an issue with `Trainer.from_argparse_args` when passing in unknown Trainer args ([#1932](https://github.com/Lightning-AI/lightning/pull/1932))
- Fixed bug related to logger not being reset correctly for model after tuner algorithms ([#1933](https://github.com/Lightning-AI/lightning/pull/1933))
- Fixed root node resolution for SLURM cluster with dash in host name ([#1954](https://github.com/Lightning-AI/lightning/pull/1954))
- Fixed `LearningRateLogger` in multi-scheduler setting ([#1944](https://github.com/Lightning-AI/lightning/pull/1944))
- Fixed test configuration check and testing ([#1804](https://github.com/Lightning-AI/lightning/pull/1804))
- Fixed an issue with Trainer constructor silently ignoring unknown/misspelled arguments ([#1820](https://github.com/Lightning-AI/lightning/pull/1820))
- Fixed `save_weights_only` in ModelCheckpoint ([#1780](https://github.com/Lightning-AI/lightning/pull/1780))
- Allow use of same `WandbLogger` instance for multiple training loops ([#2055](https://github.com/Lightning-AI/lightning/pull/2055))
- Fixed an issue with `_auto_collect_arguments` collecting local variables that are not constructor arguments and not working for signatures that have the instance not named `self` ([#2048](https://github.com/Lightning-AI/lightning/pull/2048))
- Fixed mistake in parameters' grad norm tracking ([#2012](https://github.com/Lightning-AI/lightning/pull/2012))
- Fixed CPU and hanging GPU crash ([#2118](https://github.com/Lightning-AI/lightning/pull/2118))
- Fixed an issue with the model summary and `example_input_array` depending on a specific ordering of the submodules in a LightningModule ([#1773](https://github.com/Lightning-AI/lightning/pull/1773))
- Fixed Tpu logging ([#2230](https://github.com/Lightning-AI/lightning/pull/2230))
- Fixed Pid port + duplicate `rank_zero` logging ([#2140](https://github.com/Lightning-AI/lightning/pull/2140),
     [#2231](https://github.com/Lightning-AI/lightning/pull/2231))

## [0.7.6] - 2020-05-16

### Added

- Added callback for logging learning rates ([#1498](https://github.com/Lightning-AI/lightning/pull/1498))
- Added transfer learning example (for a binary classification task in computer vision) ([#1564](https://github.com/Lightning-AI/lightning/pull/1564))
- Added type hints in `Trainer.fit()` and `Trainer.test()` to reflect that also a list of dataloaders can be passed in ([#1723](https://github.com/Lightning-AI/lightning/pull/1723)).
- Added auto scaling of batch size ([#1638](https://github.com/Lightning-AI/lightning/pull/1638))
- The progress bar metrics now also get updated in `training_epoch_end` ([#1724](https://github.com/Lightning-AI/lightning/pull/1724))
- Enable `NeptuneLogger` to work with `distributed_backend=ddp` ([#1753](https://github.com/Lightning-AI/lightning/pull/1753))
- Added option to provide seed to random generators to ensure reproducibility ([#1572](https://github.com/Lightning-AI/lightning/pull/1572))
- Added override for hparams in `load_from_ckpt` ([#1797](https://github.com/Lightning-AI/lightning/pull/1797))
- Added support multi-node distributed execution under `torchelastic` ([#1811](https://github.com/Lightning-AI/lightning/pull/1811),
     [#1818](https://github.com/Lightning-AI/lightning/pull/1818))
- Added using `store_true` for bool args ([#1822](https://github.com/Lightning-AI/lightning/pull/1822),
     [#1842](https://github.com/Lightning-AI/lightning/pull/1842))
- Added dummy logger for internally disabling logging for some features ([#1836](https://github.com/Lightning-AI/lightning/pull/1836))

### Changed

- Enable `non-blocking` for device transfers to GPU ([#1843](https://github.com/Lightning-AI/lightning/pull/1843))
- Replace mata_tags.csv with hparams.yaml ([#1271](https://github.com/Lightning-AI/lightning/pull/1271))
- Reduction when `batch_size < num_gpus` ([#1609](https://github.com/Lightning-AI/lightning/pull/1609))
- Updated LightningTemplateModel to look more like Colab example ([#1577](https://github.com/Lightning-AI/lightning/pull/1577))
- Don't convert `namedtuple` to `tuple` when transferring the batch to target device ([#1589](https://github.com/Lightning-AI/lightning/pull/1589))
- Allow passing hparams as keyword argument to LightningModule when loading from checkpoint ([#1639](https://github.com/Lightning-AI/lightning/pull/1639))
- Args should come after the last positional argument ([#1807](https://github.com/Lightning-AI/lightning/pull/1807))
- Made ddp the default if no backend specified with multiple GPUs ([#1789](https://github.com/Lightning-AI/lightning/pull/1789))

### Deprecated

- Deprecated `tags_csv` in favor of `hparams_file` ([#1271](https://github.com/Lightning-AI/lightning/pull/1271))

### Fixed

- Fixed broken link in PR template ([#1675](https://github.com/Lightning-AI/lightning/pull/1675))
- Fixed ModelCheckpoint not None checking filepath ([#1654](https://github.com/Lightning-AI/lightning/pull/1654))
- Trainer now calls `on_load_checkpoint()` when resuming from a checkpoint ([#1666](https://github.com/Lightning-AI/lightning/pull/1666))
- Fixed sampler logic for ddp with iterable dataset ([#1734](https://github.com/Lightning-AI/lightning/pull/1734))
- Fixed `_reset_eval_dataloader()` for IterableDataset ([#1560](https://github.com/Lightning-AI/lightning/pull/1560))
- Fixed Horovod distributed backend to set the `root_gpu` property ([#1669](https://github.com/Lightning-AI/lightning/pull/1669))
- Fixed wandb logger `global_step` affects other loggers ([#1492](https://github.com/Lightning-AI/lightning/pull/1492))
- Fixed disabling progress bar on non-zero ranks using Horovod backend ([#1709](https://github.com/Lightning-AI/lightning/pull/1709))
- Fixed bugs that prevent lr finder to be used together with early stopping and validation dataloaders ([#1676](https://github.com/Lightning-AI/lightning/pull/1676))
- Fixed a bug in Trainer that prepended the checkpoint path with `version_` when it shouldn't ([#1748](https://github.com/Lightning-AI/lightning/pull/1748))
- Fixed lr key name in case of param groups in LearningRateLogger ([#1719](https://github.com/Lightning-AI/lightning/pull/1719))
- Fixed accumulation parameter and suggestion method for learning rate finder ([#1801](https://github.com/Lightning-AI/lightning/pull/1801))
- Fixed num processes wasn't being set properly and auto sampler was ddp failing ([#1819](https://github.com/Lightning-AI/lightning/pull/1819))
- Fixed bugs in semantic segmentation example ([#1824](https://github.com/Lightning-AI/lightning/pull/1824))
- Fixed saving native AMP scaler state ([#1777](https://github.com/Lightning-AI/lightning/pull/1777))
- Fixed native amp + ddp ([#1788](https://github.com/Lightning-AI/lightning/pull/1788))
- Fixed `hparam` logging with metrics ([#1647](https://github.com/Lightning-AI/lightning/pull/1647))

## [0.7.5] - 2020-04-27

### Changed

- Allow logging of metrics together with `hparams` ([#1630](https://github.com/Lightning-AI/lightning/pull/1630))

### Removed

- Removed Warning from trainer loop ([#1634](https://github.com/Lightning-AI/lightning/pull/1634))

### Fixed

- Fixed ModelCheckpoint not being fixable ([#1632](https://github.com/Lightning-AI/lightning/pull/1632))
- Fixed CPU DDP breaking change and DDP change ([#1635](https://github.com/Lightning-AI/lightning/pull/1635))
- Tested pickling ([#1636](https://github.com/Lightning-AI/lightning/pull/1636))


## [0.7.4] - 2020-04-26

### Added

- Added flag `replace_sampler_ddp` to manually disable sampler replacement in DDP  ([#1513](https://github.com/Lightning-AI/lightning/pull/1513))
- Added `auto_select_gpus` flag to trainer that enables automatic selection of available GPUs on exclusive mode systems.
- Added learning rate finder ([#1347](https://github.com/Lightning-AI/lightning/pull/1347))
- Added support for DDP mode in clusters without SLURM ([#1387](https://github.com/Lightning-AI/lightning/pull/1387))
- Added `test_dataloaders` parameter to `Trainer.test()` ([#1434](https://github.com/Lightning-AI/lightning/pull/1434))
- Added `terminate_on_nan` flag to trainer that performs a NaN check with each training iteration when set to `True` ([#1475](https://github.com/Lightning-AI/lightning/pull/1475))
- Added speed parity tests (max 1 sec difference per epoch)([#1482](https://github.com/Lightning-AI/lightning/pull/1482))
- Added `ddp_cpu` backend for testing ddp without GPUs ([#1158](https://github.com/Lightning-AI/lightning/pull/1158))
- Added [Horovod](http://horovod.ai) support as a distributed backend `Trainer(distributed_backend='horovod')` ([#1529](https://github.com/Lightning-AI/lightning/pull/1529))
- Added support for 8 core distributed training on Kaggle TPU's ([#1568](https://github.com/Lightning-AI/lightning/pull/1568))
- Added support for native AMP ([#1561](https://github.com/Lightning-AI/lightning/pull/1561),
    [#1580](https://github.com/Lightning-AI/lightning/pull/1580))

### Changed

- Changed the default behaviour to no longer include a NaN check with each training iteration ([#1475](https://github.com/Lightning-AI/lightning/pull/1475))
- Decoupled the progress bar from trainer` it is a callback now and can be customized or even be replaced entirely ([#1450](https://github.com/Lightning-AI/lightning/pull/1450)).
- Changed lr schedule step interval behavior to update every backwards pass instead of every forwards pass ([#1477](https://github.com/Lightning-AI/lightning/pull/1477))
- Defines shared proc. rank, remove rank from instances (e.g. loggers) ([#1408](https://github.com/Lightning-AI/lightning/pull/1408))
- Updated semantic segmentation example with custom U-Net and logging ([#1371](https://github.com/Lightning-AI/lightning/pull/1371))
- Disabled val and test shuffling ([#1600](https://github.com/Lightning-AI/lightning/pull/1600))

### Deprecated

- Deprecated `training_tqdm_dict` in favor of `progress_bar_dict` ([#1450](https://github.com/Lightning-AI/lightning/pull/1450)).

### Removed

- Removed `test_dataloaders` parameter from `Trainer.fit()` ([#1434](https://github.com/Lightning-AI/lightning/pull/1434))

### Fixed

- Added the possibility to pass nested metrics dictionaries to loggers ([#1582](https://github.com/Lightning-AI/lightning/pull/1582))
- Fixed memory leak from opt return ([#1528](https://github.com/Lightning-AI/lightning/pull/1528))
- Fixed saving checkpoint before deleting old ones ([#1453](https://github.com/Lightning-AI/lightning/pull/1453))
- Fixed loggers - flushing last logged metrics even before continue, e.g. `trainer.test()` results ([#1459](https://github.com/Lightning-AI/lightning/pull/1459))
- Fixed optimizer configuration when `configure_optimizers` returns dict without `lr_scheduler` ([#1443](https://github.com/Lightning-AI/lightning/pull/1443))
- Fixed `LightningModule` - mixing hparams and arguments in `LightningModule.__init__()` crashes load_from_checkpoint() ([#1505](https://github.com/Lightning-AI/lightning/pull/1505))
- Added a missing call to the `on_before_zero_grad` model hook ([#1493](https://github.com/Lightning-AI/lightning/pull/1493)).
- Allow use of sweeps with `WandbLogger` ([#1512](https://github.com/Lightning-AI/lightning/pull/1512))
- Fixed a bug that caused the `callbacks` Trainer argument to reference a global variable ([#1534](https://github.com/Lightning-AI/lightning/pull/1534)).
- Fixed a bug that set all boolean CLI arguments from `Trainer.add_argparse_args` always to True ([#1571](https://github.com/Lightning-AI/lightning/pull/1571))
- Fixed do not copy the batch when training on a single GPU ([#1576](https://github.com/Lightning-AI/lightning/pull/1576),
    [#1579](https://github.com/Lightning-AI/lightning/pull/1579))
- Fixed soft checkpoint removing on DDP ([#1408](https://github.com/Lightning-AI/lightning/pull/1408))
- Fixed automatic parser bug ([#1585](https://github.com/Lightning-AI/lightning/pull/1585))
- Fixed bool conversion from string ([#1606](https://github.com/Lightning-AI/lightning/pull/1606))

## [0.7.3] - 2020-04-09

### Added

- Added `rank_zero_warn` for warning only in rank 0 ([#1428](https://github.com/Lightning-AI/lightning/pull/1428))

### Fixed

- Fixed default `DistributedSampler` for DDP training ([#1425](https://github.com/Lightning-AI/lightning/pull/1425))
- Fixed workers warning not on windows ([#1430](https://github.com/Lightning-AI/lightning/pull/1430))
- Fixed returning tuple from `run_training_batch` ([#1431](https://github.com/Lightning-AI/lightning/pull/1431))
- Fixed gradient clipping ([#1438](https://github.com/Lightning-AI/lightning/pull/1438))
- Fixed pretty print ([#1441](https://github.com/Lightning-AI/lightning/pull/1441))


## [0.7.2] - 2020-04-07

### Added

- Added same step loggers' metrics aggregation ([#1278](https://github.com/Lightning-AI/lightning/pull/1278))
- Added parity test between a vanilla MNIST model and lightning model ([#1284](https://github.com/Lightning-AI/lightning/pull/1284))
- Added parity test between a vanilla RNN model and lightning model ([#1351](https://github.com/Lightning-AI/lightning/pull/1351))
- Added Reinforcement Learning - Deep Q-network (DQN) lightning example ([#1232](https://github.com/Lightning-AI/lightning/pull/1232))
- Added support for hierarchical `dict` ([#1152](https://github.com/Lightning-AI/lightning/pull/1152))
- Added `TrainsLogger` class ([#1122](https://github.com/Lightning-AI/lightning/pull/1122))
- Added type hints to `pl.core` ([#946](https://github.com/Lightning-AI/lightning/pull/946))
- Added support for `IterableDataset` in validation and testing ([#1104](https://github.com/Lightning-AI/lightning/pull/1104))
- Added support for non-primitive types in `hparams` for `TensorboardLogger` ([#1130](https://github.com/Lightning-AI/lightning/pull/1130))
- Added a check that stops the training when loss or weights contain `NaN` or `inf` values. ([#1097](https://github.com/Lightning-AI/lightning/pull/1097))
- Added support for `IterableDataset` when `val_check_interval=1.0` (default), this will trigger validation at the end of each epoch. ([#1283](https://github.com/Lightning-AI/lightning/pull/1283))
- Added `summary` method to Profilers. ([#1259](https://github.com/Lightning-AI/lightning/pull/1259))
- Added informative errors if user defined dataloader has zero length ([#1280](https://github.com/Lightning-AI/lightning/pull/1280))
- Added testing for python 3.8 ([#915](https://github.com/Lightning-AI/lightning/pull/915))
- Added model configuration checking ([#1199](https://github.com/Lightning-AI/lightning/pull/1199))
- Added support for optimizer frequencies through `LightningModule.configure_optimizers()` ([#1269](https://github.com/Lightning-AI/lightning/pull/1269))
- Added option to run without an optimizer by returning `None` from `configure_optimizers`. ([#1279](https://github.com/Lightning-AI/lightning/pull/1279))
- Added a warning when the number of data loader workers is small. ([#1378](https://github.com/Lightning-AI/lightning/pull/1378))

### Changed

- Changed (renamed and refatored) `TensorRunningMean` -> `TensorRunningAccum`: running accumulations were generalized. ([#1278](https://github.com/Lightning-AI/lightning/pull/1278))
- Changed `progress_bar_refresh_rate` trainer flag to disable progress bar when set to 0. ([#1108](https://github.com/Lightning-AI/lightning/pull/1108))
- Enhanced `load_from_checkpoint` to also forward params to the model ([#1307](https://github.com/Lightning-AI/lightning/pull/1307))
- Updated references to `self.forward()` to instead use the `__call__` interface. ([#1211](https://github.com/Lightning-AI/lightning/pull/1211))
- Changed default behaviour of `configure_optimizers` to use no optimizer rather than Adam. ([#1279](https://github.com/Lightning-AI/lightning/pull/1279))
- Allow to upload models on W&B ([#1339](https://github.com/Lightning-AI/lightning/pull/1339))
- On DP and DDP2 unsqueeze is automated now ([#1319](https://github.com/Lightning-AI/lightning/pull/1319))
- Did not always create a DataLoader during reinstantiation, but the same type as before (if subclass of DataLoader) ([#1346](https://github.com/Lightning-AI/lightning/pull/1346))
- Did not interfere with a default sampler ([#1318](https://github.com/Lightning-AI/lightning/pull/1318))
- Remove default Adam optimizer ([#1317](https://github.com/Lightning-AI/lightning/pull/1317))
- Give warnings for unimplemented required lightning methods ([#1317](https://github.com/Lightning-AI/lightning/pull/1317))
- Made `evaluate` method private >> `Trainer._evaluate(...)`. ([#1260](https://github.com/Lightning-AI/lightning/pull/1260))
- Simplify the PL examples structure (shallower and more readable) ([#1247](https://github.com/Lightning-AI/lightning/pull/1247))
- Changed min max gpu memory to be on their own plots ([#1358](https://github.com/Lightning-AI/lightning/pull/1358))
- Remove `.item` which causes sync issues ([#1254](https://github.com/Lightning-AI/lightning/pull/1254))
- Changed smoothing in TQDM to decrease variability of time remaining between training / eval ([#1194](https://github.com/Lightning-AI/lightning/pull/1194))
- Change default logger to dedicated one ([#1064](https://github.com/Lightning-AI/lightning/pull/1064))

### Deprecated

- Deprecated Trainer argument `print_nan_grads` ([#1097](https://github.com/Lightning-AI/lightning/pull/1097))
- Deprecated Trainer argument `show_progress_bar` ([#1108](https://github.com/Lightning-AI/lightning/pull/1108))

### Removed

- Removed test for no test dataloader in .fit ([#1495](https://github.com/Lightning-AI/lightning/pull/1495))
- Removed duplicated module `pl.utilities.arg_parse` for loading CLI arguments ([#1167](https://github.com/Lightning-AI/lightning/pull/1167))
- Removed wandb logger's `finalize` method ([#1193](https://github.com/Lightning-AI/lightning/pull/1193))
- Dropped `torchvision` dependency in tests and added own MNIST dataset class instead ([#986](https://github.com/Lightning-AI/lightning/pull/986))

### Fixed

- Fixed `model_checkpoint` when saving all models ([#1359](https://github.com/Lightning-AI/lightning/pull/1359))
- `Trainer.add_argparse_args` classmethod fixed. Now it adds a type for the arguments ([#1147](https://github.com/Lightning-AI/lightning/pull/1147))
- Fixed bug related to type checking of `ReduceLROnPlateau` lr schedulers([#1126](https://github.com/Lightning-AI/lightning/pull/1126))
- Fixed a bug to ensure lightning checkpoints to be backward compatible ([#1132](https://github.com/Lightning-AI/lightning/pull/1132))
- Fixed a bug that created an extra dataloader with active `reload_dataloaders_every_epoch` ([#1196](https://github.com/Lightning-AI/lightning/pull/1196))
- Fixed all warnings and errors in the docs build process ([#1191](https://github.com/Lightning-AI/lightning/pull/1191))
- Fixed an issue where `val_percent_check=0` would not disable validation ([#1251](https://github.com/Lightning-AI/lightning/pull/1251))
- Fixed average of incomplete `TensorRunningMean` ([#1309](https://github.com/Lightning-AI/lightning/pull/1309))
- Fixed `WandbLogger.watch` with `wandb.init()` ([#1311](https://github.com/Lightning-AI/lightning/pull/1311))
- Fixed an issue with early stopping that would prevent it from monitoring training metrics when validation is disabled / not implemented ([#1235](https://github.com/Lightning-AI/lightning/pull/1235)).
- Fixed a bug that would cause `trainer.test()` to run on the validation set when overloading `validation_epoch_end` and `test_end` ([#1353](https://github.com/Lightning-AI/lightning/pull/1353))
- Fixed `WandbLogger.watch` - use of the watch method without importing `wandb` ([#1311](https://github.com/Lightning-AI/lightning/pull/1311))
- Fixed `WandbLogger` to be used with 'ddp' - allow reinits in sub-processes ([#1149](https://github.com/Lightning-AI/lightning/pull/1149),
     [#1360](https://github.com/Lightning-AI/lightning/pull/1360))
- Made `training_epoch_end` behave like `validation_epoch_end` ([#1357](https://github.com/Lightning-AI/lightning/pull/1357))
- Fixed `fast_dev_run` running validation twice ([#1365](https://github.com/Lightning-AI/lightning/pull/1365))
- Fixed pickle error from quick patch `__code__` ([#1352](https://github.com/Lightning-AI/lightning/pull/1352))
- Fixed memory leak on GPU0 ([#1094](https://github.com/Lightning-AI/lightning/pull/1094),
     [#1349](https://github.com/Lightning-AI/lightning/pull/1349))
- Fixed checkpointing interval ([#1272](https://github.com/Lightning-AI/lightning/pull/1272))
- Fixed validation and training loops run the partial dataset ([#1192](https://github.com/Lightning-AI/lightning/pull/1192))
- Fixed running `on_validation_end` only on main process in DDP ([#1125](https://github.com/Lightning-AI/lightning/pull/1125))
- Fixed `load_spawn_weights` only in proc rank 0 ([#1385](https://github.com/Lightning-AI/lightning/pull/1385))
- Fixes using deprecated `use_amp` attribute ([#1145](https://github.com/Lightning-AI/lightning/pull/1145))
- Fixed Tensorboard logger error: lightning_logs directory not exists in multi-node DDP on nodes with rank != 0 ([#1377](https://github.com/Lightning-AI/lightning/pull/1377))
- Fixed `Unimplemented backend XLA` error on TPU ([#1387](https://github.com/Lightning-AI/lightning/pull/1387))

## [0.7.1] - 2020-03-07

### Fixed

- Fixes `print` issues and `data_loader` ([#1080](https://github.com/Lightning-AI/lightning/pull/1080))

## [0.7.0] - 2020-03-06

### Added

- Added automatic sampler setup. Depending on DDP or TPU, lightning configures the sampler correctly (user needs to do nothing) ([#926](https://github.com/Lightning-AI/lightning/pull/926))
- Added `reload_dataloaders_every_epoch=False` flag for trainer. Some users require reloading data every epoch ([#926](https://github.com/Lightning-AI/lightning/pull/926))
- Added `progress_bar_refresh_rate=50` flag for trainer. Throttle refresh rate on notebooks ([#926](https://github.com/Lightning-AI/lightning/pull/926))
- Updated governance docs
- Added a check to ensure that the metric used for early stopping exists before training commences ([#542](https://github.com/Lightning-AI/lightning/pull/542))
- Added `optimizer_idx` argument to `backward` hook ([#733](https://github.com/Lightning-AI/lightning/pull/733))
- Added `entity` argument to `WandbLogger` to be passed to `wandb.init` ([#783](https://github.com/Lightning-AI/lightning/pull/783))
- Added a tool for profiling training runs ([#782](https://github.com/Lightning-AI/lightning/pull/782))
- Improved flexibility for naming of TensorBoard logs, can now set `version` to a `str` to just save to that directory, and use `name=''` to prevent experiment-name directory ([#804](https://github.com/Lightning-AI/lightning/pull/804))
- Added option to specify `step` key when logging metrics ([#808](https://github.com/Lightning-AI/lightning/pull/808))
- Added `train_dataloader`, `val_dataloader` and `test_dataloader` arguments to `Trainer.fit()`, for alternative data parsing ([#759](https://github.com/Lightning-AI/lightning/pull/759))
- Added Tensor Processing Unit (TPU) support ([#868](https://github.com/Lightning-AI/lightning/pull/868))
- Added semantic segmentation example ([#751](https://github.com/Lightning-AI/lightning/pull/751),[#876](https://github.com/Lightning-AI/lightning/pull/876),
     [#881](https://github.com/Lightning-AI/lightning/pull/881))
- Split callbacks in multiple files ([#849](https://github.com/Lightning-AI/lightning/pull/849))
- Support for user defined callbacks ([#889](https://github.com/Lightning-AI/lightning/pull/889) and [#950](https://github.com/Lightning-AI/lightning/pull/950))
- Added support for multiple loggers to be passed to `Trainer` as an iterable (e.g. list, tuple, etc.) ([#903](https://github.com/Lightning-AI/lightning/pull/903))
- Added support for step-based learning rate scheduling ([#941](https://github.com/Lightning-AI/lightning/pull/941))
- Added support for logging `hparams` as dict ([#1029](https://github.com/Lightning-AI/lightning/pull/1029))
- Checkpoint and early stopping now work without val. step ([#1041](https://github.com/Lightning-AI/lightning/pull/1041))
- Support graceful training cleanup after Keyboard Interrupt ([#856](https://github.com/Lightning-AI/lightning/pull/856),
     [#1019](https://github.com/Lightning-AI/lightning/pull/1019))
- Added type hints for function arguments ([#912](https://github.com/Lightning-AI/lightning/pull/912), )
- Added default `argparser` for `Trainer` ([#952](https://github.com/Lightning-AI/lightning/pull/1023),
     [#1023](https://github.com/Lightning-AI/lightning/pull/1023))
- Added TPU gradient clipping ([#963](https://github.com/Lightning-AI/lightning/pull/963))
- Added max/min number of steps in `Trainer` ([#728](https://github.com/Lightning-AI/lightning/pull/728))

### Changed

- Improved `NeptuneLogger` by adding `close_after_fit` argument to allow logging after training([#908](https://github.com/Lightning-AI/lightning/pull/1084))
- Changed default TQDM to use `tqdm.auto` for prettier outputs in IPython notebooks ([#752](https://github.com/Lightning-AI/lightning/pull/752))
- Changed `pl.logging` to `pl.loggers` ([#767](https://github.com/Lightning-AI/lightning/pull/767))
- Moved the default `tqdm_dict` definition from Trainer to `LightningModule`, so it can be overridden by the user ([#749](https://github.com/Lightning-AI/lightning/pull/749))
- Moved functionality of `LightningModule.load_from_metrics` into `LightningModule.load_from_checkpoint` ([#995](https://github.com/Lightning-AI/lightning/pull/995))
- Changed Checkpoint path parameter from `filepath` to `dirpath` ([#1016](https://github.com/Lightning-AI/lightning/pull/1016))
- Freezed models `hparams` as `Namespace` property ([#1029](https://github.com/Lightning-AI/lightning/pull/1029))
- Dropped `logging` config in package init ([#1015](https://github.com/Lightning-AI/lightning/pull/1015))
- Renames model steps ([#1051](https://github.com/Lightning-AI/lightning/pull/1051))
  - `training_end` >> `training_epoch_end`
  - `validation_end` >> `validation_epoch_end`
  - `test_end` >> `test_epoch_end`
- Refactor dataloading, supports infinite dataloader ([#955](https://github.com/Lightning-AI/lightning/pull/955))
- Create single file in `TensorBoardLogger` ([#777](https://github.com/Lightning-AI/lightning/pull/777))

### Deprecated

- Deprecated `pl.logging` ([#767](https://github.com/Lightning-AI/lightning/pull/767))
- Deprecated `LightningModule.load_from_metrics` in favour of `LightningModule.load_from_checkpoint` ([#995](https://github.com/Lightning-AI/lightning/pull/995),
     [#1079](https://github.com/Lightning-AI/lightning/pull/1079))
- Deprecated `@data_loader` decorator ([#926](https://github.com/Lightning-AI/lightning/pull/926))
- Deprecated model steps `training_end`, `validation_end` and `test_end` ([#1051](https://github.com/Lightning-AI/lightning/pull/1051),
     [#1056](https://github.com/Lightning-AI/lightning/pull/1056))

### Removed

- Removed dependency on `pandas` ([#736](https://github.com/Lightning-AI/lightning/pull/736))
- Removed dependency on `torchvision` ([#797](https://github.com/Lightning-AI/lightning/pull/797))
- Removed dependency on `scikit-learn` ([#801](https://github.com/Lightning-AI/lightning/pull/801))

### Fixed

- Fixed a bug where early stopping `on_end_epoch` would be called inconsistently when `check_val_every_n_epoch == 0` ([#743](https://github.com/Lightning-AI/lightning/pull/743))
- Fixed a bug where the model checkpointer didn't write to the same directory as the logger ([#771](https://github.com/Lightning-AI/lightning/pull/771))
- Fixed a bug where the `TensorBoardLogger` class would create an additional empty log file during fitting ([#777](https://github.com/Lightning-AI/lightning/pull/777))
- Fixed a bug where `global_step` was advanced incorrectly when using `accumulate_grad_batches > 1` ([#832](https://github.com/Lightning-AI/lightning/pull/832))
- Fixed a bug when calling `self.logger.experiment` with multiple loggers ([#1009](https://github.com/Lightning-AI/lightning/pull/1009))
- Fixed a bug when calling `logger.append_tags` on a `NeptuneLogger` with a single tag ([#1009](https://github.com/Lightning-AI/lightning/pull/1009))
- Fixed sending back data from `.spawn` by saving and loading the trained model in/out of the process ([#1017](https://github.com/Lightning-AI/lightning/pull/1017)
- Fixed port collision on DDP ([#1010](https://github.com/Lightning-AI/lightning/pull/1010))
- Fixed/tested pass overrides ([#918](https://github.com/Lightning-AI/lightning/pull/918))
- Fixed comet logger to log after train ([#892](https://github.com/Lightning-AI/lightning/pull/892))
- Remove deprecated args to learning rate step function ([#890](https://github.com/Lightning-AI/lightning/pull/890))

## [0.6.0] - 2020-01-21

### Added

- Added support for resuming from a specific checkpoint via `resume_from_checkpoint` argument ([#516](https://github.com/Lightning-AI/lightning/pull/516))
- Added support for `ReduceLROnPlateau` scheduler ([#320](https://github.com/Lightning-AI/lightning/pull/320))
- Added support for Apex mode `O2` in conjunction with Data Parallel ([#493](https://github.com/Lightning-AI/lightning/pull/493))
- Added option (`save_top_k`) to save the top k models in the `ModelCheckpoint` class ([#128](https://github.com/Lightning-AI/lightning/pull/128))
- Added `on_train_start` and `on_train_end` hooks to `ModelHooks` ([#598](https://github.com/Lightning-AI/lightning/pull/598))
- Added `TensorBoardLogger` ([#607](https://github.com/Lightning-AI/lightning/pull/607))
- Added support for weight summary of model with multiple inputs ([#543](https://github.com/Lightning-AI/lightning/pull/543))
- Added `map_location` argument to `load_from_metrics` and `load_from_checkpoint` ([#625](https://github.com/Lightning-AI/lightning/pull/625))
- Added option to disable validation by setting `val_percent_check=0` ([#649](https://github.com/Lightning-AI/lightning/pull/649))
- Added `NeptuneLogger` class ([#648](https://github.com/Lightning-AI/lightning/pull/648))
- Added `WandbLogger` class ([#627](https://github.com/Lightning-AI/lightning/pull/627))

### Changed

- Changed the default progress bar to print to stdout instead of stderr ([#531](https://github.com/Lightning-AI/lightning/pull/531))
- Renamed `step_idx` to `step`, `epoch_idx` to `epoch`, `max_num_epochs` to `max_epochs` and `min_num_epochs` to `min_epochs` ([#589](https://github.com/Lightning-AI/lightning/pull/589))
- Renamed `total_batch_nb` to `total_batches`, `nb_val_batches` to `num_val_batches`, `nb_training_batches` to `num_training_batches`, `max_nb_epochs` to `max_epochs`, `min_nb_epochs` to `min_epochs`, `nb_test_batches` to `num_test_batches`, and `nb_val_batches` to `num_val_batches` ([#567](https://github.com/Lightning-AI/lightning/pull/567))
- Changed gradient logging to use parameter names instead of indexes ([#660](https://github.com/Lightning-AI/lightning/pull/660))
- Changed the default logger to `TensorBoardLogger` ([#609](https://github.com/Lightning-AI/lightning/pull/609))
- Changed the directory for tensorboard logging to be the same as model checkpointing ([#706](https://github.com/Lightning-AI/lightning/pull/706))

### Deprecated

- Deprecated `max_nb_epochs` and `min_nb_epochs` ([#567](https://github.com/Lightning-AI/lightning/pull/567))
- Deprecated the `on_sanity_check_start` hook in `ModelHooks` ([#598](https://github.com/Lightning-AI/lightning/pull/598))

### Removed

- Removed the `save_best_only` argument from `ModelCheckpoint`, use `save_top_k=1` instead ([#128](https://github.com/Lightning-AI/lightning/pull/128))

### Fixed

- Fixed a bug which occurred when using Adagrad with cuda ([#554](https://github.com/Lightning-AI/lightning/pull/554))
- Fixed a bug where training would be on the GPU despite setting `gpus=0` or `gpus=[]` ([#561](https://github.com/Lightning-AI/lightning/pull/561))
- Fixed an error with `print_nan_gradients` when some parameters do not require gradient ([#579](https://github.com/Lightning-AI/lightning/pull/579))
- Fixed a bug where the progress bar would show an incorrect number of total steps during the validation sanity check when using multiple validation data loaders ([#597](https://github.com/Lightning-AI/lightning/pull/597))
- Fixed support for PyTorch 1.1.0 ([#552](https://github.com/Lightning-AI/lightning/pull/552))
- Fixed an issue with early stopping when using a `val_check_interval < 1.0` in `Trainer` ([#492](https://github.com/Lightning-AI/lightning/pull/492))
- Fixed bugs relating to the `CometLogger` object that would cause it to not work properly ([#481](https://github.com/Lightning-AI/lightning/pull/481))
- Fixed a bug that would occur when returning `-1` from `on_batch_start` following an early exit or when the batch was `None` ([#509](https://github.com/Lightning-AI/lightning/pull/509))
- Fixed a potential race condition with several processes trying to create checkpoint directories ([#530](https://github.com/Lightning-AI/lightning/pull/530))
- Fixed a bug where batch 'segments' would remain on the GPU when using `truncated_bptt > 1` ([#532](https://github.com/Lightning-AI/lightning/pull/532))
- Fixed a bug when using `IterableDataset` ([#547](https://github.com/Lightning-AI/lightning/pull/547))
- Fixed a bug where `.item` was called on non-tensor objects ([#602](https://github.com/Lightning-AI/lightning/pull/602))
- Fixed a bug where `Trainer.train` would crash on an uninitialized variable if the trainer was run after resuming from a checkpoint that was already at `max_epochs` ([#608](https://github.com/Lightning-AI/lightning/pull/608))
- Fixed a bug where early stopping would begin two epochs early ([#617](https://github.com/Lightning-AI/lightning/pull/617))
- Fixed a bug where `num_training_batches` and `num_test_batches` would sometimes be rounded down to zero ([#649](https://github.com/Lightning-AI/lightning/pull/649))
- Fixed a bug where an additional batch would be processed when manually setting `num_training_batches` ([#653](https://github.com/Lightning-AI/lightning/pull/653))
- Fixed a bug when batches did not have a `.copy` method ([#701](https://github.com/Lightning-AI/lightning/pull/701))
- Fixed a bug when using `log_gpu_memory=True` in Python 3.6 ([#715](https://github.com/Lightning-AI/lightning/pull/715))
- Fixed a bug where checkpoint writing could exit before completion, giving incomplete checkpoints ([#689](https://github.com/Lightning-AI/lightning/pull/689))
- Fixed a bug where `on_train_end` was not called when ealy stopping ([#723](https://github.com/Lightning-AI/lightning/pull/723))

## [0.5.3] - 2019-11-06

### Added

- Added option to disable default logger, checkpointer, and early stopping by passing `logger=False`, `checkpoint_callback=False` and `early_stop_callback=False` respectively
- Added `CometLogger` for use with Comet.ml
- Added `val_check_interval` argument to `Trainer` allowing validition to be performed at every given number of batches
- Added functionality to save and load hyperparameters using the standard checkpoint mechanism
- Added call to `torch.cuda.empty_cache` before training starts
- Added option for user to override the call t `backward`
- Added support for truncated backprop through time via the `truncated_bptt_steps` argument in `Trainer`
- Added option to operate on all outputs from `training_step` in DDP2
- Added a hook for modifying DDP init
- Added a hook for modifying Apex

### Changed

- Changed experiment version to be padded with zeros (e.g. `/dir/version_9` becomes `/dir/version_0009`)
- Changed callback metrics to include any metrics given in logs or progress bar
- Changed the default for `save_best_only` in `ModelCheckpoint` to `True`
- Added `tng_data_loader` for backwards compatibility
- Renamed `MLFlowLogger.client` to `MLFlowLogger.experiment` for consistency
- Moved `global_step` increment to happen after the batch has been processed
- Changed weights restore to first attempt HPC weights before restoring normally, preventing both weights being restored and running out of memory
- Changed progress bar functionality to add multiple progress bars for train/val/test
- Changed calls to `print` to use `logging` instead

### Deprecated

- Deprecated `tng_dataloader`

### Fixed

- Fixed an issue where the number of batches was off by one during training
- Fixed a bug that occurred when setting a checkpoint callback and `early_stop_callback=False`
- Fixed an error when importing CometLogger
- Fixed a bug where the `gpus` argument had some unexpected behaviour
- Fixed a bug where the computed total number of batches was sometimes incorrect
- Fixed a bug where the progress bar would sometimes not show the total number of batches in test mode
- Fixed a bug when using the `log_gpu_memory='min_max'` option in `Trainer`
- Fixed a bug where checkpointing would sometimes erase the current directory

## [0.5.2] - 2019-10-10

### Added

- Added `weights_summary` argument to `Trainer` to be set to `full` (full summary), `top` (just top level modules) or other
- Added `tags` argument to `MLFlowLogger`

### Changed

- Changed default for `amp_level` to `O1`

### Removed

- Removed the `print_weights_summary` argument from `Trainer`

### Fixed

- Fixed a bug where logs were not written properly
- Fixed a bug where `logger.finalize` wasn't called after training is complete
- Fixed callback metric errors in DDP
- Fixed a bug where `TestTubeLogger` didn't log to the correct directory

## [0.5.1] - 2019-10-05

### Added

- Added the `LightningLoggerBase` class for experiment loggers
- Added `MLFlowLogger` for logging with `mlflow`
- Added `TestTubeLogger` for logging with `test_tube`
- Added a different implementation of DDP (`distributed_backed='ddp2'`) where every node has one model using all GPUs
- Added support for optimisers which require a closure (e.g. LBFGS)
- Added automatic `MASTER_PORT` default for DDP when not set manually
- Added new GPU memory logging options `'min_max'` (log only the min/max utilization) and `'all'` (log all the GPU memory)

### Changed

- Changed schedulers to always be called with the current epoch
- Changed `test_tube` to an optional dependency
- Changed data loaders to internally use a getter instead of a python property
- Disabled auto GPU loading when restoring weights to prevent out of memory errors
- Changed logging, early stopping and checkpointing to occur by default

### Fixed

- Fixed a bug with samplers that do not specify `set_epoch`
- Fixed a bug when using the `MLFlowLogger` with unsupported data types, this will now raise a warning
- Fixed a bug where gradient norms were always zero using `track_grad_norm`
- Fixed a bug which causes a crash when logging memory

## [0.5.0] - 2019-09-26

### Changed

- Changed `data_batch` argument to `batch` throughout
- Changed `batch_i` argument to `batch_idx` throughout
- Changed `tng_dataloader` method to `train_dataloader`
- Changed `on_tng_metrics` method to `on_training_metrics`
- Changed `gradient_clip` argument to `gradient_clip_val`
- Changed `add_log_row_interval` to `row_log_interval`

### Fixed

- Fixed a bug with tensorboard logging in multi-gpu setup

## [0.4.9] - 2019-09-16

### Added

- Added the flag `log_gpu_memory` to `Trainer` to deactivate logging of GPU memory utilization
- Added SLURM resubmit functionality (port from test-tube)
- Added optional weight_save_path to trainer to remove the need for a checkpoint_callback when using cluster training
- Added option to use single gpu per node with `DistributedDataParallel`

### Changed

- Changed functionality of `validation_end` and `test_end` with multiple dataloaders to be given all of the dataloaders at once rather than in separate calls
- Changed print_nan_grads to only print the parameter value and gradients when they contain NaN
- Changed gpu API to take integers as well (e.g. `gpus=2` instead of `gpus=[0, 1]`)
- All models now loaded on to CPU to avoid device and out of memory issues in PyTorch

### Fixed

- Fixed a bug where data types that implement `.to` but not `.cuda` would not be properly moved onto the GPU
- Fixed a bug where data would not be re-shuffled every epoch when using a `DistributedSampler`

## [0.4.8] - 2019-08-31

### Added

- Added `test_step` and `test_end` methods, used when `Trainer.test` is called
- Added `GradientAccumulationScheduler` callback which can be used to schedule changes to the number of accumulation batches
- Added option to skip the validation sanity check by setting `nb_sanity_val_steps = 0`

### Fixed

- Fixed a bug when setting `nb_sanity_val_steps = 0`

## [0.4.7] - 2019-08-24

### Changed

- Changed the default `val_check_interval` to `1.0`
- Changed defaults for `nb_val_batches`, `nb_tng_batches` and `nb_test_batches` to 0

### Fixed

- Fixed a bug where the full validation set as used despite setting `val_percent_check`
- Fixed a bug where an `Exception` was thrown when using a data set containing a single batch
- Fixed a bug where an `Exception` was thrown if no `val_dataloader` was given
- Fixed a bug where tuples were not properly transferred to the GPU
- Fixed a bug where data of a non standard type was not properly handled by the trainer
- Fixed a bug when loading data as a tuple
- Fixed a bug where `AttributeError` could be suppressed by the `Trainer`

## [0.4.6] - 2019-08-15

### Added

- Added support for data to be given as a `dict` or `list` with a single gpu
- Added support for `configure_optimizers` to return a single optimizer, two list (optimizers and schedulers), or a single list

### Fixed

- Fixed a bug where returning just an optimizer list (i.e. without schedulers) from `configure_optimizers` would throw an `Exception`

## [0.4.5] - 2019-08-13

### Added

- Added `optimizer_step` method that can be overridden to change the standard optimizer behaviour

## [0.4.4] - 2019-08-12

### Added

- Added support for multiple validation dataloaders
- Added support for latest test-tube logger (optimised for `torch==1.2.0`)

### Changed

- `validation_step` and `val_dataloader` are now optional
- `lr_scheduler` is now activated after epoch

### Fixed

- Fixed a bug where a warning would show when using `lr_scheduler` in `torch>1.1.0`
- Fixed a bug where an `Exception` would be thrown if using `torch.DistributedDataParallel` without using a `DistributedSampler`, this now throws a `Warning` instead

## [0.4.3] - 2019-08-10

### Fixed

- Fixed a bug where accumulate gradients would scale the loss incorrectly

## [0.4.2] - 2019-08-08

### Changed

- Changed install requirement to `torch==1.2.0`

## [0.4.1] - 2019-08-08

### Changed

- Changed install requirement to `torch==1.1.0`

## [0.4.0] - 2019-08-08

### Added

- Added 16-bit support for a single GPU
- Added support for training continuation (preserves epoch, global step etc.)

### Changed

- Changed `training_step` and `validation_step`, outputs will no longer be automatically reduced

### Removed

- Removed need for `Experiment` object in `Trainer`

### Fixed

- Fixed issues with reducing outputs from generative models (such as images and text)

## [0.3.6] - 2019-07-25

### Added

- Added a decorator to do lazy data loading internally

### Fixed

- Fixed a bug where `Experiment` object was not process safe, potentially causing logs to be overwritten

## [0.3.5] - 2019-07-25

## [0.3.4] - 2019-07-22

## [0.3.3] - 2019-07-22

## [0.3.2] - 2019-07-21

## [0.3.1] - 2019-07-21

## [0.2.x] - 2019-07-09

## [0.1.x] - 2019-06-DD
