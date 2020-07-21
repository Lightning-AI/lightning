"""
Validation loop
===============

The lightning validation loop handles everything except the actual computations of your model.
To decide what will happen in your validation loop, define the `validation_step` function.
Below are all the things lightning automates for you in the validation loop.

.. note:: Lightning will run 5 steps of validation in the beginning of training as a sanity
 check so you don't have to wait until a full epoch to catch possible validation issues.

Check validation every n epochs
-------------------------------

If you have a small dataset you might want to check validation every n epochs

.. code-block:: python

    # DEFAULT
    trainer = Trainer(check_val_every_n_epoch=1)

Set how much of the validation set to check
-------------------------------------------

If you don't want to check 100% of the validation set (for debugging or if it's huge), set this flag.

limit_val_batches will be overwritten by overfit_batches if `overfit_batches > 0`

.. code-block:: python

    # DEFAULT
    trainer = Trainer(limit_val_batches=1.0)

    # check 10% only
    trainer = Trainer(limit_val_batches=0.1)

Set how much of the test set to check
-------------------------------------

If you don't want to check 100% of the test set (for debugging or if it's huge), set this flag.

limit_test_batches will be overwritten by overfit_batches if `overfit_batches > 0`

.. code-block:: python

    # DEFAULT
    trainer = Trainer(limit_test_batches=1.0)

    # check 10% only
    trainer = Trainer(limit_test_batches=0.1)

Set validation check frequency within 1 training epoch
------------------------------------------------------

For large datasets it's often desirable to check validation multiple times within a training loop.
 Pass in a float to check that often within 1 training epoch.
 Pass in an int k to check every k training batches. Must use an int if using an IterableDataset.

.. code-block:: python

    # DEFAULT
    trainer = Trainer(val_check_interval=0.95)

    # check every .25 of an epoch
    trainer = Trainer(val_check_interval=0.25)

    # check every 100 train batches (ie: for IterableDatasets or fixed frequency)
    trainer = Trainer(val_check_interval=100)


Set the number of validation sanity steps
-----------------------------------------

Lightning runs a few steps of validation in the beginning of training.
 This avoids crashing in the validation loop sometime deep into a lengthy training loop.

.. code-block:: python

    # DEFAULT
    trainer = Trainer(num_sanity_val_steps=5)


You can use `Trainer(num_sanity_val_steps=0)` to skip the sanity check.

# Testing loop

To ensure you don't accidentally use test data to guide training decisions Lightning
 makes running the test set deliberate.

**test**

You have two options to run the test set.
First case is where you test right after a full training routine.

.. code-block:: python

    # run full training
    trainer.fit(model)

    # run test set
    trainer.test()


Second case is where you load a model and run the test set

.. code-block:: python

    model = MyLightningModule.load_from_checkpoint(
        checkpoint_path='/path/to/pytorch_checkpoint.ckpt',
        hparams_file='/path/to/test_tube/experiment/version/hparams.yaml',
        map_location=None
    )

    # init trainer with whatever options
    trainer = Trainer(...)

    # test (pass in the model)
    trainer.test(model)

In this second case, the options you pass to trainer will be used when running
 the test set (ie: 16-bit, dp, ddp, etc...)

"""

from abc import ABC, abstractmethod
from pprint import pprint
from typing import Callable, List, Union

import torch
from torch.utils.data import DataLoader

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel, LightningDataParallel
from pytorch_lightning.utilities import rank_zero_warn, NATIVE_AMP_AVALAIBLE
from torch import distributed as dist
from pytorch_lightning.core.step_result import Result, EvalResult
from pytorch_lightning.utilities.exceptions import MisconfigurationException


try:
    import torch_xla.distributed.parallel_loader as xla_pl
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True

try:
    import horovod.torch as hvd
except (ModuleNotFoundError, ImportError):
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True


class TrainerEvaluationLoopMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    on_gpu: bool
    use_ddp: bool
    use_dp: bool
    use_ddp2: bool
    use_horovod: bool
    single_gpu: bool
    data_parallel_device_ids: ...
    model: LightningModule
    num_test_batches: List[int]
    num_val_batches: int
    world_size: int
    fast_dev_run: ...
    process_output: ...
    progress_bar_dict: ...
    global_rank: int
    current_epoch: int
    callback_metrics: ...
    test_dataloaders: DataLoader
    val_dataloaders: DataLoader
    use_tpu: bool
    reload_dataloaders_every_epoch: ...
    tpu_id: int
    verbose_test: bool

    # Callback system
    on_validation_batch_start: Callable
    on_validation_batch_end: Callable
    on_test_batch_start: Callable
    on_test_batch_end: Callable
    on_validation_start: Callable
    on_validation_end: Callable
    on_test_start: Callable
    on_test_end: Callable

    @abstractmethod
    def copy_trainer_model_properties(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def get_model(self) -> LightningModule:
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def is_overridden(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def transfer_batch_to_tpu(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def transfer_batch_to_gpu(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def add_progress_bar_metrics(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def log_metrics(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def reset_test_dataloader(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def reset_val_dataloader(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    def __call_eval_loop_hook_start(self, test_mode, model):
        # on_[train/validation]_epoch_start hook
        hook_root_name = 'test' if test_mode else 'validation'
        hook_name = f'on_{hook_root_name}_epoch_start'
        with self.profiler.profile(hook_name):
            # call hook
            getattr(self, hook_name)()

            # model hooks
            if self.is_function_implemented(hook_name):
                getattr(model, hook_name)()

    def __call_eval_loop_hook_end(self, test_mode, model):
        # on_[train/validation]_epoch_start hook
        hook_root_name = 'test' if test_mode else 'validation'
        hook_name = f'on_{hook_root_name}_epoch_end'
        with self.profiler.profile(hook_name):
            # call hook
            getattr(self, hook_name)()

            # model hooks
            if self.is_function_implemented(hook_name):
                getattr(model, hook_name)()

    def _evaluate(
        self,
        model: LightningModule,
        dataloaders: List[DataLoader],
        max_batches: Union[int, List[int]],
        test_mode: bool = False
    ):
        """Run evaluation code.

        Args:
            model: The model to evaluate.
            dataloaders: A list of PyTorch dataloaders.
            max_batches: An integer or list of integers with length of the number of dataloaders. Each
                entry is the number of batches to process in the corresponding dataloader.
            test_mode:
        """
        # enable eval mode
        model.zero_grad()
        model.eval()

        # copy properties for forward overrides
        self.copy_trainer_model_properties(model)

        # disable gradients to save memory
        torch.set_grad_enabled(False)

        # bookkeeping
        outputs = []

        # convert max_batches to list
        if isinstance(max_batches, int):
            max_batches = [max_batches] * len(dataloaders)

        # --------------------------
        # ON_EVAL_EPOCH_START hook
        # --------------------------
        self.__call_eval_loop_hook_start(test_mode, model)

        # run validation
        for dataloader_idx, dataloader in enumerate(dataloaders):
            dl_outputs = []

            # on TPU we have to wrap it under the ParallelLoader
            if self.use_tpu:
                device = xm.xla_device(self.tpu_id)
                dataloader = xla_pl.ParallelLoader(dataloader, [device])
                dataloader = dataloader.per_device_loader(device)

            # each dataloader has a max num batches
            dl_max_batches = max_batches[dataloader_idx]

            for batch_idx, batch in enumerate(dataloader):
                if batch is None:
                    continue

                # stop short when on fast_dev_run (sets max_batch=1)
                if batch_idx >= dl_max_batches:
                    break

                # callbacks
                if test_mode:
                    self.on_test_batch_start()
                else:
                    self.on_validation_batch_start()

                # -----------------
                # RUN EVALUATION STEP
                # -----------------
                if self.use_amp and NATIVE_AMP_AVALAIBLE and not self.use_tpu:
                    with torch.cuda.amp.autocast():
                        output = self.evaluation_forward(model, batch, batch_idx, dataloader_idx, test_mode)
                else:
                    output = self.evaluation_forward(model, batch, batch_idx, dataloader_idx, test_mode)

                # allow only EvalResult when using structured results (from val_step)
                if isinstance(output, Result) and not isinstance(output, EvalResult):
                    m = 'only EvalResults or dicts are allowed from validation_step'
                    raise MisconfigurationException(m)

                # on dp / ddp2 might still want to do something with the batch parts
                if test_mode:
                    if self.is_overridden('test_step_end'):
                        model_ref = self.get_model()
                        with self.profiler.profile('test_step_end'):
                            output = model_ref.test_step_end(output)
                    self.on_test_batch_end()
                else:
                    if self.is_overridden('validation_step_end'):
                        model_ref = self.get_model()
                        with self.profiler.profile('validation_step_end'):
                            output = model_ref.validation_step_end(output)
                    self.on_validation_batch_end()

                # track outputs for collation
                if output is not None:
                    dl_outputs.append(output)

            outputs.append(dl_outputs)

        eval_results = outputs

        # with a single dataloader don't pass an array
        if len(dataloaders) == 1:
            eval_results = outputs[0]

        # give model a chance to do something with the outputs (and method defined)
        if isinstance(model, (LightningDistributedDataParallel, LightningDataParallel)):
            model = model.module

        user_reduced = False

        if test_mode:
            if self.is_overridden('test_end', model=model):
                # TODO: remove in v1.0.0
                eval_results = model.test_end(eval_results)
                user_reduced = True
                rank_zero_warn('Method `test_end` was deprecated in v0.7 and will be removed in v1.0.'
                               ' Use `test_epoch_end` instead.', DeprecationWarning)

            elif self.is_overridden('test_epoch_end', model=model):
                eval_results = model.test_epoch_end(eval_results)
                user_reduced = True

        else:
            if self.is_overridden('validation_end', model=model):
                # TODO: remove in v1.0.0
                eval_results = model.validation_end(eval_results)
                user_reduced = True
                rank_zero_warn('Method `validation_end` was deprecated in v0.7 and will be removed in v1.0.'
                               ' Use `validation_epoch_end` instead.', DeprecationWarning)

            elif self.is_overridden('validation_epoch_end', model=model):
                eval_results = model.validation_epoch_end(eval_results)
                user_reduced = True

        # when user didn't reduce and it's an EvalResult, auto-reduce
        using_eval_result = len(outputs) > 0 and len(outputs[0]) > 0 and isinstance(outputs[0][0], EvalResult)
        if using_eval_result and not user_reduced:
            eval_results = self.__auto_reduce_result_objs(outputs)

        # enable train mode again
        model.train()

        # enable gradients to save memory
        torch.set_grad_enabled(True)

        # --------------------------
        # ON_EVAL_EPOCH_END hook
        # --------------------------
        self.__call_eval_loop_hook_end(test_mode, model)

        return eval_results

    def __auto_reduce_result_objs(self, outputs):
        # outputs has a list of results per dataloader
        eval_results = []
        for dl_output in outputs:
            result = dl_output[0]
            result = result.__class__.reduce_on_epoch_end(dl_output)
            if 'checkpoint_on' in result:
                result.checkpoint_on = result.checkpoint_on.mean()
            if 'early_stop_on' in result:
                result.early_stop_on = result.early_stop_on.mean()
            eval_results.append(result)

            # update callback metrics
            self.callback_metrics = result.callback_metrics

        return eval_results

    def run_evaluation(self, test_mode: bool = False):
        # TODO: finish TrainResult support in eval loop
        # hook
        model = self.get_model()
        model.on_pre_performance_check()

        # select dataloaders
        if test_mode:
            self.reset_test_dataloader(model)

            dataloaders = self.test_dataloaders
            max_batches = self.num_test_batches
        else:
            # val
            if self.val_dataloaders is None:
                self.reset_val_dataloader(model)

            dataloaders = self.val_dataloaders
            max_batches = self.num_val_batches

        # enable fast_dev_run without val loop
        if dataloaders is None:
            return

        # cap max batches to 1 when using fast_dev_run
        if self.fast_dev_run:
            max_batches = [1]

        # Validation/Test begin callbacks
        if test_mode:
            self.on_test_start()
        else:
            self.on_validation_start()

        # enable disabling validation step with limit_val_batches = 0
        should_skip = sum(max_batches) == 0
        if should_skip:
            return [], []

        # run evaluation (val_step + val_step_end + val_epoch_end)
        eval_results = self._evaluate(self.model, dataloaders, max_batches, test_mode)

        # log the final eval loop metrics
        eval_loop_results = self.__log_evaluation_epoch_metrics(eval_results, test_mode)

        # hook
        model.on_post_performance_check()

        # eventual dataset reloading
        if test_mode:
            if self.reload_dataloaders_every_epoch:
                self.reset_test_dataloader(model)
        else:
            # val
            if self.reload_dataloaders_every_epoch:
                self.reset_val_dataloader(model)

        # Validation/Test end callbacks
        if test_mode:
            self.on_test_end()
        else:
            self.on_validation_end()

        return eval_loop_results, eval_results

    def __log_evaluation_epoch_metrics(self, eval_results, test_mode):
        eval_loop_results = []
        if eval_results is not None and len(eval_results) > 0:

            # in eval, the user may return something at every validation step without final reduction
            if not isinstance(eval_results, list):
                eval_results = [eval_results]

            for result in eval_results:
                if isinstance(result, EvalResult):
                    prog_bar_metrics = result.epoch_pbar_metrics
                    log_metrics = result.epoch_log_metrics
                    callback_metrics = result.callback_metrics
                else:
                    _, prog_bar_metrics, log_metrics, callback_metrics, _ = self.process_output(result)

                # add metrics to prog bar
                self.add_progress_bar_metrics(prog_bar_metrics)

                # log results of test
                if test_mode and self.is_global_zero and self.verbose_test:
                    print('-' * 80)
                    print('TEST RESULTS')
                    pprint(callback_metrics)
                    print('-' * 80)

                # log metrics
                self.log_metrics(log_metrics, {})

                # track metrics for callbacks
                self.callback_metrics.update(callback_metrics)

                if len(callback_metrics) > 0:
                    eval_loop_results.append(callback_metrics)

        return eval_loop_results

    def evaluation_forward(self, model, batch, batch_idx, dataloader_idx, test_mode: bool = False):
        # make dataloader_idx arg in validation_step optional
        args = [batch, batch_idx]

        if (test_mode and len(self.test_dataloaders) > 1) \
                or (not test_mode and len(self.val_dataloaders) > 1):
            args.append(dataloader_idx)

        # handle DP, DDP forward
        if self.use_ddp or self.use_dp or self.use_ddp2:
            output = model(*args)
            return output

        # Horovod
        if self.use_horovod and self.on_gpu:
            batch = self.transfer_batch_to_gpu(batch, hvd.local_rank())
            args[0] = batch

        # single GPU data transfer
        if self.single_gpu:
            # for single GPU put inputs on gpu manually
            root_gpu = 0
            if isinstance(self.data_parallel_device_ids, list):
                root_gpu = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, root_gpu)
            args[0] = batch

        # TPU data  transfer
        if self.use_tpu:
            batch = self.transfer_batch_to_tpu(batch, self.tpu_id)
            args[0] = batch

        # CPU, TPU or gpu step
        if test_mode:
            output = model.test_step(*args)
        else:
            output = model.validation_step(*args)

        return output
