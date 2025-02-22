.. role:: hidden
    :class: hidden-section

.. _callbacks:

########
Callback
########

Callbacks allow you to add arbitrary self-contained programs to your training.
At specific points during the flow of execution (hooks), the Callback interface allows you to design programs that encapsulate a full set of functionality.
It de-couples functionality that does not need to be in the :doc:`lightning module <../common/lightning_module>` and can be shared across projects.

Lightning has a callback system to execute them when needed. Callbacks should capture NON-ESSENTIAL
logic that is NOT required for your :doc:`lightning module <../common/lightning_module>` to run.

A complete list of Callback hooks can be found in :class:`~lightning.pytorch.callbacks.callback.Callback`.

An overall Lightning system should have:

1. Trainer for all engineering
2. LightningModule for all research code.
3. Callbacks for non-essential code.

|

Example:

.. testcode::

    from lightning.pytorch.callbacks import Callback


    class MyPrintingCallback(Callback):
        def on_train_start(self, trainer, pl_module):
            print("Training is starting")

        def on_train_end(self, trainer, pl_module):
            print("Training is ending")


    trainer = Trainer(callbacks=[MyPrintingCallback()])

We successfully extended functionality without polluting our super clean
:doc:`lightning module <../common/lightning_module>` research code.

You can do pretty much anything with callbacks.

--------------

******************
Built-in Callbacks
******************
Lightning has a few built-in callbacks.

.. note::
    For a richer collection of callbacks, check out our
    `bolts library <https://lightning-bolts.readthedocs.io/en/stable/index.html>`_.

.. currentmodule:: lightning.pytorch.callbacks

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    BackboneFinetuning
    BaseFinetuning
    BasePredictionWriter
    BatchSizeFinder
    Callback
    DeviceStatsMonitor
    EarlyStopping
    GradientAccumulationScheduler
    LambdaCallback
    LearningRateFinder
    LearningRateMonitor
    ModelCheckpoint
    ModelPruning
    ModelSummary
    ProgressBar
    RichModelSummary
    RichProgressBar
    StochasticWeightAveraging
    Timer
    TQDMProgressBar
    WeightAveraging

----------

.. include:: callbacks_state.rst

----------


**************
Best Practices
**************
The following are best practices when using/designing callbacks.

1. Callbacks should be isolated in their functionality.
2. Your callback should not rely on the behavior of other callbacks in order to work properly.
3. Do not manually call methods from the callback.
4. Directly calling methods (eg. `on_validation_end`) is strongly discouraged.
5. Whenever possible, your callbacks should not depend on the order in which they are executed.


-----------

.. include:: entry_points.rst

-----------

.. _callback_hooks:

************
Callback API
************
Here is the full API of methods available in the Callback base class.

The :class:`~lightning.pytorch.callbacks.Callback` class is the base for all the callbacks in Lightning just like the :class:`~lightning.pytorch.core.LightningModule` is the base for all models.
It defines a public interface that each callback implementation must follow, the key ones are:

Properties
==========

state_key
^^^^^^^^^

.. autoattribute:: lightning.pytorch.callbacks.Callback.state_key
    :noindex:


Hooks
=====

setup
^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.setup
    :noindex:

teardown
^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.teardown
    :noindex:

on_fit_start
^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_fit_start
    :noindex:

on_fit_end
^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_fit_end
    :noindex:

on_sanity_check_start
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_sanity_check_start
    :noindex:

on_sanity_check_end
^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_sanity_check_end
    :noindex:

on_train_batch_start
^^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_train_batch_start
    :noindex:

on_train_batch_end
^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_train_batch_end
    :noindex:

on_train_epoch_start
^^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_train_epoch_start
    :noindex:

on_train_epoch_end
^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_train_epoch_end
    :noindex:

on_validation_epoch_start
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_validation_epoch_start
    :noindex:

on_validation_epoch_end
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_validation_epoch_end
    :noindex:

on_test_epoch_start
^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_test_epoch_start
    :noindex:

on_test_epoch_end
^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_test_epoch_end
    :noindex:

on_predict_epoch_start
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_predict_epoch_start
    :noindex:

on_predict_epoch_end
^^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_predict_epoch_end
    :noindex:

on_validation_batch_start
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_validation_batch_start
    :noindex:

on_validation_batch_end
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_validation_batch_end
    :noindex:

on_test_batch_start
^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_test_batch_start
    :noindex:

on_test_batch_end
^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_test_batch_end
    :noindex:

on_predict_batch_start
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_predict_batch_start
    :noindex:

on_predict_batch_end
^^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_predict_batch_end
    :noindex:

on_train_start
^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_train_start
    :noindex:

on_train_end
^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_train_end
    :noindex:

on_validation_start
^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_validation_start
    :noindex:

on_validation_end
^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_validation_end
    :noindex:

on_test_start
^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_test_start
    :noindex:

on_test_end
^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_test_end
    :noindex:

on_predict_start
^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_predict_start
    :noindex:

on_predict_end
^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_predict_end
    :noindex:


on_exception
^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_exception
    :noindex:

state_dict
^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.state_dict
    :noindex:

on_save_checkpoint
^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_save_checkpoint
    :noindex:

load_state_dict
^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.load_state_dict
    :noindex:

on_load_checkpoint
^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_load_checkpoint
    :noindex:

on_before_backward
^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_before_backward
    :noindex:

on_after_backward
^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_after_backward
    :noindex:

on_before_optimizer_step
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_before_optimizer_step
    :noindex:

on_before_zero_grad
^^^^^^^^^^^^^^^^^^^

.. automethod:: lightning.pytorch.callbacks.Callback.on_before_zero_grad
    :noindex:
