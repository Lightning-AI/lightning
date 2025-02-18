.. _loggers:

###############################
Track and Visualize Experiments
###############################

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Basic
   :description: Learn how to track and visualize metrics, images and text.
   :col_css: col-md-4
   :button_link: logging_basic.html
   :height: 150
   :tag: basic

.. displayitem::
   :header: Intermediate
   :description: Enable third-party experiment managers with advanced visualizations.
   :col_css: col-md-4
   :button_link: logging_intermediate.html
   :height: 150
   :tag: intermediate

.. displayitem::
   :header: Advanced
   :description: Optimize model speed with advanced self.log arguments and cloud logging.
   :col_css: col-md-4
   :button_link: logging_advanced.html
   :height: 150
   :tag: advanced

.. displayitem::
   :header: Expert
   :description: Make your own progress-bar or integrate a new experiment manager.
   :col_css: col-md-4
   :button_link: logging_expert.html
   :height: 150
   :tag: expert

.. displayitem::
   :header: LightningModule.log API
   :description: Dig into the LightningModule.log API in depth
   :col_css: col-md-4
   :button_link: ../common/lightning_module.html#log
   :height: 150

.. raw:: html

        </div>
    </div>

.. _mlflow_logger:

MLflow Logger
-------------

The MLflow logger in PyTorch Lightning now includes a `checkpoint_path_prefix` parameter. This parameter allows you to prefix the checkpoint artifact's path when logging checkpoints as artifacts.

Example usage:

.. code-block:: python

    import lightning as L
    from lightning.pytorch.loggers import MLFlowLogger

    mlf_logger = MLFlowLogger(
        experiment_name="lightning_logs",
        tracking_uri="file:./ml-runs",
        checkpoint_path_prefix="my_prefix"
    )
    trainer = L.Trainer(logger=mlf_logger)

    # Your LightningModule definition
    class LitModel(L.LightningModule):
        def training_step(self, batch, batch_idx):
            # example
            self.logger.experiment.whatever_ml_flow_supports(...)

        def any_lightning_module_function_or_hook(self):
            self.logger.experiment.whatever_ml_flow_supports(...)

    # Train your model
    model = LitModel()
    trainer.fit(model)
