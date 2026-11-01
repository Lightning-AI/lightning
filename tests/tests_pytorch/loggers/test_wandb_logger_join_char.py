from lightning.pytorch.loggers import WandbLogger


def test_wandb_logger_custom_join_char(wandb_mock):
    """Verify that WandbLogger correctly formats metric keys and logs them according to the specified join_char
    value."""

    # Case 1: Default join_char -> metrics should be logged with hyphens
    wandb_mock.run = None
    logger = WandbLogger()  # default join_char = "-"
    logger.log_metrics({"train/loss": 0.5}, step=1)

    print("\nACTUAL:", wandb_mock.init().log.call_args)

    # Expected output: metrics logged using hyphenated keys
    wandb_mock.init().log.assert_called_with({"train-loss": 0.5, "trainer/global_step": 1})

    # Case 2: Custom join_char="/" -> metrics should keep their original keys
    wandb_mock.init().log.reset_mock()
    logger = WandbLogger(join_char="/")
    logger.log_metrics({"train/loss": 0.7}, step=2)

    # Expected output: metrics logged with unmodified key names
    wandb_mock.init().log.assert_called_with({"train/loss": 0.7, "trainer/global_step": 2})

    # Case 3: Custom join_char="_" -> metrics should use underscores
    wandb_mock.init().log.reset_mock()
    logger = WandbLogger(join_char="_")
    logger.log_metrics({"train/loss": 0.9}, step=3)

    # Expected output: metrics logged with underscores in key names
    wandb_mock.init().log.assert_called_with({"train_loss": 0.9, "trainer/global_step": 3})


def test_join_char_persists_across_runs(wandb_mock):
    """Ensure that a user-defined join_char is consistently applied across multiple logs."""
    wandb_mock.run = None
    logger = WandbLogger(join_char="-")

    # Log multiple metric sets with the same logger instance
    logger.log_metrics({"val/loss": 0.1}, step=1)
    logger.log_metrics({"val/acc": 0.95}, step=2)

    # Retrieve the logged calls for inspection
    calls = wandb_mock.init().log.call_args_list
    logged_1 = calls[0].args[0] if calls[0].args else calls[0].kwargs
    logged_2 = calls[1].args[0] if calls[1].args else calls[1].kwargs

    # Expected output: both logs use the same join_char formatting (hyphens)
    assert "val-loss" in logged_1
    assert "val-acc" in logged_2
