# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
EmailCallback
===============

Sends an email to a list of emails when training is complete.
"""

import logging
import smtplib
import textwrap
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import List, Optional

from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback

log = logging.getLogger(__name__)


class SMTPProvider(Enum):
    """Enum representing different SMTP providers with their server address and port.

    Attributes:
        GMAIL (tuple): Gmail SMTP server address and port.

    """

    GMAIL = ("smtp.gmail.com", 587)
    # YAHOO = ("smtp.mail.yahoo.com", 587)
    # OUTLOOK = ("smtp.office365.com", 587)
    # ZOHO = ("smtp.zoho.com", 587)
    # Add more providers as needed


class EmailCallback(Callback):
    r"""Send an email notification when training is complete.

    Args:
        sender_email: Email address of the sender.
        password: Password for the sender's email.
        receiver_emails: List of email addresses to send the notification to. Defaults to sender_email if None.
        smtp_provider: SMTP provider to use for sending the email. Defaults to SMTPProvider.GMAIL.
        metric_precision: Number of decimal places to use for metric values in the email. Defaults to 5.

    Example:

        >>> import os
        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.callbacks import EmailCallback
        >>> your_password = os.getenv("EMAIL_PASSWORD")  # strongly suggest not to hardcode password
        >>> email_callback = EmailCallback(
        ...     sender_email = "your_email@example.com",
        ...     password = your_password,
        ...     receiver_emails = ["receiver@example.com"]
        ... )
        >>> trainer = Trainer(callbacks=[email_callback])

    SMTP Providers:
        Currently supported SMTP servers

        - GMAIL: Gmail SMTP server address and port.

    Attributes:
        EMAIL_BODY_TEMPLATE (str): Template for the body of the email.

    Methods:
        on_train_end(trainer, pl_module): Called when training ends to send an email notification.

    Raises:
        Exception: If there is an error while sending the email.

    """

    EMAIL_BODY_TEMPLATE = textwrap.dedent(
        """
    Hello,

    The training for the model {module} has been completed.

    - Final Epoch: {final_epoch}
    - Total Steps: {total_steps}

    Logged Metrics:
    """
    )

    def __init__(
        self,
        sender_email: str,
        password: str,
        receiver_emails: Optional[List[str]] = None,
        smtp_provider: SMTPProvider = SMTPProvider.GMAIL,
        metric_precision: int = 5,
    ):
        self.sender_email = sender_email
        self.receiver_emails = receiver_emails if receiver_emails else [sender_email]
        self.password = password
        self.smtp_server, self.smtp_port = smtp_provider.value
        self.metric_precision = metric_precision

    @override
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.fast_dev_run:
            return
        try:
            # Create the email message
            msg = MIMEMultipart()
            msg["From"] = self.sender_email
            msg["To"] = ", ".join(self.receiver_emails)
            msg["Subject"] = f"Training for {pl_module.__class__.__name__} completed"

            # Gather detailed training information
            final_epoch = trainer.current_epoch
            total_steps = trainer.global_step
            metrics = trainer.callback_metrics

            # Format the body of the email with named placeholders
            body = self.EMAIL_BODY_TEMPLATE.format(
                module=pl_module.__class__.__name__,
                final_epoch=final_epoch,
                total_steps=total_steps,
            )

            for key, value in metrics.items():
                if isinstance(value, (float, int)):  # Ensure value is numeric
                    value = round(value, self.metric_precision)
                elif hasattr(value, "item"):  # For tensors or numpy values
                    value = round(value.item(), self.metric_precision)
                body += f"- {key}: {value}\n"

            body += "\nBest regards,\nPytorch Lightning"

            # Attach the body with the msg instance
            msg.attach(MIMEText(body, "plain"))

            # Set up the SMTP server
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.password)

            # Send the email to each recipient
            for recipient in self.receiver_emails:
                server.sendmail(self.sender_email, recipient, msg.as_string())

            # Quit the server
            server.quit()
            log.info(f"Completion email successfully sent to: {', '.join(self.receiver_emails)}")
        except Exception as e:
            log.exception(f"An error occurred while sending an email to confirm training completion: {e}")
