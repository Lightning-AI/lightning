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
import logging
import os
from typing import TYPE_CHECKING, Any, Literal, Optional

import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import get_args, override

from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_6
from lightning.fabric.utilities.types import _PATH

if TYPE_CHECKING:
    from deepspeed.io import FastFileWriterConfig
    from deepspeed.io.base_file_writer import BaseFileWriter
    from deepspeed.ops.op_builder.async_io import AsyncIOBuilder


log = logging.getLogger(__name__)

_DEEPSPEED_AVAILABLE = RequirementCache("deepspeed")
_DEEPSPEED_GREATER_EQUAL_0_16 = RequirementCache("deepspeed>=0.16.0")

_SUPPORTED_HANDLER_TYPES = Literal["aio", "gds"]
_SUPPORTED_WRITERS = Literal["mock", "pyfile", "fast"]


class DeepNVMECheckpointIO(TorchCheckpointIO):
    """CheckpointIO that utilizes DeepSpeed's DeepNVME & FastPersist for optimized checkpoint persistence.

    DeepNVMe delivers significant I/O performance improvements for deep learning workloads by leveraging
    advanced storage technologies including:
    - Local NVMe SSDs
    - NVIDIA Magnum IO™ GPUDirect® Storage (GDS)
    - Linux Asynchronous I/O (AIO)

    This implementation provides faster checkpoint save/load operations compared to standard PyTorch I/O,
    making it ideal for large-scale distributed training scenarios with I/O bottlenecks.

    .. warning::
        This is an :ref:`experimental <versioning:Experimental API>` feature and may be subject to
        breaking changes in future releases.

    Note:
        Requires DeepSpeed to be installed with DeepNVMe support enabled.

    """

    def __init__(
        self,
        *,
        use_gpu: bool = False,
        handler: Optional[_SUPPORTED_HANDLER_TYPES] = None,
        use_double_io_buffer: bool = False,
        use_zipfile_format: bool = False,
        writer: _SUPPORTED_WRITERS = "fast",
        config: Optional["FastFileWriterConfig"] = None,
        aio_queue_depth: int = 8,
        aio_block_size: int = 8 * (1024**2),
        aio_intra_op_parallel: int = 1,
        aio_single_submit: bool = False,
        aio_overlap_events: bool = False,
        pinned_buffer_mb: int = 64,
    ) -> None:
        """Initialize the DeepNVMECheckpointIO plugin.

        Args:
            use_gpu: Whether checkpoint persistence should leverage GPU-aware I/O paths.
                When enabled and supported by the system, GPUDirect Storage (GDS) may be used
                to accelerate transfers between GPU memory and local NVMe storage.
            handler: The DeepNVMe I/O backend to use. Supported values are ``"aio"`` for
                Linux asynchronous I/O or ``"gds"`` for GPUDirect Storage. If ``None``,
                the handler is selected automatically based on system compatibility.
            use_double_io_buffer: Whether to enable double-buffering during writes. This may
                improve overlap between serialization and disk I/O.
            use_zipfile_format: Whether to use PyTorch's newer zipfile-based serialization
                format when saving checkpoints.
            writer: The file writer backend to use. Supported values are ``"mock"``,
                ``"pyfile"``, or ``"fast"``. Default is ``"fast"``.
            config: Optional configuration object for the file writer (e.g., ``FastFileWriterConfig``).
                If ``None``, default configuration is used.
            aio_queue_depth: Queue depth for asynchronous I/O operations when using the AIO handler.
                Controls the number of pending I/O operations. Default is 8.
            aio_block_size: Block size in bytes for asynchronous I/O operations when using the AIO handler.
                Default is 8 MB.
            aio_intra_op_parallel: Number of intra-operation parallel tasks for the AIO handler.
                Default is 1.
            aio_single_submit: Whether to submit a single I/O request per operation with the AIO handler.
                Default is False.
            aio_overlap_events: Whether to enable event overlapping with the AIO handler.
                May improve performance by overlapping I/O and computation. Default is False.
            pinned_buffer_mb: Size of the pinned I/O buffer in megabytes used during checkpoint
                writes. Larger buffers can improve throughput at the cost of additional memory usage.
                Default is 64 MB.

        Raises:
            ImportError: If DeepSpeed is not installed or the installed version is incompatible
                with the current PyTorch version.
            RuntimeError: If no compatible DeepNVMe handler is available for the system.
            ValueError: If an unsupported handler type is provided.

        """
        if not _DEEPSPEED_AVAILABLE:
            raise ImportError(
                "To use the `DeepNVMECheckpointIO`, you must have DeepSpeed installed."
                " Install it by running `pip install -U deepspeed`."
            )

        if _TORCH_GREATER_EQUAL_2_6 and not _DEEPSPEED_GREATER_EQUAL_0_16:
            import deepspeed

            deepspeed_version = deepspeed.__version__

            raise ImportError(
                f"PyTorch >= 2.6 requires DeepSpeed >= 0.16.0. "
                f"Detected DeepSpeed version: {deepspeed_version}. "
                "Please upgrade by running `pip install -U 'deepspeed>=0.16.0'`."
            )

        from deepspeed.ops.op_builder.async_io import AsyncIOBuilder
        from deepspeed.ops.op_builder.gds import GDSBuilder

        super().__init__()
        self.use_gpu = use_gpu
        if handler is None:
            # Auto-select handler based on system capabilities
            if use_gpu and GDSBuilder().is_compatible():
                handler = "gds"
            elif AsyncIOBuilder().is_compatible():
                handler = "aio"
            else:
                raise RuntimeError(
                    "No compatible I/O handler found for DeepNVMe. "
                    "Please ensure your system meets the requirements for either AIO or GDS."
                )
        if handler not in get_args(_SUPPORTED_HANDLER_TYPES):
            raise ValueError(
                f"Unsupported handler type: {handler}. Supported types are: {get_args(_SUPPORTED_HANDLER_TYPES)}."
            )
        if writer not in get_args(_SUPPORTED_WRITERS):
            raise ValueError(f"Unsupported writer type: {writer}. Supported types are: {get_args(_SUPPORTED_WRITERS)}.")

        self._handler_type = handler
        self._use_double_io_buffer = use_double_io_buffer
        self._use_zipfile_format = use_zipfile_format
        self._writer: Optional[BaseFileWriter] = None
        self._writer_key: _SUPPORTED_WRITERS = writer
        self._writer_config = config
        self._aio_queue_depth = aio_queue_depth
        self._aio_block_size = aio_block_size
        self._aio_intra_op_parallel = aio_intra_op_parallel
        self._aio_single_submit = aio_single_submit
        self._aio_overlap_events = aio_overlap_events
        self._pinned_buffer_mb = pinned_buffer_mb

        self._load_io_ops()
        self._dnvme_handler, self._pinned_memory = self._get_handler_and_pinned_memory()

    @override
    def save_checkpoint(self, checkpoint: dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: not used in ``TorchCheckpointIO.save_checkpoint``

        Raises:
            TypeError:
                If ``storage_options`` arg is passed in

        """
        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}`. Please implement your custom `CheckpointIO`"
                " to define how you'd like to use `storage_options`."
            )

        self._flush_writer()  # Ensure any previous writer is flushed before creating a new one

        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)

        # don’t close writer immediately: allow IO overlap
        self._writer = self._get_writer(path)

        torch.save(f=self._writer, obj=checkpoint, _use_new_zipfile_serialization=self._use_zipfile_format)

    @override
    def teardown(self) -> None:
        """This method is called to teardown the process."""
        self._flush_writer()

    def _flush_writer(self) -> None:
        """Force flush the writer to ensure all data is written to storage."""
        if self._writer is not None:
            # close the writer to ensure all data is flushed before we attempt to store again
            self._writer.close()
            self._writer = None

    def _load_io_ops(self) -> None:
        from deepspeed.ops.op_builder.async_io import AsyncIOBuilder
        from deepspeed.ops.op_builder.gds import GDSBuilder

        if self._handler_type == "aio":
            if not AsyncIOBuilder().is_compatible():
                raise RuntimeError(
                    "The selected I/O handler 'aio' is not compatible with the current system configuration. "
                    "Please ensure your system meets the requirements for AIO or select a different handler."
                )
            AsyncIOBuilder().load(verbose=False)
        if self._handler_type == "gds":
            if not GDSBuilder().is_compatible():
                raise RuntimeError(
                    "The selected I/O handler 'gds' is not compatible with the current system configuration. "
                    "Please ensure your system meets the requirements for GDS or select a different handler."
                )
            GDSBuilder().load(verbose=False)

    def _get_handler(self) -> "AsyncIOBuilder":
        from deepspeed.ops.op_builder.async_io import AsyncIOBuilder
        from deepspeed.ops.op_builder.gds import GDSBuilder

        if self._handler_type == "aio":
            return (
                AsyncIOBuilder()
                .load(verbose=False)
                .aio_handle(
                    block_size=self._aio_block_size,
                    queue_depth=self._aio_queue_depth,
                    single_submit=self._aio_single_submit,
                    overlap_events=self._aio_overlap_events,
                    intra_op_parallelism=self._aio_intra_op_parallel,
                )
            )
        if self._handler_type == "gds":
            return (
                GDSBuilder()
                .load(verbose=False)
                .gds_handle(
                    block_size=self._aio_block_size,
                    queue_depth=self._aio_queue_depth,
                    single_submit=self._aio_single_submit,
                    overlap_events=self._aio_overlap_events,
                    intra_op_parallelism=self._aio_intra_op_parallel,
                )
            )
        raise ValueError(f"Unsupported handler type: {self._handler_type}")

    def _get_handler_and_pinned_memory(self) -> tuple["AsyncIOBuilder", torch.Tensor]:
        from deepspeed.accelerator import get_accelerator

        handler = self._get_handler()
        pinned_memory = torch.empty(
            self._pinned_buffer_mb * (1024**2), dtype=torch.uint8, device=get_accelerator().current_device_name()
        )

        handler.pin_device_tensor(pinned_memory)

        return handler, pinned_memory

    def _get_writer(
        self,
        filename: _PATH,
    ) -> "BaseFileWriter":
        from deepspeed.io import BaseFileWriter, FastFileWriter, FastFileWriterConfig, MockFileWriter, PyFileWriter

        _WRITER_MAP: dict[str, type[BaseFileWriter]] = {
            "mock": MockFileWriter,
            "pyfile": PyFileWriter,
            "fast": FastFileWriter,
        }
        assert all(_WRITER_MAP[_w] is not None for _w in _SUPPORTED_WRITERS), (
            "Expected all writer keys to be mapped to a valid writer class."
        )

        writer = _WRITER_MAP.get(self._writer_key)

        if writer is None:
            raise ValueError(
                f"Writer type {self._writer_key} is not recognized. "
                f"Supported types are: {get_args(_SUPPORTED_WRITERS)}."
            )

        if not issubclass(writer, BaseFileWriter):
            raise ValueError(
                f"Writer type {writer} is not a subclass of BaseFileWriter. Please ensure the writer is valid."
            )

        if writer is FastFileWriter:
            if self._writer_config is None:
                self._writer_config = FastFileWriterConfig(
                    dnvme_handle=self._dnvme_handler,
                    pinned_tensor=self._pinned_memory,
                    double_buffer=self._use_double_io_buffer,
                    num_parallel_writers=1,
                    writer_rank=0,
                )
            return writer(filename, self._writer_config)
        return writer(filename)
