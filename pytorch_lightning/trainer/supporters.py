import csv
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor


class TensorRunningAccum(object):
    """Tracks a running accumulation values (min, max, mean) without graph
    references.

    Examples:
        >>> accum = TensorRunningAccum(5)
        >>> accum.last(), accum.mean()
        (None, None)
        >>> accum.append(torch.tensor(1.5))
        >>> accum.last(), accum.mean()
        (tensor(1.5000), tensor(1.5000))
        >>> accum.append(torch.tensor(2.5))
        >>> accum.last(), accum.mean()
        (tensor(2.5000), tensor(2.))
        >>> accum.reset()
        >>> _= [accum.append(torch.tensor(i)) for i in range(13)]
        >>> accum.last(), accum.mean(), accum.min(), accum.max()
        (tensor(12.), tensor(10.), tensor(8.), tensor(12.))
    """

    def __init__(self, window_length: int):
        self.window_length = window_length
        self.memory = torch.Tensor(self.window_length)
        self.current_idx: int = 0
        self.last_idx: Optional[int] = None
        self.rotated: bool = False

    def reset(self) -> None:
        """Empty the accumulator."""
        self = TensorRunningAccum(self.window_length)

    def last(self):
        """Get the last added element."""
        if self.last_idx is not None:
            return self.memory[self.last_idx]

    def append(self, x):
        """Add an element to the accumulator."""
        # ensure same device and type
        if self.memory.device != x.device or self.memory.type() != x.type():
            x = x.to(self.memory)

        # store without grads
        with torch.no_grad():
            self.memory[self.current_idx] = x
            self.last_idx = self.current_idx

        # increase index
        self.current_idx += 1

        # reset index when hit limit of tensor
        self.current_idx = self.current_idx % self.window_length
        if self.current_idx == 0:
            self.rotated = True

    def mean(self):
        """Get mean value from stored elements."""
        return self._agg_memory('mean')

    def max(self):
        """Get maximal value from stored elements."""
        return self._agg_memory('max')

    def min(self):
        """Get minimal value from stored elements."""
        return self._agg_memory('min')

    def _agg_memory(self, how: str):
        if self.last_idx is not None:
            if self.rotated:
                return getattr(self.memory, how)()
            else:
                return getattr(self.memory[:self.current_idx], how)()


class Accumulator(object):
    def __init__(self):
        self.num_values = 0
        self.total = 0

    def accumulate(self, x):
        with torch.no_grad():
            self.total += x
            self.num_values += 1

    def mean(self):
        return self.total / self.num_values


class PredictionCollection(object):

    def __init__(self, global_rank: int, world_size: int):
        self.global_rank = global_rank
        self.world_size = world_size
        self.predictions = {}
        self.num_predictions = 0

    def _add_prediction(self, name, values, filename):
        if filename not in self.predictions:
            self.predictions[filename] = {name: values}
        elif name not in self.predictions[filename]:
            self.predictions[filename][name] = values
        elif isinstance(values, Tensor):
            self.predictions[filename][name] = torch.cat((self.predictions[filename][name], values.detach()))
        elif isinstance(values, list):
            self.predictions[filename][name].extend(values)

    def add(self, predictions):

        if predictions is None:
            return

        for filename, pred_dict in predictions.items():
            for feature_name, values in pred_dict.items():
                self._add_prediction(feature_name, values, filename)

    def to_disk(self):
        """Write predictions to file(s).
        """
        for filename, predictions in self.predictions.items():

            # Filename or filename with rank extension in multi-gpu environment
            outfile = Path(filename)
            outfile = Path(f"{outfile.stem}{f'_rank_{self.global_rank}' if self.world_size > 1 else ''}{outfile.suffix}")
            outfile.parent.mkdir(exist_ok=True, parents=True)

            # Convert any tensor values to list
            predictions = {k: v if not isinstance(v, Tensor) else v.tolist() for k, v in predictions.items()}

            # This checks all values have same len so we can write csv
            # TODO - change message to say something like:
            # 'got len x for feature_name, and len y for other_feature_name'
            assert len(set([len(v) for v in predictions.values()])) == 1, "prediction values not equal"

            # Write predictions for current file to disk
            with outfile.open(mode='w', newline='\n') as f:
                fieldnames = list(predictions.keys())
                writer = csv.writer(f)
                writer.writerow(fieldnames)
                writer.writerows(zip(*predictions.values()))
