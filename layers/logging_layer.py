import torch
import torch.nn
from typing import Dict, Any, Callable
from framework.utils import U


class LoggingLayer:
    def __init__(self) -> None:
        super().__init__()
        self._logs = {}
        self._log_counts = {}
        self._custom_reductions = {}

    def custom_reduction(self, name: str, reduction):
        self._custom_reductions[name] = reduction

    def log(self, name: str, value: Any, drop_old: bool = False):
        value = U.apply_to_tensors(value, lambda x: x.detach().cpu())

        if name in self._custom_reductions:
            if name not in self._logs:
                self._logs[name] = []

            self._logs[name].append(value)
        else:
            if name not in self._logs or drop_old:
                self._logs[name] = value
                self._log_counts[name] = 1
            else:
                self._logs[name] = self._logs[name] + value
                self._log_counts[name] = self._log_counts[name] + 1

    def get_logs(self) -> Dict[str, Any]:
        res = {}
        for k, v in self._logs.items():
            if k in self._custom_reductions:
                res[k] = self._custom_reductions[k](v)
            else:
                res[k] = v / self._log_counts[k]

        self._logs = {}
        self._log_counts = {}
        return res


def get_logs(module: torch.nn.Module) -> Dict[str, Any]:
    res = {}
    for n, m in module.named_modules():
        if isinstance(m, LoggingLayer):
            logs = m.get_logs()
            res.update({f"{n}/{k}": v for k, v in logs.items()})
    return res
