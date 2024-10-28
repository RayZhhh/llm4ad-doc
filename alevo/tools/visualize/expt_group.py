from __future__ import annotations

from typing import List, Dict


class ExptGroup:
    def __init__(self,
                 label: str,
                 paths: List[str],
                 optimal_values: List[float | int] = None):
        """
        Args:
            label: label of the experiment.
            paths: experiment paths.
            optimal_values: optimal vales of each path.
        """
        self.label = label
        self.paths = paths
        self.optimal_values = optimal_values

        if optimal_values is None:
            self.optimal_values = [None] * len(paths)
        elif isinstance(optimal_values, (int, float)):
            self.optimal_values = [optimal_values] * len(paths)
