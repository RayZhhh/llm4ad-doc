# Constructive Heuristics for 2D Bin Packing Problem

## Problem

The **2D Bin Packing Problem** involves packing rectangular items into rectangular bins.

- **Given:** A set of rectangular items with specific widths and heights, bins of fixed dimensions
- **Objective:** Minimize the number of bins used
- **Constraints:** Items must not overlap, items must fit within bin boundaries

## Algorithm Design Task

Constructive heuristics iteratively select items and bins for placement.

- **Inputs:** Remaining items (width, height), point matrices (occupancy grids)
- **Outputs:** (selected_item, selected_bin)

## Evaluation

- **Dataset:** 8 instances, 100 items, 100x100 bins
- **Fitness:** Average number of bins used (minimized)

## Template

```python
template_program = '''
import numpy as np

def determine_next_assignment(remaining_items: List[Tuple[int, int]],
                             point_matrices: List[List[List[int]]]) -> Tuple[Tuple[int, int], int]:
    """
    A simple heuristic function to select the next item and bin for packing.

    Args:
        remaining_items: List of (width, height) tuples.
        point_matrices: List of 2D matrices (0=empty, 1=occupied).

    Returns:
        A tuple containing: (selected_item, selected_bin)
    """
    selected_item = max(remaining_items, key=lambda x: x[0] * x[1])

    for bin_idx, point_matrix in enumerate(point_matrices):
        bin_width = len(point_matrix)
        bin_height = len(point_matrix[0]) if bin_width > 0 else 0
        if bin_width >= selected_item[0] and bin_height >= selected_item[1]:
            for x in range(bin_width - selected_item[0] + 1):
                for y in range(bin_height - selected_item[1] + 1):
                    if all(point_matrix[x + dx][y + dy] == 0
                           for dx in range(selected_item[0])
                           for dy in range(selected_item[1])):
                        return selected_item, bin_idx
    return selected_item, None
'''

task_description = '''
Given a set of rectangular bins and rectangular items, iteratively assign each item to a
feasible position in one of the bins. Design a constructive heuristic that, in each
iteration, selects the best item and placement from the remaining items and feasible
corners, with the objective of minimizing the number of used bins.
'''
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `timeout_seconds` | Maximum evaluation time | 120 |
| `n_bins` | Number of bins | 100 |
| `n_instance` | Number of problem instances | 8 |
| `n_items` | Number of items | 100 |
| `bin_width` | Bin width | 100 |
| `bin_height` | Bin height | 100 |

## Example Usage

```python
from llm4ad.task.optimization.bp_2d_construct import BP2DEvaluation

evaluator = BP2DEvaluation()

def determine_next_assignment(remaining_items, point_matrices):
    selected_item = max(remaining_items, key=lambda x: x[0] * x[1])
    for bin_idx, point_matrix in enumerate(point_matrices):
        bin_width = len(point_matrix)
        bin_height = len(point_matrix[0]) if bin_width > 0 else 0
        if bin_width >= selected_item[0] and bin_height >= selected_item[1]:
            for x in range(bin_width - selected_item[0] + 1):
                for y in range(bin_height - selected_item[1] + 1):
                    if all(point_matrix[x + dx][y + dy] == 0
                           for dx in range(selected_item[0])
                           for dy in range(selected_item[1])):
                        return selected_item, bin_idx
    return selected_item, None

result = evaluator.evaluate_program('', determine_next_assignment)
```
