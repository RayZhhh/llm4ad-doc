# Constructive Heuristics for Set Covering Problem (SCP)

## Problem

The **Set Covering Problem (SCP)** is a classic combinatorial optimization problem.

- **Given:** A universal set of elements and a collection of subsets
- **Objective:** Select minimum number of subsets that cover all elements
- **Constraints:** Each element must be covered by at least one selected subset

## Algorithm Design Task

Constructive heuristics iteratively select subsets to cover remaining elements.

- **Inputs:** Selected subsets, remaining subsets, remaining elements
- **Outputs:** Next subset to select

## Evaluation

- **Dataset:** 16 instances, 50 elements, 50 subsets
- **Fitness:** Average number of subsets used (minimized)

## Template

```python
template_program = '''
import numpy as np

def select_next_subset(selected_subsets: List[List[int]], remaining_subsets: List[List[int]], remaining_elements: List[int]) -> List[int] | None:
    """
    A heuristic for the Set Covering Problem.

    Args:
        selected_subsets: List of already selected subsets.
        remaining_subsets: List of remaining subsets to choose from.
        remaining_elements: List of elements still to be covered.

    Returns:
        The next subset to select, or None if no subset can cover any remaining elements.
    """
    max_covered = 0
    best_subset = None

    for subset in remaining_subsets:
        covered = len(set(subset).intersection(remaining_elements))
        if covered > max_covered:
            max_covered = covered
            best_subset = subset

    return best_subset
'''

task_description = '''
The task involves selecting a minimum number of subsets from a collection that
covers all elements in a universal set.
Help me design a novel algorithm to select the next subset in each step.
'''
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `timeout_seconds` | Maximum evaluation time | 30 |
| `n_instance` | Number of problem instances | 16 |
| `n_elements` | Number of elements | 50 |
| `n_subsets` | Number of subsets | 50 |
| `max_subset_size` | Maximum subset size | 8 |

## Example Usage

```python
from llm4ad.task.optimization.set_cover_construct import SCPEvaluation

evaluator = SCPEvaluation()

def select_next_subset(selected_subsets, remaining_subsets, remaining_elements):
    max_covered = 0
    best_subset = None
    for subset in remaining_subsets:
        covered = len(set(subset).intersection(remaining_elements))
        if covered > max_covered:
            max_covered = covered
            best_subset = subset
    return best_subset

result = evaluator.evaluate_program('', select_next_subset)
```
