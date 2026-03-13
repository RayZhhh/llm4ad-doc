# Constructive Heuristics for Quadratic Assignment Problem (QAP)

## Problem

The **Quadratic Assignment Problem (QAP)** is one of the most challenging combinatorial optimization problems.

- **Given:** Two matrices - flow matrix (between facilities) and distance matrix (between locations)
- **Objective:** Assign facilities to locations to minimize total cost
- **Constraints:** Each facility must be assigned to exactly one location

## Algorithm Design Task

The task is to design a constructive heuristic that assigns all facilities to locations in one call.

- **Inputs:** Flow matrix, distance matrix
- **Outputs:** Complete assignment (permutation of locations)

## Evaluation

- **Dataset:** 8 instances, 20 facilities
- **Fitness:** Average total cost (minimized)

## Template

```python
template_program = '''
import numpy as np
from typing import List

def select_next_assignment(flow_matrix: np.ndarray, distance_matrix: np.ndarray) -> List[int]:
    """
    Constructive heuristic for the Quadratic Assignment Problem (QAP).

    Args:
        flow_matrix (np.ndarray): shape (n, n)
        distance_matrix (np.ndarray): shape (n, n)

    Returns:
        assignment (List[int]): length-n permutation of {0, ..., n-1}
    """
    n = flow_matrix.shape[0]
    assignment = list(range(n))
    return assignment
'''

task_description = '''
The task is to assign a set of facilities to a set of locations in such a way that
the total cost of interactions between facilities is minimized.
'''
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `timeout_seconds` | Maximum evaluation time | 60 |
| `n_facilities` | Number of facilities | 20 |
| `n_instance` | Number of problem instances | 8 |

## Example Usage

```python
from llm4ad.task.optimization.qap_construct import QAPEvaluation

evaluator = QAPEvaluation()

def select_next_assignment(flow_matrix, distance_matrix):
    n = flow_matrix.shape[0]
    # Simple greedy assignment
    assignment = list(range(n))
    return assignment

result = evaluator.evaluate_program('', select_next_assignment)
```
