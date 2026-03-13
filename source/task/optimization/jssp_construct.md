# Constructive Heuristics for Job Shop Scheduling Problem (JSSP)

## Problem

The **Job Shop Scheduling Problem (JSSP)** is a classic scheduling problem where jobs must be processed on machines in specific orders.

- **Given:** A set of jobs, each consisting of operations that must be processed on specific machines for fixed durations
- **Objective:** Minimize the makespan (total completion time)
- **Constraints:** Each machine can process one operation at a time, each job's operations must be processed in order

## Algorithm Design Task

Constructive heuristics iteratively select the next operation to schedule from feasible operations.

- **Inputs:** Current status (machine_status, job_status), feasible operations
- **Outputs:** Selected operation (job_id, machine_id, processing_time)

## Evaluation

- **Dataset:** 16 instances, 50 jobs, 10 machines
- **Fitness:** Average makespan (minimized)

## Template

```python
template_program = '''
from typing import Sequence, Tuple, TypedDict, TypeAlias

JobId: TypeAlias = int
MachineId: TypeAlias = int
Time: TypeAlias = int
ProcessingTime: TypeAlias = int
Operation: TypeAlias = Tuple[JobId, MachineId, ProcessingTime]

def determine_next_operation(
    current_status: CurrentStatus,
    feasible_operations: Sequence[Operation],
) -> Operation:
    """
    Choose one operation from `feasible_operations` to schedule next.

    Args:
        current_status: Dict with 'machine_status' and 'job_status'
        feasible_operations: List of (job_id, machine_id, processing_time)

    Returns:
        One Operation tuple from feasible_operations
    """
    return min(feasible_operations, key=lambda op: op[2])
'''

task_description = '''
Given jobs and machines, schedule jobs on machines to minimize the total makespan.
Design an algorithm to select the next operation in each step.
'''
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `timeout_seconds` | Maximum evaluation time | 20 |
| `n_instance` | Number of problem instances | 16 |
| `n_jobs` | Number of jobs | 50 |
| `n_machines` | Number of machines | 10 |

## Example Usage

```python
from llm4ad.task.optimization.jssp_construct import JSSPEvaluation

evaluator = JSSPEvaluation()

def determine_next_operation(current_status, feasible_operations):
    # Shortest processing time rule
    return min(feasible_operations, key=lambda op: op[2])

result = evaluator.evaluate_program('', determine_next_operation)
```
