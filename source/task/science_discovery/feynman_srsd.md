# Equation Discovery for Feynman Equations (SRSD)

## Problem

The **Feynman Equation Discovery** task involves discovering mathematical equations from the Feynman Lectures on Physics.

- **Given:** Observational data (inputs and outputs)
- **Objective:** Find the mathematical function that fits the data
- **Constraints:** Function must use allowed mathematical operators

## Algorithm Design Task

Design a mathematical function skeleton that can be fitted to data.

- **Inputs:** Array of input variables, parameter array
- **Outputs:** Predicted output values

## Evaluation

- **Dataset:** Various Feynman physics equations
- **Fitness:** Mean squared error between predicted and actual values

## Template

```python
template_program = '''
import numpy as np

def equation(xs: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function
    Args:
        xs: A 2-d numpy array.
        params: Array of numeric constants or parameters to be optimized.

    Return:
        A numpy array.
    """
    return params[0] * x[0] + params[1] * x[0] + params[2]
'''

task_description = "Find the mathematical function skeleton to fit a dataset."
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `timeout_seconds` | Maximum evaluation time | 20 |

## Example Usage

```python
from llm4ad.task.science_discovery.feynman_srsd import FeynmanEvaluation

evaluator = FeynmanEvaluation()

def equation(xs, params):
    # Example: Linear function
    return params[0] * xs[:, 0] + params[1]

result = evaluator.evaluate_program('', equation)
```
