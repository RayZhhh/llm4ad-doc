# WandBProfiler

## Overview

`WandBProfiler` is a profiler implementation that integrates with Weights & Biases (wandb) for experiment tracking and visualization. It extends the `ProfilerBase` class and provides functionality to log experimental results, track function evaluations, and visualize performance metrics in the wandb dashboard.

The profiler automatically tracks:
- Best score of evaluated functions
- Number of valid/invalid functions
- Total sample time and evaluation time

## Constructor

```python
WandBProfiler(
    wandb_project_name: str,
    log_dir: Optional[str] = None,
    *,
    initial_num_samples: int = 0,
    log_style: Literal['simple', 'complex'] = 'complex',
    create_random_path: bool = True,
    fork_proc: Literal['auto'] | bool = 'auto',
    **wandb_init_kwargs
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wandb_project_name` | `str` | Required | The project name in Weights & Biases to sync results |
| `log_dir` | `Optional[str]` | `None` | The directory for storing run logs |
| `initial_num_samples` | `int` | `0` | The starting number for sample ordering |
| `log_style` | `Literal['simple', 'complex']` | `'complex'` | Verbosity level of logging output |
| `create_random_path` | `bool` | `True` | Whether to create a random log path based on evaluation name, method name, and timestamp |
| `fork_proc` | `Literal['auto'] \| bool` | `'auto'` | Whether to fork the wandb process. Use `'auto'` for automatic platform detection, `True` to force fork, `False` to disable |
| `**wandb_init_kwargs` | `Any` | - | Additional keyword arguments passed to `wandb.init()` |

## Methods

### `get_logger()`

Returns the underlying wandb logger instance.

```python
def get_logger() -> wandb.sdk.wandb_run.Run
```

**Returns:**
- `wandb.sdk.wandb_run.Run`: The wandb run object

### `register_function(function: Function, program: str = '', *, resume_mode: bool = False)`

Records an evaluated function. This is a synchronized function that tracks the function evaluation results and logs them to wandb.

```python
def register_function(
    function: Function,
    program: str = '',
    *,
    resume_mode: bool = False
) -> None
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `function` | `Function` | Required | The function object containing score and metadata |
| `program` | `str` | `''` | The program code corresponding to the function |
| `resume_mode` | `bool` | `False` | If `True`, skips writing JSON records (for resuming experiments) |

### `finish()`

Closes the wandb run and finishes logging.

```python
def finish() -> None
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `_wandb_project_name` | `str` | The wandb project name |
| `_logger_wandb` | `wandb.sdk.wandb_run.Run` | The wandb run instance |
| `_num_samples` | `int` | Total number of samples evaluated |
| `_cur_best_program_score` | `float` | Current best score |
| `_evaluate_success_program_num` | `int` | Number of valid functions |
| `_evaluate_failed_program_num` | `int` | Number of invalid functions |
| `_tot_sample_time` | `float` | Total time spent on sampling |
| `_tot_evaluate_time` | `float` | Total time spent on evaluation |

## Integration with Experiments

### Basic Usage

```python
from llm4ad.tools.profiler import WandBProfiler

# Create profiler instance
profiler = WandBProfiler(
    wandb_project_name='my-experiment',
    log_dir='./logs',
    initial_num_samples=0,
    log_style='complex',
    create_random_path=True
)

# Record parameters (required for logging)
profiler.record_parameters(llm=llm, prob=problem, method=method)

# After each function evaluation
profiler.register_function(function=function_obj, program=program_code)

# When experiment is complete
profiler.finish()
```

### Complete Working Example

```python
from llm4ad.tools.profiler import WandBProfiler
from llm4ad.base import Function

# Initialize the profiler with wandb project
profiler = WandBProfiler(
    wandb_project_name='algorithm-design-experiment',
    log_dir='./experiment_logs',
    initial_num_samples=0,
    log_style='complex',
    create_random_path=True,
    fork_proc='auto',
    name='experiment-run-001',  # Additional wandb init kwargs
    tags=['experiment', 'baseline']
)

# Record LLM, problem, and method parameters
# These are typically obtained from your experiment setup
profiler.record_parameters(llm=llm_model, prob=problem_def, method=search_method)

# Simulate function evaluation
class MockFunction(Function):
    def __init__(self):
        self.score = 0.85
        self.sample_time = 0.5
        self.evaluate_time = 0.3

    def __str__(self):
        return "def solution(x):\n    return x ** 2"

# Register the evaluated function
mock_func = MockFunction()
program_code = "def solution(x):\n    return x ** 2"
profiler.register_function(function=mock_func, program=program_code)

# Finish the profiler (closes wandb run)
profiler.finish()
```

### Wandb Dashboard Metrics

The profiler logs the following metrics to wandb:

1. **Best Score of Function**: The best score achieved so far
2. **Valid Function Num**: Count of successfully evaluated functions
3. **Invalid Function Num**: Count of failed/invalid function evaluations
4. **Total Sample Time**: Cumulative time spent on generating samples
5. **Total Evaluate Time**: Cumulative time spent on evaluating functions

### Platform-Specific Behavior

- **macOS/Linux**: When `fork_proc='auto'`, uses the `'fork'` start method for wandb
- **Windows**: When `fork_proc='auto'`, uses the default initialization without forking

## Notes

- Ensure `wandb` is installed: `pip install wandb`
- Run `wandb login` before using the profiler
- The profiler creates JSON files in `log_dir/samples/` containing all evaluated function details
- A run log file `run_log.txt` is created in the log directory with experiment parameters
