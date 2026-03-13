# TensorboardProfiler

## Overview

`TensorboardProfiler` is a profiler implementation that integrates with TensorBoard for experiment tracking and visualization. It extends the `ProfilerBase` class and provides functionality to log experimental results, track function evaluations, and visualize performance metrics in TensorBoard dashboards.

The profiler automatically tracks:
- Best score of evaluated functions
- Number of legal/illegal functions
- Total sample time and evaluation time

## Constructor

```python
TensorboardProfiler(
    log_dir: Optional[str] = None,
    *,
    initial_num_samples: int = 0,
    log_style: Literal['simple', 'complex'] = 'complex',
    create_random_path: bool = True,
    **kwargs
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_dir` | `Optional[str]` | `None` | The directory for storing TensorBoard logs |
| `initial_num_samples` | `int` | `0` | The starting number for sample ordering |
| `log_style` | `Literal['simple', 'complex']` | `'complex'` | Verbosity level of logging output |
| `create_random_path` | `bool` | `True` | Whether to create a random log path based on evaluation name, method name, and timestamp |
| `**kwargs` | `Any` | - | Additional keyword arguments passed to the parent class |

## Methods

### `get_logger()`

Returns the underlying TensorBoard SummaryWriter instance.

```python
def get_logger() -> torch.utils.tensorboard.SummaryWriter
```

**Returns:**
- `torch.utils.tensorboard.SummaryWriter`: The TensorBoard summary writer object

### `register_function(function: Function, program: str = '', *, resume_mode: bool = False)`

Records an evaluated function. This is a synchronized function that tracks the function evaluation results and logs them to TensorBoard.

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

Closes the TensorBoard summary writer.

```python
def finish() -> None
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `_writer` | `torch.utils.tensorboard.SummaryWriter` | The TensorBoard summary writer |
| `_num_samples` | `int` | Total number of samples evaluated |
| `_cur_best_program_score` | `float` | Current best score |
| `_evaluate_success_program_num` | `int` | Number of valid (legal) functions |
| `_evaluate_failed_program_num` | `int` | Number of invalid (illegal) functions |
| `_tot_sample_time` | `float` | Total time spent on sampling |
| `_tot_evaluate_time` | `float` | Total time spent on evaluation |

## Integration with Experiments

### Basic Usage

```python
from llm4ad.tools.profiler import TensorboardProfiler

# Create profiler instance
profiler = TensorboardProfiler(
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
from llm4ad.tools.profiler import TensorboardProfiler
from llm4ad.base import Function

# Initialize the profiler with TensorBoard logging
profiler = TensorboardProfiler(
    log_dir='./experiment_logs',
    initial_num_samples=0,
    log_style='complex',
    create_random_path=True
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

# Get the TensorBoard writer for custom logging
writer = profiler.get_logger()
# Use writer.add_scalar(), writer.add_histogram(), etc. for custom metrics

# Finish the profiler (closes TensorBoard writer)
profiler.finish()
```

### TensorBoard Dashboard Metrics

The profiler logs the following scalar metrics to TensorBoard:

1. **Best Score of Function**: The best score achieved so far (logged per sample)
2. **Legal/Illegal Function**: Scalars containing:
   - `legal function num`: Count of successfully evaluated functions
   - `illegal function num`: Count of failed/invalid function evaluations
3. **Total Sample/Evaluate Time**: Scalars containing:
   - `sample time`: Cumulative time spent on generating samples
   - `evaluate time`: Cumulative time spent on evaluating functions

### Viewing Results

After running an experiment with TensorboardProfiler, view the results using:

```bash
tensorboard --logdir=./experiment_logs
```

This will open a web interface at `http://localhost:6006` where you can view:
- **SCALARS**: Best score over time, function validity counts, timing metrics
- **HISTOGRAMS**: Distribution of scores (if logged)

## Notes

- Ensure `tensorboard` and `torch` are installed: `pip install tensorboard torch`
- The profiler creates JSON files in `log_dir/samples/` containing all evaluated function details
- A run log file `run_log.txt` is created in the log directory with experiment parameters
- If `log_dir` is `None`, TensorBoard logging is disabled but JSON recording still works
- The profiler automatically disables TensorFlow's oneDNN optimizations for better performance

## Comparison with WandBProfiler

| Feature | TensorboardProfiler | WandBProfiler |
|---------|-------------------|---------------|
| Visualization | Local TensorBoard | Cloud wandb |
| Setup Required | Install tensorboard | wandb account + login |
| Customizable | More flexible | Less flexible |
| Collaboration | Share log files | Share project link |
| Offline Mode | Full support | Limited |
