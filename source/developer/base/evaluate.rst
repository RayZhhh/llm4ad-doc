.. py:module:: llm4ad.base.evaluate

==========
Evaluate Module
==========

Overview
========

The evaluate module provides classes for evaluating generated algorithms and code. It includes an abstract base class for defining evaluation logic and a secure evaluator that runs evaluations in isolated processes with timeout protection.

This module is designed to safely execute user-generated code while providing features like:
- Timeout protection to prevent infinite loops
- Process isolation for security
- Numba JIT compilation for performance
- Protected division to avoid division by zero errors
- Reproducible random number generation via seeded random states

Evaluation Class
================

.. py:class:: Evaluation

    An abstract base class that defines the interface for evaluating generated algorithms. Users must subclass this and implement the ``evaluate_program`` method to define their specific evaluation logic.

    The Evaluation class provides configuration options for code modification before execution, including adding Numba decorators, protected division, and random seeding.

Constructor
-----------

.. py:method:: __init__(self, template_program: str | Program, task_description: str = '', use_numba_accelerate: bool = False, use_protected_div: bool = False, protected_div_delta: float = 1e-5, random_seed: int | None = None, timeout_seconds: int | float = None, *, exec_code: bool = True, safe_evaluate: bool = True, daemon_eval_process: bool = False, fork_proc: Literal['auto'] | bool = 'auto')

    Initializes the Evaluation instance.

    :param template_program: The template program string or Program object that defines the function signature to be evolved.
    :type template_program: str | Program
    :param task_description: A description of the task (default: empty string).
    :type task_description: str
    :param use_numba_accelerate: Whether to wrap the function with ``@numba.jit(nopython=True)`` for acceleration (default: False).
    :type use_numba_accelerate: bool
    :param use_protected_div: Whether to replace division operations with protected division (default: False).
    :type use_protected_div: bool
    :param protected_div_delta: The delta value used in protected division (default: 1e-5).
    :type protected_div_delta: float
    :param random_seed: If not None, sets the random seed in the first line of the function body (default: None).
    :type random_seed: int | None
    :param timeout_seconds: Maximum time in seconds for evaluation (default: None, meaning no timeout).
    :type timeout_seconds: int | float | None
    :param exec_code: Whether to use ``exec()`` to compile the code and provide a callable function. If False, the callable_func argument in evaluate_program will always be None (default: True).
    :type exec_code: bool
    :param safe_evaluate: Whether to evaluate in safe mode using a new process. If False, the evaluation will not be terminated after timeout (default: True).
    :type safe_evaluate: bool
    :param daemon_eval_process: Whether to set the evaluation process as a daemon process. If True, you cannot create new processes in evaluate_program (default: False).
    :type daemon_eval_process: bool
    :param fork_proc: Determines process creation method when safe_evaluate=True. 'auto' uses OS-dependent default, True uses 'fork', False uses 'spawn' (default: 'auto').
    :type fork_proc: Literal['auto'] | bool

    Example:

    .. code-block:: python

        from llm4ad.base.code import Program
        from llm4ad.base.evaluate import Evaluation

        template = '''
        import numpy as np

        def target_func(arr):
            return 0
        '''

        class MyEvaluator(Evaluation):
            def evaluate_program(self, program_str, callable_func, **kwargs):
                # Use the callable function for fast evaluation
                test_input = [1, 2, 3, 4, 5]
                result = callable_func(test_input)
                # Return a fitness value (lower is better for minimization)
                return abs(result - 15)  # Target: sum = 15

        evaluator = MyEvaluator(
            template_program=template,
            use_protected_div=True,
            random_seed=42
        )


Methods
-------

.. py:method:: evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

    Abstract method that must be implemented by subclasses to define the evaluation logic.

    This method is called with both the program string and a compiled callable function. The callable function is available when ``exec_code=True`` in the constructor.

    :param program_str: The modified function code as a string. The code may include added imports, numba decorators, protected division, and random seeding depending on configuration.
    :type program_str: str
    :param callable_func: The compiled callable heuristic function. Can be called using ``callable_func(*args, **kwargs)``. This is None if ``exec_code=False``.
    :type callable_func: callable | None
    :param kwargs: Additional keyword arguments passed from the evaluator.
    :returns: The fitness/value result of the evaluation.
    :rtype: Any | None
    :raises NotImplementedError: This method must be implemented by subclasses.

    Code Modification Example:

    When ``use_numba_accelerate=True``, ``use_protected_div=True``, and ``random_seed=2024``, the input program_str will be transformed from:

    .. code-block:: python

        import numpy as np

        def f(a, b):
            a = np.random.random()
            return a / b

    To:

    .. code-block:: python

        import numpy as np
        import numba

        @numba.jit(nopython=True)
        def f():
            np.random.seed(2024)
            a = np.random.random()
            return _protected_div(a, b)

        def _protected_div(a, b, delta=1e-5):
            return a / (b + delta)


SecureEvaluator Class
=====================

.. py:class:: SecureEvaluator

    A wrapper class that provides secure evaluation of generated programs. It runs evaluations in a separate process with timeout protection and error handling.

    The SecureEvaluator handles code modification (adding numba decorators, protected division, random seeds) before execution and provides both synchronous and timing-enabled evaluation methods.

Constructor
-----------

.. py:method:: __init__(self, evaluator: Evaluation, debug_mode=False, **kwargs)

    Initializes the SecureEvaluator.

    :param evaluator: The Evaluation instance to wrap.
    :type evaluator: Evaluation
    :param debug_mode: If True, prints debug information including evaluated program code and errors (default: False).
    :type debug_mode: bool
    :param kwargs: Additional keyword arguments (passed to parent).

    Example:

    .. code-block:: python

        from llm4ad.base.code import Program
        from llm4ad.base.evaluate import Evaluation, SecureEvaluator

        template = '''
        import numpy as np

        def objective(x):
            return 0
        '''

        class MyEvaluator(Evaluation):
            def evaluate_program(self, program_str, callable_func, **kwargs):
                # Test with multiple inputs
                test_cases = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
                total_error = 0
                for test in test_cases:
                    result = callable_func(test)
                    total_error += abs(result - sum(test))
                return total_error

        evaluator = MyEvaluator(
            template_program=template,
            use_protected_div=True,
            timeout_seconds=10
        )
        secure_eval = SecureEvaluator(evaluator, debug_mode=False)


Methods
-------

.. py:method:: evaluate_program(self, program: str | Program, **kwargs)

    Evaluates a program in a secure manner with timeout and error handling.

    This method:
    1. Converts the program to a string if necessary
    2. Extracts the function name before modification
    3. Applies code modifications (numba, protected division, random seed)
    4. Executes the program in a safe process (if safe_evaluate=True)
    5. Returns the evaluation result or None on timeout/error

    :param program: The program to evaluate, as a string or Program object.
    :type program: str | Program
    :param kwargs: Additional keyword arguments passed to the evaluator's evaluate_program method.
    :returns: The evaluation result, or None if evaluation fails, times out, or encounters an error.
    :rtype: Any | None

    Example:

    .. code-block:: python

        from llm4ad.base.code import Program, TextFunctionProgramConverter
        from llm4ad.base.evaluate import Evaluation, SecureEvaluator

        template = '''
        import numpy as np

        def sort_array(arr):
            return arr
        '''

        class ArrayEvaluator(Evaluation):
            def evaluate_program(self, program_str, callable_func, **kwargs):
                test_arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
                result = callable_func(test_arr)
                expected = np.sort(test_arr)
                return -np.sum(np.abs(result - expected))  # Negative because we maximize

        evaluator = ArrayEvaluator(
            template_program=template,
            timeout_seconds=5
        )
        secure_eval = SecureEvaluator(evaluator)

        # Evaluate a candidate program
        candidate = '''
        import numpy as np

        def sort_array(arr):
            return np.sort(arr)
        '''

        result = secure_eval.evaluate_program(candidate)
        print(f"Evaluation result: {result}")


.. py:method:: evaluate_program_record_time(self, program: str | Program, **kwargs)

    Evaluates a program and records the time taken for evaluation.

    :param program: The program to evaluate, as a string or Program object.
    :type program: str | Program
    :param kwargs: Additional keyword arguments passed to the evaluator's evaluate_program method.
    :returns: A tuple of (evaluation result, evaluation time in seconds).
    :rtype: tuple[Any | None, float]

    Example:

    .. code-block:: python

        from llm4ad.base.evaluate import Evaluation, SecureEvaluator

        template = '''
        def square(x):
            return 0
        '''

        class SimpleEvaluator(Evaluation):
            def evaluate_program(self, program_str, callable_func, **kwargs):
                return callable_func(5)

        evaluator = SimpleEvaluator(template_program=template)
        secure_eval = SecureEvaluator(evaluator)

        candidate = '''
        def square(x):
            return x * x
        '''

        result, eval_time = secure_eval.evaluate_program_record_time(candidate)
        print(f"Result: {result}, Time: {eval_time:.4f}s")


.. py:method:: _modify_program_code(self, program_str: str) -> str

    Applies code modifications to the program string.

    This internal method applies transformations based on the Evaluation configuration:
    - Adds Numba JIT decorator if enabled
    - Replaces division with protected division if enabled
    - Adds numpy random seed if specified

    :param program_str: The original program string.
    :type program_str: str
    :returns: The modified program string.
    :rtype: str


.. py:method:: _evaluate_in_safe_process(self, program_str: str, function_name, result_queue: multiprocessing.Queue, **kwargs)

    Internal method that executes evaluation in a separate process.

    :param program_str: The program code to execute.
    :type program_str: str
    :param function_name: The name of the function to call.
    :param result_queue: Queue to put the result in.
    :type result_queue: multiprocessing.Queue
    :param kwargs: Additional keyword arguments.


.. py:method:: _evaluate(self, program_str: str, function_name, **kwargs)

    Internal method that executes evaluation without process isolation.

    :param program_str: The program code to execute.
    :type program_str: str
    :param function_name: The name of the function to call.
    :param kwargs: Additional keyword arguments.


Complete Example: Custom Evaluation
===================================

.. code-block:: python

    import numpy as np
    from llm4ad.base.code import Program, TextFunctionProgramConverter
    from llm4ad.base.evaluate import Evaluation, SecureEvaluator


    # Define the template program
    TEMPLATE = '''
    import numpy as np

    def objective(x: np.ndarray) -> float:
        """Minimize the sum of squares."""
        return 0.0
    '''


    class SquaredErrorEvaluator(Evaluation):
        """Evaluator that minimizes squared error against target sum."""

        def __init__(self, *args, target_sum=10.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.target_sum = target_sum

        def evaluate_program(self, program_str, callable_func, **kwargs):
            """Evaluate the function on test cases."""
            # Test with multiple random arrays
            errors = []
            for _ in range(5):
                test_arr = np.random.rand(10)
                result = callable_func(test_arr)
                expected = self.target_sum
                error = (result - expected) ** 2
                errors.append(error)
            return np.mean(errors)  # Return mean squared error


    # Create evaluator with custom settings
    evaluator = SquaredErrorEvaluator(
        template_program=TEMPLATE,
        task_description="Minimize squared error",
        use_numba_accelerate=True,       # Speed up with numba
        use_protected_div=True,            # Handle division by zero
        protected_div_delta=1e-8,         # Small delta for precision
        random_seed=42,                   # Reproducible results
        timeout_seconds=10,               # 10 second timeout
        safe_evaluate=True                # Run in separate process
    )

    # Wrap with SecureEvaluator
    secure_eval = SecureEvaluator(evaluator, debug_mode=False)

    # Candidate solution that computes sum
    candidate_solution = '''
    import numpy as np

    def objective(x: np.ndarray) -> float:
        return np.sum(x)
    '''

    # Evaluate
    result, eval_time = secure_eval.evaluate_program_record_time(candidate_solution)
    print(f"Evaluation result: {result}")
    print(f"Evaluation time: {eval_time:.4f}s")


Complete Example: Handling Timeout
===================================

.. code-block:: python

    from llm4ad.base.code import TextFunctionProgramConverter
    from llm4ad.base.evaluate import Evaluation, SecureEvaluator


    TEMPLATE = '''
    def infinite_loop(x):
        return 0
    '''


    class TimeoutEvaluator(Evaluation):
        def evaluate_program(self, program_str, callable_func, **kwargs):
            # This would run forever without timeout
            return callable_func(1)


    evaluator = TimeoutEvaluator(
        template_program=TEMPLATE,
        timeout_seconds=2,  # 2 second timeout
        safe_evaluate=True
    )

    secure_eval = SecureEvaluator(evaluator, debug_mode=True)

    # This program has an intentional infinite loop
    bad_candidate = '''
    def infinite_loop(x):
        while True:
            x = x + 1
        return x
    '''

    result = secure_eval.evaluate_program(bad_candidate)
    # Result will be None due to timeout
    print(f"Result (None due to timeout): {result}")
