EoH
==========

The `EoH` class implements an evolutionary optimization algorithm for function design based on a given task description and template program.

Usage
-----

To utilize the `EoH` class, initialize it with the necessary parameters and call the `run` method to start the evolutionary process.

Constructor
-----------

.. class:: EoH

    .. rubric:: Parameters

    - **llm** (LLM): An instance of `alevo.base.Sampler` for querying the LLM.
    - **evaluation** (Evaluation): An instance of `alevo.base.Evaluator` to calculate the score of generated functions.
    - **profiler** (EoHProfiler, optional): An instance of `alevo.method.eoh.EoHProfiler`. Pass `None` if profiling is not needed.
    - **max_generations** (int | None, optional): Maximum number of generations to evolve. Defaults to 10.
    - **max_sample_nums** (int | None, optional): Maximum number of samples to evaluate. Defaults to `None`.
    - **resume_mode** (bool, optional): If set to `True`, skips the initial evaluation of the template program. Defaults to `False`.
    - **initial_sample_num** (int | None, optional): Initial count of samples evaluated. Defaults to `None`.
    - **debug_mode** (bool, optional): If set to `True`, detailed information will be printed. Defaults to `False`.
    - **multi_thread_or_process_eval** (str, optional): Use 'thread' or 'process' for evaluation. Defaults to 'thread'.
    - **kwargs**: Additional arguments passed to `alevo.base.SecureEvaluator`.


Methods
-------

.. method:: run()

    Starts the evolutionary optimization process. If `resume_mode` is `False`, it initializes the population and then proceeds to evolve.

Private Methods
---------------

.. method:: _init_population()

    Initializes the population by repeatedly sampling and evaluating functions.

.. method:: _do_sample()

    Executes the evolutionary sampling and evaluation process in multiple threads.

.. method:: _sample_evaluate_register(prompt)

    Samples a function using the provided prompt, evaluates it, and registers it with the population and profiler.

Attributes
----------

- **_task_description_str** (str): The description of the task to be solved.
- **_template_program_str** (str): The string representation of the template program.
- **_function_to_evolve** (Function): The function that will be evolved.
- **_population** (Population): The population managing the current set of functions.
- **_sampler** (EoHSampler): The sampler instance used for sampling.
- **_evaluator** (Evaluator): The evaluator instance used for evaluation.
- **_profiler** (EoHProfiler): The profiler instance, if used.
- **_tot_sample_nums** (int): Total number of samples evaluated.

Exceptions
----------

- **RuntimeError**: Raised if the specified conditions for evolution are not met.
