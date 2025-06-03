MOEAD
===============

The `MOEAD` class implements a Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D) to optimize programs using evolutionary operators and LLM-based sampling.

Usage
-----

To use the `MOEAD` class, initialize it with the required parameters and call the `run` method to start the evolutionary optimization process.

Constructor
-----------

.. class:: MOEAD

    .. rubric:: Parameters

    - **llm** (LLM): An instance of `llm4ad.base.LLM` for querying the LLM.
    - **evaluation** (Evaluation): An instance of `llm4ad.base.Evaluation` to calculate scores of generated functions.
    - **profiler** (MOEADProfiler, optional): An instance of `llm4ad.method.moead.MOEADProfiler`. Pass `None` if profiling is not needed.
    - **max_generations** (int, optional): Maximum number of generations to evolve. Defaults to 10.
    - **max_sample_nums** (int, optional): Maximum number of functions to evaluate. Defaults to 100.
    - **pop_size** (int, optional): Population size. Defaults to 20.
    - **selection_num** (int, optional): Number of selected individuals for crossover. Defaults to 5.
    - **use_e2_operator** (bool, optional): Whether to use the E2 evolutionary operator. Defaults to True.
    - **use_m1_operator** (bool, optional): Whether to use the M1 evolutionary operator. Defaults to True.
    - **use_m2_operator** (bool, optional): Whether to use the M2 evolutionary operator. Defaults to True.
    - **num_samplers** (int, optional): Number of sampler threads. Defaults to 1.
    - **num_evaluators** (int, optional): Number of evaluator threads. Defaults to 1.
    - **num_objs** (int, optional): Number of objectives. Defaults to 2.
    - **resume_mode** (bool, optional): If True, skips initial evaluation and initialization. Defaults to False.
    - **initial_sample_num** (int, optional): Initial sample count. Defaults to None.
    - **debug_mode** (bool, optional): If True, prints detailed information. Defaults to False.
    - **multi_thread_or_process_eval** (str): Use 'thread' or 'process' for evaluation. Defaults to 'thread'.
    - **kwargs**: Additional arguments passed to `llm4ad.base.SecureEvaluator`.

Methods
-------

.. method:: run()

    Starts the MOEA/D optimization process. If `resume_mode` is False, initializes the population before evolution.

Private Methods
---------------

.. method:: _sample_evaluate_register(prompt)

    Samples a function using the given prompt, evaluates it, and registers it to the population.

.. method:: _continue_sample() -> bool

    Checks if sampling should continue based on generation and sample count limits.

.. method:: _thread_do_evolutionary_operator()

    Thread worker for performing evolutionary operations.

.. method:: _thread_init_population()

    Thread worker for initializing the population.

.. method:: _init_population()

    Initializes the population using multiple threads.

.. method:: _do_sample()

    Performs the evolutionary sampling process using multiple threads.

Attributes
----------

- **_template_program_str** (str): The template program string.
- **_task_description_str** (str): The task description string.
- **_num_objs** (int): Number of objectives.
- **_max_generations** (int | None): Maximum generations.
- **_max_sample_nums** (int | None): Maximum samples.
- **_pop_size** (int): Population size.
- **_selection_num** (int): Selection number for crossover.
- **_use_e2_operator** (bool): E2 operator flag.
- **_use_m1_operator** (bool): M1 operator flag.
- **_use_m2_operator** (bool): M2 operator flag.
- **_num_samplers** (int): Number of samplers.
- **_num_evaluators** (int): Number of evaluators.
- **_resume_mode** (bool): Resume mode flag.
- **_debug_mode** (bool): Debug mode flag.
- **_function_to_evolve** (Function): Function being evolved.
- **_function_to_evolve_name** (str): Name of function being evolved.
- **_template_program** (Program): Template program instance.
- **_population** (Population): Population instance.
- **_sampler** (MOEADSampler):
- **_evaluator** (SecureEvaluator): Evaluator instance.
- **_profiler** (MOEADProfiler): Profiler instance.
- **_tot_sample_nums** (int): Total number of samples evaluated.
- **_evaluation_executor** (concurrent.futures.Executor): Executor for parallel evaluation.

Exceptions
----------

- **RuntimeError**: Raised if the initial population cannot be properly initialized.
- **ValueError**: Raised if invalid parameters are provided (e.g., invalid multi_thread_or_process_eval value).
