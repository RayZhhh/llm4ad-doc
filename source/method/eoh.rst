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


    - **llm** (llm4ad.base.LLM): An instance of 'llm4ad.base.LLM', which provides the way to query LLM.  
    - **evaluation** (llm4ad.base.Evaluator): An instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.  
    - **profiler** (llm4ad.method.eoh.EoHProfiler | None): An instance of 'llm4ad.method.eoh.EoHProfiler'. If not needed, pass 'None'.  
    - **max_generations** (int | None): Terminate after evolving 'max_generations' generations or reaching 'max_sample_nums'. Pass 'None' to disable.  
    - **max_sample_nums** (int | None): Terminate after evaluating 'max_sample_nums' functions (valid or not) or reaching 'max_generations'. Pass 'None' to disable.  
    - **pop_size** (int | None): Population size. If 'None', EoH will auto-adjust this parameter.  
    - **selection_num** (int): Number of selected individuals during crossover.  
    - **use_e2_operator** (bool): Whether to use the e2 operator.  
    - **use_m1_operator** (bool): Whether to use the m1 operator.  
    - **use_m2_operator** (bool): Whether to use the m2 operator.  
    - **resume_mode** (bool): In resume_mode, 'randsample' skips evaluating 'template_program' and initialization.  
    - **debug_mode** (bool): If True, detailed information will be printed.  
    - **multi_thread_or_process_eval** (str): Use 'thread' (ThreadPoolExecutor) or 'process' (ProcessPoolExecutor) for multi-core CPU evaluation. Default is 'thread'. Note: If 'safe_evaluate=False' and this is 'thread', multi-core CPU won't be utilized.  
    - **kwargs** (dict): Additional args passed to 'llm4ad.base.SecureEvaluator', such as 'fork_proc'.  


    .. code-block:: python

        def your_algo(arg1: int, arg2: float) -> float:
            """Description about this function.
            Args:
                arg1: xxx.
                arg2: xxx.
            Returns:
                xxx.
            """
            pass



Methods
-------

.. method:: run()

    Starts the evolutionary optimization process. If `resume_mode` is `False`, it initializes the population and then proceeds to evolve.

Private Methods
---------------

.. method:: _iteratively_use_eoh_operator()

    Performs the core evolutionary loop using E1, E2, M1, and M2 operators (when enabled) to generate new candidate functions.

.. method:: _iteratively_init_population()

    Initializes the population by repeatedly sampling and evaluating functions using the I1 operator until reaching either the population size or `_initial_sample_nums_max`.

.. method:: _sample_evaluate_register(prompt: str)

    Samples a function using the provided prompt, evaluates it, and registers it with the population and profiler. Records timing and performance metrics.

.. method:: _adjust_pop_size()

    Automatically adjusts the population size based on `max_sample_nums` if no explicit population size was provided.

.. method:: _continue_loop() -> bool

    Determines whether the evolutionary process should continue based on termination conditions.

.. method:: _multi_threaded_sampling(fn: callable, *args, **kwargs)

    Executes the given function (either initialization or evolution) using multiple threads for parallel sampling.

Attributes
----------

- **_template_program_str** (str): String representation of the template program to evolve
- **_task_description_str** (str): Description of the optimization task
- **_function_to_evolve** (Function): The base function being evolved
- **_function_to_evolve_name** (str): Name of the function being evolved
- **_template_program** (Program): Parsed template program structure
- **_population** (Population): Manages current population of candidate functions
- **_sampler** (EoHSampler): Handles LLM-based function sampling
- **_evaluator** (SecureEvaluator): Evaluates function performance
- **_profiler** (EoHProfiler): Optional profiler for tracking evolution metrics
- **_tot_sample_nums** (int): Total number of samples evaluated
- **_initial_sample_nums_max** (int): Maximum samples for initialization phase
- **_evaluation_executor** (Executor): Thread/process pool for parallel evaluation

Configuration Parameters
------------------------

- **_max_generations** (Optional[int]): Maximum generations to evolve
- **_max_sample_nums** (Optional[int]): Maximum total samples to evaluate  
- **_pop_size** (int): Population size
- **_selection_num** (int): Number of parents for crossover
- **_use_e2_operator** (bool): Whether to use E2 operator
- **_use_m1_operator** (bool): Whether to use M1 operator  
- **_use_m2_operator** (bool): Whether to use M2 operator
- **_num_samplers** (int): Number of parallel samplers
- **_num_evaluators** (int): Number of parallel evaluators
- **_resume_mode** (bool): Whether to resume from existing population
- **_debug_mode** (bool): Enable debug output
- **_multi_thread_or_process_eval** (str): 'thread' or 'process' for evaluation

Exceptions
----------

- **AssertionError**: Raised if invalid configuration parameters are provided
- **RuntimeError**: Raised if initialization fails to produce sufficient valid functions
- **KeyboardInterrupt**: Catches user interrupt during evolution
- **Exception**: General exceptions during sampling/evaluation (continues unless debug_mode)
