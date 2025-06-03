FunSearch
===============

The `FunSearch` class implements an evolutionary function search algorithm that optimizes a given program through iterative sampling and evaluation using Large Language Models (LLMs). This class manages the complete optimization pipeline including prompt generation, parallel sampling, evaluation, and database registration.

Usage
-----

To use `FunSearch`, initialize it with the required components (LLM interface, evaluator, etc.) and call the `run()` method to start the optimization process. The algorithm will automatically handle parallel sampling and evaluation.

Constructor
-----------

.. class:: FunSearch

    .. rubric:: Parameters

    - **llm** (LLM): An instance of `llm4ad.base.LLM` for querying the LLM.
    - **evaluation** (Evaluation): An instance of `llm4ad.base.Evaluation` defining how to score generated functions.
    - **profiler** (ProfilerBase, optional): Profiling instance. Pass `None` to disable profiling.
    - **num_samplers** (int): Number of parallel samplers. Defaults to 4.
    - **num_evaluators** (int): Number of parallel evaluators. Defaults to 4.
    - **samples_per_prompt** (int): Samples generated per prompt. Defaults to 4.
    - **max_sample_nums** (int, optional): Maximum functions to evaluate. Defaults to 20.
    - **resume_mode** (bool): Skip initial template evaluation if True. Defaults to False.
    - **debug_mode** (bool): Enable detailed debug output if True. Defaults to False.
    - **multi_thread_or_process_eval** (str): Use 'thread' or 'process' for parallel evaluation. Defaults to 'thread'.
    - **kwargs**: Additional arguments for `SecureEvaluator`.

.. important::
    The template program (provided via the `evaluation` parameter) must be a fully executable function that returns a valid score during initial evaluation.


Methods
-------

.. method:: run()

    Starts the function search optimization process. If `resume_mode` is `False`, it initializes the algorithm by evaluating the template program and then starts sampling using multiple threads.

Private Methods
---------------

.. method:: _sample_evaluate_register()

    Continuously samples new functions, evaluates them, and registers the results until the maximum sample count is reached.

Attributes
----------

- **_template_program_str**: Original template program string
- **_function_to_evolve**: Function object being optimized
- **_database**: ProgramsDatabase instance storing candidates
- **_sampler**: SampleTrimmer for LLM interactions
- **_evaluator**: SecureEvaluator for scoring functions
- **_profiler**: Performance tracking instance
- **_evaluation_executor**: Thread/Process pool for parallel eval
- **_sampler_threads**: List of active sampling threads
- **_tot_sample_nums**: Total samples evaluated so far

Exceptions
----------

- **RuntimeError**: Raised if template program evaluation fails (score is None)