RandSample
================

Background
-----------

Random Sampling (RandSample) is the simplest baseline method for automatic algorithm design using LLMs. Unlike evolutionary methods that maintain a population and iteratively improve solutions, RandSample simply generates multiple candidate heuristics independently using the LLM and evaluates them to find the best one.

This method serves as a fundamental baseline in LLM4AD for comparing against more sophisticated evolutionary approaches. It demonstrates the basic capability of LLMs to generate algorithm implementations from scratch without relying on evolutionary selection pressure.

Reference paper:

- Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).

Algorithm Overview
-----------------

The RandSample method in LLM4AD is straightforward:

1. **Template Evaluation**: First, evaluate the template program to establish a baseline score. This ensures the template is valid and provides a reference point.

2. **Prompt Generation**: Create a prompt that includes the task description and the template function signature with an empty body, asking the LLM to implement the heuristic.

3. **Sampling**: Use the LLM to generate multiple candidate implementations. Each sample is independent - the LLM does not have knowledge of previously generated solutions.

4. **Evaluation**: Evaluate each generated candidate using the defined objective functions.

5. **Selection**: Keep track of all evaluated solutions and return the best one(s) found.

The key characteristics of RandSample:

- **No Population**: Each sample is generated independently without knowledge of previous attempts
- **No Evolution**: No selection pressure or crossover/mutation operators
- **Parallel Sampling**: Multiple independent sampler threads can generate candidates simultaneously
- **Simple but Effective**: Works well when the LLM has strong prior knowledge of the problem domain

Pseudocode
----------

.. code-block:: text

    Input: LLM, Evaluation, Max Samples M, Num Samplers S, Num Evaluators E
    Output: Best heuristic found

    // Initialize
    Initialize template_program
    Initialize evaluation executor with E workers
    Initialize sampler threads (S threads)

    // Evaluate template program
    template_score = evaluate(template_program)
    if template_score is None:
        raise RuntimeError("Template program must be valid")

    register_to_profiler(template_program, template_score)

    // Start parallel sampling
    start_sampler_threads(S)

    // Wait for all threads to complete
    join_all_threads()

    // Get best solution
    best_solution = get_best_from_profiler()

    return best_solution

// Thread worker function
def sampler_worker():
    while total_samples < M:
        // Generate prompt
        prompt = generate_prompt(task_description, template_signature)

        // Sample from LLM
        heuristic = LLM.generate(prompt)

        // Evaluate
        score = evaluate(heuristic)

        // Register
        register_to_profiler(heuristic, score)

Usage
-----

To utilize the `RandSample` class, initialize it with the necessary parameters and call the `run` method to start the sampling and evaluation process.

Constructor
-----------

.. class:: RandSample

    .. rubric:: Parameters

    - **llm** (LLM): An instance of `llm4ad.base.LLM` for querying the LLM.
    - **evaluation** (Evaluation): An instance of `llm4ad.base.Evaluation` to calculate the scores of generated functions.
    - **profiler** (RandSampleProfiler, optional): An instance of `llm4ad.method.randsample.RandSampleProfiler`. Pass `None` if profiling is not needed. Defaults to None.
    - **num_samplers** (int, optional): Number of sampling threads. Defaults to 4.
    - **num_evaluators** (int, optional): Number of evaluation threads. Defaults to 4.
    - **max_sample_nums** (int | None, optional): Maximum number of samples to evaluate. Defaults to 20.
    - **resume_mode** (bool, optional): If set to `True`, skips the initial evaluation of the template program. Defaults to `False`.
    - **debug_mode** (bool, optional): If set to `True`, detailed information will be printed. Defaults to `False`.
    - **multi_thread_or_process_eval** (str, optional): Use 'thread' or 'process' for evaluation. Defaults to 'thread'.
    - **kwargs**: Additional arguments passed to `llm4ad.base.SecureEvaluator`.

.. important::
    **template_program**: The template program must be a valid algorithm that obtains a valid score during evaluation. If the template program fails to execute or produces errors, the initialization will raise a RuntimeError.

Methods
-------

.. method:: run()

    Starts the sampling and evaluation process. If `resume_mode` is `False`, it evaluates the template program first to establish a baseline.

    The run method performs the following steps:

    1. If resume_mode is False:
       - Evaluate the template program to ensure it's valid
       - Register the template program to the profiler
    2. Start multiple sampler threads that independently generate and evaluate candidates
    3. Wait for all threads to complete (or until max_sample_nums is reached)
    4. Return control to the caller (best solutions can be retrieved from the profiler)

Private Methods
--------------

.. method:: _get_prompt() -> str

    Generates a prompt based on the template program and the function to be evolved.

    The prompt includes:
    - Task description
    - Template function signature
    - Empty function body (to be filled by LLM)

    Returns:
        - str: The generated prompt

.. method:: _sample_evaluate_register()

    Thread worker function that repeatedly:
    1. Generates a candidate heuristic using the LLM
    2. Evaluates the candidate
    3. Registers the result to the profiler
    4. Repeats until max_sample_nums is reached

Attributes
----------

- **_template_program_str** (str): The string representation of the template program.
- **_max_sample_nums** (int | None): Maximum number of samples to evaluate.
- **_num_samplers** (int): Number of sampling threads.
- **_num_evaluators** (int): Number of evaluation threads.
- **_debug_mode** (bool): Debug mode flag.
- **_resume_mode** (bool): Resume mode flag.
- **_function_to_evolve** (Function): The function that will be evolved.
- **_function_to_evolve_name** (str): Name of the function to evolve.
- **_template_program** (Program): Template program instance.
- **_sampler** (SampleTrimmer): The sampler instance used for sampling.
- **_evaluator** (SecureEvaluator): The evaluator instance used for evaluation.
- **_profiler** (RandSampleProfiler): The profiler instance, if used.
- **_tot_sample_nums** (int): Total number of samples evaluated.
- **_prompt_content** (str): The generated prompt content for sampling.
- **_evaluation_executor** (concurrent.futures.Executor): Executor for parallel evaluation.
- **_sampler_threads** (list): List of sampler threads.

Example
-------

.. code-block:: python

    from llm4ad.method import RandSample
    from llm4ad.base import LLM, Evaluation
    from llm4ad.tool import Profiler

    # Define task
    task_description = "Design a heuristic for the traveling salesman problem that minimizes total travel distance."

    template_program = '''
    def tsp_heuristic(distance_matrix):
        """
        A heuristic for the traveling salesman problem.

        Args:
            distance_matrix: A 2D list where distance_matrix[i][j] represents
                           the distance from city i to city j

        Returns:
            A list of city indices representing the tour
        """
        # TODO: Implement your heuristic here
        tour = []
        return tour
    '''

    # Initialize LLM (e.g., OpenAI GPT-4)
    llm = LLM(
        model_name='gpt-4',
        api_key='your-api-key'
    )

    # Define evaluation
    evaluation = Evaluation(
        task_description=task_description,
        template_program=template_program,
        num_test_cases=100,
        objective_functions=['total_distance']
    )

    # Initialize profiler
    profiler = Profiler(
        log_dir='./logs/randsample',
        save_frequency=10
    )

    # Create RandSample optimizer
    optimizer = RandSample(
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        max_sample_nums=50,
        num_samplers=4,
        num_evaluators=4
    )

    # Run sampling
    optimizer.run()

    # Get best solution from profiler
    best = profiler.get_best_function()
    print(f"Best score: {best.score}")
    print(f"Best algorithm: {best.algorithm}")
    print(f"Code:\n{str(best)}")

Exceptions
----------

- **RuntimeError**: Raised if the score of the template function is `None` during initialization. This indicates the template program is invalid and cannot be evaluated.

References
-----------

1. Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).

Comparison with Evolutionary Methods
------------------------------------

RandSample differs from evolutionary methods (MOEAD, NSGA2) in several key aspects:

| Aspect | RandSample | MOEAD | NSGA2 |
|--------|-----------|-------|-------|
| Population | No | Yes | Yes |
| Evolution | No | Yes | Yes |
| Selection | None | Weight vector-based | Non-dominated sorting |
| Diversity | None | Via decomposition | Via crowding distance |
| Parent-offspring relation | None | Yes | Yes |
| Memory of past attempts | No | Yes | Yes |

RandSample is useful as a:
1. **Baseline**: Simple baseline for comparison
2. **Quick exploration**: Generate many diverse ideas quickly
3. **Domain with strong LLM prior**: When LLM has strong knowledge of the problem domain

Evolutionary methods are preferred when:
1. Building upon previous solutions is beneficial
2. Maintaining diversity across solutions is important
3. Multi-objective optimization is required
