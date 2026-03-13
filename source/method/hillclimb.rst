HillClimb
=========

Background
----------

HillClimb is a simple yet effective algorithm for automated heuristic design using Large Language Models. Proposed in the paper "Understanding the importance of evolutionary search in automated heuristic design with large language models" (PPSN 2024), HillClimb represents the simplest form of LLM-based algorithm optimization - iteratively improving upon the best solution found so far.

Unlike evolutionary methods (EoH, FunSearch) that maintain a population of diverse solutions, HillClimb follows a greedy hill-climbing strategy: starting from a template algorithm, it repeatedly attempts to improve the current best solution. This approach is simpler to implement and understand, making it a good baseline for LLM-based algorithm design.

The key insight from the PPSN paper is that even this simple approach can be effective, though evolutionary methods generally find better solutions by maintaining diversity and exploring multiple solution pathways simultaneously.

Algorithm Overview
------------------

HillClimb follows a straightforward iterative improvement strategy:

1. **Initialization**: Evaluate the template program to establish a baseline score.

2. **Iteration**:
   - Generate a prompt containing the current best algorithm.
   - Ask the LLM to create an improved version.
   - Evaluate the new candidate.
   - If the new candidate is better, update the best solution.
   - Otherwise, keep the current best.

3. **Termination**: Stop after a fixed number of samples or when no improvement is found.

The algorithm maintains only the single best solution found so far, which is why it's called "hill climbing" - it greedily climbs toward the local optimum.

Pseudocode
----------

.. code-block:: text

    # Initialize
    best_func = template_function
    best_score = evaluate(template_function)

    sample_count = 0
    while sample_count < max_sample_nums:
        # Generate prompt from best function
        prompt = generate_prompt(best_func, template)

        # Request improved version from LLM
        new_func = LLM.generate(prompt)

        # Evaluate new candidate
        new_score = evaluate(new_func)

        # Greedy selection: only accept improvements
        if new_score > best_score:
            best_func = new_func
            best_score = new_score

        sample_count += 1

    return best_func, best_score

Usage
-----

To use the `HillClimb` class, you need to initialize it with the required parameters and then call the `run` method to start the optimization process.

Constructor
-----------

.. class:: HillClimb

    .. rubric:: Parameters

    - **llm** (LLM): An instance of `llm4ad.base.LLM` for querying the LLM.
    - **evaluation** (Evaluation): An instance of `llm4ad.base.Evaluation` to calculate the score of the generated function.
    - **profiler** (HillClimbProfiler, optional): An instance of `llm4ad.method.hillclimb.HillClimbProfiler`. Pass `None` if profiling is not needed.
    - **max_sample_nums** (int, optional): Maximum number of functions to evaluate. Defaults to 20.
    - **num_samplers** (int, optional): Number of sampler threads. Defaults to 4.
    - **num_evaluators** (int, optional): Number of evaluator threads. Defaults to 4.
    - **resume_mode** (bool, optional): If set to `True`, skips the initial evaluation of the template program. Defaults to `False`.
    - **debug_mode** (bool, optional): If set to `True`, detailed information will be printed. Defaults to `False`.
    - **multi_thread_or_process_eval** (str): Use 'thread' or 'process' for evaluation. Defaults to 'thread'.
    - **kwargs**: Additional arguments passed to `llm4ad.base.SecureEvaluator`.

.. important::
    **template_program**: The template program must be a valid algorithm that obtains a valid score during evaluation.

Methods
-------

.. method:: run()

    Start the hill climbing optimization process. If `resume_mode` is `False`, it initializes the algorithm and then starts sampling using multiple threads.

Private Methods
---------------

.. method:: _init()

    Initializes the hill climbing process by evaluating the template program and registering it as the initial best solution.

.. method:: _get_prompt() -> str

    Generates the prompt for the next sampling iteration. The prompt contains the current best function and asks the LLM to create an improved version.

.. method:: _sample_evaluate_register()

    Continuously samples new functions, evaluates them, and updates the best function found until the maximum sample count is reached. Uses greedy selection - only accepts improvements.

Complete Example
----------------

.. code-block:: python

    from llm4ad.base import LLM, Evaluation
    from llm4ad.method import HillClimb
    from llm4ad.method.hillclimb import HillClimbProfiler

    # Define your task
    task_description = '''
    Design a heuristic algorithm for the Traveling Salesman Problem.
    Given a set of cities, find the shortest route that visits each city
    exactly once and returns to the starting city.
    '''

    # Define the template function
    template_program = '''
    def tsp_greedy(distances: list) -> list:
        """Solve TSP using greedy nearest neighbor heuristic.
        Args:
            distances: 2D distance matrix.
        Returns:
            List of city indices representing the route.
        """
        n = len(distances)
        visited = [False] * n
        route = [0]
        visited[0] = True

        for _ in range(n - 1):
            current = route[-1]
            nearest = min(
                [(j, distances[current][j]) for j in range(n) if not visited[j]],
                key=lambda x: x[1]
            )[0]
            route.append(nearest)
            visited[nearest] = True

        route.append(0)  # Return to start
        return route
    '''

    # Create evaluation instance
    evaluation = Evaluation(
        task_description=task_description,
        template_program=template_program,
        num_test_cases=100
    )

    # Create LLM instance
    llm = LLM(model_name='gpt-4')

    # Create profiler (optional)
    profiler = HillClimbProfiler(save_dir='./hillclimb_results')

    # Initialize HillClimb
    hillclimb = HillClimb(
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        max_sample_nums=50,
        num_samplers=2,
        num_evaluators=2
    )

    # Run optimization
    hillclimb.run()

    # Get results
    print(f"Total samples: {profiler.total_samples}")
    print(f"Best score: {profiler.best_score}")
    print(f"Best program: {profiler.best_program}")

Comparison with Other Methods
----------------------------

HillClimb is simpler than evolutionary methods like EoH and FunSearch:

- **EoH**: Maintains a population, uses crossover and mutation operators, explores multiple solution pathways.
- **FunSearch**: Uses island-based evolution, maintains diverse solutions across clusters, periodic island reset.
- **HillClimb**: Greedy single-solution approach, simple prompt-based improvement.

For research and development, HillClimb serves as a good baseline. If evolutionary methods significantly outperform it, this indicates that diversity and exploration are important for the given problem.

References
----------

- Rui Zhang, Fei Liu, Xi Lin, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. "Understanding the importance of evolutionary search in automated heuristic design with large language models." In International Conference on Parallel Problem Solving from Nature (PPSN), pp. 185-202. 2024.

- Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. "LLM4AD: A Platform for Algorithm Design with Large Language Model." arXiv preprint arXiv:2412.17287 (2024).
