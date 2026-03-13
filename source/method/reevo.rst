ReEvo
==========

ReEvo (Reflective Evolution) is an LLM-based evolutionary algorithm that designs optimization heuristics through a novel two-stage reflection mechanism. The algorithm combines short-term and long-term reflections to guide the evolutionary process, enabling effective exploration and exploitation of the heuristic search space.

Background
----------

ReEvo was developed to address the challenge of automatically designing optimization heuristics using Large Language Models. Unlike traditional evolutionary algorithms that rely on random variation operators, ReEvo leverages the reasoning capabilities of LLMs to reflect on past solutions and generate improved heuristics. The key innovation is the introduction of reflective prompts that guide the LLM to analyze differences between solutions and synthesize insights across multiple generations.

The algorithm maintains a population of candidate heuristics and uses genetic operators (crossover and mutation) guided by reflective prompts. Short-term reflections analyze pairs of solutions to generate immediate improvement hints, while long-term reflections aggregate insights from multiple short-term reflections to provide broader guidance for mutation operations.

Algorithm Overview
------------------

ReEvo operates through the following main stages:

1. **Population Initialization**: Generate initial heuristics using the template program as a starting point.

2. **Short-term Reflection**: For each iteration, select two parent solutions and prompt the LLM to analyze their differences, generating improvement hints.

3. **Crossover**: Use the short-term reflection to guide the combination of two parent solutions, creating a new offspring heuristic.

4. **Long-term Reflection**: Periodically aggregate recent short-term reflections to synthesize broader insights about successful heuristic patterns.

5. **Mutation**: Apply mutation to elite solutions guided by long-term reflection prompts.

6. **Selection**: Maintain population size through survival selection based on fitness.

The algorithm uses simulated annealing-inspired acceptance to balance exploration and exploitation, allowing exploration of the search space even when immediate improvements are not apparent.

Pseudocode
----------

.. code-block:: text

    Initialize population with initial heuristics
    WHILE total_samples < max_sample_nums:
        // Short-term reflection
        Select two parent solutions (indivs)
        short_term_reflection = LLM.analyze(worse_solution, better_solution)

        // Crossover
        offspring = LLM.crossover(parents, short_term_reflection)
        Evaluate offspring
        Register to population

        // Long-term reflection (every pop_size generations)
        IF generation % pop_size == 0:
            long_term_reflection = LLM.synthesize(recent_short_term_reflections)

            // Mutation on elite
            FOR i in range(mutation_rate * pop_size):
                mutated = LLM.mutate(elite, long_term_reflection)
                Evaluate mutated
                Register to population

        Perform population survival

    Return best solution

Constructor Parameters
---------------------

.. class:: ReEvo

    .. rubric:: Parameters

    - **llm** (llm4ad.base.LLM): An instance of 'llm4ad.base.LLM', which provides the way to query LLM.
    - **evaluation** (llm4ad.base.Evaluation): An instance of 'llm4ad.base.Evaluation', which defines the way to calculate the score of a generated function.
    - **profiler** (llm4ad.method.reevo.ReEvoProfiler | None): An instance of 'llm4ad.method.reevo.ReEvoProfiler'. If not needed, pass 'None'.
    - **max_sample_nums** (int | None): Terminate after evaluating 'max_sample_nums' functions (valid or not). Pass 'None' to disable.
    - **pop_size** (int | None): Population size. If 'None', ReEvo will auto-adjust this parameter. Default is 20.
    - **mutation_rate** (float): Rate of mutation applied to elite solutions. Default is 0.5.
    - **num_samplers** (int): Number of parallel samplers. Default is 1.
    - **num_evaluators** (int): Number of parallel evaluators. Default is 1.
    - **resume_mode** (bool): In resume_mode, skips evaluating 'template_program' and initialization. Default is False.
    - **debug_mode** (bool): If True, detailed information will be printed. Default is False.
    - **multi_thread_or_process_eval** (str): Use 'thread' (ThreadPoolExecutor) or 'process' (ProcessPoolExecutor) for multi-core CPU evaluation. Default is 'thread'.
    - **kwargs** (dict): Additional args passed to 'llm4ad.base.SecureEvaluator', such as 'fork_proc'.

Methods
-------

.. method:: run()

    Starts the evolutionary optimization process. If `resume_mode` is `False`, it initializes the population and then proceeds to evolve using the reflective genetic operators.

Private Methods
---------------

.. method:: _iteratively_ga_evolve()

    Performs the core evolutionary loop with short-term reflection, crossover, long-term reflection, and mutation operators.

.. method:: _iteratively_init_population()

    Initializes the population by repeatedly sampling and evaluating functions using the initialization prompt.

.. method:: _sample_evaluate_register(prompt: str)

    Samples a function using the provided prompt, evaluates it, and registers it with the population and profiler. Records timing and performance metrics.

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
- **_sampler** (SampleTrimmer): Handles LLM-based function sampling
- **_evaluator** (SecureEvaluator): Evaluates function performance
- **_profiler** (ReEvoProfiler): Optional profiler for tracking evolution metrics
- **_tot_sample_nums** (int): Total number of samples evaluated
- **_MAX_SHORT_TERM_REFLECTION_PROMPT** (int): Maximum short-term reflections to consider for long-term reflection
- **_evaluation_executor** (Executor): Thread/process pool for parallel evaluation

Configuration Parameters
------------------------

- **_max_sample_nums** (Optional[int]): Maximum total samples to evaluate
- **_pop_size** (int): Population size
- **_mutation_rate** (float): Rate of mutation applied to elite solutions
- **_num_samplers** (int): Number of parallel samplers
- **_num_evaluators** (int): Number of parallel evaluators
- **_resume_mode** (bool): Whether to resume from existing population
- **_debug_mode** (bool): Enable debug output
- **_multi_thread_or_process_eval** (str): 'thread' or 'process' for evaluation

Example Usage
-------------

.. code-block:: python

    from llm4ad.base import Evaluation, LLM
    from llm4ad.method import ReEvo
    from llm4ad.tools.llm import OpenAI

    # Define your task
    task_description = "Design a heuristic for the Traveling Salesman Problem..."

    # Define template program
    template_program = '''
    def tsp_heuristic(dist_matrix, start=0):
        """A heuristic for solving TSP.
        Args:
            dist_matrix: Distance matrix between cities.
            start: Starting city index.
        Returns:
            Route and total distance.
        """
        n = len(dist_matrix)
        visited = [False] * n
        route = [start]
        visited[start] = True
        current = start

        for _ in range(n - 1):
            nearest = min(range(n), key=lambda x: dist_matrix[current][x] if not visited[x] else float('inf'))
            route.append(nearest)
            visited[nearest] = True
            current = nearest

        total_distance = sum(dist_matrix[route[i]][route[i+1]] for i in range(n-1))
        return route, total_distance
    '''

    # Create evaluation
    evaluation = Evaluation(
        task_description=task_description,
        template_program=template_program
    )

    # Create LLM
    llm = OpenAI(model='gpt-4')

    # Create and run ReEvo
    reevo = ReEvo(
        llm=llm,
        evaluation=evaluation,
        max_sample_nums=100,
        pop_size=20,
        mutation_rate=0.5
    )
    reevo.run()

References
----------

- Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. "LLM4AD: A Platform for Algorithm Design with Large Language Model." arXiv preprint arXiv:2412.17287 (2024).
