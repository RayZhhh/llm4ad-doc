FunSearch
=========

Background
----------

FunSearch is an evolutionary program search method that uses Large Language Models (LLMs) to evolve solutions for challenging mathematical and combinatorial optimization problems. Published in the Nature paper "Mathematical discoveries from program search with large language models" (2024), FunSearch achieved notable success by discovering new algorithms for fundamental combinatorial problems, including the cap set problem and online bin packing.

The key innovation of FunSearch is the combination of:
1. **Island-based Evolutionary Model**: Maintains multiple "islands" (sub-populations) that evolve independently, promoting diversity.
2. **Pareto-based Clustering**: Groups programs by performance score, sampling within clusters using softmax probabilities.
3. **Temperature-scheduled Sampling**: Uses a decaying temperature schedule to gradually shift from exploration to exploitation.
4. **Periodic Island Reset**: Periodically resets underperforming islands by copying the best programs from successful islands.

FunSearch represents a significant advance in LLM-based program synthesis, demonstrating that evolutionary search can discover novel algorithms that outperform human-designed baselines.

Algorithm Overview
------------------

FunSearch uses an island-based evolutionary algorithm with the following key components:

1. **Programs Database**: Maintains multiple islands, each containing clusters of programs grouped by performance scores.

2. **Island Operations**:
   - Each island maintains several clusters, where each cluster contains programs with similar scores.
   - Programs are sampled from clusters using softmax probabilities with temperature.
   - The temperature decreases over time to shift from exploration to exploitation.

3. **Prompt Generation**:
   - For each sampling iteration, an island is selected randomly.
   - Programs are sampled from that island's clusters based on their scores.
   - The sampled programs are formatted into a prompt asking for an improved version.

4. **Evaluation and Registration**:
   - Generated programs are evaluated on test cases.
   - Successful programs are registered back to the same island in the appropriate cluster.

5. **Island Resetting**:
   - Periodically, the weaker half of islands are reset.
   - The reset islands are initialized with copies of the best programs from the stronger islands.

Pseudocode
----------

.. code-block:: text

    # Initialize
    db = ProgramsDatabase(config)
    template_func = load_template()

    # Evaluate and register template
    template_score = evaluate(template_func)
    db.register(template_func, score=template_score)

    while not termination:
        # Select island
        island = db.select_island()

        # Generate prompt from island's programs
        prompt = island.generate_prompt()

        # Sample programs from LLM
        new_programs = LLM.sample(prompt, n=samples_per_prompt)

        # Evaluate each program
        for prog in new_programs:
            score = evaluate(prog)

            # Register to database
            db.register(prog, score=score)

            # Periodically reset weak islands
            if should_reset():
                db.reset_weak_islands()

    return db.best_program()

Usage
-----

To use `FunSearch`, initialize it with the required components (LLM interface, evaluator, etc.) and call the `run()` method to start the optimization process.

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

    Continuously samples new functions, evaluates them, and registers the results until the maximum sample count is reached. This method:
    1. Gets a prompt from the programs database
    2. Samples programs using the LLM
    3. Evaluates each program
    4. Registers successful programs to the database

Complete Example
----------------

.. code-block:: python

    from llm4ad.base import LLM, Evaluation
    from llm4ad.method import FunSearch
    from llm4ad.tool import FunSearchProfiler

    # Define your task
    task_description = '''
    Design an algorithm to find the maximum independent set in a graph.
    Given an undirected graph, find the largest set of vertices such that
    no two vertices in the set are adjacent.
    '''

    # Define the template function
    template_program = '''
    def max_independent_set(graph: list) -> list:
        """Find maximum independent set.
        Args:
            graph: Adjacency list representation of the graph.
        Returns:
            List of vertices forming a maximum independent set.
        """
        vertices = list(range(len(graph)))
        independent_set = []
        return independent_set
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
    profiler = FunSearchProfiler(save_dir='./funsearch_results')

    # Initialize FunSearch
    funsearch = FunSearch(
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        num_samplers=4,
        num_evaluators=4,
        samples_per_prompt=4,
        max_sample_nums=100
    )

    # Run optimization
    funsearch.run()

    # Get results
    print(f"Total samples: {profiler.total_samples}")
    print(f"Best score: {profiler.best_score}")
    print(f"Best program: {profiler.best_program}")

References
----------

- Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco JR Ruiz et al. "Mathematical discoveries from program search with large language models." Nature 625, no. 7995 (2024): 468-475.

- Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. "LLM4AD: A Platform for Algorithm Design with Large Language Model." arXiv preprint arXiv:2412.17287 (2024).
