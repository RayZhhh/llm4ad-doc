LLaMEA
==========

LLaMEA (Large Language Model Evolutionary Algorithm) is an LLM-based evolutionary algorithm that evolves algorithm designs using evolutionary strategies. It combines the reasoning capabilities of Large Language Models with traditional evolutionary computation to automatically generate and improve algorithms for solving optimization problems.

Background
----------

LLaMEA was proposed as a framework for using Large Language Models as evolutionary optimizers. The key insight is that LLMs can understand algorithmic patterns and generate meaningful variations of existing solutions. Unlike prompt-based methods that generate solutions from scratch, LLaMEA uses LLMs to perform evolutionary operators (crossover and mutation) on parent solutions.

The algorithm follows the (λ + μ) or (λ, μ) evolutionary strategy framework:
- (λ + μ): Offspring compete with parents, with selection from combined pool (elitist)
- (λ, μ): Offspring only compete among themselves (non-elitist)

LLaMEA integrates with the external LLaMEA library and adapts it for the LLM4AD platform, providing a seamless way to use LLM-based evolutionary optimization.

Algorithm Overview
------------------

LLaMEA operates through the following main stages:

1. **Initialization**: Create an initial population of algorithms using the template program.

2. **Parent Selection**: Select λ parent individuals from the population.

3. **Offspring Generation**: Use the LLM to generate μ offspring through crossover and mutation operators.

4. **Evaluation**: Evaluate all offspring using the provided evaluation function.

5. **Selection**: Select the best individuals to form the next generation based on the elitism setting.

6. **Termination**: Repeat until the budget (number of iterations) is exhausted.

The algorithm leverages the LLM's understanding of code structure to perform meaningful crossover and mutation operations, guided by the task description and examples.

Pseudocode
----------

.. code-block:: text

    Initialize population with λ individuals
    FOR iteration = 1 to budget:
        // Parent selection
        parents = select(population, λ)

        // Offspring generation using LLM
        offspring = []
        FOR i = 1 to μ:
            offspring_i = LLM.evolve(parents, task_description, examples)
            offspring.append(offspring_i)

        // Evaluation
        FOR each o in offspring:
            o.fitness = evaluate(o)

        // Selection
        IF elitism (λ + μ):
            combined = parents + offspring
            population = select_best(combined, λ)
        ELSE (λ, μ):
            population = select_best(offspring, λ)

    Return best solution found

Constructor Parameters
---------------------

.. class:: LLaMEA

    .. rubric:: Parameters

    - **llm** (llm4ad.base.LLM): An instance of 'llm4ad.base.LLM', which provides the way to query LLM.
    - **evaluator** (llm4ad.base.Evaluation): An instance of 'llm4ad.base.Evaluation', which defines the way to calculate the score of a generated function.
    - **iterations** (int): Number of iterations for the evolution process. Default is 50.
    - **n_parents** (int): Number of individuals in parent population (λ). Default is 5.
    - **n_offsprings** (int): Number of individuals in offspring population (μ). Default is 5.
    - **role_prompt** (str): LLM role prompt describing the task domain. Default is empty string.
    - **task_prompt** (str): Task prompt describing the problem to solve. Default is empty string.
    - **example_prompt** (str | None): Example prompt showing the template program. Default is None.
    - **minimization** (bool): Flag to define optimization direction. If True, minimize; otherwise maximize. Default is False.
    - **elitism** (bool): Use (λ + μ) if True, else (λ, μ). Default is True.
    - **kwargs** (dict): Additional arguments passed to the underlying LLaMEA algorithm.

Methods
-------

.. method:: run()

    Starts the evolutionary optimization process. This method calls the underlying LLaMEA algorithm's run method to begin the evolutionary process.

Attributes
----------

- **evaluator** (Evaluation): The evaluation instance used for assessing generated functions
- **sampler** (LLaMEASampler): The sampler for generating solutions using LLM

Example Usage
-------------

.. code-block:: python

    from llm4ad.base import Evaluation, LLM
    from llm4ad.method import LLaMEA
    from llm4ad.tools.llm import OpenAI

    # Define your task
    task_description = "Design a heuristic for the Bin Packing Problem..."

    # Define template program
    template_program = '''
    def bin_packing_heuristic(items, bin_capacity):
        """A heuristic for bin packing.
        Args:
            items: List of item sizes.
            bin_capacity: Maximum capacity of each bin.
        Returns:
            List of bins with assigned items.
        """
        bins = []
        current_bin = []
        current_sum = 0

        for item in sorted(items, reverse=True):
            if current_sum + item <= bin_capacity:
                current_bin.append(item)
                current_sum += item
            else:
                bins.append(current_bin)
                current_bin = [item]
                current_sum = item

        if current_bin:
            bins.append(current_bin)

        return bins
    '''

    # Create evaluation
    evaluation = Evaluation(
        task_description=task_description,
        template_program=template_program
    )

    # Create LLM
    llm = OpenAI(model='gpt-4')

    # Create and run LLaMEA
    llamaea = LLaMEA(
        llm=llm,
        evaluator=evaluation,
        iterations=50,
        n_parents=5,
        n_offsprings=5,
        role_prompt="You are an excellent scientific programmer tasked to solve the Bin Packing Problem.",
        task_prompt=task_description,
        minimization=False,
        elitism=True
    )
    llamaea.run()

References
----------

- LLaMEA: Large Language Model Evolutionary Algorithm (https://github.com/llamea/llamea)
- Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. "LLM4AD: A Platform for Algorithm Design with Large Language Model." arXiv preprint arXiv:2412.17287 (2024).
