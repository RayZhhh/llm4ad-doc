.. py:module:: llm4ad.base

================
Base Module
================

The base module provides core components for the LLM4AD platform, including:

* **Code Representation** - Classes for representing Python functions and programs as structured objects
* **Evaluation** - Interfaces for evaluating generated algorithms
* **Sampling** - Utilities for interacting with Large Language Models
* **Code Modification** - Tools for programmatically modifying Python source code

Submodules
==========

.. toctree::
   :maxdepth: 1

   code
   evaluate
   sample
   modify_code

Module Overview
===============

Code Module (code.py)
---------------------

The code module defines the fundamental data structures for representing Python code:

* :py:class:`Function <llm4ad.base.code.Function>` - A parsed Python function with name, arguments, body, return type, and docstring
* :py:class:`Program <llm4ad.base.code.Program>` - A complete Python program containing imports and functions
* :py:class:`TextFunctionProgramConverter <llm4ad.base.code.TextFunctionProgramConverter>` - Utilities for converting between text and structured code objects

Evaluate Module (evaluate.py)
-----------------------------

The evaluate module provides evaluation capabilities:

* :py:class:`Evaluation <llm4ad.base.evaluate.Evaluation>` - Abstract base class for defining evaluation logic
* :py:class:`SecureEvaluator <llm4ad.base.evaluate.SecureEvaluator>` - Wrapper for safe evaluation with timeout and process isolation

Sample Module (sample.py)
-------------------------

The sample module handles LLM interaction:

* :py:class:`LLM <llm4ad.base.sample.LLM>` - Abstract interface for language models
* :py:class:`SampleTrimmer <llm4ad.base.sample.SampleTrimmer>` - Utilities for cleaning LLM-generated code

ModifyCode Module (modify_code.py)
----------------------------------

The modify_code module provides code transformation utilities:

* :py:class:`ModifyCode <llm4ad.base.modify_code.ModifyCode>` - Static methods for modifying Python source code

Quick Start
===========

Working with Functions and Programs
-----------------------------------

.. code-block:: python

    from llm4ad.base.code import Function, Program, TextFunctionProgramConverter

    # Parse code string to Program
    code_str = '''
    import numpy as np

    def target_func(arr):
        return np.sum(arr)
    '''

    program = TextFunctionProgramConverter.text_to_program(code_str)
    func = program.functions[0]
    print(f"Function: {func.name}, Body: {func.body}")

Defining Evaluation
-------------------

.. code-block:: python

    from llm4ad.base.evaluate import Evaluation, SecureEvaluator
    from llm4ad.base.code import Program

    template = '''
    def objective(x):
        return 0
    '''

    class MyEvaluator(Evaluation):
        def evaluate_program(self, program_str, callable_func, **kwargs):
            return callable_func(10)

    evaluator = MyEvaluator(template_program=template, timeout_seconds=10)
    secure_eval = SecureEvaluator(evaluator)

Processing LLM Output
---------------------

.. code-block:: python

    from llm4ad.base.sample import LLM, SampleTrimmer

    class CustomLLM(LLM):
        def draw_sample(self, prompt, *args, **kwargs):
            # Call your LLM API here
            return "def target(x): return x + 1"

    llm = CustomLLM(do_auto_trim=True)
    trimmer = SampleTrimmer(llm)
    result = trimmer.draw_sample("Generate a function")
