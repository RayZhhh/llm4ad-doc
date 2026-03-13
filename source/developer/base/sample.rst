.. py:module:: llm4ad.base.sample

=========
Sample Module
=========

Overview
========

The sample module provides interfaces for interacting with Large Language Models (LLMs) and processing their outputs. It includes:

* ``LLM`` - An abstract base class defining the interface for language models
* ``SampleTrimmer`` - A utility class for cleaning and processing LLM-generated code samples

These classes enable the LLM4AD platform to:
- Interact with various LLM APIs or locally deployed models
- Automatically clean up generated code by removing extraneous text
- Convert generated content into structured Function or Program objects

LLM Class
=========

.. py:class:: LLM

    An abstract base class that defines the interface for interacting with Large Language Models. Subclasses must implement the ``draw_sample`` method to define how to communicate with specific LLM APIs.

Constructor
-----------

.. py:method:: __init__(self, *, do_auto_trim=True, debug_mode=False)

    Initializes the LLM instance.

    :param do_auto_trim: If True, automatically trim the code from response content (default: True).
    :type do_auto_trim: bool
    :param debug_mode: If True, enables debug output (default: False).
    :type debug_mode: bool

    Example:

    .. code-block:: python

        from llm4ad.base.sample import LLM

        class MyCustomLLM(LLM):
            def __init__(self, api_key, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.api_key = api_key

            def draw_sample(self, prompt, *args, **kwargs):
                # Implement API call here
                pass


Methods
-------

.. py:method:: draw_sample(self, prompt: str | Any, *args, **kwargs) -> str

    Abstract method that returns a predicted continuation of the prompt. Must be implemented by subclasses.

    :param prompt: The input prompt to the LLM.
    :type prompt: str | Any
    :param args: Additional positional arguments.
    :param kwargs: Additional keyword arguments.
    :returns: The LLM's generated text response.
    :rtype: str

    Example Response Format:

    .. code-block:: text

        Here is the function.
        def priority_v2(..., ...) -> Any:
            a = np.array([1, 2, 3])
            if len(a) > 2:
                return a / a.sum()
            else:
                return a / a.mean()
        This function is going to ..., and returns ...[Descriptions by LLM]


.. py:method:: draw_samples(self, prompts: List[str | Any], *args, **kwargs) -> List[str]

    Returns multiple predicted continuations for a list of prompts.

    :param prompts: A list of input prompts.
    :type prompts: List[str | Any]
    :param args: Additional positional arguments.
    :param kwargs: Additional keyword arguments.
    :returns: A list of generated text responses.
    :rtype: List[str]

    Default Implementation:

    .. code-block:: python

        def draw_samples(self, prompts, *args, **kwargs):
            return [self.draw_sample(p, *args, **kwargs) for p in prompts]


.. py:method:: close(self)

    Defines how to close the connection to the API or release GPU resources at the end of the program search.

    The default implementation does nothing. Subclasses should override to clean up resources.


SampleTrimmer Class
===================

.. py:class:: SampleTrimmer

    A wrapper class that processes LLM-generated code samples. It wraps an LLM instance and automatically trims extraneous content from generated outputs.

    The SampleTrimmer can:
    - Detect if the output is from a code completion model or instruct model
    - Remove descriptions and extraneous text before the function body
    - Extract the function body from generated code
    - Convert cleaned code into Function or Program objects

Constructor
-----------

.. py:method:: __init__(self, llm: LLM)

    Initializes the SampleTrimmer with an LLM instance.

    :param llm: The LLM instance to wrap.
    :type llm: LLM

    Example:

    .. code-block:: python

        from llm4ad.base.sample import LLM, SampleTrimmer

        # Assume CustomLLM is implemented
        llm = CustomLLM(api_key="...")
        trimmer = SampleTrimmer(llm)


Methods
-------

.. py:method:: draw_sample(self, prompt: str | Any, *args, **kwargs) -> str

    Gets a sample from the LLM and automatically trims it if do_auto_trim is enabled.

    :param prompt: The input prompt.
    :type prompt: str | Any
    :param args: Additional positional arguments.
    :param kwargs: Additional keyword arguments.
    :returns: The trimmed/generated code.
    :rtype: str

    Example:

    .. code-block:: python

        # If inner LLM has do_auto_trim=True:
        generated = trimmer.draw_sample("Write a function that adds two numbers")
        # Returns only the function body, trimmed of any descriptions


.. py:method:: draw_samples(self, prompts: List[str | Any], *args, **kwargs) -> List[str]

    Gets multiple samples from the LLM and automatically trims them.

    :param prompts: A list of input prompts.
    :type prompts: List[str | Any]
    :param args: Additional positional arguments.
    :param kwargs: Additional keyword arguments.
    :returns: A list of trimmed/generated code.
    :rtype: List[str]


.. py:method:: _check_indent_if_code_completion(cls, generated_code: str) -> bool

    Class method that judges whether the content was generated through a code completion model or instruct model.

    :param generated_code: The generated code to check.
    :type generated_code: str
    :returns: True if the first line starts with indentation (tab or 2/4 spaces), indicating code completion model output.
    :rtype: bool


.. py:method:: trim_preface_of_function(cls, generated_code: str)

    Class method that trims redundant descriptions/symbols/'def' declaration BEFORE the function body.

    :param generated_code: The generated code containing potential extra content.
    :type generated_code: str
    :returns: The code starting from the function body.
    :rtype: str

    Example Transformation:

    .. code-block:: python

        # Input:
        """
        This is the optimized function ...

        def priority_v2(...) -> ...:
            a = random.random()
            return a * a

        This function aims to ...
        """

        # Output:
        """
            a = random.random()
            return a * a

        This function aims to ...
        """


.. py:method:: auto_trim(cls, generated_code: str) -> str

    Class method that automatically trims the preface of generated content based on the output type.

    :param generated_code: The generated code.
    :type generated_code: str
    :returns: The trimmed code.
    :rtype: str

    Logic:
    - If output appears to be from code completion model (indented first line), return as-is
    - Otherwise, trim the preface of the function


.. py:method:: sample_to_function(cls, generated_code: str, template_program: str | Program) -> Function | None

    Class method that converts generated content to a Function instance.

    :param generated_code: The LLM-generated code with potential redundant components.
    :type generated_code: str
    :param template_program: The template program providing the function signature.
    :type template_program: str | Program
    :returns: A Function instance with the generated body, or None if conversion fails.
    :rtype: Function | None

    Note:
        The returned Function is not directly executable as it lacks import statements. Use sample_to_program() or combine with a Program's preface.


.. py:method:: sample_to_program(cls, generated_code: str, template_program: str | Program) -> Program | None

    Class method that converts generated content to a Program instance.

    :param generated_code: The LLM-generated code with potential redundant components.
    :type generated_code: str
    :param template_program: The template program providing imports and function signature.
    :type template_program: str | Program
    :returns: A Program instance with the generated function body, or None if conversion fails.
    :rtype: Program | None

    This method:
    1. Trims the function body from generated code
    2. Creates a copy of the template program
    3. Replaces the template function body with the generated body
    4. Removes any docstrings from the generated body
    5. Preserves the original docstring from the template


.. py:method:: trim_function_body(cls, generated_code: str) -> str | None

    Class method that extracts the body of the generated function, trimming anything after it.

    :param generated_code: The generated function code.
    :type generated_code: str
    :returns: The extracted function body, or empty string if extraction fails.
    :rtype: str | None

    This method uses incremental parsing to handle incomplete or malformed code:


.. py:method:: remove_docstrings(cls, func: Function | str)

    Class method that removes docstrings from a Function.

    :param func: The Function or function string to remove docstrings from.
    :type func: Function | str
    :returns: The Function or string with docstrings removed.
    :rtype: Function | str


Usage Examples
==============

Example: Basic LLM Usage
------------------------

.. code-block:: python

    from llm4ad.base.sample import LLM

    class OpenAIChat(LLM):
        def __init__(self, model="gpt-4", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = model

        def draw_sample(self, prompt, *args, **kwargs):
            # Implementation would call OpenAI API
            # This is a placeholder
            response = f"""
            def solution(arr):
                return sum(arr)
            """
            return response

        def close(self):
            # Cleanup if needed
            pass


Example: Using SampleTrimmer
----------------------------

.. code-block:: python

    from llm4ad.base.sample import LLM, SampleTrimmer
    from llm4ad.base.code import Program, TextFunctionProgramConverter

    # Mock LLM that returns generated code
    class MockLLM(LLM):
        def draw_sample(self, prompt, *args, **kwargs):
            return '''
            Here is the optimized function:

            def target_func(arr):
                # Sort the array
                return sorted(arr)

            This function sorts the input array in ascending order.
            '''

    # Create LLM and wrap with trimmer
    llm = MockLLM(do_auto_trim=True)
    trimmer = SampleTrimmer(llm)

    # Generate
    prompt = "Write a function that sorts an array"
    generated = trimmer.draw_sample(prompt)
    print(generated)


Example: Converting Generated Code to Program
---------------------------------------------

.. code-block:: python

    from llm4ad.base.sample import SampleTrimmer
    from llm4ad.base.code import TextFunctionProgramConverter

    # Template that provides imports and function signature
    template = '''
    import numpy as np

    def objective(x: np.ndarray) -> float:
        """Compute the objective value."""
        return 0.0
    '''

    # LLM-generated content (may contain extra text)
    generated_content = '''
    Here is the optimized version:

    def objective(x: np.ndarray) -> float:
        return np.sum(x ** 2)

    This minimizes the sum of squares.
    '''

    # Convert to Program
    program = SampleTrimmer.sample_to_program(generated_content, template)
    if program:
        print("Converted program:")
        print(program)
        print("\nFunction body:")
        print(program.functions[0].body)


Example: Converting to Function
-------------------------------

.. code-block:: python

    from llm4ad.base.sample import SampleTrimmer
    from llm4ad.base.code import TextFunctionProgramConverter

    template = '''
    def process(data):
        """Process the data."""
        pass
    '''

    generated = '''
    def process(data):
        return [x * 2 for x in data]
    '''

    # Convert to Function
    func = SampleTrimmer.sample_to_function(generated, template)
    if func:
        print(f"Function name: {func.name}")
        print(f"Function args: {func.args}")
        print(f"Function body: {func.body}")


Complete Example: Full Pipeline
-------------------------------

.. code-block:: python

    from llm4ad.base.sample import LLM, SampleTrimmer
    from llm4ad.base.code import (
        Program,
        Function,
        TextFunctionProgramConverter
    )
    from llm4ad.base.evaluate import Evaluation, SecureEvaluator


    # 1. Define LLM
    class SimpleLLM(LLM):
        def draw_sample(self, prompt, *args, **kwargs):
            # Return a simple candidate
            return '''
            def target(arr):
                return sum(arr)
            '''

    # 2. Create components
    llm = SimpleLLM(do_auto_trim=True)
    trimmer = SampleTrimmer(llm)

    # 3. Template for the algorithm
    template_program = '''
    import numpy as np

    def target(arr):
        """Compute sum of array elements."""
        return 0
    '''

    # 4. Generate and convert
    prompt = "Write a function that computes sum"
    generated = trimmer.draw_sample(prompt)
    program = SampleTrimmer.sample_to_program(generated, template_program)

    print("Generated Program:")
    print(program)


Custom LLM Implementation Example
----------------------------------

.. code-block:: python

    from llm4ad.base.sample import LLM

    class LocalLLM(LLM):
        """Example of using a locally deployed LLM."""

        def __init__(self, model_path, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_path = model_path
            # Initialize local model here

        def draw_sample(self, prompt, *args, **kwargs):
            # Use local inference
            # response = self.model.generate(prompt)
            pass

        def draw_samples(self, prompts, *args, **kwargs):
            # Batch inference for efficiency
            # return self.model.batch_generate(prompts)
            pass

        def close(self):
            # Release model resources
            # del self.model
            pass
