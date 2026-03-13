.. _openai_api:

OpenAI API
==========

The OpenAIAPI class provides an interface to interact with OpenAI-compatible LLM APIs.
It supports both OpenAI's official API and third-party services that expose OpenAI-compatible endpoints.

Overview
--------

The ``OpenAIAPI`` class extends the base ``LLM`` class and provides a unified interface for making
chat completion requests to OpenAI-compatible APIs. It handles API authentication, request formatting,
and response parsing.

Setup Requirements
------------------

Before using the OpenAIAPI, ensure you have the following:

1. **Python Package**: Install the OpenAI Python package:

   .. code:: bash

       pip install openai

2. **API Access**: You need one of the following:
   - OpenAI API key from `platform.openai.com <https://platform.openai.com/>`_
   - Third-party service with OpenAI-compatible API (e.g., Azure OpenAI, local proxies)

API Reference
-------------

Class Definition
~~~~~~~~~~~~~~~~

.. class:: OpenAIAPI

   A concrete implementation of the LLM base class that interfaces with OpenAI-compatible APIs.

Constructor
~~~~~~~~~~~

.. method:: __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60, **kwargs)

   Initialize the OpenAI API client.

   :param base_url: The base URL of the API endpoint. For OpenAI's official API, use
                    ``"https://api.openai.com/v1"``. For third-party services, use their
                    provided endpoint URL.
   :type base_url: str
   :param api_key: Your API key for authentication. For OpenAI, this is your secret API key.
   :type api_key: str
   :param model: The model identifier to use (e.g., ``"gpt-4"``, ``"gpt-3.5-turbo"``).
   :type model: str
   :param timeout: Request timeout in seconds. Defaults to 60.
   :type timeout: int, optional
   :param kwargs: Additional parameters passed to the ``openai.OpenAI`` client constructor.
                  See `OpenAI Python SDK documentation <https://platform.openai.com/docs/libraries/python-library>`_
                  for available options.
   :type kwargs: Any, optional

   .. note::

      For Azure OpenAI, the base_url should follow the format:
      ``"https://<your-resource-name>.openai.azure.com/openai/deployments/<your-deployment-name>"``

Methods
~~~~~~~

.. method:: draw_sample(self, prompt: str | Any, *args, **kwargs) -> str

   Generate a response from the LLM based on the provided prompt.

   :param prompt: The input prompt. Can be either:
                  - A string containing the user message
                  - A list of message dictionaries with 'role' and 'content' keys
   :type prompt: str | Any
   :param args: Additional positional arguments (unused).
   :param kwargs: Additional keyword arguments passed to the API call.
                  See OpenAI's `Chat Completion API <https://platform.openai.com/docs/api-reference/chat/create>`_
                  for available parameters.
   :return: The generated text content from the LLM response.
   :rtype: str

   :raises openai.APIError: If the API returns an error.
   :raises openai.APIConnectionError: If there's a connection issue.

   .. note::

      When passing a string prompt, it is automatically wrapped in a user message format:
      ``[{'role': 'user', 'content': prompt.strip()}]``

Examples
--------

Example 1: Using OpenAI's Official API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.llm_api_openai import OpenAIAPI

    # Initialize with OpenAI API credentials
    sampler = OpenAIAPI(
        base_url="https://api.openai.com/v1",
        api_key="your-api-key-here",
        model="gpt-4",
        timeout=60
    )

    # Simple prompt
    response = sampler.draw_sample("What is the capital of France?")
    print(response)  # Output: Paris

    # Using chat message format
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a function to calculate factorial."}
    ]
    response = sampler.draw_sample(messages)

Example 2: Using Third-Party OpenAI-Compatible API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.llm_api_openai import OpenAIAPI

    # Using a third-party service (e.g., local proxy or alternative provider)
    sampler = OpenAIAPI(
        base_url="https://api.example.com/v1",  # Third-party endpoint
        api_key="your-third-party-api-key",
        model="gpt-3.5-turbo",
        timeout=30
    )

    response = sampler.draw_sample("Hello, how are you?")

Example 3: Using with Additional API Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.llm_api_openai import OpenAIAPI

    sampler = OpenAIAPI(
        base_url="https://api.openai.com/v1",
        api_key="your-api-key-here",
        model="gpt-4",
        timeout=60
    )

    # Passing additional parameters via kwargs
    response = sampler.draw_sample(
        prompt="Explain quantum computing in simple terms.",
        temperature=0.7,
        max_tokens=500,
        top_p=0.9
    )

Example 4: Integration with LLM4AD Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
    import llm4ad

    # Create LLM sampler
    sampler = OpenAIAPI(
        base_url="https://api.openai.com/v1",
        api_key="your-api-key-here",
        model="gpt-4"
    )

    # Use with an LLM4AD method (example with EoH)
    task = llm4ad.tasks.optimization.SymReg(  # Example task
        dimension=5,
        num_samples=100,
        eval_budget=1000
    )

    method = llm4ad.methods.eoh.EoH(
        task=task,
        sampler=sampler,
        num_iterations=50
    )

    result = method.run()
    print(f"Best solution: {result.best_solution}")
    print(f"Best fitness: {result.best_fitness}")

Common Issues and Troubleshooting
----------------------------------

1. **Authentication Error**: Ensure your API key is correct and has sufficient credits.

2. **Rate Limiting**: If you encounter rate limit errors, implement retry logic or reduce
   request frequency.

3. **Timeout Issues**: Increase the timeout parameter if network latency is high.

4. **Model Not Found**: Verify that the model name is correct and available for your API key.

See Also
--------

- :ref:`https_api` - For custom HTTPS API implementations
- :ref:`ollama_api` - For local Ollama deployments
- :ref:`vllm_api` - For local vLLM deployments
