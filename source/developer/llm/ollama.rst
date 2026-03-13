.. _ollama_api:

Ollama API
==========

The LocalOllamaLLM class provides an interface to interact with locally deployed Ollama models.
Ollama allows you to run large language models locally on your machine.

Overview
--------

The ``LocalOllamaLLM`` class extends the base ``LLM`` class and provides integration with
the Ollama platform for running LLMs locally. It uses the langchain-ollama library for
seamless interaction with local Ollama instances.

Setup Requirements
------------------

Before using LocalOllamaLLM, ensure you have the following:

1. **Install Ollama**: Download and install Ollama from `ollama.ai <https://ollama.ai/>`_

   .. code:: bash

       # On macOS/Linux
       curl -fsSL https://ollama.com/install.sh | sh

2. **Pull Required Models**: Pull the desired model from Ollama library:

   .. code:: bash

       ollama pull qwen3:14b
       ollama pull llama3.2
       ollama pull mistral

3. **Python Packages**: Install the required Python packages:

   .. code:: bash

       pip install ollama langchain-ollama langchain

4. **Start Ollama Service**: Ensure the Ollama service is running:

   .. code:: bash

       ollama serve

   .. note::

      By default, Ollama runs on ``http://localhost:11434``. The LocalOllamaLLM
      connects to this endpoint automatically.

API Reference
-------------

Class Definition
~~~~~~~~~~~~~~~

.. class:: LocalOllamaLLM

   A concrete implementation of the LLM base class that interfaces with local Ollama models.

Constructor
~~~~~~~~~~~

.. method:: __init__(self, model_name: str, **ollama_llm_init_params)

   Initialize the local Ollama LLM client.

   :param model_name: The name of the Ollama model to use. This should match a model
                      you've pulled using ``ollama pull <model>``.
                      Examples: ``"qwen3:14b"``, ``"llama3.2"``, ``"mistral"``.
   :type model_name: str
   :param ollama_llm_init_params: Additional initialization parameters passed to
                                  the ``langchain_ollama.OllamaLLM`` constructor.
                                  See `langchain-ollama documentation <https://python.langchain.com/docs/integrations/llms/ollama/>`_
                                  for available options.
   :type ollama_llm_init_params: Any, optional

   .. note::

      Common parameters include:
      - ``base_url``: Custom endpoint URL (defaults to ``http://localhost:11434``)
      - ``temperature``: Sampling temperature (0.0 to 2.0)
      - ``top_p``: Nucleus sampling parameter
      - ``top_k``: Top-k sampling parameter
      - ``num_ctx``: Context window size
      - ``num_gpu``: Number of GPUs to use
      - ``num_thread``: Number of CPU threads

Methods
~~~~~~~

.. method:: draw_sample(self, prompt: str | Any, *args, **kwargs) -> str

   Generate a response from the local Ollama model based on the provided prompt.

   :param prompt: The input prompt. Can be either:
                  - A string containing the user message
                  - Any format accepted by the Ollama model
   :type prompt: str | Any
   :param args: Additional positional arguments (passed to langchain).
   :param kwargs: Additional keyword arguments passed to the model invocation.
   :return: The generated text content from the LLM response.
   :rtype: str

   .. note::

      This method uses ``model.invoke()`` under the hood, which is a synchronous call.
      For streaming responses, you would need to modify this method or use the
      underlying langchain_ollama.OllamaLLM directly.

Examples
--------

Example 1: Basic Usage with Qwen Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.local_ollama import LocalOllamaLLM

    # Initialize with a specific model
    sampler = LocalOllamaLLM(model_name="qwen3:14b")

    # Generate a response
    response = sampler.draw_sample("What is machine learning?")
    print(response)

Example 2: Using Different Ollama Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.local_ollama import LocalOllamaLLM

    # Using Llama 3.2 model
    sampler = LocalOllamaLLM(model_name="llama3.2")

    # Using Mistral model
    sampler = LocalOllamaLLM(model_name="mistral")

    # Using a smaller model for faster inference
    sampler = LocalOllamaLLM(model_name="llama3.2:1b")

Example 3: Customizing Model Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.local_ollama import LocalOllamaLLM

    # Initialize with custom parameters
    sampler = LocalOllamaLLM(
        model_name="qwen3:14b",
        temperature=0.7,      # Control randomness (0.0 to 2.0)
        top_p=0.9,           # Nucleus sampling
        top_k=40,            # Top-k sampling
        num_ctx=8192,        # Context window size
        num_thread=8,        # CPU threads
        base_url="http://localhost:11434"  # Custom endpoint
    )

    response = sampler.draw_sample("Write a Python function to sort a list.")
    print(response)

Example 4: Integration with LLM4AD Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.local_ollama import LocalOllamaLLM
    import llm4ad

    # Create LLM sampler with local Ollama
    sampler = LocalOllamaLLM(
        model_name="qwen3:14b",
        temperature=0.8
    )

    # Use with an LLM4AD method
    task = llm4ad.tasks.optimization.SymbolicRegression(
        dimension=5,
        num_samples=100
    )

    method = llm4ad.methods.eoh.EoH(
        task=task,
        sampler=sampler,
        num_iterations=50
    )

    result = method.run()
    print(f"Best solution: {result.best_solution}")

Example 5: Multi-Turn Conversation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.local_ollama import LocalOllamaLLM

    sampler = LocalOllamaLLM(model_name="llama3.2")

    # First prompt
    response1 = sampler.draw_sample("What is Python?")
    print(f"Response 1: {response1}")

    # For multi-turn, you would need to maintain conversation history manually
    # since the basic implementation doesn't maintain state

Available Ollama Models
-----------------------

Here are some popular models available on Ollama:

================== =================== ================
Model              Size                Description
================== =================== ================
llama3.2           3.8GB               Latest Llama 3.2 model
llama3.2:1b        1.3GB               Lightweight Llama 3.2
qwen3:14b          ~9GB                Qwen 3 14B model
qwen3:8b           ~5GB                Qwen 3 8B model
mistral            ~4GB                Mistral 7B model
codellama          ~3.8GB              Code-focused Llama
phi3               ~2.3GB              Microsoft's Phi-3
================== =================== ================

For the full list, visit `ollama.com/library <https://ollama.com/library>`_

Common Issues and Troubleshooting
----------------------------------

1. **Ollama Not Running**: Ensure ``ollama serve`` is running before using the API.

2. **Model Not Found**: Make sure you've pulled the model with ``ollama pull <model_name>``.

3. **Out of Memory**: Use a smaller model or reduce context size.

4. **Slow Inference**: Adjust ``num_thread`` parameter or use GPU acceleration.

5. **Connection Refused**: Verify Ollama is running on the correct port (default: 11434).

See Also
--------

- :ref:`openai_api` - For OpenAI API integration
- :ref:`https_api` - For custom HTTPS API implementations
- :ref:`vllm_api` - For local vLLM deployments
