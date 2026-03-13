.. _llm_tools:

LLM Tools
=========

This section provides detailed API documentation for the LLM (Large Language Model)
tools available in the LLM4AD platform. These tools allow you to connect to various
LLM backends for algorithm design and optimization tasks.

Overview
--------

The LLM4AD platform provides multiple LLM backends to suit different deployment scenarios:

- **OpenAIAPI**: Connect to OpenAI's official API or any OpenAI-compatible endpoint
- **OllamaAPI**: Run models locally using the Ollama platform
- **vLLM API**: High-performance local inference with vLLM engine
- **HTTPS API**: Generic HTTPS-based API for any compatible LLM service

All LLM tools inherit from the base ``LLM`` class and implement the ``draw_sample()``
method for generating responses.

.. toctree::
   :maxdepth: 2
   :caption: LLM Backends

   openai
   ollama
   vllm
   https

Quick Start
-----------

Choose the appropriate LLM backend based on your setup:

.. list-table::
   :header-rows: 1

   * - Use Case
     - Recommended Backend
   * - Using OpenAI's official API
     - :ref:`openai_api`
   * - Using third-party OpenAI-compatible APIs
     - :ref:`openai_api` or :ref:`https_api`
   * - Running models locally on Mac/Linux
     - :ref:`ollama_api`
   * - High-performance local inference (multiple GPUs)
     - :ref:`vllm_api`
   * - Custom HTTPS-based LLM service
     - :ref:`https_api`

Base Class
----------

All LLM tools inherit from the base ``LLM`` class located at
``llm4ad.base.LLM``. This base class provides common functionality including:

- **debug_mode**: Boolean flag for enabling detailed error reporting
- **auto_trim**: Boolean flag for automatic response trimming (enabled by default)
- **draw_sample()**: Abstract method that must be implemented by subclasses

For more details on the base class, see :doc:`../../base/index`.

Common Parameters
-----------------

Most LLM backends accept these common parameters:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
     - Default
   * - ``model``
     - Model identifier (e.g., "gpt-4", "qwen3:14b")
     - Required
   * - ``timeout``
     - Request timeout in seconds
     - 60
   * - ``temperature``
     - Sampling temperature (0.0 to 2.0)
     - 1.0
   * - ``max_tokens``
     - Maximum tokens to generate
     - 4096

Response Handling
-----------------

All LLM tools automatically handle response parsing and trimming. The platform
includes an automatic trimmer that extracts the "useful part" of generated content,
removing descriptions and truncated code.

This works for both:
- **Code completion models**: StarCoder, CodeLlama-Python, etc.
- **Chat models**: GPT series, Llama series, etc.

.. note::

   Unless you encounter special issues, keep ``auto_trim`` at its default value ``True``.

Error Handling
-------------

The LLM tools include built-in error handling:

- **Automatic retries**: Failed requests are automatically retried
- **Debug mode**: Enable via ``sampler.debug_mode = True`` for detailed error information
- **Graceful degradation**: Continues operation even with temporary failures

Performance Tips
----------------

1. **Use local models** for development and testing to save API costs
2. **Enable caching** where available to reduce redundant API calls
3. **Adjust batch size** based on your API rate limits
4. **Monitor token usage** to stay within budget limits
5. **Use appropriate timeout values** based on network conditions

Migration Guide
---------------

If you're migrating from one LLM backend to another:

**From OpenAI to Ollama:**
   .. code:: python

       # Before (OpenAI)
       from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
       sampler = OpenAIAPI(base_url="...", api_key="...", model="gpt-4")

       # After (Ollama)
       from llm4ad.tools.llm.local_ollama import LocalOllamaLLM
       sampler = LocalOllamaLLM(model_name="qwen3:14b")

**From HTTPS to vLLM:**
   .. code:: python

       # Before (HTTPS API)
       from llm4ad.tools.llm.llm_api_https import HttpsApi
       sampler = HttpsApi(host="api.example.com", key="...", model="gpt-4")

       # After (vLLM)
       from llm4ad.tools.llm.local_vllm import LocalVLLMAPI
       sampler = LocalVLLMAPI(model_path="/path/to/model", tknz_path="/path/to/tokenizer",
                              gpus=[0], ports=[22001])

See Also
--------

- :doc:`../../base/index` - Base LLM class documentation
- :doc:`../../../getting_started/installation` - Installation guide
- :doc:`../../../getting_started/examples` - Usage examples
