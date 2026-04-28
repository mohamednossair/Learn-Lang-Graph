# Model Configuration Guide

## Overview
This project now uses centralized configuration for the Ollama model. You can easily switch between `llama3.2` and `qwen3:0.6b` (or any other Ollama model) across all files.

## Setup

### 1. Environment Configuration
Edit the `.env` file in the project root:
```
OLLAMA_MODEL=llama3.2
```

To use `qwen3:0.6b`, change it to:
```
OLLAMA_MODEL=qwen3:0.6b
```

### 2. Using in Python Files
Import the config module at the top of your Python files:
```python
import sys
sys.path.append('..')  # Adjust path as needed
from config import get_ollama_model

# Use it when creating the LLM
llm = ChatOllama(model=get_ollama_model(), temperature=0)
```

### 3. Using in Jupyter Notebooks
Add this at the beginning of your notebook (in a code cell):
```python
import sys
import os
sys.path.append(os.path.abspath('..'))
from config import get_ollama_model
```

Then replace hardcoded model references:
```python
# OLD:
llm = ChatOllama(model="llama3.2", temperature=0)

# NEW:
llm = ChatOllama(model=get_ollama_model(), temperature=0)
```

## Available Models
- `llama3.2` - Default model (requires ~2.3 GiB memory)
- `qwen3:0.6b` - Smaller model (requires less memory, good for testing)

## Memory Issues
If you encounter memory errors like "model requires more system memory", switch to `qwen3:0.6b` in your `.env` file.

**Current default in .env:** `qwen3:0.6b` (to avoid memory errors on systems with limited RAM)

## Files Status

### ✅ Python Files (All Updated)
All lesson Python files and task files have been updated to use centralized config:
- lesson_03_chatbot/ (main + tasks)
- lesson_04_tools_agent/ (main + tasks)
- lesson_05_multi_agent/ (main + tasks)
- lesson_06_database_agent/ (main + tasks)
- lesson_07_human_in_loop/ (main + tasks)
- lesson_08_memory_persistence/ (main + tasks)
- lesson_09_best_practices/ (main + tasks)
- lesson_10_capstone/ (main + tasks)
- lesson_11_subgraphs/
- lesson_12_rag_agent/
- lesson_13_vector_memory/
- lesson_16_postgres_async/ (postgres + oracle)
- lesson_17_auth_rbac/
- lesson_18_observability/
- lesson_19_event_driven/
- lesson_20_enterprise_capstone/
- lesson_21_aws_bedrock/
- lesson_22_aws_s3/
- lesson_23_conversation_api/

### ✅ Jupyter Notebooks (All Updated)
All notebooks have been updated to use centralized config:
- lesson_03_chatbot/lesson_03_chatbot.ipynb
- lesson_04_tools_agent/lesson_04_tools_agent.ipynb
- lesson_05_multi_agent/lesson_05_multi_agent.ipynb
- lesson_06_database_agent/lesson_06_database_agent.ipynb
- lesson_07_human_in_loop/lesson_07_human_in_loop.ipynb
- lesson_08_memory_persistence/lesson_08_memory_persistence.ipynb
- lesson_09_best_practices/lesson_09_best_practices.ipynb
- lesson_10_capstone/lesson_10_capstone.ipynb
- lesson_11_subgraphs/lesson_11_subgraphs.ipynb
- lesson_12_rag_agent/lesson_12_rag_agent.ipynb
- lesson_13_vector_memory/lesson_13_vector_memory.ipynb
- lesson_15_deployment/lesson_15_deployment.ipynb
- lesson_16_postgres_async/lesson_16_postgres_async.ipynb
- lesson_17_auth_rbac/lesson_17_auth_rbac.ipynb
- lesson_18_observability/lesson_18_observability.ipynb
- lesson_19_event_driven/lesson_19_event_driven.ipynb
- lesson_20_enterprise_capstone/lesson_20_enterprise_capstone.ipynb
- lesson_21_aws_bedrock/lesson_21_aws_bedrock.ipynb
- lesson_23_conversation_api/lesson_23_conversation_api.ipynb

## Configuration Files
- `config.py` - Centralized configuration module
- `.env` - Environment variables (current: OLLAMA_MODEL=qwen3:0.6b)
- `.env.example` - Template for environment variables
