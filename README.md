# langchain-textual

[![PyPI version](https://img.shields.io/pypi/v/langchain-textual)](https://pypi.org/project/langchain-textual/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/TonicAI/langchain-textual/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

PII redaction tools for LangChain, powered by [Tonic Textual](https://textual.tonic.ai).

Strip names, emails, addresses, and other sensitive data from text before it hits your LLM — or on the way back out. Drop it into any LangChain chain or agent as a standard tool.

## Installation

```bash
pip install langchain-textual
```

## Quick start

```bash
export TONIC_TEXTUAL_API_KEY="your-api-key"
```

```python
from langchain_textual import TonicTextualRedact

tool = TonicTextualRedact()
tool.invoke("My name is John Smith and my email is john@example.com.")
# "My name is [NAME_GIVEN_xxxx] [NAME_FAMILY_xxxx] and my email is [EMAIL_ADDRESS_xxxx]."
```

## Configuration

**Synthesis mode** — replace PII with realistic fake data instead of placeholders:

```python
tool = TonicTextualRedact(generator_default="Synthesis")
tool.invoke("Contact Jane Doe at jane.doe@example.com.")
# "Contact Maria Chen at maria.chen@gmail.com."
```

**Self-hosted deployment:**

```python
tool = TonicTextualRedact(tonic_textual_base_url="https://textual.your-company.com")
```

**Explicit API key** (instead of env var):

```python
tool = TonicTextualRedact(tonic_textual_api_key="your-api-key")
```

## Using with a LangChain agent

`TonicTextualRedact` is a standard LangChain [tool](https://python.langchain.com/docs/concepts/tools/), so it works anywhere tools do:

```python
from langchain_textual import TonicTextualRedact
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, [TonicTextualRedact()])
```

## Development

```bash
# install dependencies
uv sync --group dev --group test --group lint --group typing

# run unit tests
make test

# run integration tests (requires TONIC_TEXTUAL_API_KEY)
make integration_tests

# lint & format
make lint
make format
```

## License

MIT
