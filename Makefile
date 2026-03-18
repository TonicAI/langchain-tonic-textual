.PHONY: test integration_tests lint format

test:
	uv run --group test pytest tests/unit_tests/

integration_tests:
	uv run --group test pytest tests/integration_tests/

lint:
	uv run --group lint ruff check langchain_textual/ tests/
	uv run --group lint ruff format --diff langchain_textual/ tests/

format:
	uv run --group lint ruff format langchain_textual/ tests/
	uv run --group lint ruff check --fix langchain_textual/ tests/
