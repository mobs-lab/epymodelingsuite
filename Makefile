.PHONY: install-dev install-dev-pip test lint format lock clean

# Install with uv sync (creates .venv)
install-dev:
	uv sync --extra dev

# Install in current env (conda, venv, etc.)
install-dev-pip:
	uv pip install -r requirements.lock.txt -e .

# Update lock files
# uv.lock is the single source of truth; requirements.lock.txt is generated from it. 
lock:
	uv lock && uv export --frozen --extra dev > requirements.lock.txt