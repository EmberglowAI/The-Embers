# config.py
import os

# --- LLM Configuration ---
LLM_PROVIDER = "anthropic" # Changed
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_CLAUDE_API_KEY_HERE") # Specific key
LLM_MODEL = "claude-3-5-sonnet-20240620" # Specific model
LLM_TEMPERATURE = 0.3 # Maybe lower temp for more predictable planning/coding
LLM_MAX_TOKENS = 4000 # Sonnet 3.5 supports long contexts, but max_tokens refers to the *output* size

# --- Target Application ---
TARGET_APP_DIR = "./sample_app"
TARGET_FILES = None # Analyze all *.py files initially

# --- Tool Configuration ---
PYTEST_PATH = "pytest"
PYLINT_PATH = "pylint"
MYPY_PATH = "mypy"
PYTHON_EXECUTABLE = "python"

# --- Agent Settings ---
MAX_ITERATIONS = 15 # Allow slightly more iterations for dynamic planning
TEST_FILE_SUFFIX = "_claude_test.py"
REPORT_FILE = "claude_agent_report.md"
GOAL = "Analyze the Python application, identify potential bugs or areas lacking tests, generate necessary pytest unit tests, run them, and report the findings."
