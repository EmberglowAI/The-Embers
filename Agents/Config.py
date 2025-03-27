# config.py
import os
from google.colab import userdata # Use Colab secrets for the API key

# --- LLM Configuration ---
LLM_PROVIDER = "google" # Changed to google

# !!! IMPORTANT: Store your Gemini API key in Colab Secrets (key icon on the left) !!!
# Name the secret 'GEMINI_API_KEY'. userdata.get() will retrieve it securely.
GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')

LLM_MODEL = "gemini-1.5-flash-latest" # Or "gemini-1.5-pro-latest", "gemini-pro"
# Adjust generation config for Gemini
GENERATION_CONFIG = {
    "temperature": 0.3,
        "top_p": 0.95,
            "top_k": 40,
                # "max_output_tokens": 4000, # Often handled by model limits, but can be set
                }
                # Safety settings can be adjusted if needed (e.g., if code generation gets blocked)
SAFETY_SETTINGS = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                                ]

                                # --- Target Application ---
                                # !!! IMPORTANT: Make sure this directory exists in your Colab environment !!!
TARGET_APP_DIR = "./sample_app"
TARGET_FILES = None # None means analyze all *.py files in TARGET_APP_DIR

                                # --- Tool Configuration ---
                                # Adjust paths if needed, Colab might require specific paths or installation locations
PYTEST_PATH = "pytest"
PYLINT_PATH = "pylint"
MYPY_PATH = "mypy" # Set to None or "" if you don't want to use mypy
PYTHON_EXECUTABLE = "python3" # Colab typically uses python3

                                # --- Agent Settings ---
MAX_ITERATIONS = 15 # Max steps the agent can take before stopping
TEST_FILE_SUFFIX = "_gemini_test.py" # Suffix for generated test files
REPORT_FILE = "gemini_agent_report.md" # Output report file
GOAL = "Analyze the Python application, identify potential bugs or areas lacking tests, generate necessary pytest unit tests, run them, and report the findings using Gemini."
