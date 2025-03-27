import subprocess
import os
import json
import re
from typing import Dict, Any, Tuple, Callable, Optional

# Import config variables
from config import (
    LLM_PROVIDER, GEMINI_API_KEY, LLM_MODEL, GENERATION_CONFIG, SAFETY_SETTINGS,
    PYTEST_PATH, PYLINT_PATH, MYPY_PATH, PYTHON_EXECUTABLE
)
# Import prompts to potentially use system prompt during LLM calls
import prompts

# --- LLM Client ---
llm_client = None
llm_model_instance = None # Store the generative model instance

def initialize_llm_client():
    """Initializes the Google Gemini client."""
    global llm_client, llm_model_instance
    if llm_client: # Avoid re-initialization
        return
    if LLM_PROVIDER == "google":
        try:
            import google.generativeai as genai
            # Ensure API key is present
            api_key = GEMINI_API_KEY
            if not api_key:
                 # Attempt to get from environment if Colab secret failed somehow
                 api_key = os.getenv('GEMINI_API_KEY')
                 if not api_key:
                      raise ValueError("Gemini API Key not found. Set it in Colab Secrets as 'GEMINI_API_KEY' or as an environment variable.")

            genai.configure(api_key=api_key)
            llm_client = genai # Store the configured module
            # Create the specific model instance
            llm_model_instance = llm_client.GenerativeModel(
                LLM_MODEL,
                # system_instruction=prompts.SYSTEM_PROMPT # Set system instruction at model level
                # System prompt/instruction might be better handled per-request depending on API/model version
            )

            # Optional: Test connection (e.g., list models or simple generation)
            try:
                 # list_models is lightweight way to check auth
                 # models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                 # if not any(m.name.endswith(LLM_MODEL) for m in models):
                 #     print(f"Warning: Configured model {LLM_MODEL} not found in list_models().")
                 # Or a dummy generation:
                 # llm_model_instance.generate_content("test", generation_config={"max_output_tokens": 5})
                 print(f"Google Gemini client initialized for model: {LLM_MODEL}")
            except Exception as test_err:
                 print(f"Warning: Gemini client initialized but validation check failed: {test_err}")

        except ImportError:
            print("Error: Google Generative AI library not installed. `pip install google-generativeai`")
            llm_client = None
            raise ConnectionError("Google Generative AI library not found.")
        except ValueError as e:
            print(f"Configuration Error: {e}")
            llm_client = None
            raise ConnectionError(f"Gemini configuration error: {e}")
    else:
        print(f"Error: Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Expected 'google'.")
        llm_client = None
        raise ConnectionError(f"Unsupported LLM Provider: {LLM_PROVIDER}")

def call_llm(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Calls the configured Google Gemini API."""
    global llm_model_instance
    if llm_client is None or llm_model_instance is None:
        try:
            initialize_llm_client()
            if llm_client is None or llm_model_instance is None:
                 raise ConnectionError("LLM client/model is None after initialization attempt.")
        except ConnectionError as init_err:
             raise ConnectionError(f"LLM client not initialized and failed to initialize: {init_err}")

    try:
        # Construct content correctly, potentially including system prompt if not set on model
        # Newer Gemini APIs might prefer system prompt via generate_content parameter
        contents = [prompt]
        # Note: System instructions set at the model level are generally preferred for Gemini 1.5+
        # If system_prompt needs to be passed per request (older models or specific needs):
        # model_to_use = llm_client.GenerativeModel(LLM_MODEL, system_instruction=system_prompt)
        # Or structure the 'contents' list/dict according to the API if needed.

        response = llm_model_instance.generate_content(
            contents,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS,
            # stream=False # Ensure non-streaming response
        )

        # Handle potential safety blocks or empty responses
        if not response.candidates:
             # Check prompt feedback for safety issues
             block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
             safety_ratings = response.prompt_feedback.safety_ratings if response.prompt_feedback else "N/A"
             print(f"Warning: LLM response blocked or empty. Reason: {block_reason}. Ratings: {safety_ratings}")
             return f"Error: LLM response blocked due to safety settings (Reason: {block_reason})."

        # Assuming a successful response, extract text
        # Handle potential multi-part responses if necessary, but usually text is in one part
        if response.text:
             return response.text.strip()
        else:
            # If no .text but parts exist, try to reconstruct (unlikely for simple text prompts)
            full_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, "text"))
            if full_text:
                 print("Warning: Reconstructed text from response parts.")
                 return full_text.strip()
            else:
                 print(f"Warning: LLM response received but no text content found. Response: {response}")
                 # Return raw response structure might break parsing, return error string instead
                 # return json.dumps(response.to_dict()) # to_dict() is useful for debugging
                 return "Error: LLM response received but contained no usable text content."


    except Exception as e:
        print(f"Error calling Google Gemini API: {e}")
        # More specific error handling for google.api_core.exceptions might be useful
        return f"Error: LLM API call failed. Details: {e}"


# --- Tool Execution Logic (Unchanged from Claude version) ---
# (Includes run_subprocess, run_static_analysis, run_python_file, run_tests,
# read_file, write_file, parse_code_snippet, get_python_files)

def run_subprocess(command: list[str], cwd: str = ".") -> Tuple[int, str, str]:
    """Runs a command as a subprocess and returns exit code, stdout, stderr."""
    print(f"Running command: {' '.join(command)} in {cwd}")
    try:
        if not os.path.isdir(cwd):
             print(f"Error: Working directory not found: {cwd}")
             return -1, "", f"Error: Working directory not found: {cwd}"
        process = subprocess.run(
            command, capture_output=True, text=True, cwd=cwd, check=False, timeout=120
        )
        print(f"Command finished with exit code: {process.returncode}")
        return process.returncode, process.stdout, process.stderr
    except FileNotFoundError:
        cmd_name = command[0]
        print(f"Error: Command not found: {cmd_name}")
        # Provide hint about installation in Colab
        hint = ""
        if cmd_name in ["pytest", "pylint", "mypy"]:
            hint = f" Try running `!pip install {cmd_name}` in a Colab cell."
        return -1, "", f"Error: Command '{cmd_name}' not found.{hint}"
    except subprocess.TimeoutExpired:
        print(f"Error: Command timed out: {' '.join(command)}")
        return -1, "", f"Error: Command timed out after 120s: {' '.join(command)}"
    except Exception as e:
        print(f"Error running subprocess {command[0]}: {e}")
        return -1, "", f"Error running subprocess {command[0]}: {e}"

def run_static_analysis(filepath: str, project_root: str) -> str:
    """Runs pylint and optionally mypy on a file relative to project_root."""
    reports = []
    abs_filepath = os.path.abspath(os.path.join(project_root, filepath))
    if not os.path.isfile(abs_filepath):
        return f"Error: File not found for static analysis: {filepath} (resolved to {abs_filepath})"

    if PYLINT_PATH:
        cmd_lint = [PYLINT_PATH, "--output-format=text", abs_filepath]
        exit_code_lint, stdout_lint, stderr_lint = run_subprocess(cmd_lint, cwd=project_root)
        lint_output = stdout_lint + ("\n" + stderr_lint if stderr_lint else "")
        reports.append(f"--- Pylint Report on {filepath} (Exit Code: {exit_code_lint}) ---\n{lint_output.strip()}")
    else:
        reports.append("--- Pylint skipped (PYLINT_PATH not configured) ---")

    if MYPY_PATH:
        cmd_mypy = [MYPY_PATH, abs_filepath]
        exit_code_mypy, stdout_mypy, stderr_mypy = run_subprocess(cmd_mypy, cwd=project_root)
        mypy_output = (stdout_mypy + "\n" + stderr_mypy).strip() or "No output."
        reports.append(f"--- MyPy Report on {filepath} (Exit Code: {exit_code_mypy}) ---\n{mypy_output}")
    else:
         reports.append("--- MyPy skipped (MYPY_PATH not configured) ---")
    return "\n\n".join(reports)

def run_python_file(filepath: str, project_root: str) -> Tuple[int, str, str]:
    """Executes a python file relative to project_root."""
    abs_filepath = os.path.abspath(os.path.join(project_root, filepath))
    if not os.path.isfile(abs_filepath):
        return -1, "", f"Error: Python file not found: {filepath} (resolved to {abs_filepath})"
    cmd = [PYTHON_EXECUTABLE, abs_filepath]
    return run_subprocess(cmd, cwd=project_root)

def run_tests(test_dir_or_file: str, project_root: str) -> Tuple[int, str, str]:
    """Runs pytest on a specific directory or file relative to project_root."""
    abs_path = os.path.abspath(os.path.join(project_root, test_dir_or_file))
    if not os.path.exists(abs_path):
        return -1, "", f"Error: Test path not found: {test_dir_or_file} (resolved to {abs_path})"
    cmd = [PYTEST_PATH, "-v", abs_path]
    return run_subprocess(cmd, cwd=project_root)

def read_file(filepath: str, project_root: Optional[str] = None) -> str | None:
    """Reads the content of a file, resolving path relative to project_root if provided."""
    if project_root:
        abs_filepath = os.path.abspath(os.path.join(project_root, filepath))
    else:
        abs_filepath = os.path.abspath(filepath)
    try:
        if not os.path.isfile(abs_filepath):
             print(f"Error: File not found: {filepath} (resolved to {abs_filepath})")
             return None
        with open(abs_filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {filepath} (at {abs_filepath}): {e}")
        return None

def write_file(filepath: str, content: str, project_root: Optional[str] = None) -> bool:
    """Writes content to a file, resolving path relative to project_root if provided."""
    if project_root:
        abs_filepath = os.path.abspath(os.path.join(project_root, filepath))
    else:
        abs_filepath = os.path.abspath(filepath)
    try:
        os.makedirs(os.path.dirname(abs_filepath), exist_ok=True)
        with open(abs_filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully wrote to {abs_filepath}")
        return True
    except Exception as e:
        print(f"Error writing file {filepath} (at {abs_filepath}): {e}")
        return False

def parse_code_snippet(llm_response: str, lang: str = "python") -> str | None:
    """Extracts the first code snippet ```lang ... ``` from LLM response."""
    pattern = rf"```{lang}\s*([\s\S]+?)\s*```"
    match = re.search(pattern, llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Basic fallback if response seems to BE the code
    lines = llm_response.strip().splitlines()
    if len(lines) > 1 and (lines[0].startswith(("import ", "from ", "def ", "class ", "#"))):
         print("Warning: Assuming entire LLM response is code snippet due to lack of ``` markers.")
         return llm_response.strip()
    print("Warning: Could not parse code snippet from LLM response.")
    return None

def get_python_files(start_dir: str) -> list[str]:
    """Finds all .py files recursively in a directory, excluding common ignores."""
    py_files = []
    ignore_dirs = {'.git', '.venv', 'venv', 'env', '__pycache__', '.ipynb_checkpoints'}
    for root, dirs, files in os.walk(start_dir):
        # Modify dirs in-place to prevent walking into ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files


# --- Tool Definitions and Execution (Unchanged from Claude version) ---
TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "read_file": read_file,
    "write_file": write_file,
    "run_static_analysis": run_static_analysis,
    "run_python_file": run_python_file,
    "run_tests": run_tests,
}

TOOL_SCHEMAS = {
    "read_file": {
        "description": "Reads the content of a specified file relative to the project root.",
        "params": {"filepath": "string (relative path within project)"}
    },
    "write_file": {
        "description": "Writes content to a specified file relative to the project root, overwriting if it exists, creating directories if needed.",
        "params": {
            "filepath": "string (relative path within project)",
            "content": "string (the content to write)"
        }
    },
    "run_static_analysis": {
        "description": "Runs static analysis tools (pylint, mypy) on a Python file relative to the project root.",
        "params": {"filepath": "string (relative path to the .py file)"}
        # project_root is added automatically by execute_tool
    },
    "run_python_file": {
        "description": "Executes a specified Python file relative to the project root and captures its output/exit code.",
        "params": {"filepath": "string (relative path to the .py file)"}
        # project_root is added automatically by execute_tool
    },
    "run_tests": {
        "description": "Runs pytest on a specified test file or directory relative to the project root.",
        "params": {"test_dir_or_file": "string (relative path to test file/directory)"}
        # project_root is added automatically by execute_tool
    },
    # --- Actions handled directly by the Agent using the LLM ---
    "analyze_code": {
        "description": "Performs an LLM-based analysis of a Python file's content for bugs, quality, and testability.",
        "params": {"filepath": "string (relative path to the .py file)"}
    },
    "generate_tests": {
        "description": "Uses the LLM to generate pytest unit tests for a given Python file.",
        "params": {"filepath": "string (relative path to the .py file)"}
    },
    "debug_code": {
        "description": "Uses the LLM to analyze an error traceback and suggest a fix for a Python file.",
        "params": {
            "filepath": "string (relative path to the .py file)",
            "traceback": "string (the error traceback)",
        }
    },
     # --- Control Flow ---
    "finish": {
        "description": "Indicates the agent should stop processing because the goal is met or no further actions are productive.",
        "params": {"reason": "string (brief justification)"}
    }
}

def get_tool_schemas_json() -> str:
    """Returns the tool schemas as a formatted JSON string for the prompt."""
    # Ensure keys are strings for JSON compatibility, although they should be already
    clean_schemas = {str(k): v for k, v in TOOL_SCHEMAS.items()}
    return json.dumps(clean_schemas, indent=2)

def execute_tool(action_name: str, params: Dict[str, Any], agent_instance) -> Any:
    """Looks up and executes an external tool function based on the action name."""
    if action_name in TOOL_REGISTRY:
        func = TOOL_REGISTRY[action_name]
        try:
            # Add project_root context for tools that need it
            params_with_context = params.copy()
            if action_name in ["run_static_analysis", "run_python_file", "run_tests", "read_file", "write_file"]:
                 params_with_context["project_root"] = agent_instance.app_dir
                 # Ensure filepath exists if required by the function signature
                 if 'filepath' not in params_with_context and func.__name__ in ['run_static_analysis', 'run_python_file', 'read_file', 'write_file']:
                      return {"error": f"Missing 'filepath' parameter for tool '{action_name}'"}
                 if 'test_dir_or_file' not in params_with_context and func.__name__ == 'run_tests':
                      return {"error": f"Missing 'test_dir_or_file' parameter for tool '{action_name}'"}

            # Filter params to only those expected by the function? Maybe not needed if using **kwargs
            result = func(**params_with_context)

            # Standardize result format slightly for logging/state
            if isinstance(result, tuple) and len(result) == 3 and isinstance(result[0], int): # subprocess results
                return {"exit_code": result[0], "stdout": result[1], "stderr": result[2]}
            elif isinstance(result, str) and func.__name__ == 'run_static_analysis':
                return {"report": result} # Static analysis returns one string report
            elif isinstance(result, str) and func.__name__ == 'read_file':
                 return {"content": result} # read_file returns content string
            elif isinstance(result, bool) and func.__name__ == 'write_file':
                 return {"success": result}
            elif result is None and func.__name__ == 'read_file': # File not found case
                 return {"error": "File not found by read_file tool."}
            else:
                # Fallback for unexpected results
                return {"result": result}
        except TypeError as e:
            print(f"Error: Incorrect or missing parameters for tool '{action_name}': {e}. Provided: {params}")
            return {"error": f"Parameter mismatch for tool '{action_name}': {e}"}
        except Exception as e:
            print(f"Error executing tool '{action_name}': {e}")
            return {"error": f"Tool execution failed: {e}"}
    else:
        # This case happens if the LLM hallucinates a tool or requests an agent-internal action
        return {"error": f"Tool '{action_name}' is not a recognized external tool."}
