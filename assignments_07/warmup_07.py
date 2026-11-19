import json
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent


if load_dotenv(find_dotenv()):
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()
print('OpenAI client created.')

RESOURCES_DIR = Path("resources")

# Q1
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


tools = [
    {
        'type': 'function',
        'function': {
            'name': 'celsius_to_fahrenheit',
            'description': 'Converts a temperature from Celsius to Fahrenheit.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'celsius': {
                        'type': 'number',
                        'description': 'The temperature in Celsius.'
                    }
                },
                'required': ['celsius'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    }
]
print("Converting 0°C to Fahrenheit:", celsius_to_fahrenheit(0))
print("Converting 100°C to Fahrenheit:", celsius_to_fahrenheit(100))
print("Converting -40°C to Fahrenheit:", celsius_to_fahrenheit(-40))

# Q2 and Q3
def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call? Why or why not?
# No, it will not trigger a tool call because tool uses different function name. it should be "celsius_to_fahrenheit" instead "get_current_time".
# How many API calls will be made to answer this query? - based on the code below 2 API calls will be made. The first call is to determine if a tool is needed, 
# and the second call is to provide the final answer after the tool's output is included in the conversation.
def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time or convert temperatures.
                     Use the tool get_current_time whenever a user asks about the time.
                     Use the tool celsius_to_fahrenheit whenever a user asks to convert Celsius to Fahrenheit.
                     If the user asks for something else, answer directly without using any tools.'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            elif function_name == 'celsius_to_fahrenheit':
                try:
                    # Parse the JSON string into a Python dictionary
                    args = json.loads(tool_call.function.arguments)
                    celsius_value = args.get('celsius')
                
                    if celsius_value is not None:
                        tool_result = celsius_to_fahrenheit(celsius_value)
                    else:
                        tool_result = 'Error: missing required argument "celsius".'
                except json.JSONDecodeError:
                    tool_result = 'Error: failed to parse tool arguments as JSON.'
                
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''


response = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print("Response:", response)

response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# Agent mode engaged, the model called the celsius_to_fahrenheit tool, and the final answer was returned in the second API call.

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# No tools were needed, so the model answered directly in the first API call.

class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------
    def list_csv_files(self):
        """List available CSV files in resources/.

        Returns:
            A dict with a "files" list, or a message if none are found.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """Load a CSV file from resources/ and make it the active dataset.

        Args:
            filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

        Returns:
            A dict with a status message and column names, or an error dict.
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """Return column names for the currently loaded CSV.

        Returns:
            A list of column names, or an error dict if no CSV is loaded.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """Return summary stats for selected columns (or all columns). 
        This includes count, mean, std, min, max, and percentiles for numeric columns,
        or count, unique, top, freq for categorical columns.

        Args:
            columns: Column names to summarize. If None, summarizes all columns.

        Returns:
            A dict of summary statistics (from pandas.describe), or an error dict.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """Describe a single column (basic stats) for the requested column.
        This includes count, mean, std, min, max, and percentiles for numeric column,
        or count, unique, top, freq for categorical column.

        Args:
            column: The name of the column to describe.

        Returns:
            A dict of basic stats for the column, or an error dict.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    # Q4: Compute the Pearson correlation between two columns in the loaded DataFrame.
    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.

        Args:
            col1: The first column name.
            col2: The second column name.  
        Returns:
            A dict with keys: col1, col2, pearson_r, p_value, or an error dict if the columns are not found or no CSV is loaded.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {
                "error": f"One or both columns are not in the data. Options: {self.df.columns.tolist()}"
            }

        person_corr, p_value = pearsonr(self.df[col1], self.df[col2])
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(person_corr, 4),
            "p_value": round(p_value, 4),
        }

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """Plot from the active CSV.

        Args:
            y: Column name to plot on the y-axis. 
            x: Column name to plot on the x-axis. If None, use row index.
            plot_type: "line" or "scatter". Scatter requires x and y.

        Returns:
            Generates and shows the plot. 
            Retirms a short success message string, or an error dict/string.
        """
        error = self._ensure_loaded()
        if error:
            return error
    
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."
    
        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
    
        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None
    
        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."
    
        title_csv = self.csv_name or "current CSV"
    
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."
    
        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
    
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()
        
        return f"Plotted {y} vs {x} as a {plot_type}."

print("Class defined")

csv_backend = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation between two columns in the loaded DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "First column name for correlation.",
                    },
                    "col2": {
                        "type": "string",
                        "description": "Second column name for correlation.",
                    },
                },
                "required": ["col1", "col2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
]

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt, 
    and then user text will be appended.  

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the 
        final answer for this react cycle, this implies that reasoning alone without 
        tool calls was enough.  
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,}
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        # Append the assistant message to the conversation history.
        # Use a plain dict so `messages` stays simple and inspectable.
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        # No tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content 

        # ACT + OBSERVE: run each tool call, then append its result.
        # Note there may be multiple tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
                    
            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))
            
            # After we appending information about all tool outputs, we loop back and REASON again.

    return "I hit the tool-round limit. Try a simpler request."


# Q5
SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
print(result)

# Q6
# Role SYSTEM: sets the agent's persona, instructions, and context.
# Role USER: the human's request to the agent.
# Role ASSISTANT: Represents the model's responses, including reasoning, decisions, tool calls, and the final answer.
# Role TOOL: Returns the execution output from a called function back to the model.
print(json.dumps(messages, indent=2, default=str))

#Q7
csv_manager = CsvManager(resources_dir=RESOURCES_DIR)
@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns). 
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot from the active CSV.

    Args:
        y: Column name to plot on the y-axis. 
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.

    Returns:
        Generates and shows the plot. 
        Retirms a short success message string, or an error dict/string.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation between two columns in the loaded DataFrame.
    Returns the correlation coefficient and p-value.

    Args:
        col1: The first column name.
        col2: The second column name.  
    Returns:
        A dict with keys: col1, col2, pearson_r, p_value, or an error dict if the columns are not found or no CSV is loaded.
    """
    error = csv_manager._ensure_loaded()
    if error:
        return error

    if col1 not in csv_manager.df.columns or col2 not in csv_manager.df.columns:
        return {
            "error": f"One or both columns are not in the data. Options: {csv_manager.df.columns.tolist()}"
            }

    person_corr, p_value = pearsonr(csv_manager.df[col1], csv_manager.df[col2])
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(person_corr, 4),
        "p_value": round(p_value, 4),
    }


print(compute_correlation.description)
# Comparison with Q4 manual JSON schema:
# In Q4, we had to manually write a nested JSON Schema dictionary specifying the tool's name, description, parameters, and required fields.
# With @tool, smolagents inspects the Python function and generates all of that JSON schema
# metadata automatically (including parameter types, descriptions, required vs. optional fields,
# and the overall tool description), eliminating boilerplate and keeping code and schema in sync.
#
# What smolagents needs from the developer to produce a good description/schema:
# 1. A descriptive function name (becomes the tool's name in the schema).
# 2. Strict type annotations on every parameter and the return value.
# 3. A clear top-level docstring summary explaining what the tool does and when to use it.
# 4. A Google-style Args: section in the docstring documenting each parameter by its exact name.
# 5. Default values (if any) to determine which parameters are optional vs. required.

# Q8
TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]
model_to_use = "gpt-4o-mini"  # default model ID
model = OpenAIServerModel(
    api_key=api_key,
    model_id=model_to_use,
)

SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)
CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.

Rules:
- Prefer tools for simple tasks.
- IMPORTANT: If the user requests plot styling (color, marker, title text, labels, grid, etc.)
  that the plot_data tool cannot control, DO NOT call plot_data.
  Instead, write matplotlib code directly so the plot matches the request.
  If code execution fails, do not fall back to plot_data when the user requested styling (like color). 
  Explain what failed and what you would need to proceed.
- Be honest: only claim you did something if the code or tool actually did it.
- Assume the active dataset lives in csv_manager.df after a CSV is loaded.
"""

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

tool_agent = ToolCallingAgent(tools=TOOLS,
                         model=model,
                         instructions=SYSTEM_PROMPT,)
code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)


response_tool = tool_agent.run(prompt)
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})
print("Tool Agent Response:", response_tool)
print("Code Agent Response:", response_code)
print("ToolCallingAgent response:", response_tool)
print("CodeAgent response:", response_code)

# 1. What did each agent actually produce? Did each agent change the dot color?
# - ToolCallingAgent: Called the predefined load_csv and plot tools sequentially via JSON.
#   It could NOT change the dot color to green because the plot tool schema does not accept
#   a color parameter—it was restricted to the tool's built-in defaults.
# - CodeAgent: Wrote and executed Python code directly. Because it has access to Python
#   it was able to customize the matplotlib scatter plot to
#   fulfill the exact styling request.
#
# 2. What does this reveal about when each type of agent is more useful?
# - CodeAgent is better for dynamic, open-ended tasks (like exploratory data analysis or custom
#   visualizations) where users make unanticipated requests, need to chain complex logic, or
#   manipulate data directly in code without being limited by rigid tool parameters.
# - ToolCallingAgent is better for predictable, tightly controlled workflows where safety,
#   security, and strict guardrails matter—ensuring the model can only execute pre-approved,
#   validated operations rather than arbitrary Python code.

# Q9
# Q9:
# 1. Task where ToolCallingAgent is a better choice & its key property:
#    - Task example: Issuing customer refunds or updating records in a production database.
#    - Key property: The task has high-stakes external side effects and requires strict,
#      deterministic guardrails. Every action must follow a fixed, pre-validated schema
#      with zero room for the agent to improvise or bypass business rules.
#
# 2. Meaningful risk of CodeAgent that does not apply to ToolCallingAgent:
#    - Arbitrary code execution: Because CodeAgent generates and executes Python code
#      at runtime, a model mistake or prompt injection attack could run unintended or
#      destructive operations—such as modifying/deleting files, accessing environment variables
#      and API keys and  so on. ToolCallingAgent avoids
#      this because the LLM never executes code directly; it only outputs JSON parameters
#      for pre-approved tools, which are sandboxed and validated before execution.