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


# Load environment variables from a .env file if present
load_dotenv(find_dotenv())

# Retrieve the API key
api_key = os.getenv("OPENAI_API_KEY")

# Validate that the key exists
if not api_key:
    raise ValueError(
        "API key not found. Please make sure 'OPENAI_API_KEY' is set in your .env file."
    )

print("API key loaded successfully.")

df = None  # Global variable to hold the loaded DataFrame
DATA_DIR = Path("resources/happiness_project")  # Directory containing the CSV files

COLUMN_NAMES = {
    "Country": "country",
    "Regional indicator": "region",
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",   # the 2024 file uses this name
    "GDP per capita": "gdp_per_capita",
}

# Task 1: Define Your Tools
# Tool 1: load_happiness_data
@tool
def load_happiness_data() -> dict:
    """
    Load the happiness data from CSV files in the resources/happiness_project directory
    into the dataframe.
    Returns:
        dict: A dictionary containing such keys as `shape` and `columns` and `data`,
        data containing the loaded DataFrame.

    """
    global df
    frames = []
    for path in sorted(DATA_DIR.glob("world_happiness_*.csv")):
        year_df = pd.read_csv(path, sep=";", decimal=",")
        year_df = year_df.rename(columns=COLUMN_NAMES)
        year_df["year"] = int(path.stem.split("_")[-1])  # Extract year from filename
        frames.append(year_df)
    df = pd.concat(frames, ignore_index=True)
    return {"shape": df.shape, 
            "columns": list(df.columns), 
            "data": df
    }

# Tool 2: summarize_column
def _available_csv_files() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted(
        [
            f.name
            for f in DATA_DIR.iterdir()
            if f.is_file() and f.suffix.lower() == ".csv"
        ]
    )


def _ensure_loaded():
    if df is None:
        files = _available_csv_files()
        example = files[0] if files else "your_file.csv"
        return {
            "error": (
                "No CSV is loaded yet. First load one from resources/happiness_project/. "
                f"For example: load_happiness_data '{example}'."
            )
        }
    return None


@tool
def summarize_column(column_name: str) -> dict:
    """
    Summarize a specific column in the loaded DataFrame.
    Args:
        column_name (str): The name of the column to summarize.
    Returns:
        dict: A dictionary containing summary statistics of the column.
    """
    # global df
    error_response = _ensure_loaded()
    if error_response:
        return error_response

    if column_name not in df.columns:
        return {"error": f"Column '{column_name}' does not exist in the DataFrame."}

    summary = df[column_name].describe().to_dict()
    return summary


# Tool 3: compute_correlation
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation between two columns in the loaded DataFrame.
    Args:
        col1 (str): The first column name.
        col2 (str): The second column name.
    Returns:
        dict: A dictionary containing the correlation coefficient and p-value, column names, or an error message if applicable.
    """
    #global df
    error_response = _ensure_loaded()
    if error_response:
        return error_response

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": f"One or both columns are not in the data. Options: {df.columns.tolist()}"}

    person_corr, p_value = pearsonr(df[col1], df[col2])
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(person_corr, 4),
        "p_value": round(p_value, 4),
    }

# Tool 4: get_top_n_countries
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """
    Get the top N countries based on a specific column for a given year.
    Args:
        column (str): The column to sort by.
        year (int): The year to filter the data.
        n (int): The number of top countries to return, default is 5.
    Returns:
        dict: A dictionary containing the top N countries and their values.
    """
    # global df
    error_response = _ensure_loaded()
    if error_response:
        return error_response

    if column not in df.columns:
        return {"error": f"Column '{column}' does not exist in the DataFrame."}

    filtered_df = df[df["year"] == year]
    if filtered_df.empty:
        return {"error": f"No data available for the year {year}."}

    top_countries = (
        filtered_df.sort_values(by=column, ascending=False)
        .head(n)[["country", column]]
        .to_dict(orient="records")
    )
    return top_countries


# Task 2: Build the Agent
model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")
SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""
TOOLS = [
    load_happiness_data,
    summarize_column,
    compute_correlation,
    get_top_n_countries,
]

agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats", "numpy"],
    max_steps=8,
)

if __name__ == "__main__":

    # Task 3: Run Guided Queries
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        (
            "What is the correlation between gdp_per_capita and happiness_score? Is"
            " it statistically significant?"
        ),
        "Show me the top 5 happiest countries in 2020.",
        (
            "Plot happiness_score over the years as a line chart, with one line per region. "
            "Write direct top-level Python script code that calls load_happiness_data() and extracts 'data' into df like df = load_happiness_data()['data'], "
            "groups and averages the data by year and region, plots it using matplotlib, "
            "and saves the plot to outputs/happiness_by_region.png."
        ),
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)


    # Task 4: Your Own Questions
    # My query 1
    my_query_1 = "What are the top 3 countries with the highest GDP per capita in 2021?" 
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: Yes, the agent used the get_top_n_countries tool to retrieve the top 3 countries with the highest GDP per capita in 2021.

    # My query 2
    my_query_2 = "Generate dummy data and compute the correlation between two columns, then plot the results."
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: The agent generated dummy data, computed the correlation between two columns, and plotted the results using matplotlib.


# Task 5: Reflection
# --- Reflection ---
#
# 1. In Query 3, how did the agent report the correlation? The `compute_correlation` tool
#    returns both a Pearson r and a p-value — did the agent compute and communicate the
#    result sensibly (for example, describing the strength and direction of the
#    relationship, and noting how confident we can be in it)? - Agent compute the result properly, but it did not provide a 
#    detailed interpretation of the correlation strength and direction. It reported the Pearson r and p-value, 
#    but did not explain what those values mean in terms of the relationship between GDP per capita and happiness score.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example. - agent used the tools correcly, was able to translate the human query into
#    tool calls and code generation. it was surprising. In some sence agent depends on the tools design heavily, 
#    like for example global data, it is better to avoid using global variables and instead pass the data as parameters to the tools.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    (You do not need to implement it.) -  A tool that cleans and preprocess the data, handling missing values, outliers, and normalizing columns. 
#    This would help the agent answer questions about data quality and allow for more accurate analyses and visualizations.