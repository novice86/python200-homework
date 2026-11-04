from dotenv import load_dotenv
from openai import OpenAI
import json


# API Question 1
load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}
    ]
)
print(f"Response Text: {response.choices[0].message.content}")
print(f"Model Used: {response.model}")
print(f"Total Tokens: {response.usage.total_tokens}")

# API Question 2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=temp
    )

    print(f"Temperature: {temp}")
    print(f"Response Text: {response.choices[0].message.content}")
    print("\n")
    # With temperature = 1.5 the output becomes eccentric in comparison to straightforward output when temperature is equal to 0.
    # I would use a temperature of 0. This forces the model to almost always choose the most probable next token, 
    # resulting in highly consistent and reproducible outputs across multiple runs.

# API Question 3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", 
         "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."
         }
    ],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response.choices):
    print(f"Response {i + 1}: {choice.message.content}")

# API Question 4
prompt = "Explain how neural networks work."
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=15
)
print(f"Response Text: {response.choices[0].message.content}")
# with max_tokens = 15 the output sentence is unfinished. 
# In a real application, we might want to use max_tokens to control API costs, ensure faster response times, 
# and restrict the overall length of the generated text."

# System Question 1
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
print(f"Response Text: {response.choices[0].message.content}")
messages = [
    {"role": "system", "content": "You are Python tutor. You always explain things in a complex way without a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
print(f"Response Text: {response.choices[0].message.content}")
# Adding prompt for the system role change the tone of the returned text based on the provided information.
# it acts as a high-level instruction manual that governs how the AI should respond.

# System Question 2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
print(f"Response Text: {response.choices[0].message.content}")
# The model knows Jordan's name becuse we included the conversation history into the messages and sent it as a single API call.

# Prompt Question 1 — Zero-Shot
prompt = """Classify the sentiment of each review below as strictly 'positive', 'negative', 'mixed'.
Print each result labeled with the review number.
Review 1: The onboarding process was smooth and the team was welcoming.
Review 2: The software crashes constantly and support never responds.
Review 3: Great price, but the documentation is nearly impossible to follow.
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)

# Prompt Question 2 — One-Shot
prompt = """Classify the sentiment of each review below as strictly 'positive', 'negative', 'mixed'.
Print each result labeled with the review number. Use the format below to structure the response:
Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Review 1: The onboarding process was smooth and the team was welcoming.
Review 2: The software crashes constantly and support never responds.
Review 3: Great price, but the documentation is nearly impossible to follow.
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print("\nResponse for One-Shot\n")
print(response.choices[0].message.content)
# Adding one exampke changed the format of the model response compared to the Q1.

# Prompt Question 3 — Few-Shot
prompt = """Classify the sentiment of each review below as strictly 'positive', 'negative', 'mixed'.
Print each result labeled with the review number. Use the format below to structure the response:
Example1:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed
Example2:
Review: "Great quality and delivery was on time."
Sentiment: positive
Example3:
Review: "Not happy with the product quality."
Sentiment: negative


Review 1: The onboarding process was smooth and the team was welcoming.
Review 2: The software crashes constantly and support never responds.
Review 3: Great price, but the documentation is nearly impossible to follow.
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print(f"\nResponse for Few-Shot:\n")
print(response.choices[0].message.content)
# More examples equl better pattern recognition and consistency.
# We can usee one-shot or few-shot when the task is nuanced or you need a specific output format.
# Zero-shot is fastest but less reliable for complex tasks.

# Prompt Question 4 — Chain of Thought
prompt = """
Please solve the following problem, show the reasoning step by step before giving a final answer. 
Label the final answer clearly as "Final Answer: [your answer]".
Problem:
A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary? 
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print(f"\nResponse for Chain of Thought:\n")
print(response.choices[0].message.content)
# Instructing the model to reason step-by-step enforces a strict order of 
# operations. This prevents the LLM from hallucinating the final calculation 
# by forcing it to establish the intermediate context first.


# Prompt Question 5 — Structured Output
prompt = """
Classify the sentiment of the review and respond ONLY with valid JSON.
Keys: sentiment (positive/negative/mixed), confidence (0–1 scale), reason (one short sentence).

review = "I've been using this tool for three months. It handles large datasets well,
but the UI is clunky and the export options are limited."
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
    response_format={ "type": "json_object" }
)
message_content = response.choices[0].message.content
try:
    result = json.loads(message_content)
    print("Parsed sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])
except json.JSONDecodeError:
    print("Error: response was not valid JSON")
    print(f"Raw response:{message_content}")

# Prompt Question 6 — Delimiters
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print(f"\nResponse for Delimiters step-by-step instructions case:\n")
print(response.choices[0].message.content)

user_text = "The sun slowly dipped below the horizon, painting the clouds in vibrant shades of pink and orange as the city quieted down for the night."
prompt = f"""You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print(f"\nResponse for Delimiters regular prose case:\n")
print(response.choices[0].message.content)
# we use delimiters to clearly separate user instructions from data, 
# reducing the risk of prompt injection and misinterpretation.

# Ollama Question 1
prompt = """
Explain what a large language model is in two sentences.
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print(f"\nResponse for Ollama question:\n")
print(response.choices[0].message.content)

# Thinking...
# Okay, the user wants me to explain what a large language model is in two sentences. Let me start by breaking down the key
# elements. First, a large language model is a type of artificial intelligence that can understand and generate human-like text.
# That's the basic definition.

# Next, I need to highlight its capabilities. It can process a lot of text, understand context, and even create new content.
# Plus, it's trained on vast amounts of data, so it's very powerful and adaptable. I should make sure to mention that it's used
# in various fields like AI, healthcare, etc. Oh, and I should end with a friendly note to keep the conversation going. Let me
# check if that's two sentences. Yes, first sentence defines it, the second explains its capabilities and uses, and then a
# closing. That should work.
# ...done thinking.

# A large language model is a type of artificial intelligence that can understand and generate human-like text, learning from
# vast amounts of data to process context and create new content. It’s trained to understand complex language structures and
# adapt to various domains, making it powerful for tasks like writing, translation, and research.

# In two sentences:
# 1. A large language model is an AI system that can understand and generate human-like text.
# 2. It’s trained on large datasets and can process context, create new content, and adapt to diverse tasks.

# Differences noticed:
# The local Ollama model typed out its "thinking" steps before giving the answer, 
# and it followed the two-sentence rule very strictly. 
# The OpenAI model just gave a smooth, short answer right away without showing its work.

# Advantages of running locally:
# 1. No token fees (completely free to use).
# 2. Complete privacy (data never leaves your machine) and offline capability.


# Disadvantages of running locally:
# A 0.6B parameter model is significantly less powerful, prone to more 
# hallucinations, and has less world knowledge than OpenAI's large models.