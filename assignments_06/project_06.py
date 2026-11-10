# Step 1: Setup
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator


if load_dotenv(find_dotenv()):
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# Explicitly set the chat LLM to avoid the deprecated gpt-3.5-turbo default
Settings.llm = OpenAI(model="gpt-4o-mini")

ROOT_DIR = Path(__file__).resolve().parents[1]

docs_dir = ROOT_DIR / "groundwork_docs"
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# Step 2: Load the Documents
documents = SimpleDirectoryReader(docs_dir).load_data()
# Print how many documents were loaded
print(f"\nTotal documents loaded: {len(documents)}")

# Print the file name of each document using the metadata dictionary
print("Loaded files:")
for doc in documents:
    file_name = doc.metadata.get("file_name", "Unknown")
    print(f"- {file_name}")

# Step 3: Build the Index and Query Engine
index = VectorStoreIndex.from_documents(documents)
print("Index built successfully. Ready to answer questions.")

query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Document Name: {node_with_score.metadata.get('file_name')}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
        print("-" * 30)
# Reflection: all five answers were confident and specific, with no hedging,
# and accurate.
#
# What surprised me: the weekend-hours answer was right even though the
# top chunk was wrong — our_story.txt (0.5162) beat faq.txt (0.4697),
# which actually contains the hours. Scores were low overall, especially
# loyalty (top score only 0.2858), yet the answers still held up. The
# cleanest retrieval was Q4, our_story.txt at 0.7763. 
# The confident tone gives no signal that the retrieval was weak — the
# answer sounds just as certain at a top score of 0.28 as at 0.77. In
# this small case that is not a big risk, since the documents are short
# and easy to check by hand. But with a large collection, where reading
# everything is not practical, that same confident tone would make a
# wrong answer much harder to spot.

# Step 5: Find a Failure
query = "How many visitors do you have every day?"
print(f"Q: {query}")
response = query_engine.query(query)
print("A:", response)

for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Document Name: {node_with_score.metadata.get('file_name')}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
        print("-" * 30)


# What I asked and why I expected it to be hard:
# The question is "How many visitors do you have every day?"
# The groundwork documents contain no information about daily visitor
# numbers, so it should be hard for the model to answer correctly.
# The correct response would be to say that no such information is
# available in the documents.
#
# What went wrong:
# Retrieval failed — no relevant information was returned. The top
# chunk scored only 0.2035 (faq.txt), followed by menu.txt at 0.1782
# and wholesale_catering.txt at 0.1644, and none of them mentions
# visitor numbers. This is expected: the documents contain no such
# information, so there was nothing relevant to retrieve, but the
# retriever still returns its top 3 chunks.
#
# Did the model's tone change when the retrieval failed?
#  Model abstained correctly, but it did so in declarative voice it
# uses for well-grounded answers, with no sign that retrieval was
# weak (top score only 0.2035, versus 0.77 for a good query): 
# tone tells you nothing about whether the answer is grounded in the documents,
#  so the response must be checked against its sources instead of trusted
# on tone alone.
#
# What I would change about the system to improve it:
# I would add faithfulness and relevancy validation, and use the
# results to decide whether to answer at all — to make
# the abstention reliable. 
# A FaithfulnessEvaluator would check whether every claim in the
# answer is supported by the retrieved context and a
# RelevancyEvaluator would check whether the answer actually
# addresses the question. Combined with an explicit instruction
# to answer only from the retrieved context — and a similarity
# threshold below which the system says "this information is not
# in the documents" instead of generating (the top score here was
# only 0.2035, far below the 0.5–0.77 of the good queries) — this
# would make the honest abstention happen every time, instead of
# depending on how the model happens to phrase one particular run.


# Step 6: Reflection
# 1. What LlamaIndex automates, and how much code it took
#    LlamaIndex automates chunking, embedding, and indexing. In my
#    project the core of that took just three lines:
#        documents = SimpleDirectoryReader(...).load_data()
#        index = VectorStoreIndex.from_documents(documents)
#        query_engine = index.as_query_engine(similarity_top_k=3)
#    The first of those three lines alone — VectorStoreIndex.from_documents
#    — does the chunking, the embedding calls, and the indexing.
#    By comparison, the keyword RAG code from the earlier lesson needed
#    a whole hand-written function (tokenising, filtering stopwords,
#    counting overlap, sorting, picking the best document) just to do
#    retrieval — and it still only matched exact words, with no
#    semantics and no generated answer. Writing chunking, embedding,
#    and indexing myself would be far longer still: splitting text with
#    overlap, calling the embedding API in batches, storing the vectors,
#    and implementing cosine-similarity search, plus PDF parsing and
#    error handling — easily 100+ lines. 
#    The trade-off is less control and less visibility into
#    exactly how chunking and retrieval work under the hood.
#
# 2. A different use case where this would add genuine value
#    An internal HR assistant for a mid-size company. Employees
#    constantly ask the same questions — how much parental leave do I
#    get, how do I claim expenses, can I work remotely from another
#    country — and the answers live in a handbook, policy PDFs, and
#    benefit documents that are hundreds of pages long and updated
#    every quarter. A RAG system over those documents would give an
#    instant, grounded answer with the source document named, instead
#    of an employee waiting days for HR to reply, or acting on an
#    outdated answer remembered from a colleague. The value is not
#    just speed: it is that everyone gets the current, written policy,
#    and HR stops spending its time answering the same questions.
#
# 3. One failure mode RAG cannot fully prevent, even when retrieval
#    is working correctly
#    It cannot go beyond its own documents. If a person does not know
#    an answer, they can take time to search other sources, ask a
#    colleague, or point to someone who knows more — and come back
#    with the answer later. This RAG system cannot do that: it has
#    only the indexed documents to work from, with no other source
#    to check and no one to escalate to. So when the answer is not
#    in the documents, its only honest option is to say so. Retrieval
#    working correctly does not fix this, because the problem is not
#    that the wrong chunk was retrieved — it is that the right
#    information was never in the collection at all.





