from dotenv import load_dotenv
import os
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator


env_path = Path(__file__).resolve().parent.parent / ".env"
if load_dotenv(dotenv_path=env_path):
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# Explicitly set the chat LLM to avoid the deprecated gpt-3.5-turbo default
Settings.llm = OpenAI(model="gpt-4o-mini")

# Concepts Question 1
# Scenario A — RAG
# The policy library is too large to fit in a prompt and changes every
# quarter, so retrieve only the relevant passages at query time and
# re-index on each update instead of retraining. This keeps answers
# grounded in current internal policy rather than memorized, possibly
# outdated, text.
#
# Scenario B — Fine-tuning
# 3,000 in-house examples are enough to teach a consistent, rare brand
# voice that prompting with a few examples cannot reliably reproduce.
# The task is style, not retrieving facts, so the voice needs to be
# baked into the model itself.
#
# Scenario C — Prompt engineering
# A single two-page report fits easily in context, so we just paste it with
# the question and skip retrieval infrastructure or training entirely.
# There is no reuse case to justify anything heavier.

# Concepts Question 2
# A confidently wrong answer is more harmful than one that says "I am not sure" because confidence 
# changes how we treat the answer, not just what it says. Much like in a conversation with a real person,
# we read tone as a signal of reliability: a fluent, assertive answer with specific details feels trustworthy, 
# so we tend to accept it and skip verification. An uncertain answer does the opposite — it triggers caution, 
# double-checking, and a second opinion, which is the step that would catch the error.
# Example of problematic sutuation: A hospital patient asks a general-text chatbot about chest pain, 
# and it confidently reassures them it is probably just indigestion and they can wait and rest. 
# Trusting that confident tone, the patient delays emergency care that could have mattered.

# Concepts Question 3
steps = [
    "Extract text from source documents", #  Pull the readable text out of PDFs, docs, or web pages, stripping away formatting and layout.
    "Split text into chunks", # Break long documents into smaller parts
    "Convert text chunks into embeddings", # Turn each chunk into a numeric vector that captures its meaning, so chunks with similar meanings end up close together.
    "Receive the user's query", # Accept the question the user asks, which starts the retrieval side of the pipeline.
    "Embed the user's query", # Convert the query into a vector with the same embedding model so it can be compared directly with the stored chunk vectors.
    "Retrieve the most relevant chunks", # Search the vector database for the chunks whose vectors are closest to the query vector and return the top few.
    "Inject retrieved chunks into the prompt", # Place those chunks into the LLM's prompt as context, alongside the question and an instruction to answer from them.
    "Generate a response from the LLM", # Produce the final answer grounded in the retrieved chunks rather than relying only on what the model memorised in training.
]

# Keyword RAG
import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# Keyword Question 1
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

docs = simple_keyword_retrieval(query=query, documents=documents, verbose=True)
name, content = docs[0]
print(f"Name of the selected document: {name}")
# Python compares tuples element by element, so it sorts mainly by score. If two scores tie, it then compares name, then content.
# loyalty.txt is returned first — not because it's more relevant, but because among the three equally-scored documents, 
# its name sorts highest in descending order.

# Keyword Question 2
query = "Do you have anything without caffeine?"
docs = simple_keyword_retrieval(query=query, documents=documents, verbose=True)
name, content = docs[0]
print(f"Name of the selected document: {name}")

# Which document was selected?- None of the documents was selected
# Whether keyword RAG got this right — and why or why not? - RAG wasn't able to catch similarities between query and documents
# since it relies on the exact word matching(content overlap) without underlying meaning. 
# What kind of retrieval would do better here? - Semantic retrieval can understand the context of the query and search the corresponding meaning in 
# documents.

# Keyword Question 3
# None of the documents was selected. Results didn't surprise. Since the exact word matching technique
# has been used no relevant documents were found.
query = "How do I sign up for rewards?"
docs = simple_keyword_retrieval(query=query, documents=documents, verbose=True)
name, content = docs[0]
print(f"Name of the selected document: {name}")

# Semantic Question 1
# Vector - is the list of numbers that repsents a meaning in the multidimensional space.
# Thus words with similar meaning end up near each other.

# chunk with the highest score (0.85) is the most semantically similar to the question.
# Cosine similarity measures how similar the direction of two vectors is — in other words, how much they point the same way.

# Why can semantic search find a relevant chunk even when none of the exact words from the query appear in the chunk?
# Semantic search doesn't compare words — it compares meanings, represented as vectors.
# An embedding model turns a whole piece of text into a dense vector — a list of numbers that captures what the text is about. 
# The model learned this from huge amounts of text, where words that appear in similar contexts end up with similar vectors. 
# That's the key idea: synonyms and paraphrases live close together in vector space, even if they share no letters.
# So when we embed the query and the chunk with the same model, and then measure cosine similarity between the two vectors, 
# a high score means "these point in the same direction" — i.e. they're about the same thing — regardless of the exact words used.

# Semantic Question 2
# | Feature                    | Keyword RAG                       | Semantic RAG      |
# |----------------------------|-----------------------------------|--------------     |
# | What is compared?          | Exact word overlap                | compares meanings |
# | What is retrieved?         | Full document                     | chunck of text    |
# | Can it handle synonyms?    | No                                | Yes               |
# | Storage format             | Plain text dictionary             | Vector embeddings |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity between two vectors|


documents = SimpleDirectoryReader(
    "../resources/brightleaf_pdfs"
).load_data()

# LlamaIndex Question 1
# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(documents)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]
query_engine = index.as_query_engine(similarity_top_k=3)
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:100]}...")
        print("-" * 30)

# Q1: What employee benefits does BrightLeaf offer?
# 1. Relevance: Partly. The top chunk (0.7667, Introduction) is relevant —
#    it opens with employee well-being. But the other two are weak: the
#    EcoVolt Energy partnership chunk (0.5903) is not about benefits at
#    all, and the Overview chunk (0.5700) is just general company background.
# 2. Tone: Confident and specific — a detailed answer with named providers
#    and numbers (Blue Cross, 401(k) match up to 5%, 12 weeks paid parental
#    leave). No hedging like "based on the context" or "I'm not sure".
# 3. Unexpected retrieval: Yes — the EcoVolt partnership chunk. It has
#    nothing to do with benefits, so it looks like a semantic near-miss.
#    Note: the snippets are truncated, so the specific numbers in the answer
#    can't be fully verified from the snippets shown alone.

# Q2: What are BrightLeaf's security policies?
# 1. Relevance: Mostly yes. The top chunk (0.6824, Network and Data
#    Security) is directly on topic and matches the answer. The other two
#    are not: the Introduction chunk (0.6139) is about employee well-being
#    and the Overview chunk (0.5872) is general company background.
# 2. Tone: Confident and specific — MFA, VPN, 90-day credential rotation,
#    NIST 800-61, ISO 27001, all stated as facts. No hedging like
#    "based on the context" or "I'm not sure".
# 3. Unexpected retrieval: Yes — the same Introduction and Overview chunks
#    as in Q1. They seem to be retrieved for almost any query, probably
#    because they are broad, general text, not because they answer
#    the security question.


# LlamaIndex Question 2
question = "What employee benefits does BrightLeaf offer?"

for sim_top_k in [1, 5]:
    query_engine = index.as_query_engine(similarity_top_k=sim_top_k)

    print(f"\nsimilarity_top_k: {sim_top_k}")
    response = query_engine.query(question)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")

# similarity_top_k comparison — "What employee benefits does BrightLeaf offer?"
#
# How the response changed:
# Barely. With top_k=1, the answer already covered all four areas —
# health/vision/wellness, financial security, parental leave/flexibility,
# and learning/career growth — with the same key facts: Blue Cross,
# $600 wellness reimbursement, 401(k) match up to 5%, 12 weeks paid
# parental leave. With top_k=5, the answer is longer and a little more
# detailed and it switches from a numbered list to prose — but it adds
# no new benefit category and contradicts nothing from the top_k=1 answer.
# The top chunk (0.7668) already contained most of the answer; the extra
# chunks (0.5903, 0.5700, 0.4768, 0.4335) only filled in small details.
#
# Is more retrieved context always better? No. More context helps only
# while the extra chunks are actually relevant. Past that point it adds
# noise — here, generic or unrelated chunks. The goal
# is the most relevant context, not the most context: a small top_k is
# enough when the top chunk is strong, as it was here.


# LlamaIndex Question 3
q = "What are BrightLeaf's plans for preparing mission to the Moon?"

response = query_engine.query(q)

print("Q:", q)
print("A:", response.response)
print("\nRetrieved chunks:")
for i, node_with_score in enumerate(response.source_nodes, 1):
    print(f"\n--- Chunk {i} | score: {node_with_score.score:.4f} ---")
    print(node_with_score.node.get_content())

# Hard query test: "What are BrightLeaf's plans for preparing mission
# to the Moon?"
#
# What I expected: The information is not in the documents at all, so
# retrieval should return only loosely related chunks, and the risk was
# that the model would invent a confident, specific Moon-mission plan
# from them.
#
# What actually happened: The model did the right thing — it abstained
# instead of hallucinating. It said there are no details about a Moon
# mission in the provided information, and listed what the documents
# actually cover (renewable energy, community engagement, partnerships,
# financial performance, employee benefits, solar product specs).
#
# What I would change: Make this correct behaviour reliable rather than
# lucky. 1) Add an explicit instruction to answer only from the retrieved
# context and say "not in the documents" when the answer is absent.
# 2) Add a similarity threshold 
# 3) Consider filtering out very generic chunks

# LlamaIndex Question 3

judge_llm = OpenAI(model="gpt-4o-mini", temperature=0)

faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

def evaluate(q):
    response = query_engine.query(q)

    faith = faithfulness_evaluator.evaluate_response(response=response)
    rel = relevancy_evaluator.evaluate_response(query=q, response=response)

    print(f"Q: {q}")
    print(f"Faithfulness: {faith.score} (passing={faith.passing})")
    print(f"  Feedback: {faith.feedback}")
    print(f"Relevancy: {rel.score} (passing={rel.passing})")
    print(f"  Feedback: {rel.feedback}")
    print("-" * 60)

evaluate("What employee benefits does BrightLeaf offer?")
evaluate("What are BrightLeaf's plans for preparing mission to the Moon?")
# 1. Faithfulness 1.0 means every claim in the answer is supported by
#    the retrieved chunks — nothing invented. 0.0 means at least one claim
#    is not supported by the context, i.e. a hallucination.
#
# 2. Relevancy measures whether the response actually addresses the
#    question asked. It differs from faithfulness because a response can
#    be faithful but irrelevant (true facts that do not answer the
#    question), or relevant but unfaithful (it answers directly, but with
#    invented facts). Faithfulness = grounded in the sources; relevancy
#    = useful for the question.
#
# 3. Did the scores change between the two queries? Why?
#    Expected: benefits should score 1.0 on both — grounded in the
#    benefits chunk (0.7668) and directly answering the question. For
#    the Moon query, faithfulness should stay at 1.0, because the answer
#    ("there are no details...") invents nothing. Relevancy is the one
#    likely to drop, because the answer gives no Moon-mission plans —
#    it cannot, since they are not in the documents. A low relevancy
#    score here does not mean hallucination; it means the question was
#    unanswerableю
#
# 4. LLM-as-a-judge means using another LLM (here gpt-4o-mini) as the
#    evaluator, prompted with the question, response, and retrieved
#    context, to judge faithfulness and relevancy. It is used for RAG
#    because there is no single correct answer string — a good answer
#    can be paraphrased in many valid ways, so exact-match accuracy or
#    word-overlap metrics would mark correct answers wrong and cannot
#    detect hallucination. A judge LLM assesses meaning instead of
#    wording.
