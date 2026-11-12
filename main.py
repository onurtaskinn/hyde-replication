"""HyDE vs. plain dense retrieval, using only OpenAI models.

    plain:  query -> embedding -> nearest documents
    HyDE:   query -> LLM writes N hypothetical documents -> embeddings
            -> average them together with the query embedding (Eq. 8 in the paper) -> nearest documents

Paper: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels", ACL 2023.

Setup:  pip install -r requirements.txt
        set the OPENAI_API_KEY environment variable
Run:    python main.py                                                       (default embedding model)
        python main.py text-embedding-3-small text-embedding-ada-002         (repeat with older models)

Everything expensive is cached in ./cache. The hypothetical documents are generated once and reused,
so every embedding model is tested with exactly the same generated text.
"""
import csv, io, json, math, os, sys, zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------- settings ----------------
DATASET      = "scifact"        # BEIR dataset: "scifact" (used in the paper) or "nfcorpus" (very short queries)
EMBED_MODELS = sys.argv[1:] or ["text-embedding-3-large"]
LLM_MODEL    = "gpt-6-luna"     # writes the hypothetical documents
N_HYPO       = 8                # hypothetical documents per query (the official HyDE code uses 8)
MAX_QUERIES = None          # 20 = quick first try. Set to None to use all test queries.

PROMPTS = {
    # from Appendix A.1 of the paper
    "scifact": "Please write a scientific paper passage to support or refute the claim\nClaim: {}\nPassage:",
    # not from the paper (it does not evaluate NFCorpus); written in the same style
    "nfcorpus": "Please write a medical scientific paper passage about the topic\nTopic: {}\nPassage:",
}

client = OpenAI(max_retries=8)  # reads OPENAI_API_KEY; retries automatically when rate-limited


# ---------------- data ----------------
def load_beir(name):
    """Returns docs {id: text}, queries {id: text}, qrels {query id: {relevant doc id: relevance}}."""
    if not os.path.isdir(f"data/{name}"):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
        print("downloading", url)
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        zipfile.ZipFile(io.BytesIO(response.content)).extractall("data")
    docs, queries, qrels = {}, {}, {}
    for line in open(f"data/{name}/corpus.jsonl", encoding="utf-8"):
        d = json.loads(line)
        docs[d["_id"]] = (d.get("title", "") + " " + d["text"]).strip()
    for line in open(f"data/{name}/queries.jsonl", encoding="utf-8"):
        q = json.loads(line)
        queries[q["_id"]] = q["text"]
    for line in list(open(f"data/{name}/qrels/test.tsv", encoding="utf-8"))[1:]:   # skip the header line
        query_id, doc_id, relevance = line.split()
        if int(relevance) > 0:
            qrels.setdefault(query_id, {})[doc_id] = int(relevance)
    return docs, queries, qrels


# ---------------- OpenAI calls ----------------
def write_hypothetical_docs(query):
    """Asks the LLM N_HYPO times for a passage that answers the query. The passages may contain made-up facts."""
    texts = []
    for _ in range(N_HYPO):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": PROMPTS[DATASET].format(query)}],
            temperature=0.7,            # as in the paper
            max_completion_tokens=512,  # as in the official HyDE code
            reasoning_effort="none",    # plain generation; temperature only applies when reasoning is off
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError(f"{LLM_MODEL} returned an empty passage")
        texts.append(text)
    return texts


def embed(texts, model):
    """Returns one embedding per text as a matrix. OpenAI embeddings have length 1, so dot product = cosine."""
    vectors = []
    for i in range(0, len(texts), 64):
        batch = [t.replace("\n", " ")[:16000] for t in texts[i:i + 64]]     # stay below the input token limit
        response = client.embeddings.create(model=model, input=batch)
        vectors += [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        print(f"\r  embedded {len(vectors)}/{len(texts)} texts", end="", flush=True)
    print()
    return np.array(vectors, dtype=np.float32)


# ---------------- retrieval and metrics ----------------
def evaluate(query_vecs, doc_vecs):
    """Ranks all documents for every query and returns nDCG@10 and Recall@100 per query."""
    ranking = np.argsort(-(query_vecs @ doc_vecs.T), axis=1)[:, :100]      # exact search, best 100 documents
    ndcg, recall = [], []
    for query_id, row in zip(query_ids, ranking):
        relevant = qrels[query_id]
        gains = [relevant.get(doc_ids[j], 0) for j in row]
        dcg = sum(g / math.log2(rank + 2) for rank, g in enumerate(gains[:10]))
        best = sum(g / math.log2(rank + 2) for rank, g in enumerate(sorted(relevant.values(), reverse=True)[:10]))
        ndcg.append(dcg / best)
        recall.append(sum(g > 0 for g in gains) / len(relevant))
    return np.array(ndcg), np.array(recall)


# ---------------- experiment ----------------
os.makedirs("cache", exist_ok=True)
docs, queries, qrels = load_beir(DATASET)
doc_ids = list(docs)
query_ids = sorted(qrels)[:MAX_QUERIES]
query_words = np.array([len(queries[q].split()) for q in query_ids])
short = query_words <= int(np.median(query_words))       # the shorter half of the queries

# Step 1: hypothetical documents (cached)
hypo_file = f"cache/{DATASET}_{LLM_MODEL}_hypothetical_docs.json"
hypo = json.load(open(hypo_file)) if os.path.exists(hypo_file) else {}
todo = [q for q in query_ids if len(hypo.get(q, [])) < N_HYPO]
with ThreadPoolExecutor(16) as pool:
    for query_id, texts in zip(todo, pool.map(write_hypothetical_docs, [queries[q] for q in todo])):
        hypo[query_id] = texts
        json.dump(hypo, open(hypo_file, "w"), indent=1)
        print(f"\r  hypothetical documents written for {len(hypo)} queries", end="", flush=True)
hypo_texts = [text for q in query_ids for text in hypo[q][:N_HYPO]]

# Step 2: for each embedding model, search with the query vector and with the HyDE vector
table = []
for model in EMBED_MODELS:
    print(f"\n{model}")
    doc_file = f"cache/{DATASET}_{model}_docs.npy"
    if not os.path.exists(doc_file):
        np.save(doc_file, embed(list(docs.values()), model))
    doc_vecs = np.load(doc_file)
    query_vecs = embed([queries[q] for q in query_ids], model)
    hypo_vecs = embed(hypo_texts, model).reshape(len(query_ids), N_HYPO, -1)
    hyde_vecs = (query_vecs + hypo_vecs.sum(axis=1)) / (N_HYPO + 1)        # Eq. 8

    ndcg_query, recall_query = evaluate(query_vecs, doc_vecs)
    ndcg_hyde, recall_hyde = evaluate(hyde_vecs, doc_vecs)

    # Is the difference real or noise? Resample the queries 10,000 times (paired bootstrap).
    diff = ndcg_hyde - ndcg_query
    resampled_means = np.random.default_rng(0).choice(diff, size=(10000, len(diff))).mean(axis=1)
    low, high = np.percentile(resampled_means, [2.5, 97.5])

    confidence = f"[{100 * low:+.1f}, {100 * high:+.1f}]"
    counts = f"{(diff > 0).sum()}/{(diff == 0).sum()}/{(diff < 0).sum()}"
    table.append(f"{model:<24}{100 * ndcg_query.mean():>9.1f}{100 * ndcg_hyde.mean():>9.1f}{100 * diff.mean():>+8.1f}"
                 f"   {confidence:<17}{counts:<20}{100 * recall_query.mean():>9.1f}{100 * recall_hyde.mean():>9.1f}"
                 f"{100 * diff[short].mean():>+12.1f}{100 * diff[~short].mean():>+12.1f}")

    with open(f"results_{DATASET}_{model}.csv", "w", newline="", encoding="utf-8") as f:   # per-query scores
        rows = csv.writer(f)
        rows.writerow(["query_id", "query", "ndcg10_query", "ndcg10_hyde", "recall100_query", "recall100_hyde"])
        for i, q in enumerate(query_ids):
            rows.writerow([q, queries[q], ndcg_query[i], ndcg_hyde[i], recall_query[i], recall_hyde[i]])

# ---------------- report ----------------
print(f"\n{DATASET}: {len(query_ids)} queries, {len(doc_ids)} documents, LLM = {LLM_MODEL}, N = {N_HYPO}")
print(f"average query length: {query_words.mean():.1f} words; 'short' = at most {int(np.median(query_words))} words\n")
print(f"{'':<24}{'nDCG@10':>9}{'nDCG@10':>9}{'':>8}   {'95% CI':<17}{'HyDE better/':<20}{'Recall@100':>9}{'Recall@100':>11}{'diff':>10}{'diff':>12}")
print(f"{'embedding model':<24}{'query':>9}{'HyDE':>9}{'diff':>8}   {'of diff':<17}{'same/worse':<20}{'query':>9}{'HyDE':>11}{'short q.':>10}{'long q.':>12}")
print("\n".join(table))
print("\nIf the 95% CI of diff contains 0, the difference cannot be told apart from noise on these queries.")
if MAX_QUERIES:
    print(f"NOTE: only the first {MAX_QUERIES} test queries were used. Set MAX_QUERIES = None for the real run.")