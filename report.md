### Layer10 report

[cite_start]This system transforms unstructured technical communication into a grounded, queryable memory graph[cite: 13]. [cite_start]I used the `pandas-dev/pandas` GitHub issue tracker as the corpus, providing a realistic mix of structured metadata and messy human dialogue. Below are the core architectural decisions, the math behind them, and how they adapt to a production Layer10 environment[cite: 84].

#### 1. Claim Extraction
[cite_start]To construct a memory graph, raw text must be parsed into typed, grounded objects[cite: 36]. 

I implemented a strict, rule-based regex pipeline for extraction rather than calling an LLM. 

[cite_start]**Why:** An extraction system's primary goal is grounding so every memory item must trace back to evidence reliably. LLMs are prone to abstraction and vagueness, which breaks clear evidence mapping. By using regex to target explicit markers (`#123`, `dup of`, `should/could`), I guarantee that the extracted relation is 100% anchored to the raw substring and thus, purely deterministic.

```python
# Rule 2: Duplicate detection
for pattern in duplicate_patterns:
    matches = re.findall(pattern, text_lower)
    for ref in matches:
        claim = {
            "subject": f"issue_{issue_id}",
            "relation": "DUPLICATE_OF",
            "object": f"issue_{ref}",
            "evidence": text[:200] # Hard pointer to the exact excerpt
        }
```


**Layer10 Adaptation:** Regex is brittle. In a live environment (email, Slack), this module must be swapped for an LLM (e.g., `gpt-4o-mini` with forced JSON schemas). [cite_start]To maintain extraction quality and versioning[cite: 103], the LLM prompt must strictly return string offsets alongside the extracted claim. [cite_start]Furthermore, every extracted claim must carry a `pipeline_hash` (model + prompt version)[cite: 41]. If the schema drifts, we can safely backfill specific extraction runs without nuking the graph.

#### 2. Deduplication: Agglomerative Clustering vs. K-Means
[cite_start]A memory system fails if it stores the exact same fact 100 times[cite: 44]. 

I used `SentenceTransformer` (`all-MiniLM-L6-v2`) to generate 384-dimensional dense vectors for semantic claim deduplication, paired exclusively with Agglomerative Clustering[cite: 52].


**Why:** Keyword matching fails on paraphrases. K-Means clustering is fundamentally broken for this use case because it requires pre-defining the number of clusters. In an expanding knowledge base, the cluster count is an unknown, dynamic variable. Agglomerative clustering operates bottom-up. By setting a hard semantic boundary (`distance_threshold=0.25`), claims are mathematically prohibited from merging if they deviate too far in meaning.

```python
distance_matrix = 1 - similarity_matrix

clustering = AgglomerativeClustering(
    metric="precomputed",
    linkage="average",
    distance_threshold=0.25,
    n_clusters=None # Dynamic cluster discovery
)
labels = clustering.fit_predict(distance_matrix)
```

**Merge Safety & Reversibility:** Automatic merges are risky. [cite_start]Because the raw claims and embeddings are stored immutably, clustering is treated as a materialized view, not a destructive overwrite, ensuring merges remain safe and reversible[cite: 105]. [cite_start]If a merge is audited and deemed unsafe, the threshold is adjusted, or the edge is dropped, and the graph safely recalculates[cite: 55].

#### 3. Canonicalization: Matrix Operations vs. Iterative Loops
Each cluster needs a single representative statement to act as the node in the final graph.

Vectorized numpy operations to find the centroid.

**Why:** Looping through string similarities in Python is computationally expensive. By slicing the existing similarity matrix, I isolate the cluster, mask the diagonal (self-similarity), and calculate the row-wise mean. The `argmax()` of that mean is the exact mathematical centroid of the cluster.

```python
# Isolate the similarity matrix for just this cluster
cluster_sim = similarity_matrix[np.ix_(cluster, cluster)].copy()
np.fill_diagonal(cluster_sim, 0) # Ignore self-similarity

# The claim with highest average similarity to peers becomes canonical
canonical_position = cluster_sim.mean(axis=1).argmax()
canonical_text = rows.iloc[canonical_position]["clean_evidence"]
```

This ensures the canonical claim is the most universally aligned phrasing, aggregating the total support count and mapping back to all related artifact IDs.

#### 4. Graph Architecture: Bipartite Network vs. Strict RDF Triples
[cite_start]The memory must be queryable and maintainable over time[cite: 59].

A bipartite graph utilizing `NetworkX`, featuring Entity nodes and Canonical Claim nodes[cite: 60].



**Why:** Standard Knowledge Graphs (RDF triples: Subject-Predicate-Object) force highly complex human text into overly rigid structures. [cite_start]Instead, I extracted named entities via `spaCy` and connected them directly to the Canonical Claims they participate in. The claim acts as a hyperedge—it holds the semantic truth, the support count, and the evidence pointers[cite: 60]. 

```python
# Add entity nodes
G.add_node(row["entity_id"], label=row["entity_name"], type="entity")

# Add claim nodes containing aggregated support
G.add_node(row["claim_id"], label=row["canonical_claim"], type="claim", support=row["support_count"])

# Link entities to the claims they appear in
G.add_edge(row["entity_id"], row["claim_id"])
```

#### 5. Long-Term Correctness: Static vs. Temporal State
**The Choice:** The current implementation is a static extraction snapshot.

**The Layer10 Adaptation:** Organizational knowledge mutates. [cite_start]Decisions are reversed[cite: 14]. [cite_start]To guarantee long-term correctness, the memory graph must become a temporal graph[cite: 106].
* **Revisions & Conflicts:** Graph edges must adopt `valid_from` and `valid_to` timestamps. If a Jira ticket's decision flips from "approved" to "blocked", we do not overwrite the existing claim. We cap its `valid_to` timestamp and generate a new edge. [cite_start]This prevents rewriting history and explicitly models "it used to be true" versus "it is true now"[cite: 53].
* [cite_start]**Redaction Handling:** If a source artifact is deleted for compliance, the system triggers a cascade delete via the `artifact_id` to handle deletions and redactions safely[cite: 106]. The associated raw claims drop, the cluster centroid recalculates, and if a canonical claim's support count hits zero, the node is tombstoned.
* [cite_start]**Permissions:** Retrieval must be constrained by access to underlying sources[cite: 89]. During vector search retrieval, the system intersects the user's ACL token with the array of `issue_ids` supporting the canonical claim. If the intersection is empty, the claim is silently redacted from the context pack.