# Datacreek

Tool for generating high-quality synthetic datasets to fine-tune LLMs.

Generate reasoning traces and QA pairs and save them to common fine-tuning formats.

> [Checkout our guide on using the tool to unlock task-specific reasoning in Llama-3 family](https://github.com/meta-llama/datacreek/tree/main/use-cases/adding_reasoning_to_llama_3)

# What does Datacreek offer? 

Fine-Tuning Large Language Models is easy. There are many mature tools that you can use to fine-tune Llama model family using various post-training techniques.

### Why target data preparation?

Multiple tools support standardized formats. However, most of the times your dataset is not structured in "user", "assistant" threads or in a certain format that plays well with a fine-tuning packages. 

This toolkit simplifies the journey of:

- Using a LLM (vLLM or any local/external API endpoint) to generate examples
- Converting your existing files to fine-tuning friendly formats
- Parsing additional formats like images (via [unstructured](https://github.com/Unstructured-IO/unstructured)) and audio files (transcribed with SpeechRecognition + pocketsphinx). Extracted text from these sources is inserted in the knowledge graph while preserving the original file path
- `unstructured` is used to parse PDF, DOCX, PPTX, HTML and image files, ensuring consistent handling across formats
- Files are partitioned with `unstructured` so text and images become individual elements linked in the graph
- Text is cleaned with `unstructured` before being inserted into the knowledge graph
- Images are captioned with BLIP and stored with the caption as `alt_text`
- Audio files are transcribed via Whisper and linked back to the originating chunk
- Quantities are converted to SI units when `quantulum3` and `pint` are installed
- Optionally extracting named entities and standalone facts during ingestion
- Advanced chunking options including semantic, contextual and summarized splitting
- Creating synthetic datasets
- Extracting standalone facts into a knowledge graph
- Supporting various formats of post-training fine-tuning

## Workflow Overview

1. **Ingestion** – parse documents or URLs from multiple sources, clean the text and store everything in a knowledge graph.
2. **Knowledge graph operations** – deduplicate, resolve entities and link related chunks to improve data quality.
3. **Dataset generation** – choose a dataset type and training goal. The application selects the appropriate pipeline and formats the output accordingly.
4. **Initial curation** – a first filter removes low quality results.
5. **Dataset cleanup** – additional operations can be applied interactively to refine the dataset.
6. **Export** – download locally or upload to services like Hugging Face in the desired format.

| Stage | Options | Purpose |
|-------|---------|---------|
| **Ingestion** | Multiple file formats (PDF, DOCX, PPTX, TXT), web pages and YouTube URLs. Automatic cleaning and optional entity extraction. | Produce clean text chunks stored in the knowledge graph. |
| **Knowledge graph ops** | Helpers like `deduplicate_chunks`, `resolve_entities` and various linking utilities. | Improve quality and connectivity of the graph. |
| **Dataset generation** | Choose dataset type, training goal and output format. | The correct pipeline generates samples and formats them as requested. |
| **Curation** | Configure a quality threshold and batch size. | Drop obviously bad samples directly after generation. |
| **Dataset cleanup** | Reuse graph helper functions plus formatting utilities. | Finalize the dataset before exporting. |
| **Export** | Local download or HF upload in JSONL, Alpaca, ChatML or OpenAI FT style. | Deliver the dataset in the desired place and format. |

### Step Reference

Below is a quick overview of the main options and operations exposed at each stage.

**Ingestion**

- `extract_entities`, `extract_facts` – add entity and fact nodes while parsing
- `high_res`, `ocr` – improved PDF and image handling
- `chunk_method` – `basic`, `sliding`, `semantic`, `contextual` or `summary`

**Knowledge graph operations**

- `deduplicate_chunks()` – drop identical chunks
- `resolve_entities()` – merge aliases into canonical entities
- `link_*()` – connect nodes that mention the same entities
- `prune_sources([...])` – remove unwanted data from the graph
- `compute_graph_embeddings()` – build Node2Vec embeddings
- `mark_conflicting_facts()` – flag contradictory statements

**Dataset generation**

- `dataset_type` – qa, cot, kg, vqa, pref_pair…
- `training_goal` – SFT, DPO, PPO, etc.
- `fmt` – jsonl, alpaca, chatml, openai-ft

**Curation**

- `threshold` – minimum quality rating
- `batch_size` – number of pairs rated together
- `temperature` – sampling temperature for the rating prompt

**Dataset cleanup**

- `clean_chunks()` – normalize whitespace and remove markup
- `normalize_date_fields()` – standardize date attributes
- `deduplicate_pairs()` – remove near-duplicate examples

**Export**

- `fmt` – choose the output format
- `repo` – optionally push to a Hugging Face repo

### SaaS Pipeline Summary

Below is a concise walkthrough of the SaaS flow and which helpers implement each
step:

1. **Multimodal ingestion** – functions in `ingest.py` rely on
   `unstructured` to partition documents and on BLIP/Whisper to caption images
   and transcribe audio (`partition_pdf`, `partition_html`, `caption_image`,
   `transcribe_audio`).
2. **Atomic splitting** – `molecule_from_atoms()` groups contiguous elements and
   preserves relations such as `NEXT` or `CAPTION_OF`.
3. **Knowledge graph build** – `KnowledgeGraph.add_document()` and
   `add_chunk()` insert nodes and edges, while linking helpers (e.g.
   `link_chunks_by_entity`) establish semantics.
4. **Neo4j quality checks** – `DatasetBuilder.gds_quality_check()` runs
   `wcc`, `triangleCount` and `nodeSimilarity` to remove duplicates and weak
   links.
5. **Fractalization & embeddings** – `build_mdl_hierarchy()` performs box
   covering and `compute_graph_embeddings()` materializes Node2Vec vectors.
6. **Topological Perception Layer** – `optimize_topology()` minimizes
   bottleneck distance and can inject NetGAN/GraphRNN edges when needed.
7. **Semantic perception** – `apply_perception()` modifies node text and now
   automatically triggers `node_similarity()` to flag near duplicates.
8. **Export & datasets** – `run_generation_pipeline()` produces QA pairs or
   other dataset types which are curated via `curate.py` and saved through
   `save_as.py`.

## Fractal metrics and confidence

Knowledge graph utilities provide MDL-guided box covering to assign `fractal_level` annotations. QA generation records a `confidence` score computed by searching up to three hops between subject and object.
Useful helpers:
- `build_mdl_hierarchy` returns successive coarse graphs until the description length increases.
- `annotate_mdl_levels` tags each node with its fractal level based on that hierarchy.
- `fact_confidence` searches up to three hops to rate the reliability of a statement.
- `DatasetBuilder.export_prompts(auto_fractal=True)` automatically annotates
  nodes with fractal levels using the MDL hierarchy when none are present.
- Generated QA pairs include a `confidence` field so downstream
  applications can score factual reliability directly.



# How does Datacreek offer it?

The toolkit exposes a REST API that mirrors the main data preparation steps. All
operations are asynchronous and keyed by users:

- `/tasks/ingest` for converting raw files to text
- `/tasks/generate` for creating datasets
- `/tasks/curate` for quality filtering
- `/tasks/save` for exporting in common fine-tuning formats
- `/datasets` to manage generated datasets

 All behaviour is driven from a YAML configuration file that you can override with your own values.



### Installation

#### From PyPI

```bash
# Create a new environment

conda create -n synthetic-data python=3.10 

conda activate synthetic-data

pip install datacreek
```

Optional helpers such as quantity normalization during text cleanup require
additional packages:

```bash
pip install quantulum3 pint
```

#### (Alternatively) From Source

```bash
git clone https://github.com/meta-llama/datacreek.git
cd datacreek
pip install -e .
pip install pre-commit
pre-commit install
```

### 1. Tool Setup

The application persists all state in Redis and Neo4j so no local data
directories are required. User and dataset records are cached in Redis for
quick access. Ensure both services are running and accessible via the
configuration file. If you plan to use the bundled vLLM helpers
start the server as shown below:

```bash
# Start vLLM server
# Note you will need to grab your HF Authentication from: https://huggingface.co/settings/tokens
vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000
```

### 2. Usage

Start the REST API server and interact with the endpoints.

All operations run asynchronously. Use `/tasks/ingest`, `/tasks/generate`,
`/tasks/curate` and `/tasks/save` to launch long running processes in the
background. Each request returns a `task_id` which can be polled via
`/tasks/{task_id}`.

Tasks are executed by [Celery](https://docs.celeryq.dev/). By default an in-memory
broker is used, but in production you should set `CELERY_BROKER_URL` and
`CELERY_RESULT_BACKEND` to a Redis or RabbitMQ instance.

Datasets can be managed through `/datasets` (create, list, update, delete and
download). Each dataset records its history in Redis. Retrieve the current
progress via `/datasets/<name>/progress` and past events via
`/datasets/<name>/history`. Generation runs are versioned so you can review each
attempt. List them via `/datasets/<name>/versions`, fetch a single run with
`/datasets/<name>/versions/{n}` and remove unwanted entries with
`DELETE /datasets/<name>/versions/{n}`. Each version stores the parameters and
summary information for that generation. Every request must include an
`X-API-Key` header issued when creating a user via `/users`.
Datasets are persisted in Redis and Neo4j only. Use `DatasetBuilder.to_redis` and `save_neo4j` for persistence.
## Configuration

The toolkit uses a YAML configuration file (default: `configs/config.yaml`).
Database connection settings can be provided either through the
`DATABASE_URL` environment variable or a `database.url` entry in the YAML
file. By default a local SQLite file `datacreek.db` is used, but you can
point this to any SQLAlchemy compatible database.

### Database initialization

Run `python -m datacreek.cli init-db` to create the tables before starting the
server if they do not already exist. The API container executes this step on
startup so the database is ready when the services come online.
The command line interface is reserved for such maintenance operations (database
setup, test runs) and is not intended for dataset generation.

### Development build

The stack can be used directly from source when hacking on the project. Install
the Python package in editable mode and set up pre-commit hooks:

```bash
git clone https://github.com/meta-llama/datacreek.git
cd datacreek
pip install -e .
pip install pre-commit
pre-commit install
```

The front-end lives in `frontend` and can be started separately for a faster
iteration loop:

```bash
cd frontend
npm install
npm run dev
```

This will watch for file changes and serve the interface on
`http://localhost:5173` while the API runs on port 8000.

### Starting the stack with Docker

Use the provided `docker-compose.yml` to launch the API, Celery worker,
Redis, Neo4j and the front-end:

The stack consists of six main services:

- **api** – FastAPI server exposing ingestion, generation, curation and export
  endpoints.
- **worker** – Celery background worker executing long-running tasks.
- **redis** – message broker and task result backend.
- **neo4j** – knowledge graph database.
- **backend** – Flask web interface.
- **frontend** – React web application built with Vite.

```bash
./scripts/start_services.sh
```

If no `.env` file exists the script will copy `.env.example` so you only
need to adjust values for production. The default configuration stores
the SQLite database and generated datasets in `./data` which is mounted
inside the containers.

The API will be available on `http://localhost:8000` and the Flask backend on
`http://localhost:5000` while the front-end is served on `http://localhost:3000`.
Redis listens on `6379` and Neo4j exposes `7474` and `7687`.

To rebuild the images after modifying the code simply run:

```bash
docker compose build
```

### Deployment

Use the `scripts/deploy.sh` helper to update a remote host. The CI pipeline
builds container images for the API, worker and front-end and pushes them to
GitHub Container Registry. On deployment the remote host pulls the latest
images defined in `.env` and restarts the stack. Export `DEPLOY_HOST`,
`DEPLOY_USER`, `DEPLOY_KEY` and `DEPLOY_PATH` before executing the script.
Docker Compose automatically loads variables from an optional `.env` file
located next to `docker-compose.yml`:

```bash
export DEPLOY_HOST=example.com
export DEPLOY_USER=ubuntu
export DEPLOY_PATH=/opt/datacreek
export DEPLOY_KEY=~/.ssh/id_rsa
scripts/deploy.sh
```

Environment variables can be configured via a `.env` file. See
`.env.example` for defaults.

At a minimum, set `NEO4J_URI`, `NEO4J_USER` and `NEO4J_PASSWORD` so the
API can reach the Neo4j instance. `DATABASE_URL` defaults to storing the
SQLite file inside the mounted `./data` directory. `IMAGE_NAME` and
`FRONTEND_IMAGE_NAME` define the container images pulled during deployment.

Before running the deployment script ensure your CI pipeline built and pushed
the container images to the registry referenced by `IMAGE_NAME` and
`FRONTEND_IMAGE_NAME`. The remote host only pulls and restarts the services;
all configuration is controlled by the `.env` file copied alongside
`docker-compose.yml`.

You can override any value by providing a custom YAML file to the server.

```yaml
# Example configuration using vLLM
llm:
  provider: "vllm"

vllm:
  api_base: "http://localhost:8000/v1"
  model: "meta-llama/Llama-3.3-70B-Instruct"

generation:
  temperature: 0.7
  chunk_size: 4000
  chunk_method: sliding  # basic|sliding|semantic|contextual|summary
  retrieval_top_k: 3
  num_pairs: 25

curate:
  threshold: 7.0
  batch_size: 8
```

or using an API endpoint:

```yaml
# Example configuration using the llama API
llm:
  provider: "api-endpoint"

api-endpoint:
  api_base: "https://api.llama.com/v1"
  api_key: "llama-api-key"
  model: "Llama-4-Maverick-17B-128E-Instruct-FP8"
```

### Customizing Configuration

Create a custom configuration file and pass it via the `X-Config-Path` header:

The `generation` section now exposes advanced chunking and retrieval options:

```
  generation:
    chunk_method: semantic  # or "sliding" for fixed windows, "contextual" or "summary" for prefixed chunks
  similarity_drop: 0.25   # threshold when using semantic splitting
    retrieval_top_k: 5      # number of chunks fetched using embeddings
```

Most options can also be overridden with environment variables. For example set
`GEN_TEMPERATURE=0.5` to change the default temperature.

| Variable | Description |
|----------|-------------|
| `GEN_TEMPERATURE` | Default sampling temperature |
| `GEN_TOP_P` | Default top-p value |
| `GEN_CHUNK_SIZE` | Override chunk size |
| `GEN_OVERLAP` | Override overlap between chunks |
| `GEN_CHUNK_METHOD` | Chunking method (`basic`, `sliding`, `semantic, contextual, summary`) |
| `GEN_SIMILARITY_DROP` | Similarity threshold for semantic splits |
| `GEN_RETRIEVAL_TOP_K` | Top-k chunks retrieved |
| `GEN_NUM_PAIRS` | Default number of QA pairs |
| `GEN_NUM_COT_EXAMPLES` | Default number of CoT examples |
| `GEN_NUM_COT_ENHANCE_EXAMPLES` | Max conversations to enhance |
| `GEN_BATCH_SIZE` | Batch size for generation |
| `GEN_MAX_TOKENS` | Maximum tokens in completions |
| `GEN_FREQUENCY_PENALTY` | Sampling frequency penalty |
| `GEN_PRESENCE_PENALTY` | Sampling presence penalty |
| `GEN_SUMMARY_TEMPERATURE` | Temperature for summaries |
| `GEN_SUMMARY_MAX_TOKENS` | Max tokens for summaries |
| `SDK_VERBOSE` | Enable detailed logs |
| `DATACREEK_CONFIG` | Path to YAML configuration file |
| `DATACREEK_PIPELINES_CONFIG` | Path to pipeline definitions |
| `REDIS_HOST` | Redis hostname |
| `REDIS_PORT` | Redis port |
| `S3_BUCKET` | Upload dataset exports to this bucket |
| `S3_PREFIX` | Optional prefix for uploaded objects |
| `S3_ENDPOINT_URL` | Custom S3-compatible endpoint |
| `AWS_ACCESS_KEY_ID` | AWS credential for S3 uploads |
| `AWS_SECRET_ACCESS_KEY` | AWS secret for S3 uploads |
| `USE_REDIS_GRAPH` | Enable RedisGraph integration |
| `DATASET_MAX_VERSIONS` | Maximum versions kept per dataset |
| `SDK_DEBUG` | Log full model responses |

### Model Profiles

Define multiple model profiles in your configuration to easily switch between
providers or models:

```yaml
models:
  local-llama:
    provider: vllm
    api_base: "http://localhost:8000/v1"
    model: "meta-llama/Llama-3.3-70B-Instruct"
  llama-api:
    provider: api-endpoint
    api_base: "https://api.llama.com/v1"
    model: "Llama-4-Maverick-17B-128E-Instruct-FP8"
```

Select a profile by passing `profile` when calling the API or constructing an
``LLMClient``.

You can override prompt templates per request by sending a `prompts` object to
`/tasks/generate`:

```bash
curl -X POST localhost:8000/tasks/generate \
     -H "Content-Type: application/json" \
     -H "X-API-Key: <key>" \
     -d '{"src_id": 1, "prompts": {"qa_generation": "Ask three short questions"}}'
```

```bash
curl -X POST localhost:8000/tasks/ingest \
     -H "X-Config-Path: custom_config.yaml" \
     -H "X-API-Key: <key>" \
     -d "path=docs/paper.pdf"
```

## Examples

### Processing a PDF Document

```bash
# Ingest PDF with entity extraction
curl -X POST localhost:8000/tasks/ingest \
     -H "Content-Type: application/json" \
     -H "X-API-Key: <key>" \
     -d '{"path": "research_paper.pdf", "extract_entities": true}'  # unstructured parsing enabled by default

# Generate QA pairs (assuming source ID 1)
curl -X POST localhost:8000/tasks/generate -d "src_id=1&num_pairs=30" -H "X-API-Key: <key>"

# Override the QA generation prompt for this request
curl -X POST localhost:8000/tasks/generate \
     -H "Content-Type: application/json" \
     -H "X-API-Key: <key>" \
     -d '{"src_id": 1, "prompts": {"qa_generation": "Ask three short questions"}}'

# Curate data (dataset ID 1)
curl -X POST localhost:8000/tasks/curate -d "ds_id=1&threshold=8.5" -H "X-API-Key: <key>"

# Save in OpenAI fine-tuning format
curl -X POST localhost:8000/tasks/save -d "ds_id=1&fmt=jsonl" -H "X-API-Key: <key>"
```

### Processing a YouTube Video

```bash
# Extract transcript and generate QA pairs
curl -X POST localhost:8000/tasks/ingest -d "path=https://www.youtube.com/watch?v=dQw4w9WgXcQ" -H "X-API-Key: <key>"
curl -X POST localhost:8000/tasks/generate -d "src_id=1" -H "X-API-Key: <key>"
```

### Processing Multiple Files

```bash
# Bash script to process multiple files
for file in /path/to/pdfs/*.pdf; do
  filename=$(basename "$file" .pdf)

  curl -X POST localhost:8000/tasks/ingest -d "path=$file" -H "X-API-Key: <key>"
  curl -X POST localhost:8000/tasks/generate -d "src_id=1&num_pairs=20" -H "X-API-Key: <key>"
  curl -X POST localhost:8000/tasks/curate -d "ds_id=1&threshold=7.5" -H "X-API-Key: <key>"
  curl -X POST localhost:8000/tasks/save -d "ds_id=1&fmt=chatml" -H "X-API-Key: <key>"
done
```

## Advanced Usage

### Custom Prompt Templates

Edit the `prompts` section in your configuration file to customize generation behavior:

```yaml
prompts:
  qa_generation: |
    You are creating question-answer pairs for fine-tuning a legal assistant.
    Focus on technical legal concepts, precedents, and statutory interpretation.
    
    Below is a chunk of text about: {summary}...
    
    Create {num_pairs} high-quality question-answer pairs based ONLY on this text.
    
    Return ONLY valid JSON formatted as:
    [
      {
        "question": "Detailed legal question?",
        "answer": "Precise legal answer."
      },
      ...
    ]
    
    Text:
    ---
    {text}
    ---
```

Each dataset you create owns its own knowledge graph. During ingestion the
selected documents are inserted into this graph and linked to their original
source.  Generation steps query this cleaned graph instead of the raw files.
The graph exposes simple search helpers so you can explore the content:

```python
from datacreek import DatasetBuilder, DatasetType, KnowledgeGraph

ds = DatasetBuilder(DatasetType.QA, name="example")
ds.add_document("doc1", source="paper.pdf")
ds.add_chunk("doc1", "c1", "hello world")
ds.add_image("doc1", "img0", "pages/img0.png", page=1)
print(ds.search("hello"))  # ["c1"]
print(ds.search_documents("paper"))  # ["doc1"]
print(ds.get_chunks_for_document("doc1"))  # ["c1"]
print(ds.get_images_for_document("doc1"))  # ["img0"]
print(ds.get_document_for_chunk("c1"))  # "doc1"
ds.graph.index.build()
print(ds.graph.search_embeddings("hello", k=1))  # ["c1"]
print(ds.graph.search_hybrid("hello"))  # ["c1"]
print(ds.search_hybrid("paper", node_type="document"))  # ["doc1"]
print(ds.search_with_links("hello", hops=1))  # ["c1", "c2", ...]
print(ds.search_with_links_data("hello", hops=1)[0])  # includes depth and path
ds.link_similar_chunks()         # connect semantically close chunks
ds.update_embeddings()           # materialize embeddings on graph nodes
ds.extract_facts()               # populate fact nodes using an LLM or regex
fact_id = ds.get_facts_for_chunk("c1")[0]
print(ds.get_documents_for_fact(fact_id))  # ["doc1"]
print(ds.find_conflicting_facts())  # check for conflicting information

# After ingestion you can further enrich the graph:
ds.consolidate_schema()        # normalize labels
ds.detect_communities()        # cluster chunks
ds.summarize_communities()     # generate simple summaries
ds.detect_entity_groups()      # cluster entities
ds.summarize_entity_groups()   # summarize entity groups
ds.score_trust()               # compute naive trust scores
# provenance is stored on edges so you can track sources

Files can also be ingested directly via the REST API:

```bash
curl -X POST localhost:8000/api/datasets/example/ingest \
     -H "Content-Type: application/json" \
     -H "X-API-Key: <key>" \
     -d '{"path": "paper.pdf", "high_res": true, "ocr": true,
          "extract_entities": true, "extract_facts": true}'  # unstructured is default
```

You can then enrich and query the graph via the API:

```bash
curl -X POST localhost:8000/api/datasets/example/similarity -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/search_hybrid?q=hello -H "X-API-Key: <key>"
curl -X GET  "localhost:8000/api/datasets/example/search_hybrid?q=paper&type=document" -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/search_links?q=hello\&hops=1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/similar_chunks?cid=c1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/similar_chunks_data?cid=c1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/chunk_neighbors -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/chunk_neighbors_data -H "X-API-Key: <key>"
The chunk_neighbors_data endpoint returns similarity scores, neighbor text, and the owning document for each chunk.
curl -X POST localhost:8000/api/datasets/example/section_similarity -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/document_similarity -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/chunk_context?cid=c1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/similar_sections?sid=s1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/similar_documents?did=doc1 -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/extract_facts -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/extract_entities -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/conflicts -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/mark_conflicts -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/prune -H "X-API-Key: <key>" \
     -d '{"sources": ["bad_source"]}'
curl -X POST localhost:8000/api/datasets/example/deduplicate -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/clean_chunks -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/normalize_dates -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/co_mentions -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/doc_co_mentions -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/section_co_mentions -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/graph_embeddings -H "X-API-Key: <key>" \
     -d '{"dimensions": 32, "walk_length": 5, "num_walks": 20, "seed": 42}'
curl -X GET  localhost:8000/api/datasets/example/chunk_document?cid=c1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/chunk_page?cid=c1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/section_page?sid=s1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/fact_documents?fid=f1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/fact_pages?fid=f1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/entity_pages?eid=A -H "X-API-Key: <key>"
```

Provide either a local path or a URL directly. Ingestion no longer relies on
configured directories and simply reads the resource you specify.

# Clone a dataset to experiment with different cleaning steps
ds_copy = ds.clone(name="copy")


## Dataset Generation Pipelines

After ingestion, parsed content is placed into a knowledge graph. Generation
pipelines operate on this graph and are specialized for different training goals.

| Dataset type | Compatible trainings |
|--------------|---------------------|
| `qa`         | SFT, DPO, ORPO, DPO+SFT, PPO, RRHF, RLAIF, GRPO |
| `cot`        | SFT, DPO, ORPO, DPO+SFT, RRHF |
| `vqa`        | SFT |
| `text`       | CPT |
| `kg`         | SFT, DPO, ORPO, DPO+SFT, PPO, RRHF, RLAIF, GRPO |
| `pref_pair`  | PPO, DPO, ORPO, DPO+SFT, RLAIF |
| `pref_list`  | GRPO, RRHF |
| `tool`       | SFT, DPO, ORPO, DPO+SFT, PPO, RRHF, RLAIF, GRPO |
| `conversation` | SFT, DPO, ORPO, DPO+SFT, PPO, RRHF, RLAIF, GRPO |
| `multi_tool` | SFT, DPO, ORPO, DPO+SFT, PPO, RRHF, RLAIF, GRPO |

### Training-Specific Pipelines

Depending on the training objective, different dataset types are available.

| Training goal | Supported dataset types |
|---------------|-----------------------|
| SFT | qa, cot, vqa, kg, tool, conversation, multi_tool |
| DPO | qa, cot, kg, pref_pair, tool, conversation, multi_tool |
| ORPO | qa, cot, kg, pref_pair, tool, conversation, multi_tool |
| DPO+SFT | qa, cot, kg, pref_pair, tool, conversation, multi_tool |
| PPO | qa, kg, pref_pair, tool, conversation, multi_tool |
| RRHF | qa, cot, kg, pref_list, tool, conversation, multi_tool |
| RLAIF | qa, kg, pref_pair, tool, conversation, multi_tool |
| GRPO | qa, kg, pref_list, tool, conversation, multi_tool |
| CPT | text |

```python
from datacreek import get_dataset_types_for_training, TrainingGoal
print(get_dataset_types_for_training(TrainingGoal.DPO))
```

You can query pipelines programmatically:

```python
from datacreek import get_pipelines_for_training, TrainingGoal
print(get_pipelines_for_training(TrainingGoal.SFT))
```

Use `run_generation_pipeline` to execute the generation steps directly on a
knowledge graph:

```python
from datacreek import (
    run_generation_pipeline,
    run_generation_pipeline_async,
    DatasetType,
    KnowledgeGraph,
)

kg = KnowledgeGraph()
kg.add_document("doc", source="text", text="Hello world")

# Synchronous usage
qa_data = run_generation_pipeline(DatasetType.QA, kg)

# Asynchronous usage
# qa_data = await run_generation_pipeline_async(DatasetType.QA, kg)
Both functions raise `PipelineExecutionError` when a step fails.
```

## Post-Ingestion Operations

After parsing documents into the knowledge graph you can refine the data quality
with a few helper methods:

- `deduplicate_chunks()` removes chunks with identical text
- `resolve_entities(aliases={...})` merges entity nodes referring to the same concept and accepts a case-insensitive alias mapping
- `prune_sources(['src'])` deletes nodes originating from unwanted sources
- `link_chunks_by_entity()` connects chunks that mention the same entity
- `link_sections_by_entity()` connects sections that mention the same entity
- `link_documents_by_entity()` connects documents that mention the same entity
- `link_authors_organizations()` links document authors to their organizations
- `clean_chunk_texts()` removes HTML tags and excess whitespace from chunks
- `normalize_date_fields()` standardizes date attributes to ISO format
- `compute_graph_embeddings(dimensions=64, walk_length=10, num_walks=50, seed=0, workers=2)` generates Node2Vec embeddings for all nodes. Adjust these parameters to tune vector size and random walks
- `predict_links(use_graph_embeddings=True)` infers relations using graph embeddings
- `mark_conflicting_facts()` flags edges when multiple objects disagree
- `validate_coherence()` marks logically impossible relations like a parent born after a child
- `apply_perception()` updates a node and automatically runs Neo4j `gds.nodeSimilarity` to flag near duplicates
- `apply_perception_all_nodes()` transforms every node and performs the same similarity check
- `node_similarity(id, threshold=0.95)` returns nodes similar to a given ID using Neo4j GDS
- `enrich_entity_wikidata(id)` fetches label and description from Wikidata
- `enrich_entity_dbpedia(id)` fetches additional info from DBpedia

Example:

```python
builder.apply_perception(
    "chunk_1",
    "Shortened text.",
    perception_id="summary",
    strength=0.7,
    threshold=0.9,
)

similar = builder.node_similarity("chunk_1", threshold=0.95)
print(similar)
```

Each similarity query is stored as a `node_similarity_check` event when matches are found, and every call to `node_similarity()` emits a `node_similarity_query` event for traceability.

These utilities are exposed through both the REST API and the web interface.

## Troubleshooting FAQs:

### vLLM Server Issues

- Ensure vLLM is installed: `pip install vllm`
- Start server with: `vllm serve <model_name> --port 8000`
- Check connection: `curl http://localhost:8000/docs`

### Memory Issues

If you encounter CUDA out of memory errors:
- Use a smaller model
- Reduce batch size in config
- Start vLLM with `--gpu-memory-utilization 0.85`

### JSON Parsing Issues

If you encounter issues during the curation step:
- Enable verbose logging
- Set smaller batch sizes in your config.yaml
- Ensure the LLM model supports proper JSON output
- Install json5 for enhanced JSON parsing: `pip install json5`

### Parser Errors

- Ensure required dependencies are installed for specific parsers:
  - YouTube: `pip install pytubefix youtube-transcript-api`
  - Office/PDF/HTML: `pip install "unstructured[all-docs]"`

## Web Interface

A lightweight React application built with Vite lives in the `frontend`
directory. It uses Tailwind CSS v4 for styling and communicates with the Flask
API for authentication and dataset operations.

```bash
cd frontend
npm install
npm run dev  # start the Vite dev server
```

Point your browser to `http://localhost:5173` while the Flask API runs on
`http://localhost:8000`.

## License

Read more about the [License](./LICENSE)

## Contributing

Contributions are welcome! [Read our contributing guide](./CONTRIBUTING.md)
