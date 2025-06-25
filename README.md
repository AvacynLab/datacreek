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
- Creating synthetic datasets
- Supporting various formats of post-training fine-tuning

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

#### (Alternatively) From Source

```bash
git clone https://github.com/meta-llama/datacreek.git
cd datacreek
pip install -e .
pip install pre-commit
pre-commit install
```

### 1. Tool Setup

- The tool expects respective files to be put in named folders.

```bash
# Create directory structure
mkdir -p data/{pdf,html,youtube,docx,ppt,txt,output,generated,cleaned,final}
```

- You also need a LLM backend that you will utilize for generating your dataset, if using vLLM:

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
download). Every request must include an `X-API-Key` header issued when creating
a user via `/users`.
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

### Starting the stack with Docker

Use the provided `docker-compose.yml` to launch the API, Celery worker,
Redis, Neo4j and the front-end:

```bash
./scripts/start_services.sh
```

If no `.env` file exists the script will copy `.env.example` so you only
need to adjust values for production. The default configuration stores
the SQLite database and generated datasets in `./data` which is mounted
inside the containers.

The API will be available on `http://localhost:8000` while the front-end is
served on `http://localhost:3000`. Redis listens on `6379` and Neo4j exposes
`7474` and `7687`.

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
  chunk_method: sliding  # basic|sliding|semantic
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
  chunk_method: semantic  # or "sliding" for fixed windows
  similarity_drop: 0.25   # threshold when using semantic splitting
    retrieval_top_k: 5      # number of chunks fetched using embeddings
```

Most options can also be overridden with environment variables. For example set
`GEN_TEMPERATURE=0.5` to change the default temperature, or `LLM_MODEL` to use a
different model without editing the YAML file.

```bash
curl -X POST localhost:8000/tasks/ingest \
     -H "X-Config-Path: custom_config.yaml" \
     -H "X-API-Key: <key>" \
     -d "path=docs/paper.pdf"
```

## Examples

### Processing a PDF Document

```bash
# Ingest PDF
curl -X POST localhost:8000/tasks/ingest -d "path=research_paper.pdf" -H "X-API-Key: <key>"

# Generate QA pairs (assuming source ID 1)
curl -X POST localhost:8000/tasks/generate -d "src_id=1&num_pairs=30" -H "X-API-Key: <key>"

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
for file in data/pdf/*.pdf; do
  filename=$(basename "$file" .pdf)

  curl -X POST localhost:8000/tasks/ingest -d "path=$file" -H "X-API-Key: <key>"
  curl -X POST localhost:8000/tasks/generate -d "src_id=1&num_pairs=20" -H "X-API-Key: <key>"
  curl -X POST localhost:8000/tasks/curate -d "ds_id=1&threshold=7.5" -H "X-API-Key: <key>"
  curl -X POST localhost:8000/tasks/save -d "ds_id=1&fmt=chatml" -H "X-API-Key: <key>"
done
```

### Command Line Generation

You can also generate data directly without running the REST API.

```bash
python -m datacreek.cli generate ./docs/paper.txt --content-type qa \
    --model llama-3 --temperature 0.6 --output-dir out
```

Pass `--prompt-file` to override the default prompt template for the run.

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
print(ds.search("hello"))  # ["c1"]
print(ds.search_documents("paper"))  # ["doc1"]
print(ds.get_chunks_for_document("doc1"))  # ["c1"]

# Clone a dataset to experiment with different cleaning steps
ds_copy = ds.clone(name="copy")

# Persist and reload later using Redis and Neo4j
import redis
from neo4j import GraphDatabase

client = redis.Redis(host="localhost", port=6379)
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j"))

ds.to_redis(client, "dataset:example")
ds.graph.to_neo4j(driver)

ds_loaded = DatasetBuilder.from_redis(client, "dataset:example")
ds_loaded.graph = KnowledgeGraph.from_neo4j(driver)
```

Ingestion now has two phases. First `ingest_file()` parses a file into raw
text, then `to_kg()` splits it into chunks and inserts them into your
dataset's knowledge graph:

```python
from datacreek import ingest_file, to_kg

text = ingest_file("paper.pdf")
to_kg(text, ds, "paper")
```

### Mental Model:

```mermaid
graph LR
    API[REST API] --> Ingest
    API --> Generate
    API --> Curate
    API --> Save
    Ingest --> HTMLFile[HTML File]
    Ingest --> YouTubeURL[File Format]

    Generate --> CoT[CoT]
    Generate --> QA[QA Pairs]
    Generate --> Summary[Summary]

    Curate --> Filter[Filter by Quality]

    Save --> JSONL[JSONL Format]
    Save --> Alpaca[Alpaca Format]
    Save --> FT[Fine-Tuning Format]
    Save --> ChatML[ChatML Format]
```

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

You can query pipelines programmatically:

```python
from datacreek import get_pipelines_for_training, TrainingGoal
print(get_pipelines_for_training(TrainingGoal.SFT))
```

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
  - PDF: `pip install pdfminer.six`
  - HTML: `pip install beautifulsoup4`
  - YouTube: `pip install pytubefix youtube-transcript-api`
  - DOCX: `pip install python-docx`
  - PPTX: `pip install python-pptx`

## License

Read more about the [License](./LICENSE)

## Contributing

Contributions are welcome! [Read our contributing guide](./CONTRIBUTING.md)
