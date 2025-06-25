"""
Flask application for the Datacreek web interface.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from flask_wtf import FlaskForm
from neo4j import GraphDatabase
from werkzeug.security import check_password_hash, generate_password_hash
from wtforms import FileField, IntegerField, PasswordField, SelectField, StringField, SubmitField
from wtforms.validators import DataRequired

from datacreek.core.create import process_file
from datacreek.core.curate import curate_qa_pairs
from datacreek.core.dataset import DatasetBuilder
from datacreek.core.ingest import ingest_into_dataset
from datacreek.core.ingest import process_file as ingest_process_file
from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.db import SessionLocal, User, init_db
from datacreek.pipelines import DatasetType
from datacreek.services import generate_api_key, hash_key
from datacreek.utils.config import get_llm_provider, get_neo4j_config, load_config

STATIC_DIR = Path(__file__).parents[2] / "frontend" / "dist"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/")
app.config["SECRET_KEY"] = os.urandom(24)

login_manager = LoginManager(app)
login_manager.login_view = None


@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({"error": "login required"}), 401


# Set default paths
DEFAULT_DATA_DIR = Path(__file__).parents[2] / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "output"
DEFAULT_GENERATED_DIR = DEFAULT_DATA_DIR / "generated"

# Create directories if they don't exist
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Load SDK config
config = load_config()
init_db()


@login_manager.user_loader
def load_user(user_id: str) -> User | None:
    with SessionLocal() as db:
        return db.get(User, int(user_id))


def get_neo4j_driver():
    """Return a Neo4j driver using config or environment variables."""
    cfg = get_neo4j_config(config)
    uri = os.getenv("NEO4J_URI", cfg.get("uri"))
    user = os.getenv("NEO4J_USER", cfg.get("user"))
    password = os.getenv("NEO4J_PASSWORD", cfg.get("password"))
    if not uri or not user or not password:
        return None
    return GraphDatabase.driver(uri, auth=(user, password))


# In-memory store of datasets being built
DATASETS: Dict[str, DatasetBuilder] = {}
# Store knowledge graphs independently from datasets
GRAPHS: Dict[str, DatasetBuilder] = {}


# Forms
class CreateForm(FlaskForm):
    """Form for creating content from text"""

    input_file = StringField("Input File Path", validators=[DataRequired()])
    content_type = SelectField(
        "Content Type",
        choices=[
            ("qa", "Question-Answer Pairs"),
            ("summary", "Summary"),
            ("cot", "Chain of Thought"),
            ("cot-enhance", "CoT Enhancement"),
        ],
        default="qa",
    )
    num_pairs = IntegerField("Number of QA Pairs", default=10)
    provider = SelectField(
        "Provider",
        choices=[("vllm", "vLLM"), ("api-endpoint", "API Endpoint")],
        default="vllm",
    )
    model = StringField("Model Name (optional)")
    api_base = StringField("API Base URL (optional)")
    submit = SubmitField("Generate Content")


class IngestForm(FlaskForm):
    """Form for ingesting documents"""

    input_type = SelectField(
        "Input Type",
        choices=[("file", "Upload File"), ("url", "URL"), ("path", "Local Path")],
        default="file",
    )
    upload_file = FileField("Upload Document")
    input_path = StringField("File Path or URL")
    output_name = StringField("Output Filename (optional)")
    submit = SubmitField("Parse Document")


class CurateForm(FlaskForm):
    """Form for curating QA pairs"""

    input_file = StringField("Input JSON File Path", validators=[DataRequired()])
    num_pairs = IntegerField("Number of QA Pairs to Keep", default=0)
    provider = SelectField(
        "Provider",
        choices=[("vllm", "vLLM"), ("api-endpoint", "API Endpoint")],
        default="vllm",
    )
    model = StringField("Model Name (optional)")
    api_base = StringField("API Base URL (optional)")
    submit = SubmitField("Curate QA Pairs")


class UploadForm(FlaskForm):
    """Form for uploading files"""

    file = FileField("Upload File", validators=[DataRequired()])
    submit = SubmitField("Upload")


class DatasetForm(FlaskForm):
    """Form for creating a new dataset"""

    name = StringField("Dataset Name", validators=[DataRequired()])
    dataset_type = SelectField(
        "Dataset Type",
        choices=[(dt.value, dt.value) for dt in DatasetType],
        default=DatasetType.QA.value,
    )
    submit = SubmitField("Create Dataset")


# API Routes


@app.post("/api/login")
def api_login():
    data = request.get_json() or {}
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "missing credentials"}), 400
    with SessionLocal() as db:
        user = db.query(User).filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return jsonify({"message": "logged in"})
    return jsonify({"error": "invalid credentials"}), 401


@app.post("/api/logout")
@login_required
def api_logout():
    logout_user()
    return jsonify({"message": "logged out"})


@app.post("/api/register")
def api_register():
    data = request.get_json() or {}
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "missing credentials"}), 400
    with SessionLocal() as db:
        if db.query(User).filter_by(username=username).first():
            return jsonify({"error": "username exists"}), 400
        api_key = generate_api_key()
        user = User(
            username=username,
            api_key=hash_key(api_key),
            password_hash=generate_password_hash(password),
        )
        db.add(user)
        db.commit()
    return jsonify({"message": "account created", "api_key": api_key})


@app.get("/api/session")
def api_session():
    if current_user.is_authenticated:
        return jsonify({"username": current_user.username})
    return jsonify({"username": None})


@app.get("/api/datasets")
@login_required
def api_datasets():
    """Return list of dataset names."""
    return jsonify(list(DATASETS.keys()))


@app.post("/api/datasets")
@login_required
def api_create_dataset():
    data = request.get_json() or {}
    name = data.get("name")
    dtype = data.get("dataset_type")
    graph_name = data.get("graph")
    if not name or not dtype:
        abort(400)
    if name in DATASETS:
        abort(400, description="Dataset already exists")
    try:
        ds = DatasetBuilder(DatasetType(dtype), name=name)
    except ValueError:
        abort(400)
    if graph_name:
        g = GRAPHS.get(graph_name)
        if not g:
            abort(404)
        ds.graph = KnowledgeGraph.from_dict(g.graph.to_dict())
        ds.stage = max(ds.stage, 1)
        ds.history.append(f"Initialized from graph {graph_name}")
    ds.history.append("Dataset created")
    DATASETS[name] = ds
    return jsonify({"message": "Dataset created"})


@app.get("/api/datasets/<name>")
@login_required
def api_dataset_detail(name: str):
    """Return dataset details as JSON."""
    ds = DATASETS.get(name)
    if not ds:
        abort(404)
    nodes = ds.graph.graph
    num_docs = sum(1 for _, d in nodes.nodes(data=True) if d.get("type") == "document")
    num_chunks = sum(1 for _, d in nodes.nodes(data=True) if d.get("type") == "chunk")
    quality = min(100, num_chunks * 5)
    tips = []
    if num_docs == 0:
        tips.append("Add documents to your dataset")
    if num_chunks == 0:
        tips.append("Ingest text chunks to improve quality")
    info = {
        "name": ds.name,
        "type": ds.dataset_type.value,
        "created_at": ds.created_at.isoformat(),
        "num_nodes": len(nodes.nodes),
        "num_edges": len(nodes.edges),
        "num_documents": num_docs,
        "num_chunks": num_chunks,
        "quality": quality,
        "tips": tips,
        "history": ds.history,
        "versions": ds.versions,
        "stage": ds.stage,
    }
    return jsonify(info)


@app.get("/api/datasets/<name>/content")
@login_required
def api_dataset_content(name: str):
    """Return dataset documents with their chunks."""
    ds = DATASETS.get(name)
    if not ds:
        abort(404)
    docs = []
    for node, data in ds.graph.graph.nodes(data=True):
        if data.get("type") == "document":
            doc = {
                "id": node,
                "source": data.get("source"),
                "chunks": [],
            }
            for cid in ds.graph.get_chunks_for_document(node):
                cdata = ds.graph.graph.nodes[cid]
                doc["chunks"].append({"id": cid, "text": cdata.get("text", "")})
            docs.append(doc)
    return jsonify(docs)


@app.post("/api/datasets/<name>/ingest")
@login_required
def api_dataset_ingest(name: str):
    """Ingest a document into the dataset."""
    ds = DATASETS.get(name)
    if not ds:
        abort(404)
    data = request.get_json() or {}
    path = data.get("path")
    doc_id = data.get("doc_id")
    if not path:
        abort(400)
    ingest_into_dataset(path, ds, doc_id=doc_id, config=config)
    ds.history.append(f"Ingested {os.path.basename(path)}")
    ds.stage = max(ds.stage, 1)
    return jsonify({"message": "ingested"})


@app.post("/api/datasets/<name>/generate")
@login_required
def api_dataset_generate(name: str):
    """Record a generation run with optional parameters."""
    ds = DATASETS.get(name)
    if not ds:
        abort(404)
    data = request.get_json() or {}
    params = data.get("params", {})
    ds.versions.append(
        {
            "params": params,
            "time": datetime.now(timezone.utc).isoformat(),
        }
    )
    ds.stage = max(ds.stage, 2)
    ds.history.append(f"Dataset generated (v{len(ds.versions)})")
    return jsonify({"message": "generated", "version": len(ds.versions)})


@app.delete("/api/datasets/<name>/chunks/<cid>")
@login_required
def api_delete_chunk(name: str, cid: str):
    ds = DATASETS.get(name)
    if not ds:
        abort(404)
    ds.remove_chunk(cid)
    return jsonify({"message": "deleted"})


@app.get("/api/datasets/<name>/export")
@login_required
def api_export_dataset(name: str):
    """Return the dataset as JSON and mark it exported."""
    ds = DATASETS.get(name)
    if not ds:
        abort(404)
    ds.stage = max(ds.stage, 4)
    ds.history.append("Dataset exported")
    return jsonify(ds.to_dict())


# ---------------------------------------------------------------------------
# Knowledge Graph API
# ---------------------------------------------------------------------------


@app.get("/api/graphs")
@login_required
def api_graphs():
    """Return list of knowledge graph names."""
    return jsonify(list(GRAPHS.keys()))


@app.post("/api/graphs")
@login_required
def api_create_graph():
    data = request.get_json() or {}
    name = data.get("name")
    docs = data.get("documents", [])
    if not name:
        abort(400)
    if name in GRAPHS:
        abort(400, description="Graph already exists")
    kg_ds = DatasetBuilder(DatasetType.DOCUMENT, name=name)
    for path in docs:
        ingest_into_dataset(path, kg_ds, config=config)
    kg_ds.history.append("Graph created")
    GRAPHS[name] = kg_ds
    return jsonify({"message": "graph created"})


@app.get("/api/graphs/<name>")
@login_required
def api_graph_detail(name: str):
    ds = GRAPHS.get(name)
    if not ds:
        abort(404)
    graph = ds.graph.graph
    info = {
        "name": name,
        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),
    }
    return jsonify(info)


@app.get("/api/graphs/<name>/data")
@login_required
def api_graph_data(name: str):
    ds = GRAPHS.get(name)
    if not ds:
        abort(404)
    return jsonify(ds.graph.to_dict())


@app.route("/")
def index():
    """Serve the front-end application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return app.send_static_file("index.html")
    return "Frontend not built", 200


@app.route("/datasets", methods=["GET", "POST"])
@login_required
def datasets():
    """List and create datasets"""
    form = DatasetForm()
    if form.validate_on_submit():
        name = form.name.data
        ds_type = DatasetType(form.dataset_type.data)
        DATASETS[name] = DatasetBuilder(ds_type, name=name)
        DATASETS[name].history.append("Dataset created")
        flash("Dataset created", "success")
        return redirect(url_for("dataset_detail", name=name))
    return render_template("datasets.html", datasets=DATASETS, form=form)


@app.route("/datasets/<name>")
@login_required
def dataset_detail(name: str):
    ds = DATASETS.get(name)
    if not ds:
        abort(404)
    return render_template("dataset_detail.html", dataset=ds)


@app.get("/datasets/<name>/graph")
@login_required
def dataset_graph(name: str):
    """Return dataset knowledge graph as JSON."""
    ds = DATASETS.get(name)
    if not ds:
        abort(404)
    return jsonify(ds.graph.to_dict())


@app.get("/datasets/<name>/search")
@login_required
def dataset_search(name: str):
    """Return node ids matching the query."""
    ds = DATASETS.get(name)
    if not ds:
        abort(404)
    query = request.args.get("q")
    node_type = request.args.get("type", "chunk")
    if not query:
        return jsonify([])
    ids = ds.graph.search(query, node_type=node_type)
    return jsonify(ids)


@app.post("/datasets/<name>/ingest")
@login_required
def dataset_ingest(name: str):
    """Ingest a file or URL into the dataset knowledge graph."""
    ds = DATASETS.get(name)
    if not ds:
        abort(404)

    input_path = request.form.get("input_path")
    doc_id = request.form.get("doc_id") or None
    if not input_path:
        flash("Input path is required", "warning")
        return redirect(url_for("dataset_detail", name=name))

    try:
        ingest_into_dataset(input_path, ds, doc_id=doc_id, config=config)
        ds.history.append(f"Ingested {os.path.basename(input_path)}")
        ds.stage = max(ds.stage, 1)
        flash("Document ingested", "success")
    except Exception as e:  # pragma: no cover - flash message only
        flash(f"Error ingesting document: {e}", "danger")

    return redirect(url_for("dataset_detail", name=name))


@app.post("/datasets/<name>/save_neo4j")
@login_required
def save_dataset_neo4j(name: str):
    """Persist the dataset graph to Neo4j."""
    ds = DATASETS.get(name)
    if not ds:
        abort(404)
    driver = get_neo4j_driver()
    if not driver:
        abort(500, description="Neo4j not configured")
    ds.graph.to_neo4j(driver)
    driver.close()
    ds.history.append("Saved to Neo4j")
    flash("Graph saved to Neo4j", "success")
    return redirect(url_for("dataset_detail", name=name))


@app.post("/datasets/<name>/load_neo4j")
@login_required
def load_dataset_neo4j(name: str):
    """Load the dataset graph from Neo4j."""
    ds = DATASETS.get(name)
    if not ds:
        abort(404)
    driver = get_neo4j_driver()
    if not driver:
        abort(500, description="Neo4j not configured")
    ds.graph = KnowledgeGraph.from_neo4j(driver)
    driver.close()
    ds.history.append("Loaded from Neo4j")
    flash("Graph loaded from Neo4j", "success")
    return redirect(url_for("dataset_detail", name=name))


@app.post("/datasets/<name>/delete")
@login_required
def delete_dataset(name: str):
    DATASETS.pop(name, None)
    flash("Dataset deleted", "success")
    return redirect(url_for("datasets"))


@app.post("/datasets/<name>/copy")
@login_required
def copy_dataset(name: str):
    ds = DATASETS.get(name)
    if not ds:
        abort(404)
    new_name = f"{name}_copy"
    counter = 1
    while new_name in DATASETS:
        counter += 1
        new_name = f"{name}_copy{counter}"
    DATASETS[new_name] = ds.clone(name=new_name)
    DATASETS[new_name].history.append(f"Copied from {name}")
    flash("Dataset copied", "success")
    return redirect(url_for("dataset_detail", name=new_name))


@app.route("/create", methods=["GET", "POST"])
@login_required
def create():
    """Create content from text"""
    form = CreateForm()
    default_provider = get_llm_provider(config)

    if not form.provider.data:
        form.provider.data = default_provider

    provider = form.provider.data or default_provider

    if form.validate_on_submit():
        try:
            input_file = form.input_file.data
            content_type = form.content_type.data
            num_pairs = form.num_pairs.data
            model = form.model.data or None
            api_base = form.api_base.data or None

            output_path = process_file(
                file_path=input_file,
                output_dir=str(DEFAULT_GENERATED_DIR),
                content_type=content_type,
                num_pairs=num_pairs,
                provider=provider,
                api_base=api_base,
                model=model,
                config_path=None,  # Use default config
                verbose=True,
            )

            content_type_labels = {
                "qa": "QA pairs",
                "summary": "summary",
                "cot": "Chain of Thought examples",
                "cot-enhance": "CoT enhanced conversation",
            }
            content_label = content_type_labels.get(content_type, content_type)

            flash(
                f"Successfully generated {content_label}! Output saved to: {output_path}", "success"
            )
            return redirect(
                url_for(
                    "view_file",
                    file_path=str(Path(output_path).relative_to(DEFAULT_DATA_DIR.parent)),
                )
            )

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    # Get the list of available input files
    input_files = []
    if DEFAULT_OUTPUT_DIR.exists():
        input_files = [
            str(f.relative_to(DEFAULT_DATA_DIR.parent)) for f in DEFAULT_OUTPUT_DIR.glob("*.txt")
        ]

    return render_template("create.html", form=form, provider=provider, input_files=input_files)


@app.route("/curate", methods=["GET", "POST"])
@login_required
def curate():
    """Curate QA pairs interface"""
    form = CurateForm()
    default_provider = get_llm_provider(config)

    if not form.provider.data:
        form.provider.data = default_provider

    provider = form.provider.data or default_provider

    if form.validate_on_submit():
        try:
            input_file = form.input_file.data
            model = form.model.data or None
            api_base = form.api_base.data or None

            # Create output path
            filename = Path(input_file).stem
            output_file = f"{filename}_curated.json"
            output_path = str(Path(DEFAULT_GENERATED_DIR) / output_file)

            result_path = curate_qa_pairs(
                input_path=input_file,
                output_path=output_path,
                provider=provider,
                api_base=api_base,
                model=model,
                config_path=None,  # Use default config
                verbose=True,
            )

            flash(f"Successfully curated QA pairs! Output saved to: {result_path}", "success")
            return redirect(
                url_for(
                    "view_file",
                    file_path=str(Path(result_path).relative_to(DEFAULT_DATA_DIR.parent)),
                )
            )

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    # Get the list of available JSON files
    json_files = []
    if DEFAULT_GENERATED_DIR.exists():
        json_files = [
            str(f.relative_to(DEFAULT_DATA_DIR.parent))
            for f in DEFAULT_GENERATED_DIR.glob("*.json")
        ]

    return render_template("curate.html", form=form, provider=provider, json_files=json_files)


@app.route("/files")
@login_required
def files():
    """File browser"""
    # Get all files in the data directory
    output_files = []
    generated_files = []

    if DEFAULT_OUTPUT_DIR.exists():
        output_files = [
            str(f.relative_to(DEFAULT_DATA_DIR.parent)) for f in DEFAULT_OUTPUT_DIR.glob("*.*")
        ]

    if DEFAULT_GENERATED_DIR.exists():
        generated_files = [
            str(f.relative_to(DEFAULT_DATA_DIR.parent)) for f in DEFAULT_GENERATED_DIR.glob("*.*")
        ]

    return render_template("files.html", output_files=output_files, generated_files=generated_files)


@app.route("/view/<path:file_path>")
@login_required
def view_file(file_path):
    """View a file's contents"""
    full_path = Path(DEFAULT_DATA_DIR.parent, file_path)

    if not full_path.exists():
        flash(f"File not found: {file_path}", "danger")
        return redirect(url_for("files"))

    file_content = None
    file_type = "text"

    if full_path.suffix.lower() == ".json":
        try:
            with open(full_path, "r") as f:
                file_content = json.load(f)
            file_type = "json"

            # Detect specific JSON formats
            is_qa_pairs = "qa_pairs" in file_content
            is_cot_examples = "cot_examples" in file_content
            has_conversations = "conversations" in file_content
            has_summary = "summary" in file_content

        except Exception:
            # If JSON parsing fails, treat as text
            with open(full_path, "r") as f:
                file_content = f.read()
            file_type = "text"
            is_qa_pairs = False
            is_cot_examples = False
            has_conversations = False
            has_summary = False
    else:
        # Read as text
        with open(full_path, "r") as f:
            file_content = f.read()
        file_type = "text"
        is_qa_pairs = False
        is_cot_examples = False
        has_conversations = False
        has_summary = False

    return render_template(
        "view_file.html",
        file_path=file_path,
        file_type=file_type,
        content=file_content,
        is_qa_pairs=is_qa_pairs,
        is_cot_examples=is_cot_examples,
        has_conversations=has_conversations,
        has_summary=has_summary,
    )


@app.route("/ingest", methods=["GET", "POST"])
@login_required
def ingest():
    """Ingest and parse documents"""
    form = IngestForm()

    if form.validate_on_submit():
        try:
            input_type = form.input_type.data
            output_name = form.output_name.data or None

            # Get default output directory for parsed files
            output_dir = str(DEFAULT_OUTPUT_DIR)

            if input_type == "file":
                # Handle file upload
                if not form.upload_file.data:
                    flash("Please upload a file", "warning")
                    return render_template("ingest.html", form=form)

                # Save the uploaded file to a temporary location
                temp_file = form.upload_file.data
                original_filename = temp_file.filename
                file_extension = Path(original_filename).suffix

                # Use upload filename as the output name if not provided
                if not output_name:
                    output_name = Path(original_filename).stem

                # Create a temporary file path in the output directory
                temp_path = DEFAULT_OUTPUT_DIR / f"temp_{output_name}{file_extension}"
                temp_file.save(temp_path)

                # Process the file
                input_path = str(temp_path)
            else:
                # URL or local path
                input_path = form.input_path.data
                if not input_path:
                    flash("Please enter a valid path or URL", "warning")
                    return render_template("ingest.html", form=form)

            # Process the file or URL
            output_path = ingest_process_file(
                file_path=input_path, output_dir=output_dir, output_name=output_name, config=config
            )

            # Clean up temporary file if it was an upload
            if input_type == "file" and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

            flash(f"Successfully parsed document! Output saved to: {output_path}", "success")
            return redirect(
                url_for(
                    "view_file",
                    file_path=str(Path(output_path).relative_to(DEFAULT_DATA_DIR.parent)),
                )
            )

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    # Get some example URLs for different document types
    examples = {
        "PDF": "path/to/document.pdf",
        "YouTube": "https://www.youtube.com/watch?v=example",
        "Web Page": "https://example.com/article",
        "Word Document": "path/to/document.docx",
        "PowerPoint": "path/to/presentation.pptx",
        "Text File": "path/to/document.txt",
    }

    return render_template("ingest.html", form=form, examples=examples)


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    """Upload a file to the data directory"""
    form = UploadForm()

    if form.validate_on_submit():
        f = form.file.data
        filename = f.filename
        filepath = DEFAULT_OUTPUT_DIR / filename
        f.save(filepath)
        flash(f"File uploaded successfully: {filename}", "success")
        return redirect(url_for("files"))

    return render_template("upload.html", form=form)


@app.route("/api/qa_json/<path:file_path>")
@login_required
def qa_json(file_path):
    """Return QA pairs as JSON for the JSON viewer"""
    full_path = Path(DEFAULT_DATA_DIR.parent, file_path)

    if not full_path.exists() or full_path.suffix.lower() != ".json":
        abort(404)

    try:
        with open(full_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception:
        abort(500)


@app.route("/api/edit_item/<path:file_path>", methods=["POST"])
@login_required
def edit_item(file_path):
    """Edit an item in a JSON file"""
    full_path = Path(DEFAULT_DATA_DIR.parent, file_path)

    if not full_path.exists() or full_path.suffix.lower() != ".json":
        return jsonify({"success": False, "message": "File not found or not a JSON file"}), 404

    try:
        # Get the request data
        data = request.json
        item_type = data.get("item_type")  # qa_pairs, cot_examples, conversations
        item_index = data.get("item_index")
        item_content = data.get("item_content")

        if not all([item_type, item_index is not None, item_content]):
            return jsonify({"success": False, "message": "Missing required parameters"}), 400

        # Read the file
        with open(full_path, "r") as f:
            file_content = json.load(f)

        # Update the item
        if item_type == "qa_pairs" and "qa_pairs" in file_content:
            if 0 <= item_index < len(file_content["qa_pairs"]):
                file_content["qa_pairs"][item_index] = item_content
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        elif item_type == "cot_examples" and "cot_examples" in file_content:
            if 0 <= item_index < len(file_content["cot_examples"]):
                file_content["cot_examples"][item_index] = item_content
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        elif item_type == "conversations" and "conversations" in file_content:
            if 0 <= item_index < len(file_content["conversations"]):
                file_content["conversations"][item_index] = item_content
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        else:
            return jsonify({"success": False, "message": "Invalid item type"}), 400

        # Write back to the file
        with open(full_path, "w") as f:
            json.dump(file_content, f, indent=2)

        return jsonify({"success": True, "message": "Item updated successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/delete_item/<path:file_path>", methods=["POST"])
@login_required
def delete_item(file_path):
    """Delete an item from a JSON file"""
    full_path = Path(DEFAULT_DATA_DIR.parent, file_path)

    if not full_path.exists() or full_path.suffix.lower() != ".json":
        return jsonify({"success": False, "message": "File not found or not a JSON file"}), 404

    try:
        # Get the request data
        data = request.json
        item_type = data.get("item_type")  # qa_pairs, cot_examples, conversations
        item_index = data.get("item_index")

        if not all([item_type, item_index is not None]):
            return jsonify({"success": False, "message": "Missing required parameters"}), 400

        # Read the file
        with open(full_path, "r") as f:
            file_content = json.load(f)

        # Delete the item
        if item_type == "qa_pairs" and "qa_pairs" in file_content:
            if 0 <= item_index < len(file_content["qa_pairs"]):
                file_content["qa_pairs"].pop(item_index)
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        elif item_type == "cot_examples" and "cot_examples" in file_content:
            if 0 <= item_index < len(file_content["cot_examples"]):
                file_content["cot_examples"].pop(item_index)
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        elif item_type == "conversations" and "conversations" in file_content:
            if 0 <= item_index < len(file_content["conversations"]):
                file_content["conversations"].pop(item_index)
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        else:
            return jsonify({"success": False, "message": "Invalid item type"}), 400

        # Write back to the file
        with open(full_path, "w") as f:
            json.dump(file_content, f, indent=2)

        return jsonify({"success": True, "message": "Item deleted successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


def run_server(host="127.0.0.1", port=5000, debug=False):
    """Run the Flask server"""
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
