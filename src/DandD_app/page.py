from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import pathlib, shutil, uvicorn
import os
import asyncio
import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- RAG Core: LightRAG Interface (Your provided class structure) ---

# Placeholder for Ollama Embed (since we can't import the real one here)
# You MUST replace this with the real import from lightrag.llm.ollama in your environment
async def ollama_embed(texts, embed_model):
    """
    SIMULATED: In a real environment, this calls Ollama to get the embeddings.
    Here, we return a dummy embedding array.
    """
    print(f"SIMULATING: Calling Ollama embed model '{embed_model}' for {len(texts)} text(s).")
    return [np.random.rand(1024).tolist()] * len(texts)


class RagQuery:
    """
    Implements the RAG query logic using local vector store and .jsonl files.
    """
    def __init__(self, project_name, edition, embedding_dim=1024, storage_dir=None):
        self.project_name = project_name
        self.edition = edition
        self.embedding_dim = embedding_dim
        
        base_path = PROJECT_ROOT / project_name / edition
        self.storage_dir = storage_dir or base_path / "rag_storage" / "vector_store"
        self.meta_dir = base_path / "rag_storage"
        self.vectors = []
        self.meta = []

        # --- Synchronous Loading of Vectors and Metadata ---
        try:
            vec_files = sorted(Path(self.storage_dir).glob("vec_*.npy"))
            for vec_file in vec_files:
                try:
                    vectors = np.load(vec_file)
                    self.vectors.append(vectors)
                except Exception as e:
                    print(f"Error loading vector file {vec_file}: {e}")
                    continue

                idx = vec_file.stem.split("_")[1]
                meta_file = Path(self.meta_dir) / f"meta_{idx}.jsonl"
                if not meta_file.exists():
                    continue
                with open(meta_file, "r", encoding="utf-8") as f:
                    self.meta.extend([json.loads(line) for line in f])

            if self.vectors:
                self.vectors = np.vstack(self.vectors)
            else:
                self.vectors = np.zeros((0, embedding_dim), dtype=np.float32)
        
        except Exception as e:
            print(f"CRITICAL: Failed to initialize RagQuery due to file system error: {e}")
            self.vectors = np.zeros((0, embedding_dim), dtype=np.float32)


    async def query(self, query_text, top_k=5, max_chunks_per_file=3):
        """
        Query the RAG store asynchronously.
        """
        if self.vectors.shape[0] == 0:
            return "CONTEXT_NOT_FOUND: Vector store is empty."

        # Embed query
        embed_model = os.environ.get("EMBEDDED_OLLAMA_MODEL", "default-embed-model")
        query_vec = await ollama_embed([query_text], embed_model=embed_model)
        query_vec = np.asarray(query_vec, dtype=np.float32)
        
        if query_vec.shape[1] != self.embedding_dim:
            query_vec = query_vec[:, :min(query_vec.shape[1], self.embedding_dim)]
            if self.vectors.shape[1] != query_vec.shape[1]:
                return "CONTEXT_ERROR: Vector dimensions mismatch. Cannot retrieve context."

        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, self.vectors)[0]

        # Get top-k matches
        top_indices = similarities.argsort()[::-1][:top_k]
        top_meta = [self.meta[idx] for idx in top_indices]

        llm_chunks = []
        seen_files = set()

        for r in top_meta:
            file_path = Path(r["source_path"])
            
            # CRITICAL: Fix path to be relative to PROJECT_ROOT for safe file access
            try:
                if file_path.is_absolute():
                    if self.project_name in file_path.parts:
                        start_index = file_path.parts.index(self.project_name)
                        file_path = PROJECT_ROOT / Path(*file_path.parts[start_index:])
                    else:
                        file_path = PROJECT_ROOT / Path(*file_path.parts[1:])
                elif file_path.is_relative() and file_path.parts[0] != self.project_name:
                     file_path = PROJECT_ROOT / file_path
                
            except Exception as e:
                print(f"Path correction error: {e}")
                continue 

            if not file_path.exists() or file_path in seen_files:
                continue
            seen_files.add(file_path)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                mini_chunks = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
                
                for chunk in mini_chunks[:max_chunks_per_file]:
                    llm_chunks.append(f"Source: {file_path.name}\n{chunk}")

        return "\n\n---\n\n".join(llm_chunks)


class LightRAG: 
    def __init__(self, project_name, rag_storage_path: pathlib.Path):
        self.edition = rag_storage_path.parent.name
        self.rag_storage_path = rag_storage_path
        
        self.rag_query_instance = RagQuery(
            project_name=project_name,
            edition=self.edition, 
            storage_dir=self.rag_storage_path / "vector_store"
        )

    async def get_answer(self, query: str) -> str:
        """
        Runs the async query method and awaits it.
        """
        return await self.rag_query_instance.query(query)


# --- Global Variables and Main App Setup ---
PROJECT_ROOT = pathlib.Path("./")

# Users and FastAPI App setup (unchanged)
users = {
    "nigel": {"password": "1504_Nigel", "role": "admin"},
    "simon": {"password": "simon-x", "role": "user"},
    "niels": {"password": "niels-4", "role": "user"},
    "sam": {"password": "sam-7", "role": "user"},
    "joren": {"password": "joren-6", "role": "user"},
    "lynn": {"password": "lynn-2", "role": "user"},
}

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="supersecretkey123")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Utility Functions (for project and version listing) ---

def get_available_projects():
    """Returns a sorted list of top-level directories (e.g., 'DandD', 'test')."""
    try:
        exclude = {'.git', '__pycache__', 'venv', 'logs', 'node_modules', 'app.py'}
        projects = [
            d.name for d in PROJECT_ROOT.iterdir() 
            if d.is_dir() and d.name not in exclude and not d.name.startswith('.')
        ]
        return sorted(projects)
    except Exception as e:
        return []


def get_available_versions(project_name: str):
    """
    Returns a sorted list of version directories (e.g., '5e', 'versionX') 
    inside the selected project root (./<project_name>/).
    """
    if not project_name:
        return []
        
    project_path = PROJECT_ROOT / project_name
    
    try:
        if not project_path.is_dir():
            return []
            
        versions = [
            d.name for d in project_path.iterdir() 
            if d.is_dir() and (d / "rag_storage").is_dir()
        ]
        return sorted(versions, reverse=True)
    except Exception as e:
        return []


# --- Reusable HTML Generation Function ---

def generate_dashboard_html(request: Request, rag_result: str = None, selected_project: str = None, selected_version: str = None):
    """Generates the full HTML for the dashboard page."""
    user = request.session.get("user")
    role = request.session.get("role")
    
    available_projects = get_available_projects()
    # Ensure selected_project is in the list or set to default
    if selected_project and selected_project not in available_projects:
         selected_project = available_projects[0] if available_projects else None
    elif not selected_project:
        selected_project = available_projects[0] if available_projects else None
        
    
    # Get available versions based on the selected/default project
    available_versions = get_available_versions(selected_project)
    
    # Ensure selected_version is in the list or set to default
    if selected_version and selected_version not in available_versions:
        selected_version = available_versions[0] if available_versions else None
    elif not selected_version:
        selected_version = available_versions[0] if available_versions else None
    
    
    # Generate <option> tags, ensuring the selected one is marked
    def get_options_html(items, current_selection):
        html = ""
        for item in items:
            selected = 'selected' if item == current_selection else ''
            html += f'<option value="{item}" {selected}>{item}</option>'
        return html

    project_options = get_options_html(available_projects, selected_project)
    version_options = get_options_html(available_versions, selected_version)
    
    form_disabled = "disabled" if not (available_projects and available_versions) else ""

    css_style = """
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); }
        h2 { color: #6c5ce7; border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-top: 0; }
        .header-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .role-tag { background-color: #6c5ce7; color: white; padding: 5px 10px; border-radius: 5px; font-size: 0.9em; margin-left: 10px; }
        .admin-link a { padding: 10px 15px; background-color: #2ecc71; color: white; text-decoration: none; border-radius: 6px; transition: background-color 0.3s; font-weight: bold; }
        .admin-link a:hover { background-color: #27ae60; }
        .logout-link a { color: #e74c3c; text-decoration: none; font-weight: bold; margin-left: 20px; }
        .query-form { margin-top: 20px; padding: 20px; border: 1px solid #eee; border-radius: 8px; background: #f9f9fb; }
        input[type="text"], select { width: 100%; padding: 12px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 6px; box-sizing: border-box; font-size: 1em; }
        input[type="submit"] { padding: 10px 20px; background-color: #6c5ce7; color: white; border: none; border-radius: 6px; cursor: pointer; transition: background-color 0.3s; }
        input[type="submit"]:hover { background-color: #5b4dd4; }
        input[type="submit"][disabled] { background-color: #ccc; cursor: not-allowed; }
        .rag-result-box { margin-top: 25px; padding: 20px; border-radius: 8px; background: #f0f0f5; border: 1px solid #ddd; }
        .rag-result-box p { margin: 0 0 10px 0; line-height: 1.6; }
        blockquote { white-space: pre-wrap; border-left: 4px solid #6c5ce7; padding-left: 15px; margin: 15px 0; background: #fff; padding: 10px; border-radius: 4px; font-style: italic; color: #555; }
    </style>
    """

    admin_button = ""
    if role == "admin":
        admin_button = """
        <span class="admin-link">
            <a href="/gradio">Upload RAG Artifacts (Admin)</a>
        </span>
        """
        
    result_html = ""
    if rag_result:
        # The rag_result already contains the formatted HTML blockquote
        result_html = f"<div class='rag-result-box'><h3>LightRAG Lookup Result</h3>{rag_result}</div>"

    html_content = f"""
    <html>
    <head>{css_style}<title>RAG Dashboard</title></head>
    <body>
    <div class="container">
        <div class="header-bar">
            <h2>Welcome, {user}<span class="role-tag">Role: {role}</span></h2>
            <div>
                {admin_button}
                <span class="logout-link"><a href="/logout">Logout</a></span>
            </div>
        </div>
        
        <div class="query-form">
            <h3>LightRAG Query Interface</h3>
            <form action="/dashboard" method="post">
                <label for="project">Select Project:</label><br>
                <select name="project" id="project" style="padding: 10px; border: 1px solid #ddd; border-radius: 6px; margin-bottom: 15px; width: 100%;">
                    {project_options or '<option value="">No projects available</option>'}
                </select><br>
                
                <label for="version">Select Version Folder (e.g., 5e, version1) from **{selected_project or 'No Project Selected'}**:</label><br>
                <select name="version" id="version" style="padding: 10px; border: 1px solid #ddd; border-radius: 6px; margin-bottom: 15px; width: 100%;">
                    {version_options or '<option value="">No versions available</option>'}
                </select><br>

                <label for="query">Ask a question about the RAG data:</label><br>
                <input type="text" id="query" name="query" placeholder="E.g., What is the policy on data retention?" required><br>
                <input type="submit" value="Get Answer from LightRAG" {form_disabled}>
            </form>
        </div>
        
        {result_html}
    </div>
    </body></html>
    """
    return HTMLResponse(content=html_content)


# --- Dashboard Routes (Modified) ---

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_get(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/")
        
    # On a GET request, there is no RAG result yet.
    return generate_dashboard_html(request)

@app.post("/dashboard")
async def handle_query(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/")

    form = await request.form()
    query = form.get("query")
    project = form.get("project")
    version_folder = form.get("version")
    
    if not project or not version_folder:
        rag_response = "❌ Error: Please select both a Project and a Version Folder to query."
        # Render page immediately with error
        return generate_dashboard_html(request, rag_response, project, version_folder)
        
    try:
        rag_storage_path = PROJECT_ROOT / project / version_folder / "rag_storage"
        rag_instance = LightRAG(project, rag_storage_path)
        
        # AWAIT the asynchronous RAG method
        rag_answer = await rag_instance.get_answer(query)
        
        # Format the successful answer, ready to be inserted into the HTML
        rag_response = f"""
        <p><strong>LightRAG System Output (Project: {project}, Version: {version_folder}):</strong></p>
        <blockquote style='border-left: 3px solid #6c5ce7; padding-left: 15px; margin: 10px 0; background: #fff;'>
            {rag_answer}
        </blockquote>
        <p><strong>Your Original Query:</strong> {query}</p>
        """

    except Exception as e:
        rag_response = f"CRITICAL SYSTEM ERROR during RAG processing: {e}"
    
    # --- KEY CHANGE: RENDER HTML DIRECTLY (NO REDIRECT) ---
    return generate_dashboard_html(request, rag_response, project, version_folder)


# --- Login/Logout (Omitted for brevity) ---
@app.post("/", response_class=HTMLResponse)
async def login(request: Request):
    form = await request.form()
    username = form.get("username")
    password = form.get("password")
    user = users.get(username)
    if user and user["password"] == password:
        request.session["user"] = username
        request.session["role"] = user["role"]
        return RedirectResponse("/dashboard", 302)
    return HTMLResponse("<h3>Invalid credentials</h3><a href='/' style='color: #6c5ce7;'>Try again</a>")

@app.get("/", response_class=HTMLResponse)
def login_form(request: Request):
    if request.session.get("user"):
        return RedirectResponse("/dashboard", 302)
    return """
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .login-card { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); width: 300px; }
        h3 { color: #333; text-align: center; margin-bottom: 20px; }
        input[type="text"], input[type="password"] { width: 100%; padding: 10px; margin-bottom: 15px; border: 1px solid #ccc; border-radius: 6px; box-sizing: border-box; }
        input[type="submit"] { width: 100%; padding: 10px; background-color: #6c5ce7; color: white; border: none; border-radius: 6px; cursor: pointer; transition: background-color 0.3s; }
        input[type="submit"]:hover { background-color: #5b4dd4; }
    </style>
    <html><body>
    <div class="login-card">
        <h3>RAG System Login</h3>
        <form action="/" method="post">
            <label for="username">Username:</label>
            <input type="text" name="username" id="username" required/><br>
            <label for="password">Password:</label>
            <input type="password" name="password" id="password" required/><br>
            <input type="submit" value="Log In"/>
        </form>
    </div></body></html>
    """

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")

# --- Gradio Access and Logic (Unchanged) ---

@app.get("/gradio")
def gradio_access_check(request: Request):
    role = request.session.get("role")
    if role != "admin":
        return HTMLResponse("<h3>Access Denied</h3><p>Only **admin** users can access the file upload page.</p><a href='/dashboard' style='color: #6c5ce7;'>Go to Dashboard</a>")
    return RedirectResponse("/gradio_mounted")


def gradio_upload(files_dir, project, version_folder):
    """
    Handles the directory upload, which should contain .jsonl and vector store files,
    and copies the contents into ./<project>/<version_folder>/rag_storage/.
    """
    if not files_dir:
        return "No directory uploaded."
    
    safe_project = "".join(c for c in project if c.isalnum() or c in ('_', '-')).strip()
    safe_version = "".join(c for c in version_folder if c.isalnum() or c in ('_', '-')).strip()
    
    if not safe_project or not safe_version:
        return "Project and Version Folder name cannot be empty or contain invalid characters."

    source_dir = pathlib.Path(files_dir.name)
    target_dir = PROJECT_ROOT / safe_project / safe_version / "rag_storage"
    
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear existing contents
        for item in target_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        
        # Move new contents
        for item in source_dir.iterdir():
            shutil.move(str(item), str(target_dir / item.name))
        
        return f"✅ Success! RAG artifacts uploaded to: **{target_dir}**"
        
    except Exception as e:
        return f"❌ Upload failed. Error during file movement: {e}"

# Gradio Interface
demo = gr.Interface(
    fn=gradio_upload,
    inputs=[
        gr.File(label="Upload RAG Artifact Directory (contains .jsonl, vector store, etc.)", file_count="directory"),
        gr.Text(label="Project Name (e.g., DandD)", value="DandD"), 
        gr.Text(label="Version Folder Name (e.g., 5e, version1)", value="version1")
    ],
    outputs="markdown",
    title="RAG Artifact Uploader (ADMIN ONLY)",
    description="Uploads the contents of a directory (e.g., .jsonl and vector store files) to: `./<Project Name>/<Version Folder>/rag_storage/`"
)

# Mount Gradio
app = gr.mount_gradio_app(app, demo, path="/gradio_mounted")

# --- Run Application ---

if __name__ == "__main__":
    print("Ensuring default project structure exists and populating dummy data...")
    os.environ["EMBEDDED_OLLAMA_MODEL"] = "nomic-embed-text" 
    
    # Create the full path for testing
    for p, v in [("DandD", "version1"), ("DandD", "versionX"), ("test", "version_a")]:
        storage_path = PROJECT_ROOT / p / v / "rag_storage"
        storage_path.mkdir(parents=True, exist_ok=True)
        
        vector_store_path = storage_path / "vector_store"
        vector_store_path.mkdir(parents=True, exist_ok=True)

        # Create dummy .jsonl and .npy files for the simulation to work
        
        # Dummy .jsonl (meta data)
        dummy_meta_file = storage_path / "meta_0.jsonl"
        with open(dummy_meta_file, 'w') as f:
            dummy_source_path = Path(f"./{p}/{v}/source_data.txt")
            # Write relative path for meta data source
            f.write(json.dumps({"source_path": str(dummy_source_path), "text_start": 0, "text_end": 500}) + "\n")
        
        # Dummy .npy vector file
        dummy_vec_file = vector_store_path / "vec_0.npy"
        dummy_vectors = np.random.rand(1, 1024).astype(np.float32)
        np.save(dummy_vec_file, dummy_vectors)
        
        # Create the dummy source text file the metadata points to
        dummy_source_file = PROJECT_ROOT / p / v / "source_data.txt"
        dummy_source_file.parent.mkdir(parents=True, exist_ok=True)
        with open(dummy_source_file, 'w') as f:
            f.write(f"This is the source document for project {p}, version {v}. It contains key information about data retention and Cleric spell slots, which should be retrieved when queried.\n\n")
            f.write(f"Clerics are divine spellcasters who gain their power from a god or faith. Their spellcasting modifier is Wisdom.")


    config = uvicorn.Config(app, host="0.0.0.0", port=80)
    server = uvicorn.Server(config)
    server.run()