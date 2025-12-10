from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import pathlib, shutil, uvicorn

# FastAPI app
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="supersecretkey123")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Users with roles
users = {
    "alice": {"password": "alice123", "role": "admin"},
    "bob": {"password": "bob123", "role": "user"},
    "charlie": {"password": "charlie123", "role": "viewer"},
}

# Login
@app.get("/", response_class=HTMLResponse)
def login_form(request: Request):
    if request.session.get("user"):
        return RedirectResponse("/gradio")
    return """
    <html><body>
    <form action="/" method="post">
        <input name="username"/>
        <input type="password" name="password"/>
        <input type="submit"/>
    </form></body></html>
    """

@app.post("/", response_class=HTMLResponse)
async def login(request: Request):
    form = await request.form()
    username = form.get("username")
    password = form.get("password")
    user = users.get(username)
    if user and user["password"] == password:
        request.session["user"] = username
        request.session["role"] = user["role"]
        return RedirectResponse("/gradio", 302)
    return HTMLResponse("<h3>Invalid credentials</h3><a href='/'>Try again</a>")

@app.get("/gradio")
def gradio_redirect(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/")
    return RedirectResponse("/gradio")

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")

def gradio_upload(file):
    return f"File uploaded: {file.name}"

# Mount Gradio once
demo = gr.Interface(fn=gradio_upload, inputs=gr.File(), outputs="text")
app = gr.mount_gradio_app(app, demo, path="/gradio")

# Protect access via a FastAPI route BEFORE Gradio
@app.get("/gradio_protected")
def gradio_protected(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/")
    # Simply redirect to the mounted Gradio app
    return RedirectResponse("/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
