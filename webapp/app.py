from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.pipeline import create_llm

from .chat_store import clear_messages, create_chat_id, ensure_chat_storage, load_messages, save_messages
from .knowledge_base import (
    UPLOADS_DIR,
    ask,
    ensure_storage,
    get_source_content,
    ingest_files,
    list_sources,
)


APP_TITLE = "RAG MVP Chat"
ADMIN_USERNAME = os.getenv("RAG_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("RAG_ADMIN_PASSWORD", "admin")
ADMIN_COOKIE_NAME = "rag_admin_auth"
ADMIN_COOKIE_VALUE = os.getenv("RAG_ADMIN_TOKEN", "mvp-admin-token")
CHAT_COOKIE_NAME = "rag_chat_id"
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen:7b")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title=APP_TITLE)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

ensure_storage()
ensure_chat_storage()
llm = create_llm(LLM_MODEL)


def _is_admin(request: Request) -> bool:
    return request.cookies.get(ADMIN_COOKIE_NAME) == ADMIN_COOKIE_VALUE


def _chat_id_from_request(request: Request) -> str:
    return request.cookies.get(CHAT_COOKIE_NAME) or create_chat_id()


def _run_chat_turn(chat_id: str, query: str) -> tuple[list[dict], dict]:
    messages = load_messages(chat_id)
    messages.append({"role": "user", "text": query})

    data = ask(query=query, llm=llm, history=messages)
    assistant_message = {
        "role": "assistant",
        "text": data["answer"],
        "sources": data["sources"],
    }
    messages.append(assistant_message)
    save_messages(chat_id, messages)
    return messages, assistant_message


def _set_chat_cookie(response, chat_id: str):
    response.set_cookie(
        key=CHAT_COOKIE_NAME,
        value=chat_id,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,
    )
    return response


def _set_admin_cookie(response):
    response.set_cookie(
        key=ADMIN_COOKIE_NAME,
        value=ADMIN_COOKIE_VALUE,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24,
    )
    return response


def _render_home(
    request: Request,
    *,
    query: str = "",
    messages=None,
    last_sources=None,
    error: str | None = None,
):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "is_admin": _is_admin(request),
            "title": APP_TITLE,
            "sources": list_sources(),
            "error": error,
            "query": query,
            "messages": messages or [],
            "last_sources": last_sources or [],
        },
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    chat_id = _chat_id_from_request(request)
    static_index = STATIC_DIR / "index.html"
    if static_index.exists():
        response = FileResponse(static_index)
    else:
        response = _render_home(request, messages=load_messages(chat_id))
    return _set_chat_cookie(response, chat_id)


@app.get("/chat", response_class=HTMLResponse)
async def chat_home(request: Request):
    chat_id = _chat_id_from_request(request)
    static_index = STATIC_DIR / "index.html"
    if static_index.exists():
        response = FileResponse(static_index)
    else:
        response = _render_home(request, messages=load_messages(chat_id))
    return _set_chat_cookie(response, chat_id)


@app.get("/spa", response_class=HTMLResponse)
async def spa_home(request: Request):
    chat_id = _chat_id_from_request(request)
    static_index = STATIC_DIR / "index.html"
    if static_index.exists():
        response = FileResponse(static_index)
    else:
        response = _render_home(request, messages=load_messages(chat_id))
    return _set_chat_cookie(response, chat_id)


@app.get("/legacy", response_class=HTMLResponse)
async def legacy_home(request: Request):
    chat_id = _chat_id_from_request(request)
    response = _render_home(request, messages=load_messages(chat_id))
    return _set_chat_cookie(response, chat_id)


@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, query: str = Form(...)):
    chat_id = _chat_id_from_request(request)
    messages, assistant_message = _run_chat_turn(chat_id, query)

    response = _render_home(
        request,
        query="",
        messages=messages,
        last_sources=assistant_message["sources"],
    )
    return _set_chat_cookie(response, chat_id)


@app.post("/api/chat")
async def chat_api(request: Request):
    payload = await request.json()
    query = (payload.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "Пустой вопрос."}, status_code=400)

    chat_id = _chat_id_from_request(request)
    _, assistant_message = _run_chat_turn(chat_id, query)
    response = JSONResponse(
        {
            "answer": assistant_message["text"],
            "sources": assistant_message.get("sources", []),
        }
    )
    return _set_chat_cookie(response, chat_id)


@app.get("/api/session")
async def session_api(request: Request):
    chat_id = _chat_id_from_request(request)
    response = JSONResponse({"is_admin": _is_admin(request)})
    return _set_chat_cookie(response, chat_id)


@app.get("/api/chat/history")
async def chat_history_api(request: Request):
    chat_id = _chat_id_from_request(request)
    response = JSONResponse({"messages": load_messages(chat_id)})
    return _set_chat_cookie(response, chat_id)


@app.post("/api/chat/clear")
async def clear_chat_api(request: Request):
    chat_id = _chat_id_from_request(request)
    clear_messages(chat_id)
    response = JSONResponse({"ok": True, "messages": []})
    return _set_chat_cookie(response, chat_id)


@app.get("/api/sources")
async def list_sources_api():
    return JSONResponse({"sources": list_sources()})


@app.get("/api/source-content")
async def source_content(source_path: str):
    try:
        payload = get_source_content(source_path)
        return JSONResponse(payload)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)


@app.post("/chat/clear")
async def clear_chat(request: Request):
    chat_id = _chat_id_from_request(request)
    clear_messages(chat_id)
    response = RedirectResponse(url="/", status_code=303)
    return _set_chat_cookie(response, chat_id)


@app.post("/api/admin/login")
async def admin_login_api(request: Request):
    payload = await request.json()
    username = (payload.get("username") or "").strip()
    password = (payload.get("password") or "").strip()
    chat_id = _chat_id_from_request(request)

    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        response = JSONResponse({"ok": True, "is_admin": True})
        _set_admin_cookie(response)
        return _set_chat_cookie(response, chat_id)

    response = JSONResponse(
        {"ok": False, "is_admin": False, "error": "Неверный логин или пароль."},
        status_code=401,
    )
    return _set_chat_cookie(response, chat_id)


@app.post("/api/admin/logout")
async def admin_logout_api(request: Request):
    chat_id = _chat_id_from_request(request)
    response = JSONResponse({"ok": True, "is_admin": False})
    response.delete_cookie(ADMIN_COOKIE_NAME)
    return _set_chat_cookie(response, chat_id)


@app.post("/api/admin/upload")
async def upload_files_api(request: Request, files: list[UploadFile] = File(...)):
    if not _is_admin(request):
        return JSONResponse({"error": "Требуется авторизация админа."}, status_code=403)

    saved_paths = []
    for file in files:
        if not file.filename:
            continue
        destination = UPLOADS_DIR / file.filename
        content = await file.read()
        destination.write_bytes(content)
        saved_paths.append(destination)

    result = ingest_files(saved_paths)
    return JSONResponse(
        {
            "ok": True,
            "result": result,
            "sources": list_sources(),
        }
    )


@app.post("/admin/login")
async def admin_login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        response = RedirectResponse(url="/", status_code=303)
        _set_admin_cookie(response)
        return response
    chat_id = _chat_id_from_request(request)
    response = _render_home(
        request,
        messages=load_messages(chat_id),
        error="Неверный логин или пароль.",
    )
    return _set_chat_cookie(response, chat_id)


@app.post("/admin/logout")
async def admin_logout(request: Request):
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(ADMIN_COOKIE_NAME)
    return response


@app.post("/admin/upload")
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    if not _is_admin(request):
        return RedirectResponse(url="/", status_code=303)

    saved_paths = []
    for file in files:
        if not file.filename:
            continue
        destination = UPLOADS_DIR / file.filename
        content = await file.read()
        destination.write_bytes(content)
        saved_paths.append(destination)

    ingest_files(saved_paths)
    return RedirectResponse(url="/", status_code=303)
