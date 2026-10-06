"""CalibGuard 后端服务（T-G3，FastAPI）：serve 前端静态页 + REST API。

路由（逐字，见 GUI 文档 T-G3；/api/settings 的 GET 为 2026-10-05 扩展）：
  GET  /api/health          {"status":"ok","model_dir":"models/","calib_ok":true}
  GET  /                    返回 web/index.html
  POST /api/jobs            body {"paths":[...]} → {"job_id":"..."}
  GET  /api/jobs/{job_id}   job_status 原样
  GET  /api/model-info      {"model":"codebert_finetuned","calib_fit_at":"..."}
  GET  /api/settings        当前生效设置（供设置面板回填；Key 明文，仅本机 127.0.0.1）
  POST /api/settings        body 为设置对象（支持部分更新）→ {"saved":true,
                            "llm_reload_deferred":false}；非法值 400

铁律：只监听 127.0.0.1；端口由 launcher 通过 --port 传入（禁止固定端口）；不吞异常。
"""
import argparse
import json
import os
import sys
import uuid

import uvicorn
from fastapi import Body, FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import audit_service

_ROOT = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = "models/"                                          # /api/health 报的目录名
_MODEL_FILE = os.path.join(_ROOT, "models", "codebert_finetuned", "model.safetensors")
_CALIB_FILE = os.path.join(_ROOT, "models", "calibration_params.json")
_SETTINGS_PATH = os.path.join(_ROOT, "gui-settings.json")
_INDEX_FILE = os.path.join(_ROOT, "web", "index.html")
_UPLOAD_DIR = os.path.join(_ROOT, "uploads")                    # 浏览器上传副本的落盘根
_UPLOAD_MAX_BYTES = 100 * 1024 * 1024                           # 单文件 100MB 上限
_UPLOAD_CHUNK_BYTES = 1024 * 1024                               # 分块读写，避免整文件进内存
_FILENAME_ILLEGAL_CHARS = '\\/:*?"<>|'


def _check_calib_ok() -> bool:
    """模型配套校验：两个产物都在，且 calibration_params.json 不早于 model.safetensors。"""
    if not (os.path.isfile(_MODEL_FILE) and os.path.isfile(_CALIB_FILE)):
        return False
    return not (os.path.getmtime(_CALIB_FILE) < os.path.getmtime(_MODEL_FILE))


def _safe_filename(name: str) -> str:
    """文件名净化：只取 basename + 替换非法字符，防目录穿越（禁止信任浏览器传来的名字）。"""
    base = os.path.basename((name or "").replace("\\", "/"))
    for char in _FILENAME_ILLEGAL_CHARS:
        base = base.replace(char, "_")
    base = base.strip().strip(".")
    return base or "unnamed"


def _unique_target(folder: str, safe_name: str) -> str:
    """同目录重名时追加 _1/_2…，避免同名文件相互覆盖。"""
    target = os.path.join(folder, safe_name)
    if not os.path.exists(target):
        return target
    stem, ext = os.path.splitext(safe_name)
    index = 1
    while os.path.exists(target):
        target = os.path.join(folder, f"{stem}_{index}{ext}")
        index += 1
    return target


# 启动时执行一次（T-G7 用例 5 靠重启服务观察该值变化）
CALIB_OK = _check_calib_ok()

app = FastAPI(title="CalibGuard GUI")


@app.exception_handler(Exception)
async def _unhandled_error(_request, exc: Exception) -> JSONResponse:
    """未捕获异常统一 500 {"error": ...}（禁止吞异常：文本回给前端显示）。"""
    return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


@app.get("/api/health")
def health() -> dict:
    """健康检查：launcher 靠它判断后端就绪。"""
    return {"status": "ok", "model_dir": _MODEL_DIR, "calib_ok": CALIB_OK}


@app.get("/")
def index() -> FileResponse:
    """返回前端单页（同源，无需 StaticFiles）。"""
    return FileResponse(_INDEX_FILE)


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """接收浏览器拖入/选择的文件，落盘 uploads/{uuid8}/，返回可供 /api/jobs 使用的相对路径。

    浏览器拿不到磁盘路径，只能上传内容（补丁 9）。副本在任务结束后保留，不自动清理。
    """
    if not files:
        return JSONResponse(status_code=400, content={"error": "files 不能为空"})
    folder = os.path.join(_UPLOAD_DIR, uuid.uuid4().hex[:8])
    os.makedirs(folder, exist_ok=True)
    saved: list[str] = []
    for item in files:
        safe_name = _safe_filename(item.filename)
        target = _unique_target(folder, safe_name)
        written = 0
        too_large = False
        with open(target, "wb") as handle:
            while True:
                chunk = await item.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > _UPLOAD_MAX_BYTES:
                    too_large = True
                    break
                handle.write(chunk)
        if too_large:
            os.remove(target)
            return JSONResponse(status_code=400,
                                content={"error": f"文件超过 100MB 上限: {safe_name}"})
        saved.append(os.path.relpath(target, _ROOT).replace(os.sep, "/"))
    return {"paths": saved}


@app.post("/api/jobs")
def create_jobs(payload: dict = Body(...)):
    """建审计任务；路径必须真实存在，否则 400。"""
    paths = payload.get("paths")
    if not isinstance(paths, list) or not paths:
        return JSONResponse(status_code=400,
                            content={"error": "paths 必须是非空数组"})
    for path in paths:
        if not isinstance(path, str) or not os.path.isfile(path):
            return JSONResponse(status_code=400,
                                content={"error": f"file not found: {path}"})
    return {"job_id": audit_service.create_job(paths)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    """查询任务状态/进度/结果（job_status 原样返回）。"""
    try:
        return audit_service.job_status(job_id)
    except KeyError:
        return JSONResponse(status_code=404,
                            content={"error": f"job not found: {job_id}"})


@app.get("/api/model-info")
def model_info() -> dict:
    """模型信息：模型名 + 校准参数的拟合时间（fit_at）。"""
    with open(_CALIB_FILE, "r", encoding="utf-8") as handle:
        params = json.load(handle)
    return {"model": "codebert_finetuned", "calib_fit_at": params.get("fit_at")}


@app.get("/api/settings")
def read_settings() -> dict:
    """返回当前生效设置（供设置面板回填）。"""
    return audit_service.get_settings()


@app.post("/api/settings")
def save_settings(payload: dict = Body(...)):
    """保存设置到 gui-settings.json（禁止写入 calibguard/config.yaml）。

    支持部分更新；非法值返回 400（附中文原因），绝不落盘。
    `llm_reload_deferred=true` 表示当前有审计任务在跑，设置应用已暂缓（补丁 11）。
    """
    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"error": "设置必须是 JSON 对象"})
    try:
        applied = audit_service.save_settings(payload)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return {"saved": True, "llm_reload_deferred": not applied,
            "settings": audit_service.get_settings()}


def main() -> None:
    """命令行入口：--port 由 launcher 分配传入。"""
    parser = argparse.ArgumentParser(prog="server.py", description="CalibGuard GUI 后端服务")
    parser.add_argument("--port", type=int, required=True, help="监听端口（launcher 分配）")
    args = parser.parse_args()

    # 一切相对路径（models/、reports/）相对项目根目录解析（铁律 6）
    os.chdir(_ROOT)
    try:
        audit_service.load_settings()            # gui-settings.json → DEEPSEEK_API_KEY
    except Exception as exc:
        print(f"[错误] 读取 gui-settings.json 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
