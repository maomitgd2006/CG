"""T-G7 端到端验收：全部走 HTTP API，不启动浏览器。

用例：
  1. test_health              服务起来后 /api/health 200 且 calib_ok 为 true
  2. test_audit_consistency   同一文件双跑结果一致（推理确定性）+ 结果键 ⊆ 引擎报告骨架键
  3. test_missing_file_400    不存在路径 → 400
  4. test_settings_roundtrip  Key 落盘 + load_settings() 注入环境变量
  5. test_settings_full_roundtrip  完整设置（开关/URL/模型名/灰区/内容源/三模式+参数）读写
  6. test_settings_invalid_400     非法设置 → 400 且不落盘
  7. test_llm_payload_modes   LLM 投递三模式分片逻辑（纯函数，不加载模型）
  8. test_calib_mismatch_flag 更新 model.safetensors → 重启服务 → calib_ok=false（用例最后执行）
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)                 # 让测试进程能 import audit_service / calibguard

PYTHON = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
MODEL_FILE = os.path.join(ROOT, "models", "codebert_finetuned", "model.safetensors")
CALIB_FILE = os.path.join(ROOT, "models", "calibration_params.json")
SETTINGS_FILE = os.path.join(ROOT, "gui-settings.json")
SERVER_LOG = os.path.join(ROOT, "logs", "test-server.log")
SAMPLE = "samples/demo.py"
JOB_TIMEOUT_S = 300                          # 首次含模型加载，单任务上限 5 分钟


# ============================== HTTP 小工具（只用标准库） ==============================

def _request(url: str, method: str = "GET", payload=None) -> tuple[int, dict]:
    """发请求并返回 (状态码, JSON)；HTTP 错误码不抛异常，交给用例断言。"""
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return response.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        return exc.code, (json.loads(body) if body else {})


def _free_port() -> int:
    """取一个空闲端口（与 launcher 同规则：socket bind 0，禁止固定端口）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Server:
    """被测试的后端进程句柄（源码方式启动 server.py）。"""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self._log = None

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def start(self, timeout_s: float = 60) -> None:
        self.port = _free_port()
        os.makedirs(os.path.dirname(SERVER_LOG), exist_ok=True)
        self._log = open(SERVER_LOG, "ab")
        self.proc = subprocess.Popen([PYTHON, "server.py", "--port", str(self.port)],
                                     cwd=ROOT, stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server 进程提前退出，退出码 {self.proc.returncode}，"
                                   f"见日志 {SERVER_LOG}")
            try:
                status, body = _request(self.url("/api/health"))
                if status == 200 and body.get("status") == "ok":
                    return
            except Exception:
                pass
            time.sleep(0.3)
        raise RuntimeError(f"server 在 {timeout_s} 秒内未就绪，见日志 {SERVER_LOG}")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self._log is not None:
            self._log.close()
            self._log = None

    def restart(self) -> None:
        self.stop()
        self.start()


def _run_job_and_wait(server: Server, paths: list[str]) -> dict:
    """提交任务并轮询到终态，返回最终 job 状态。"""
    status, created = _request(server.url("/api/jobs"), "POST", {"paths": paths})
    assert status == 200, f"POST /api/jobs 返回 {status}: {created}"
    job_id = created["job_id"]
    deadline = time.time() + JOB_TIMEOUT_S
    while time.time() < deadline:
        status, job = _request(server.url(f"/api/jobs/{job_id}"))
        assert status == 200, f"GET /api/jobs/{job_id} 返回 {status}: {job}"
        if job["state"] in ("done", "error"):
            return job
        time.sleep(1)
    raise AssertionError(f"任务 {job_id} 在 {JOB_TIMEOUT_S} 秒内未结束")


# ============================== 夹具 ==============================

@pytest.fixture(scope="session")
def server():
    """整个测试会话共用一个 server 进程（模型只加载一次）。"""
    instance = Server()
    instance.start()
    yield instance
    instance.stop()


# ============================== 用例 1 ==============================

def test_health(server):
    """服务健康且模型配套。"""
    status, body = _request(server.url("/api/health"))
    assert status == 200, f"HTTP {status}: {body}"
    assert body["status"] == "ok"
    assert body["model_dir"] == "models/"
    assert body["calib_ok"] is True


# ============================== 用例 2 ==============================

def test_audit_consistency(server):
    """同一文件双跑：决策/概率/窗口数一致；结果键集合 ⊆ 引擎报告骨架键集合。"""
    first = _run_job_and_wait(server, [SAMPLE])
    second = _run_job_and_wait(server, [SAMPLE])
    assert first["state"] == "done" and second["state"] == "done"
    assert len(first["results"]) == 1 and len(second["results"]) == 1
    left, right = first["results"][0], second["results"][0]

    # 引擎报告骨架恒含 "error" 键（成功时为 None），故断言其值为空而非键不存在
    assert left.get("error") is None, f"审计失败：{left.get('error')}"
    assert right.get("error") is None, f"审计失败：{right.get('error')}"
    assert left["final_decision"] == right["final_decision"]
    assert abs(left["calibrated_probability"] - right["calibrated_probability"]) <= 1e-6
    assert left["windows_evaluated"] == right["windows_evaluated"]

    # 基准键集合 = 引擎 pipeline._new_report() 产出的报告骨架（schemas.py 无 AuditReport 类）。
    # 说明：直接构造 AuditPipeline 会再加载一次模型，这里用最小桩对象调用该真实方法，
    # 只满足它用到的 self.calibrator.method，不修改引擎任何代码。
    from calibguard.pipeline import AuditPipeline

    class _StubCalibrator:
        method = "platt"

    class _StubPipeline:
        calibrator = _StubCalibrator()

    allowed_keys = set(AuditPipeline._new_report(_StubPipeline(), SAMPLE).keys())
    assert allowed_keys, "引擎报告骨架键集合为空，基准不可用"
    assert set(left.keys()) <= allowed_keys, (
        f"API 层篡改了报告字段：多出 {sorted(set(left.keys()) - allowed_keys)}")


# ============================== 用例 3 ==============================

def test_missing_file_400(server):
    """不存在的路径必须 400。"""
    status, body = _request(server.url("/api/jobs"), "POST",
                            {"paths": ["samples/__not_exist__.py"]})
    assert status == 400, f"期望 400，实际 {status}: {body}"
    assert "file not found" in body["error"]


# ============================== 用例 4 ==============================

def test_settings_roundtrip(server):
    """Key 落盘 gui-settings.json，并经 load_settings() 注入 DEEPSEEK_API_KEY。"""
    import audit_service

    original = None
    if os.path.isfile(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
            original = handle.read()
    test_key = "sk-test-roundtrip-0000000000"
    try:
        status, body = _request(server.url("/api/settings"), "POST",
                                {"deepseek_api_key": test_key})
        assert status == 200, f"HTTP {status}: {body}"
        assert body["saved"] is True

        assert os.path.isfile(SETTINGS_FILE), "gui-settings.json 未生成"
        with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        assert saved["deepseek_api_key"] == test_key

        os.environ.pop("DEEPSEEK_API_KEY", None)
        audit_service.load_settings()
        assert os.environ["DEEPSEEK_API_KEY"] == test_key
    finally:
        # 还原开发者原有的 Key（本用例禁止污染真实设置）
        os.environ.pop("DEEPSEEK_API_KEY", None)
        if original is None:
            if os.path.isfile(SETTINGS_FILE):
                os.remove(SETTINGS_FILE)
        else:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as handle:
                handle.write(original)


# ============================== 用例 5/6（2026-10-05 设置扩展） ==============================

def _backup_settings_file():
    """备份 gui-settings.json 原文（不存在返回 None）。"""
    if not os.path.isfile(SETTINGS_FILE):
        return None
    with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
        return handle.read()


def _restore_settings_file(server: Server, original) -> None:
    """还原 gui-settings.json，并把运行中的后端内存设置同步回原值。"""
    if original is None:
        if os.path.isfile(SETTINGS_FILE):
            os.remove(SETTINGS_FILE)
        return
    with open(SETTINGS_FILE, "w", encoding="utf-8") as handle:
        handle.write(original)
    try:
        payload = json.loads(original)
    except ValueError:
        return
    if isinstance(payload, dict):
        _request(server.url("/api/settings"), "POST", payload)


def test_settings_full_roundtrip(server):
    """完整设置落盘并可经 GET /api/settings 原样读回。"""
    original = _backup_settings_file()
    payload = {
        "llm_enabled": True,
        "deepseek_api_key": "sk-test-full-0000000000",
        "llm_base_url": "https://api.example.com/v1",
        "llm_model": "my-model",
        "allow_threshold": 0.2,
        "block_threshold": 0.8,
        "llm_content_source": "raw",
        "llm_delivery_mode": "window",
        "llm_truncate_chars": 1234,
        "llm_window_chars": 2000,
        "llm_window_overlap": 400,
    }
    try:
        status, body = _request(server.url("/api/settings"), "POST", payload)
        assert status == 200, f"HTTP {status}: {body}"
        assert body["saved"] is True

        status, got = _request(server.url("/api/settings"))
        assert status == 200, f"HTTP {status}: {got}"
        for key, value in payload.items():
            assert got[key] == value, f"{key}: 期望 {value!r}，实际 {got[key]!r}"

        # 落盘文件与内存设置一致
        with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        assert saved["llm_delivery_mode"] == "window"
        assert saved["llm_window_overlap"] == 400
    finally:
        _restore_settings_file(server, original)


def test_settings_invalid_400(server):
    """非法设置（灰区下限 ≥ 上限 / 滑窗重叠 ≥ 滑窗大小）→ 400，且不落盘。"""
    original = _backup_settings_file()
    try:
        status, body = _request(server.url("/api/settings"), "POST",
                                {"allow_threshold": 0.9, "block_threshold": 0.1})
        assert status == 400, f"期望 400，实际 {status}: {body}"
        assert "灰区" in body["error"], f"错误文本未说明原因：{body}"

        status, body = _request(server.url("/api/settings"), "POST",
                                {"llm_delivery_mode": "window",
                                 "llm_window_chars": 100, "llm_window_overlap": 500})
        assert status == 400, f"期望 400，实际 {status}: {body}"
        assert "重叠" in body["error"], f"错误文本未说明原因：{body}"

        # 非法请求不得改变已生效的设置
        status, got = _request(server.url("/api/settings"))
        assert status == 200
        assert got["allow_threshold"] < got["block_threshold"]
    finally:
        _restore_settings_file(server, original)


def test_llm_payload_modes():
    """LLM 投递三模式分片逻辑（纯逻辑，不加载模型）。"""
    from calibguard.config import AppConfig
    from calibguard.feature_extractor import extract_features
    from calibguard.pipeline import AuditPipeline

    pipeline = object.__new__(AuditPipeline)      # 只测纯逻辑，绕过模型加载
    pipeline.cfg = AppConfig()
    evidence = extract_features("import os\nos.system('x')\n")
    text = "x" * 1000

    def payloads(source, mode, truncate=100, window=300, overlap=50):
        pipeline.cfg.llm_content_source = source
        pipeline.cfg.llm_delivery_mode = mode
        pipeline.cfg.llm_truncate_chars = truncate
        pipeline.cfg.llm_window_chars = window
        pipeline.cfg.llm_window_overlap = overlap
        return pipeline._build_llm_payloads(evidence, text)

    assert payloads("raw", "all") == [text]                       # 全量：整段
    assert payloads("raw", "truncate") == [text[:100]]            # 截断：前 N 字符

    chunks = payloads("raw", "window")
    assert len(chunks) == 4, f"1000 字符 / (300-50) 应为 4 段，实际 {len(chunks)}"
    assert [len(item) for item in chunks] == [300, 300, 300, 250]
    assert chunks[0] == text[:300] and chunks[-1] == text[-250:]  # 首尾覆盖到边界

    assert len(payloads("raw", "window", window=2000)) == 1        # 内容短于窗口：单段
    assert payloads("evidence", "all")[0].startswith("【危险API调用】")  # 内容源=证据


# ============================== 用例 7（必须最后执行） ==============================

def test_calib_mismatch_flag(server):
    """换模型只换一半（model 更新、calib 偏旧）→ 重启服务后 calib_ok=false。"""
    backup = MODEL_FILE + ".testbak"
    os.replace(MODEL_FILE, backup)                 # 同盘改名，瞬时且保留原 mtime
    try:
        with open(MODEL_FILE, "wb") as handle:
            handle.write(b"dummy-not-a-real-model")   # 复制小文件改名（这里直接写小文件）
        os.utime(MODEL_FILE, None)                    # mtime = 现在，晚于 calibration_params.json
        assert os.path.getmtime(MODEL_FILE) > os.path.getmtime(CALIB_FILE)

        server.restart()
        status, body = _request(server.url("/api/health"))
        assert status == 200, f"HTTP {status}: {body}"
        assert body["calib_ok"] is False, "model 比 calib 新时 calib_ok 应为 false"
    finally:
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        os.replace(backup, MODEL_FILE)                # 还原真模型（mtime 不变）
        server.restart()

    status, body = _request(server.url("/api/health"))
    assert status == 200 and body["calib_ok"] is True, "还原真模型后 calib_ok 应恢复为 true"
