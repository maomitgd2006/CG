"""CalibGuard 启动器（launcher）——三层架构的第一层，只允许标准库 + tkinter。

职责：
  1. 【首启引导】find_python 检测 Python 3.10.x → 内嵌 uv.exe 建 .venv → 按 GPU 定档装依赖；
  2. 【每次启动】校验环境 → 用 .venv 的 python 启动 server.py → 打开浏览器。

铁律（GUI 文档第 0 节）：
  - 本文件严禁 import 任何第三方库：首启时 fastapi/uvicorn/torch 都还不存在；
  - 只监听 127.0.0.1，端口由系统分配（socket bind 0），禁止固定端口；
  - 一切路径相对 exe/项目根目录解析，禁止写死绝对路径；
  - 任何一步失败都必须让用户看到错误文本（禁止吞异常）。
"""
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.request
import webbrowser
from tkinter import messagebox, ttk

# ============ 版本与来源常量（铁律 3：写死，禁止改动） ============
PYTHON_SERIES = "3.10"          # 允许使用的 Python 系列（3.11+ 不使用：torch cp310 轮子约束）
PYTHON_SERIES_PREFIX = "Python 3.10."
PYTHON_PIN = "3.10.8"           # 未装 3.10.x 时 uv 精确安装的版本
UV_VERSION = "0.5.11"           # tools/uv.exe 版本（禁止追新）
PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
UV_PYTHON_INSTALL_MIRROR = ("https://ghproxy.cn/https://github.com/astral-sh/"
                            "python-build-standalone/releases/download")
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CUDA_MIN = (12, 8)              # CUDA 上限 ≥ 12.8 才走 cu128 轮子，否则 CPU 版
# cu128 轮子包含的最低算力（sm_75）：低于此值（如 GTX 10 系 sm_61）GPU 上无可用 kernel，
# 运行时必报 "CUDA error: no kernel image is available"（每份文件都转人工 + 报错），
# 故定档阶段在 CUDA 版本之外再做算力双检，不达标直接判 CPU 版
CUDA_MIN_COMPUTE_CAP = (7, 5)

# ============ 超时常量 ============
HEALTH_TIMEOUT_S = 120          # 等待 /api/health 就绪的上限（秒）
HEALTH_INTERVAL_S = 2           # 健康检查轮询间隔（秒）
HEALTH_REQUEST_TIMEOUT_S = 3    # 单次健康检查请求超时（秒）
CMD_TIMEOUT_S = 3600            # 单条子进程命令上限（秒）：装 torch 可能很久

# 各步骤的“镜像降级”环境变量（仅在失败重试时启用）
_PIP_MIRROR_ENV = {"PIP_INDEX_URL": PIP_MIRROR}
_UV_MIRROR_ENV = {"UV_INDEX_URL": PIP_MIRROR, "UV_DEFAULT_INDEX": PIP_MIRROR}
_UV_PYTHON_MIRROR_ENV = {"UV_PYTHON_INSTALL_MIRROR": UV_PYTHON_INSTALL_MIRROR}

# 「重试（使用镜像）」按钮按下后置 True：下一次引导优先走镜像源
_MIRROR_FIRST = False


class BootstrapError(RuntimeError):
    """首启引导失败（带用户可读的中文说明）。"""


# ============================== 路径与日志 ==============================

def _root_dir() -> str:
    """项目根目录：打包运行取 exe 所在目录，源码运行取本文件所在目录。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _venv_python(root: str) -> str:
    """用户 venv 的 python 解释器路径。"""
    return os.path.join(root, ".venv", "Scripts", "python.exe")


def _bootstrap_log_path(root: str) -> str:
    """首启引导日志路径（logs/bootstrap.log）。"""
    return os.path.join(root, "logs", "bootstrap.log")


class _BootstrapLog:
    """bootstrap 日志：时间戳 + 每步命令的 stdout/stderr（禁止记录任何 API Key）。"""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("\n" + "=" * 72 + "\n")
                handle.write(f"[{_now()}] 开始环境准备\n")
        except OSError:
            # 日志目录不可写不能阻断引导：错误仍会通过 UI 冒泡
            pass

    def write(self, text: str) -> None:
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(f"[{_now()}] {text}\n")
            except OSError:
                pass


def _now() -> str:
    """本地时间戳字符串（秒级）。"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _tail(path: str, lines: int) -> str:
    """读文件末尾若干行（用于把 server 崩溃信息显示给用户）。"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read().splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


# ============================== 子进程封装 ==============================

def _run(cmd: list[str], cwd: str | None, env_extra: dict | None = None,
         timeout: float | None = CMD_TIMEOUT_S) -> subprocess.CompletedProcess:
    """执行命令；非零退出码抛 BootstrapError（附 stderr 末尾，禁止吞异常）。"""
    env = os.environ.copy()
    if env_extra:
        env.update({str(k): str(v) for k, v in env_extra.items()})
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except FileNotFoundError as exc:
        raise BootstrapError(f"命令不存在：{cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise BootstrapError(f"命令超时（{timeout} 秒）：{' '.join(cmd)}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        lines = detail.splitlines()[-12:]
        raise BootstrapError(f"命令失败（退出码 {proc.returncode}）：{' '.join(cmd)}\n"
                             + "\n".join(lines))
    return proc


def _run_quiet(cmd: list[str], cwd: str | None = None,
               timeout: float = 60) -> str:
    """执行探测类命令；失败返回空串（用于 nvidia-smi/where 之类可缺命令的探测）。"""
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout or ""


def _uv_exe(root: str) -> str:
    """内嵌 uv.exe 路径（打包适配：onefile 解压在 _MEIPASS/tools/）。"""
    path = os.path.join(getattr(sys, "_MEIPASS", root), "tools", "uv.exe")
    if not os.path.isfile(path):
        raise BootstrapError(
            f"内嵌 uv.exe 缺失：{path}（打包需 --add-binary \"tools/uv.exe;tools\"，"
            f"源码运行需 tools/uv.exe，版本 {UV_VERSION}）")
    return path


# ============================== T-G1 步骤 1：找 Python ==============================

def _python_version_of(path: str) -> str:
    """返回 `python --version` 的输出（失败返回空串）。"""
    return (_run_quiet([path, "--version"]) or "").strip()


def find_python() -> str:
    """返回可用的 3.10.x python.exe 完整路径；找不到返回空串。"""
    # 途径 1：Windows py 启动器（py -0p 列出全部已装解释器）
    listing = _run_quiet(["py", "-0p"])
    for line in listing.splitlines():
        match = re.match(r"^\s*[*-]?\s*(?:-V:)?(3\.10[^\s]*)\s+(.+?)\s*$", line)
        if not match:
            continue
        candidate = match.group(2).strip().strip('"')
        if os.path.isfile(candidate) and _python_version_of(candidate).startswith(
                PYTHON_SERIES_PREFIX):
            return candidate
    # 途径 2：PATH 上的 python / python3，逐个 --version 过滤 3.10.x
    for name in ("python", "python3"):
        found = _run_quiet(["where", name])
        for line in found.splitlines():
            candidate = line.strip().strip('"')
            if not candidate or not os.path.isfile(candidate):
                continue
            if _python_version_of(candidate).startswith(PYTHON_SERIES_PREFIX):
                return candidate
    return ""


# ============================== T-G1 步骤 2-5：bootstrap ==============================

def _attempt_plan(mirror_env: dict | None) -> list[tuple[str, dict]]:
    """每步的重试计划：原样 → 原样重试 → 镜像重试；手动重试时镜像优先。"""
    if not mirror_env:
        return [("第 1 次尝试", {}), ("第 2 次尝试（原样重试）", {})]
    if _MIRROR_FIRST:
        return [("镜像尝试", mirror_env),
                ("镜像重试", mirror_env),
                ("原样尝试", {})]
    return [("第 1 次尝试", {}),
            ("第 2 次尝试（原样重试）", {}),
            ("第 3 次尝试（镜像降级）", mirror_env)]


def _install_step(label: str, apply_fn, log: _BootstrapLog, progress_cb,
                  mirror_env: dict | None = None):
    """执行一步：失败先原样重试一次 → 再换镜像重试一次 → 仍失败则报错。"""
    last_error = ""
    for attempt_label, env_extra in _attempt_plan(mirror_env):
        progress_cb(f"{label}（{attempt_label}）")
        log.write(f"[步骤] {label} {attempt_label}"
                  + (f" 镜像环境={sorted(env_extra)}" if env_extra else ""))
        try:
            result = apply_fn(env_extra)
        except Exception as exc:                      # 任何异常都要记录并继续重试
            last_error = f"{type(exc).__name__}: {exc}"
            log.write(f"[失败] {label} {attempt_label}: {last_error}")
            continue
        log.write(f"[成功] {label} {attempt_label}")
        return result
    raise BootstrapError(f"{label} 失败（已原样重试 + 镜像重试）：{last_error}")


def _detect_gpu() -> tuple[bool, str]:
    """检测 NVIDIA 显卡：返回 (是否存在, 显卡名)。"""
    out = _run_quiet(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    name = out.strip().splitlines()[0].strip() if out.strip() else ""
    return (True, name) if name else (False, "")


def _detect_cuda_version() -> tuple[int, int] | None:
    """取 CUDA 上限：先看 nvidia-smi 右上角 CUDA Version，再退回 nvcc --version。"""
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", _run_quiet(["nvidia-smi"], timeout=60))
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"release\s+(\d+)\.(\d+)", _run_quiet(["nvcc", "--version"], timeout=60))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _detect_compute_cap() -> tuple[int, int] | None:
    """取 GPU 算力（compute capability，如 GTX 1060 = 6.1）：nvidia-smi 直接可查。

    取第一块可见 NVIDIA 卡（与 _detect_gpu 的取名行为一致）；查询失败返回 None
    （驱动 ≥ 12.8 时该查询必然支持，失败说明探测本身异常，调用方按兼容处理）。
    """
    out = _run_quiet(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    match = re.search(r"(\d+)\.(\d+)", out or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def bootstrap(root: str, progress_cb) -> str:
    """首启环境引导：检测 Python → uv 建 venv → 定档装依赖；返回 venv python 路径。"""
    log = _BootstrapLog(_bootstrap_log_path(root))
    progress_cb("正在检查 Python 运行环境…")

    # ---------- 步骤 1：检测已装的 Python 3.10.x ----------
    python_path = _install_step("检测 Python 3.10.x", lambda _env: find_python(),
                                log, progress_cb)

    # ---------- 步骤 2：没有 3.10.x 时用内嵌 uv 精确安装 3.10.8 ----------
    if not python_path:
        def _uv_install_python(env_extra: dict) -> str:
            uv_exe = _uv_exe(root)
            _run([uv_exe, "python", "install", PYTHON_PIN], cwd=root, env_extra=env_extra)
            found = _run([uv_exe, "python", "find", PYTHON_PIN], cwd=root,
                         env_extra=env_extra).stdout.strip().splitlines()
            path = found[-1].strip() if found else ""
            if not path or not os.path.isfile(path):
                raise BootstrapError(f"uv 未返回可用的 Python 路径：{path!r}")
            if not _python_version_of(path).startswith(PYTHON_SERIES_PREFIX):
                raise BootstrapError(f"uv 安装的解释器不是 {PYTHON_SERIES} 系列：{path}")
            return path

        python_path = _install_step(f"安装 Python {PYTHON_PIN}", _uv_install_python,
                                    log, progress_cb, mirror_env=_UV_PYTHON_MIRROR_ENV)

    def _uv_create_venv(env_extra: dict) -> str:
        uv_exe = _uv_exe(root)
        # 上次引导残留的半成品 .venv（没有 python.exe）先清掉，uv 才能重建
        venv_dir = os.path.join(root, ".venv")
        if os.path.isdir(venv_dir) and not os.path.isfile(_venv_python(root)):
            log.write("[清理] 删除不完整的 .venv 后重建")
            _remove_tree(venv_dir)
        # --seed：预装 pip/setuptools，后续步骤才能用 python -m pip
        _run([uv_exe, "venv", "--seed", "--python", python_path, ".venv"],
             cwd=root, env_extra=env_extra)
        exe = _venv_python(root)
        if not os.path.isfile(exe):
            raise BootstrapError(f"uv venv 未产出解释器：{exe}")
        return exe

    # ---------- 步骤 3：建 .venv ----------
    venv_python = _install_step("创建虚拟环境 .venv", _uv_create_venv,
                                log, progress_cb, mirror_env=_UV_MIRROR_ENV)

    # ---------- 步骤 4：GPU 定档（CUDA 版本 + 算力双检） ----------
    progress_cb("检测显卡与 CUDA 版本…")
    has_nvidia, gpu_name = _detect_gpu()
    cuda_version = _detect_cuda_version() if has_nvidia else None
    compute_cap = _detect_compute_cap() if has_nvidia else None
    cuda_ok = cuda_version is not None and cuda_version >= CUDA_MIN
    # 算力未知（探测异常）按兼容放行：CUDA ≥ 12.8 的驱动必然支持该查询，
    # 走到 None 分支说明环境本身已异常，维持旧行为交由上层报错可见
    cap_ok = compute_cap is None or compute_cap >= CUDA_MIN_COMPUTE_CAP
    use_cuda = bool(has_nvidia and cuda_ok and cap_ok)
    if has_nvidia and not use_cuda:
        if not cuda_ok:
            if cuda_version is None:
                progress_cb("检测到 NVIDIA 但无法确定 CUDA 版本，使用 CPU 版")
            else:
                progress_cb("检测到 NVIDIA 但 CUDA<12.8，使用 CPU 版")
        else:
            progress_cb(f"检测到 NVIDIA（算力 {compute_cap[0]}.{compute_cap[1]}）"
                       f"低于 cu128 轮子支持的最低算力 7.5，使用 CPU 版")
    log.write(f"[定档] 显卡={gpu_name or '无'} CUDA={cuda_version} 算力={compute_cap} 选择="
              f"{'cu128' if use_cuda else 'cpu'}")

    # ---------- 步骤 5：装依赖（分两步，防 index-url 干扰非 torch 包） ----------
    requirements = os.path.join(root, "requirements-gui.txt")
    if not os.path.isfile(requirements):
        raise BootstrapError(f"依赖清单缺失：{requirements}")

    def _pip_torch(env_extra: dict) -> str:
        # torch 专用源：download.pytorch.org 不受墙影响，故本步不做 PyPI 镜像降级
        _run([_venv_python(root), "-m", "pip", "install", "torch",
              "--index-url", TORCH_CUDA_INDEX], cwd=root, env_extra=env_extra)
        return "torch (cu128) 安装完成"

    def _pip_requirements(env_extra: dict) -> str:
        _run([_venv_python(root), "-m", "pip", "install", "-r", "requirements-gui.txt"],
             cwd=root, env_extra=env_extra)
        return "requirements-gui.txt 安装完成"

    if use_cuda:
        _install_step("安装 torch（CUDA cu128）", _pip_torch, log, progress_cb)
    _install_step("安装其余依赖（requirements-gui.txt）", _pip_requirements,
                  log, progress_cb, mirror_env=_PIP_MIRROR_ENV)

    # ---------- 步骤 6：返回 venv 解释器 ----------
    progress_cb("环境准备完成")
    log.write("[完成] 环境准备完成")
    return _venv_python(root)


def _remove_tree(path: str) -> None:
    """删除目录树（标准库 shutil，只在这一处按需 import）。"""
    import shutil
    shutil.rmtree(path, ignore_errors=True)


# ============================== T-G1 步骤 3：选端口 ==============================

def pick_port() -> int:
    """socket bind ("127.0.0.1", 0) 取系统分配的空闲端口（禁止固定端口）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ============================== 进度窗口与主窗口 ==============================

class _ProgressWindow:
    """首启进度窗口：主线程跑 Tk，后台线程跑 bootstrap，队列回传进度/结果。"""

    def __init__(self, error_text: str = ""):
        self.events: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.status = "quit"          # ok / retry / quit
        self.payload = ""
        self.window = tk.Tk()
        self.window.title("CalibGuard 环境准备")
        self.window.geometry("700x320")
        self.window.resizable(False, False)

        tk.Label(self.window, text="首次运行需要准备运行环境（约 10-20 分钟，之后秒开）",
                 anchor="w", font=("Microsoft YaHei UI", 10, "bold")).pack(
            fill="x", padx=16, pady=(16, 6))
        self.step_label = tk.Label(self.window, text="正在准备运行环境…", anchor="w",
                                   justify="left", wraplength=660,
                                   font=("Microsoft YaHei UI", 10))
        self.step_label.pack(fill="x", padx=16, pady=(6, 8))
        self.bar = ttk.Progressbar(self.window, mode="indeterminate", length=660)
        self.bar.pack(padx=16, pady=4)
        self.detail_label = tk.Label(self.window, text="", anchor="nw", justify="left",
                                     wraplength=660, fg="#b00020",
                                     font=("Microsoft YaHei UI", 9))
        self.detail_label.pack(fill="both", expand=True, padx=16, pady=(8, 4))
        self.button_row = tk.Frame(self.window)
        self.button_row.pack(pady=(0, 14))
        self.retry_button = tk.Button(self.button_row, text="重试（使用镜像）",
                                      command=self._on_retry, width=18)
        self.log_button = tk.Button(self.button_row, text="打开日志目录",
                                    command=self._on_open_log, width=18)
        if error_text:
            self._show_error(error_text)
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- 对外入口 ----------
    def start(self, root: str) -> None:
        """启动后台引导线程并进入窗口事件循环（返回时 status/payload 已确定）。"""
        self.bar.start(12)
        threading.Thread(target=self._worker, args=(root,), daemon=True).start()
        self.window.after(100, self._poll)
        self.window.mainloop()

    # ---------- 后台线程 ----------
    def _worker(self, root: str) -> None:
        try:
            exe = bootstrap(root, lambda text: self.events.put(("progress", text)))
        except Exception as exc:                       # UI 必须看到错误文本
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))
        else:
            self.events.put(("ok", exe))

    # ---------- 主线程轮询 ----------
    def _poll(self) -> None:
        try:
            while True:
                kind, text = self.events.get_nowait()
                if kind == "progress":
                    self.step_label.config(text=text)
                elif kind == "ok":
                    self.status, self.payload = "ok", text
                    self.window.destroy()
                    return
                else:
                    self._show_error(text)
        except queue.Empty:
            pass
        self.window.after(100, self._poll)

    # ---------- 失败出路 ----------
    def _show_error(self, error_text: str) -> None:
        self.bar.stop()
        self.bar.config(mode="determinate", value=0)
        self.step_label.config(text="环境准备失败，可点击下方按钮重试（第 3 次失败会自动走镜像源）")
        self.detail_label.config(text=error_text)
        self.retry_button.pack(side="left", padx=8)
        self.log_button.pack(side="left", padx=8)

    def _on_retry(self) -> None:
        self.status, self.payload = "retry", self.detail_label.cget("text")
        self.window.destroy()

    def _on_open_log(self) -> None:
        log_dir = os.path.dirname(_bootstrap_log_path(_root_dir()))
        try:
            os.makedirs(log_dir, exist_ok=True)
            os.startfile(log_dir)                      # Windows 专用，打开资源管理器
        except OSError as exc:
            messagebox.showerror("无法打开日志目录", f"{log_dir}\n{exc}")

    def _on_close(self) -> None:
        self.status = "quit"
        self.window.destroy()


def _show_main_window(url: str, proc: subprocess.Popen) -> None:
    """启动完成窗口：显示访问地址；关闭窗口即终止 server 子进程并退出。"""
    window = tk.Tk()
    window.title("CalibGuard")
    window.geometry("560x200")
    window.resizable(False, False)
    tk.Label(window, text="CalibGuard 文件审计已启动", font=("Microsoft YaHei UI", 12, "bold")).pack(
        pady=(22, 8))
    tk.Label(window, text="已在浏览器打开，关闭本窗口将退出 CalibGuard",
             font=("Microsoft YaHei UI", 10)).pack(pady=4)
    url_entry = tk.Entry(window, width=52, justify="center")
    url_entry.insert(0, url)
    url_entry.config(state="readonly")
    url_entry.pack(pady=8)

    def on_close() -> None:
        try:
            if proc.poll() is None:
                proc.terminate()
        finally:
            window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_close)
    window.mainloop()
    # 兜底：mainloop 因异常退出时也要收掉子进程
    if proc.poll() is None:
        proc.terminate()


def _wait_health(port: int, proc: subprocess.Popen | None = None) -> bool:
    """轮询 /api/health 直到 200；server 进程若提前退出则立即判失败。"""
    deadline = time.time() + HEALTH_TIMEOUT_S
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=HEALTH_REQUEST_TIMEOUT_S) as response:
                if response.status == 200:
                    return True
        except Exception:                              # 未就绪属正常，继续轮询
            pass
        time.sleep(HEALTH_INTERVAL_S)
    return False


def _fatal(title: str, text: str) -> None:
    """致命错误窗口（错误文本必须让用户看到）。"""
    window = tk.Tk()
    window.withdraw()
    messagebox.showerror(title, text)
    window.destroy()


# ============================== T-G1 步骤 4：ensure_env ==============================

def ensure_env(root: str) -> str:
    """首启引导，返回 .venv\\Scripts\\python.exe 路径；已有环境则直接返回（秒开）。"""
    global _MIRROR_FIRST
    exe = _venv_python(root)
    if os.path.isfile(exe):
        return exe
    error_text = ""
    while True:
        window = _ProgressWindow(error_text)
        window.start(root)
        if window.status == "ok":
            return window.payload
        if window.status == "quit":
            raise BootstrapError(window.payload or "用户关闭了环境准备窗口")
        # 「重试（使用镜像）」：下一轮引导优先走镜像源
        _MIRROR_FIRST = True
        error_text = window.payload


# ============================== T-G1 步骤 5：main ==============================

def main() -> None:
    """主流程：准备环境 → 启动 server → 打开浏览器 → 等待窗口关闭。"""
    root = _root_dir()
    try:
        venv_python = ensure_env(root)
    except BootstrapError as exc:
        _fatal("CalibGuard 环境准备失败",
               f"{exc}\n\n日志：{_bootstrap_log_path(root)}\n"
               f"可删除 .venv 目录后重试；仍失败请把日志发给开发者。")
        return

    server_script = os.path.join(root, "server.py")
    if not os.path.isfile(server_script):
        _fatal("CalibGuard 启动失败", f"缺少后端文件：{server_script}")
        return

    port = pick_port()
    server_log = os.path.join(root, "logs", "server.log")
    os.makedirs(os.path.dirname(server_log), exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"          # 崩溃信息立刻落盘，便于显示给用户
    handle = open(server_log, "ab")
    try:
        proc = subprocess.Popen([venv_python, "server.py", "--port", str(port)], cwd=root,
                                stdout=handle, stderr=subprocess.STDOUT, env=env)
    except OSError as exc:
        handle.close()
        _fatal("CalibGuard 启动失败", f"无法启动后端进程：{exc}")
        return

    if not _wait_health(port, proc):
        proc.terminate()
        handle.close()
        detail = _tail(server_log, 30)
        _fatal("CalibGuard 启动失败",
               f"服务在 {HEALTH_TIMEOUT_S} 秒内未就绪（127.0.0.1:{port}/api/health）。\n"
               f"若环境已损坏，请删除 .venv 目录后重新运行。\n\n"
               f"日志：{server_log}\n{detail}")
        return

    webbrowser.open(f"http://127.0.0.1:{port}")
    _show_main_window(f"http://127.0.0.1:{port}", proc)
    if proc.poll() is None:
        proc.terminate()
    handle.close()


if __name__ == "__main__":
    main()
