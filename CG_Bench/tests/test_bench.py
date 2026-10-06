"""CalibGuard Bench 验收测试（TH-8）。

运行：Windows 用 test.bat；Linux 用 `.venv/bin/python -m pytest tests/ -v`
（Windows 无需产物即可通过 4 项 + 2 项跳过；产物级验收需训练产物就绪）

六项验收：
  1. test_split_identity    results/_splits/ 统一划分源结构校验（v2.5 语义：数据重建，
                            不再与 v1 基线逐行对齐；强校验由 _prep/v25_split_rebuild.py 负责）
  2. test_dry_run_sequence  TH-2 验收（2026-10-02 矩阵：9 个 pending 实验完整流程序列
                            + baseline 补跑 4-6，在当前统一划分源的 test 段上评）
  3. test_rules_baseline    TH-5 验收（已知恶意样本 classify 全 True，两域各一条）
  4. test_speed_monotonic   TH-6 验收（short < medium < long 中位数单调）
  5. test_report_fields     report 生成后 benchmark.json 每实验含文档第 7 节全部列
  6. test_api_circuit_breaker  API 熔断（2026-10-02 用户要求）：404 / 长时间无响应 →
                            抛 ApiFatalError、classify 不吞、写 api_error_skip.json 记录
"""
import json
import os
import re
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# venv 解释器：Windows 为 .venv\Scripts\python.exe，Linux 为 .venv/bin/python
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe") if os.name == "nt" \
    else os.path.join(ROOT, ".venv", "bin", "python")
SPLITS = os.path.join(ROOT, "results", "_splits")
BASELINE = os.path.join(ROOT, "models_baseline", "v0.2.2")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SPLIT_FILES = ("split_train.jsonl", "split_val.jsonl",
               "split_calibration.jsonl", "split_test.jsonl")

# 2026-10-02 修订矩阵的 12 实验（测试自带副本，与运行期 yaml 状态解耦）：
#   训练顺序（用户决策）：先依次跑完全部 *_shared，再依次跑全部 *_ft（不按模型交替）；
#   codebert_ft 最先（产 shared_hpo）；hybrid_gate_llm 路由复用 codebert_ft（不训练，
#   只补一轮 test 评测）；每个多基座 LLM 两段式（*_shared 先跑 codebert 最优超参 →
#   *_ft 再各自跑 HPO，best 与 shared 一致时不二次训练）。
#   9 个 pending（含 rules_baseline v2.5 重跑）+ 1 个已训对照（补跑 4-6）
#   + llm_zero/llm_fewshot 两个 excluded（2026-10-02 用户决策：v2.5 不跑，成本与耗时不可接受）。
DOCUMENTED_MATRIX = [
    {"name": "codebert_ft", "type": "ft", "base_model": "microsoft/codebert-base",
     "hpo": "shared", "batch": 16, "status": "pending", "expect_steps": {1, 3, 4, 5, 6}},
    {"name": "hybrid_gate_llm", "type": "hybrid", "base_model": "microsoft/codebert-base",
     "reuse_model_from": "codebert_ft", "batch": 16, "status": "pending",
     "expect_steps": {3, 4, 5, 6}},
    {"name": "qwen3_06b_shared", "type": "ft", "base_model": "base_models/Qwen3-0.6B",
     "hpo": "shared", "batch": 8, "status": "pending", "expect_steps": {3, 4, 5, 6}},
    {"name": "qwen25_coder_05b_shared", "type": "ft",
     "base_model": "base_models/Qwen2.5-Coder-0.5B-Instruct",
     "hpo": "shared", "batch": 8, "status": "pending", "expect_steps": {3, 4, 5, 6}},
    {"name": "qwen3_coder_small_shared", "type": "ft", "base_model": "base_models/Qwen3-CoderSmall",
     "hpo": "shared", "batch": 8, "status": "pending", "expect_steps": {3, 4, 5, 6}},
    {"name": "qwen3_06b_ft", "type": "ft", "base_model": "base_models/Qwen3-0.6B",
     "hpo": "per_model", "batch": 8, "status": "pending", "expect_steps": {2, 3, 4, 5, 6}},
    {"name": "qwen25_coder_05b_ft", "type": "ft",
     "base_model": "base_models/Qwen2.5-Coder-0.5B-Instruct",
     "hpo": "per_model", "batch": 8, "status": "pending", "expect_steps": {2, 3, 4, 5, 6}},
    {"name": "qwen3_coder_small_ft", "type": "ft", "base_model": "base_models/Qwen3-CoderSmall",
     "hpo": "per_model", "batch": 8, "status": "pending", "expect_steps": {2, 3, 4, 5, 6}},
    {"name": "v0.2.2_baseline", "type": "ft",
     "model_dir": "models_baseline/v0.2.2/codebert_finetuned",
     "params": "models_baseline/v0.2.2/calibration_params.json",
     "hpo": "none", "status": "trained", "expect_steps": {4, 5, 6}},
    {"name": "llm_zero", "type": "llm", "variant": "zero", "status": "excluded",
     "expect_steps": set()},
    {"name": "llm_fewshot", "type": "llm", "variant": "fewshot", "status": "excluded",
     "expect_steps": set()},
    {"name": "rules_baseline", "type": "rules", "status": "pending", "expect_steps": {4}},
]

# 文档第 7 节「report 必含列」
REQUIRED_ROW_KEYS = ("acc", "precision", "recall", "f1", "auc", "ece", "brier",
                     "silent_allow_rate", "block_purity", "gate_allow",
                     "gate_llm_review", "gate_block", "delta_f1", "per_domain",
                     "speed", "train_seconds", "total_seconds", "parse_error_rate",
                     "config")
REQUIRED_SPEED_KEYS = ("env", "files_per_min_overall", "windows_per_sec_mean",
                       "latency_sec", "per_file", "network")


def _read_lines(path: str) -> list:
    with open(path, "r", encoding="utf-8") as handle:
        return [line for line in handle.read().splitlines() if line.strip()]


def test_split_identity():
    """统一划分源结构校验（v2 语义）。

    v1 时代 canonical 与 models_baseline/v0.2.2/ 逐行相同（铁律 1）；v2 数据域重构后
    canonical = v2 重建（v1 划分已归档 data/archive/v1-splits-20260929/）。家族不跨段 /
    sha 六对交叉等强校验由 data/raw/_prep/v2_split_verify.py 在数据准备阶段执行；
    此处做结构性回归：四文件齐备非空、引用路径有效、label 合法、比例约 70/10/10/10。
    """
    for file_name in SPLIT_FILES:
        canonical = os.path.join(SPLITS, file_name)
        assert os.path.isfile(canonical), f"缺少统一划分源：{canonical}（先跑 v2_split_verify.py）"
    counts = {}
    for file_name in SPLIT_FILES:
        rows = []
        for line in _read_lines(os.path.join(SPLITS, file_name)):
            row = json.loads(line)
            assert row["label"] in (0, 1), f"{file_name} 非法 label：{row}"
            assert os.path.isfile(os.path.join(ROOT, row["path"])), \
                f"{file_name} 引用不存在的样本：{row['path']}"
            rows.append(row)
        assert rows, f"{file_name} 为空"
        counts[file_name] = len(rows)
    total = sum(counts.values())
    assert total > 0
    ratios = [counts[n] / total for n in SPLIT_FILES]
    assert 0.60 < ratios[0] < 0.80, f"train 段比例异常：{ratios[0]:.3f}"
    for r in ratios[1:]:
        assert 0.05 < r < 0.15, f"留出段比例异常：{r:.3f}"


def test_dry_run_sequence():
    """TH-2 验收：9 个 pending 实验各一条完整流程序列 + baseline 一条「补跑步骤 4-6」序列。"""
    import harness

    shared = {"lr": None, "epochs": None}
    ctx = {"overrides": {}, "needs_shared_hpo": True, "data_version": "v2",
           "experiments": DOCUMENTED_MATRIX}
    planned = {}
    for exp in DOCUMENTED_MATRIX:
        action, plan = harness.build_plan(exp, shared, has_key=False, ctx=ctx)
        planned[exp["name"]] = (action, {step_no for step_no, _n, _c in plan})
        if ctx["needs_shared_hpo"] and action == "run" and exp.get("hpo") == "shared":
            ctx["needs_shared_hpo"] = False     # 队列语义：第一个 shared 实验产出后复用

    pending = [exp for exp in DOCUMENTED_MATRIX if exp["status"] == "pending"]
    assert len(pending) == 9, "默认实验集必须是 9 个 pending + 1 个已训对照 + 2 个 v2 已训"
    for exp in pending:
        action, steps = planned[exp["name"]]
        name = exp["name"]
        assert action in ("run", "skip_key"), f"{name} 未被排入队列（action={action}）"
        assert exp["expect_steps"] <= steps, \
            f"{name} 流程序列不完整：期望含 {sorted(exp['expect_steps'])}，实际 {sorted(steps)}"
        if exp["type"] in ("llm", "rules"):
            assert 3 not in steps, f"{name}（{exp['type']}）不应有训练步骤"
        else:
            assert {3, 4, 5, 6} <= steps and 7 in steps, f"{name} ft/hybrid 流程缺步骤"

    base_action, base_steps = planned["v0.2.2_baseline"]
    assert base_action == "backfill", f"已训对照应为补跑（action={base_action}）"
    assert base_steps == {4, 5, 6}, f"对照实验只补跑步骤 4-6，实际 {sorted(base_steps)}"

    # CLI 侧冒烟：dry-run 可执行、输出格式合法、已训对照必入队、各实验步骤集合合法。
    # 注意不能断言「6 个 pending 实验都出现」——yaml 的运行期状态会变（如无 Key 时
    # llm_* 已是 skipped_api_key，本就不该入队）；初始态的完整覆盖由上面的程序化断言负责。
    result = subprocess.run([PY, "harness.py", "train", "--all", "--dry-run"],
                            cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    assert result.returncode == 0, f"dry-run 退出码 {result.returncode}：{result.stderr[-2000:]}"
    seen = {}
    for line in result.stdout.splitlines():
        match = re.match(r"^\[dry-run\] (\S+) \| STEP (\d)/7 \| (\w+) \|", line)
        if match:
            seen.setdefault(match.group(1), set()).add(int(match.group(2)))
    assert seen, "dry-run 未打印任何 STEP 行"
    by_name = {exp["name"]: exp for exp in DOCUMENTED_MATRIX}
    assert set(seen) <= set(by_name), f"dry-run 出现非实验集条目：{sorted(set(seen) - set(by_name))}"
    for name, steps in seen.items():
        assert steps, f"{name} 未打印任何步骤"
        assert all(1 <= step <= 7 for step in steps), f"{name} 步骤号越界：{sorted(steps)}"
        expect = by_name[name]["expect_steps"]
        assert expect <= steps, f"{name} 步骤集合不完整：期望含 {sorted(expect)}，实际 {sorted(steps)}"
    # status=trained 且带 model_dir 的对照实验必须始终入队（补跑步骤 4-6）
    assert "v0.2.2_baseline" in seen, "dry-run 输出缺少已训对照 v0.2.2_baseline 的补跑序列"
    assert seen["v0.2.2_baseline"] == {4, 5, 6}, \
        f"对照实验只补跑步骤 4-6，实际 {sorted(seen['v0.2.2_baseline'])}"


def test_rules_baseline():
    """TH-5 验收：3 个已知恶意样本 classify 全 True。"""
    sys.path.insert(0, ROOT)
    from adapters.rules_adapter import classify, hit_count
    from calibguard.preprocessor import preprocess_file

    # v2.5 换样本（2026-10-02）：mal_py_11Cent_1 已随 DataDog 仓库更新消失，
    # 换为抽检实锤且规则命中≥3 的现存样本（prompt 技能 / pypi / npm 三源）
    samples = ("data/train/malicious/mal_sb_0.txt",
               "data/train/malicious/mal_py_tphttplgtbrandom_0.txt",
               "data/train/malicious/mal_npm_customer-identity-mfe-dev_0.txt")
    for relative in samples:
        path = os.path.join(ROOT, relative)
        assert os.path.isfile(path), f"验收样本缺失：{relative}"
        text = preprocess_file(path).text
        result = classify(text)
        assert result["malicious"] is True, f"{relative} 被判良性：{result}"
        assert result["hits"] >= 3, f"{relative} 命中数不足：{result}"
        assert result["hits"] == hit_count(text)
        assert 0.0 <= result["confidence"] <= 0.99


def test_speed_monotonic():
    """TH-6 验收：v0.2.2_baseline 的 speed.json 三档中位数单调（short < medium < long）。

    产物级验收：数据版本重置后（SOP 3.0 第 3 条归档旧产物）暂无 speed.json 时跳过，
    实验 + speed 跑完后本测试才生效。
    """
    speed_path = os.path.join(ROOT, "results", "v0.2.2_baseline", "speed.json")
    if not os.path.isfile(speed_path):
        pytest.skip("v2 重置后暂无 speed 产物（speed --all 跑完后本测试生效）")
    payload = json.load(open(speed_path, "r", encoding="utf-8"))
    for key in REQUIRED_SPEED_KEYS:
        assert key in payload, f"speed.json 缺少字段 {key}"
    for key in ("cpu", "threads", "gpu", "torch", "date"):
        assert key in payload["env"], f"speed.json env 缺少环境戳字段 {key}"
    latency = payload["latency_sec"]
    short, medium, long_ = (latency["short"]["median"], latency["medium"]["median"],
                            latency["long"]["median"])
    assert short is not None and medium is not None and long_ is not None
    assert short < medium < long_, f"三档中位数未单调：{short} / {medium} / {long_}"
    for bucket in ("short", "medium", "long"):
        assert latency[bucket]["p90"] >= latency[bucket]["median"]
    assert payload["files_per_min_overall"] > 0
    assert payload["per_file"], "speed.json 缺少 per_file 明细"


def test_report_fields():
    """report 验收：benchmark.json 每个实验含文档第 7 节全部列。

    产物级验收：数据版本重置后暂无 benchmark.json 时跳过，report 跑完后生效。
    """
    report_path = os.path.join(ROOT, "benchmark.json")
    if not os.path.isfile(report_path):
        pytest.skip("v2.5 重置后暂无 benchmark.json（harness.py report 跑完后本测试生效）")
    payload = json.load(open(report_path, "r", encoding="utf-8"))
    rows = payload.get("experiments") or []
    names = [row["name"] for row in rows]
    for exp in DOCUMENTED_MATRIX:
        assert exp["name"] in names, f"benchmark.json 缺少实验 {exp['name']}"
    assert len(rows) == len(DOCUMENTED_MATRIX), \
        f"benchmark.json 实验数 {len(rows)} != 默认 {len(DOCUMENTED_MATRIX)} 实验"
    for row in rows:
        for key in REQUIRED_ROW_KEYS:
            assert key in row, f"{row['name']} 缺少列 {key}"
        for domain in ("code", "prompt"):
            assert domain in row["per_domain"], f"{row['name']} 分域缺 {domain}"
        speed = row["speed"]
        if speed is not None:
            for key in REQUIRED_SPEED_KEYS:
                assert key in speed, f"{row['name']} 速度项缺 {key}"
            for bucket in ("short", "medium", "long"):
                info = speed["latency_sec"][bucket]
                assert "median" in info and "p90" in info, \
                    f"{row['name']} 速度档 {bucket} 缺 median/p90"
            assert "cpu" in speed["env"] and "threads" in speed["env"]
        assert isinstance(row["config"], dict) and row["config"].get("name") == row["name"]


def test_api_circuit_breaker(tmp_path, monkeypatch):
    """API 熔断（2026-10-02 用户要求）：404 / 长时间无响应 → ApiFatalError + 记录。

    不联网：用假 client 注入异常与成功响应，验证
      ① 404（status_code=404）→ ApiFatalError(reason=not_found)；
      ② 超过 stall 阈值仍无成功响应 → ApiFatalError(reason=stall)；
      ③ classify 不吞 ApiFatalError（否则逐条空转会污染整段评测）；
      ④ write_skip_marker 写出 status=skipped_api_error 的记录文件。
    """
    from adapters import llm_adapter as LA

    class _NotFound(Exception):
        status_code = 404

    class _FakeWire:
        def __init__(self, exc=None, content='{"malicious": false, "confidence": 0.1}'):
            self.exc = exc
            self.content = content

        def create(self, **_kwargs):
            if self.exc is not None:
                raise self.exc
            message = type("Msg", (), {"content": self.content})
            choice = type("Choice", (), {"message": message})
            return type("Resp", (), {"choices": [choice]})

    def _client(exc=None):
        return type("Client", (), {"chat": type("Chat", (), {"completions": _FakeWire(exc)})})

    messages = [{"role": "user", "content": "x"}]

    # ① 404 → not_found
    LA.reset_progress()
    monkeypatch.setattr(LA, "_get_client", lambda: _client(_NotFound("404 not found")))
    with pytest.raises(LA.ApiFatalError) as excinfo:
        LA._call_api(messages)
    assert excinfo.value.reason == "not_found"

    # ② 长时间无响应 → stall（阈值由环境变量调小以便测试）
    LA.reset_progress()
    LA._PROGRESS["start"] = time.time() - 100
    monkeypatch.setenv(LA.STALL_TIMEOUT_ENV, "10")
    monkeypatch.setattr(LA, "_get_client", lambda: _client())
    with pytest.raises(LA.ApiFatalError) as excinfo:
        LA._call_api(messages)
    assert excinfo.value.reason == "stall"

    # ③ classify 不吞熔断异常
    LA.reset_progress()
    monkeypatch.setattr(LA, "_get_client", lambda: _client(_NotFound("404 not found")))
    with pytest.raises(LA.ApiFatalError):
        LA.classify("some text", variant="zero", template="v_plain")

    # ④ 熔断记录文件
    payload = LA.write_skip_marker(str(tmp_path), LA.ApiFatalError("not_found", "demo"))
    assert payload["status"] == "skipped_api_error"
    assert (tmp_path / LA.SKIP_MARKER).is_file()


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-v"]))
