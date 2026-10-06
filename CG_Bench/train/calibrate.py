"""CalibGuard 概率校准脚本（T14）。

在 T13 产出的校准集清单上推理 logit，拟合 Platt 参数并落盘。
运行方式：calibrate.bat [参数]，或先设 PYTHONPATH=src 再 python scripts/calibrate.py [参数]。
退出码沿用全局约定：1 输入不存在 / 3 校准集不可用 / 4 模型加载失败。
"""
import argparse
import json
import os
import sys

from calibguard.calibrator import PlattCalibrator
from calibguard.model import CodeBERTClassifier
from calibguard.preprocessor import preprocess_file

# 推理窗口大小固定 512（与 config.yaml 的滑动窗口一致，见 17.2 第 1 步）
_MAX_LENGTH = 512


def main() -> None:
    parser = argparse.ArgumentParser(description="在校准集上拟合 Platt 校准参数")
    parser.add_argument("--model-dir", default="models/codebert_finetuned",
                        help="微调模型目录")
    parser.add_argument("--calib-file", default="models/split_calibration.jsonl",
                        help="T13 产出的校准集清单")
    parser.add_argument("--output", default="models/calibration_params.json",
                        help="Platt 参数输出文件")
    args = parser.parse_args()

    if not os.path.isfile(args.calib_file):
        print(f"[错误] 校准集清单不存在: {args.calib_file}", file=sys.stderr)
        sys.exit(1)

    # 1. 加载微调模型
    try:
        classifier = CodeBERTClassifier(args.model_dir, max_length=_MAX_LENGTH)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(4)

    # 2. 逐行读校准集：预处理 → 推理 logit
    logits: list[float] = []
    labels: list[int] = []
    with open(args.calib_file, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            sample_path = record["path"]
            label = int(record["label"])
            try:
                result = preprocess_file(sample_path)
            except Exception as exc:
                print(f"跳过无法预处理的校准样本: {sample_path}")
                print(f"  原因: {type(exc).__name__}: {exc}")
                continue
            logits.append(classifier.predict_logit(result.text))
            labels.append(label)

    # 3. 拟合（单一类别 / 空校准集 → 退出码 3）
    calibrator = PlattCalibrator()
    try:
        calibrator.fit(logits, labels)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(3)

    # 4. 落盘
    calibrator.save(args.output)

    # 5. 摘要
    print(f"拟合完成 A={calibrator.A} B={calibrator.B} 样本数={calibrator.n_samples}，"
          f"输出文件 {args.output}")


if __name__ == "__main__":
    main()
