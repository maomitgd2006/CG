"""普通工具脚本：计算文件行数（T-G4 验收用良性样本）。"""
import sys


def count_lines(path: str) -> int:
    """返回文件行数。"""
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


if __name__ == "__main__":
    print(count_lines(sys.argv[1]))
