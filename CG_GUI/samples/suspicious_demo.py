"""可疑脚本样本（T-G4 验收演示用：仅触发证据徽章，不含真实攻击行为）。"""
import base64
import os
import socket

REPORT_URL = "http://example.com/collect"
SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_rsa")
BLOB = base64.b64decode("cHJpbnQoImhpIik=")
SOCK = socket.socket()
DEMO_SECRET_TOKEN = "demo-token-value"

print(REPORT_URL, SSH_KEY_PATH, BLOB, SOCK, DEMO_SECRET_TOKEN)
