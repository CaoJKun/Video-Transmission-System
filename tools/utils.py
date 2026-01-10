import subprocess
import logging

logger = logging.getLogger(__name__)

def validate_ffmpeg():
    """验证 ffmpeg 是否可用"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False

def run_ffmpeg_command(cmd: str, timeout: int = 300) -> tuple:
    """运行 ffmpeg 命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, 
                              text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "命令超时"
    except Exception as e:
        return -1, "", str(e)