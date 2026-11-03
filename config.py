"""
Configuration and environment management for InvoiceChecker.
Supports local private models (e.g. Ollama, vLLM) and privacy protection.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List
from openai import OpenAI
from dotenv import load_dotenv

# Load .env relative to project root
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── 隐私与离线保护 ──
# 若设置为 true，系统将严格禁止将任何发票数据/图片上传到外部第三方云端模型
OFFLINE_ONLY = os.getenv("OFFLINE_ONLY", "false").strip().lower() in ("true", "1", "yes")

# ── OCR / VLM 服务提供商配置 ──
# 可选值: "openai" (推荐本地私有模型如 Ollama / vLLM 或外部 OpenAI 兼容端点),
#        "zhipu", "aliyun", "tencent"
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "openai").strip().lower()

# ── OpenAI 兼容协议配置（支持自定义本地端口如 http://127.0.0.1:11434/v1）──
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# ── 智谱 GLM 配置 ──
ZHIPU_API_KEY   = os.getenv("ZHIPU_API_KEY", "").strip()

# ── 腾讯混元配置 ──
HUNYUAN_API_KEY  = os.getenv("HUNYUAN_API_KEY", "").strip()
HUNYUAN_BASE_URL = os.getenv("HUNYUAN_BASE_URL", "https://api.hunyuan.cloud.tencent.com/v1").strip()
HUNYUAN_MODEL    = os.getenv("HUNYUAN_MODEL", "hy4-preview").strip()

# ── 腾讯云传统 OCR 配置 ──
TENCENTCLOUD_SECRET_ID  = os.getenv("TENCENTCLOUD_SECRET_ID", "").strip()
TENCENTCLOUD_SECRET_KEY = os.getenv("TENCENTCLOUD_SECRET_KEY", "").strip()
TENCENTCLOUD_REGION     = os.getenv("TENCENTCLOUD_REGION", "ap-guangzhou").strip()

# ── 阿里云百炼配置 ──
ALIBABA_API_KEY = os.getenv("ALIBABA_API_KEY", "").strip()
VLM_MODEL       = os.getenv("VLM_MODEL", "qwen3-vl-flash").strip()

# ── 浏览器与查验配置 ──
HEADLESS = os.getenv("HEADLESS", "true").strip().lower() in ("true", "1", "yes")
CDP_URL  = os.getenv("CDP_URL", "").strip()
CHROMIUM_EXECUTABLE_PATH = os.getenv("CHROMIUM_EXECUTABLE_PATH", "").strip()

# 是否强制禁用 Chromium 沙箱。默认为 auto（非 root 运行自动启用沙箱）
CHROMIUM_NO_SANDBOX = os.getenv("CHROMIUM_NO_SANDBOX", "false").strip().lower() in ("true", "1", "yes")


def get_chromium_executable_path() -> Optional[str]:
    """获取可用的 Chromium / Chrome 可执行文件路径"""
    if CHROMIUM_EXECUTABLE_PATH and os.path.exists(CHROMIUM_EXECUTABLE_PATH):
        return CHROMIUM_EXECUTABLE_PATH

    candidates = [
        "/snap/bin/chromium",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def get_chromium_args() -> List[str]:
    """生成兼顾安全与环境适配的 Chromium 启动参数。
    默认启用原生安全沙箱，仅在显式配置或 root 用户无特权容器环境下安全回退。
    """
    args = ['--disable-dev-shm-usage']
    
    # 检测是否为 root 用户运行（Linux 下 os.geteuid() == 0）
    is_root = False
    if hasattr(os, "geteuid"):
        try:
            is_root = (os.geteuid() == 0)
        except Exception:
            pass

    if CHROMIUM_NO_SANDBOX or is_root:
        args.extend(['--no-sandbox', '--disable-setuid-sandbox'])
        
    return args


def is_ocr_configured() -> Tuple[bool, str]:
    """校验当前 OCR / VLM 引擎凭证配置"""
    if OFFLINE_ONLY:
        return True, ""

    if OCR_PROVIDER == "openai":
        # 如果是本地端点（如 localhost 或 127.0.0.1），无需强制要求真实的外部 API_KEY
        is_local = any(h in OPENAI_BASE_URL.lower() for h in ("localhost", "127.0.0.1", "0.0.0.0", "192.168."))
        if not is_local and (not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your_")):
            return False, "请先在 .env 中配置有效的 OPENAI_API_KEY，或配置本地私有模型端点 OPENAI_BASE_URL"
        return True, ""
    elif OCR_PROVIDER in ("zhipu", "glm"):
        if not ZHIPU_API_KEY or ZHIPU_API_KEY.startswith("your_"):
            return False, "请先在 .env 中配置有效的 ZHIPU_API_KEY"
        return True, ""
    elif OCR_PROVIDER == "tencent":
        if not TENCENTCLOUD_SECRET_ID or not TENCENTCLOUD_SECRET_KEY or TENCENTCLOUD_SECRET_ID.startswith("your_"):
            return False, "请先在 .env 中配置有效的 TENCENTCLOUD_SECRET_ID 和 TENCENTCLOUD_SECRET_KEY"
        return True, ""
    elif OCR_PROVIDER == "aliyun":
        if not ALIBABA_API_KEY or ALIBABA_API_KEY.startswith("your_"):
            return False, "请先在 .env 中配置有效的 ALIBABA_API_KEY"
        return True, ""
    return False, f"未知的 OCR_PROVIDER: {OCR_PROVIDER}"


def get_vlm_client() -> Tuple[Optional[OpenAI], str]:
    """获取兼容 OpenAI 的视觉大模型客户端与模型名称。
    支持自定义本地端点（如 Ollama / vLLM）以及第三方云端平台。
    """
    if OFFLINE_ONLY:
        return None, ""

    # 1. 优先检查用户显式配置的 OpenAI 兼容端点（如本地私有端点或专有网关）
    if OCR_PROVIDER == "openai":
        key = OPENAI_API_KEY or "local-key"
        return OpenAI(api_key=key, base_url=OPENAI_BASE_URL), OPENAI_MODEL

    # 2. 混元专用端点
    if HUNYUAN_API_KEY and not HUNYUAN_API_KEY.startswith("your_"):
        return OpenAI(api_key=HUNYUAN_API_KEY, base_url=HUNYUAN_BASE_URL), HUNYUAN_MODEL

    # 3. 智谱 GLM 开放平台
    if ZHIPU_API_KEY and not ZHIPU_API_KEY.startswith("your_"):
        return OpenAI(api_key=ZHIPU_API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4"), "glm-4v-flash"

    # 4. 阿里云百炼
    if OCR_PROVIDER == "aliyun" and ALIBABA_API_KEY and not ALIBABA_API_KEY.startswith("your_"):
        return OpenAI(api_key=ALIBABA_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"), VLM_MODEL

    # 兜底：如果配置了 OPENAI_API_KEY 则使用
    if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your_"):
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL), OPENAI_MODEL

    return None, ""
