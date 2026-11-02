"""
发票系统 - Flask 后端服务
功能一：发票查验（原版逻辑完全保留）
功能二：发票识别整理（新增，支持 PDF/OFD/XML/图片，优先二维码→正则零token）
        PDF 支持多页拼接（每页识别为一张独立发票）

安装：pip install flask playwright openai PyMuPDF easyofd pillow pandas openpyxl ddddocr python-dotenv pyzbar opencv-python-headless
      playwright install chromium
启动：/volume1/@appstore/python313/bin/python3 server.py
"""

import asyncio
import time
import os
import re
import io
import json
import base64
import tempfile
import shutil
import zipfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from flask import Flask, request, Response, send_from_directory, jsonify
from openai import OpenAI
import fitz  # PyMuPDF
from playwright.async_api import async_playwright
from dotenv import load_dotenv

# pyzbar 可选依赖（二维码扫描，pip install pyzbar opencv-python-headless）
try:
    from pyzbar.pyzbar import decode as qr_decode
    from PIL import Image as _PILImage
    _PYZBAR_AVAILABLE = True
except ImportError:
    _PYZBAR_AVAILABLE = False

# ── .env 加载（绝对路径，避免从不同目录调用时找不到）──
load_dotenv(Path(__file__).parent / ".env")

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30MB limit


# ==================== 配置区域（从 .env 读取）====================
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "tencent").strip().lower()

# ── 腾讯云 OCR 配置 ──
TENCENTCLOUD_SECRET_ID  = os.getenv("TENCENTCLOUD_SECRET_ID", "").strip()
TENCENTCLOUD_SECRET_KEY = os.getenv("TENCENTCLOUD_SECRET_KEY", "").strip()
TENCENTCLOUD_REGION     = os.getenv("TENCENTCLOUD_REGION", "ap-guangzhou").strip()

# ── 阿里云百炼配置 ──
ALIBABA_API_KEY = os.getenv("ALIBABA_API_KEY", "").strip()
VLM_MODEL       = os.getenv("VLM_MODEL", "qwen3-vl-flash").strip()

# ── OpenAI 兼容协议配置 ──
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# 兼容保留原有 API_KEY 变量
API_KEY = ALIBABA_API_KEY

# ── 浏览器查验配置 ──
HEADLESS = os.getenv("HEADLESS", "true").strip().lower() in ("true", "1", "yes")
CDP_URL  = os.getenv("CDP_URL", "").strip()
CHROMIUM_EXECUTABLE_PATH = os.getenv("CHROMIUM_EXECUTABLE_PATH", "").strip()

def get_chromium_executable_path() -> Optional[str]:
    """获取可用的 Chromium 可执行文件路径"""
    # 1. 优先读取环境变量配置
    if CHROMIUM_EXECUTABLE_PATH and os.path.exists(CHROMIUM_EXECUTABLE_PATH):
        return CHROMIUM_EXECUTABLE_PATH
    
    # 2. 自动搜索常见系统路径
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

def is_ocr_configured() -> Tuple[bool, str]:
    """校验当前 OCR 引擎凭证配置"""
    if OCR_PROVIDER in ("zhipu", "glm"):
        z_key = os.getenv("ZHIPU_API_KEY", "").strip()
        if not z_key or z_key.startswith("your_"):
            return False, "请先在 .env 文件中配置有效的 ZHIPU_API_KEY"
        return True, ""
    elif OCR_PROVIDER == "tencent":
        if not TENCENTCLOUD_SECRET_ID or not TENCENTCLOUD_SECRET_KEY or TENCENTCLOUD_SECRET_ID.startswith("your_"):
            return False, "请先在 .env 文件中配置有效的 TENCENTCLOUD_SECRET_ID 和 TENCENTCLOUD_SECRET_KEY"
        return True, ""
    elif OCR_PROVIDER == "aliyun":
        if not ALIBABA_API_KEY or ALIBABA_API_KEY.startswith("your_") or ALIBABA_API_KEY.startswith("sk-xxx"):
            return False, "请先在 .env 文件中配置有效的 ALIBABA_API_KEY"
        return True, ""
    elif OCR_PROVIDER == "openai":
        if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your_"):
            return False, "请先在 .env 文件中配置有效的 OPENAI_API_KEY"
        return True, ""
    return False, f"未知的 OCR_PROVIDER: {OCR_PROVIDER}"

def get_vlm_client() -> Tuple[Optional[OpenAI], str]:
    """获取兼容 OpenAI 的视觉大模型客户端与模型名称，优先支持腾讯混元TokenHub与各多模态平台"""
    hunyuan_key = os.getenv("HUNYUAN_API_KEY", "").strip()
    hunyuan_base = os.getenv("HUNYUAN_BASE_URL", "https://api.hunyuan.cloud.tencent.com/v1").strip()
    hunyuan_model = os.getenv("HUNYUAN_MODEL", "hy4-preview").strip()
    if hunyuan_key and not hunyuan_key.startswith("your_"):
        return OpenAI(api_key=hunyuan_key, base_url=hunyuan_base), hunyuan_model

    zhipu_key = os.getenv("ZHIPU_API_KEY", "").strip()
    if zhipu_key and not zhipu_key.startswith("your_"):
        return OpenAI(api_key=zhipu_key, base_url="https://open.bigmodel.cn/api/paas/v4"), "glm-4v-flash"

    if OCR_PROVIDER == "aliyun":
        if ALIBABA_API_KEY and not ALIBABA_API_KEY.startswith("your_") and not ALIBABA_API_KEY.startswith("sk-xxx"):
            return OpenAI(api_key=ALIBABA_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"), VLM_MODEL
    elif OCR_PROVIDER == "openai":
        if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your_"):
            return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL), OPENAI_MODEL

    if ALIBABA_API_KEY and not ALIBABA_API_KEY.startswith("your_") and not ALIBABA_API_KEY.startswith("sk-xxx"):
        return OpenAI(api_key=ALIBABA_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"), VLM_MODEL
    if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your_"):
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL), OPENAI_MODEL
    return None, ""

def call_tencent_vat_invoice(image_path: str) -> Optional[Dict[str, Any]]:
    """调用腾讯云 VatInvoiceOCR 接口识别发票"""
    try:
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.ocr.v20181119 import ocr_client, models

        cred = credential.Credential(TENCENTCLOUD_SECRET_ID, TENCENTCLOUD_SECRET_KEY)
        httpProfile = HttpProfile()
        httpProfile.endpoint = "ocr.tencentcloudapi.com"
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = ocr_client.OcrClient(cred, TENCENTCLOUD_REGION, clientProfile)

        req = models.VatInvoiceOCRRequest()
        req.ImageBase64 = encode_image(image_path)

        resp = client.VatInvoiceOCR(req)
        return _parse_tencent_vat_invoice_resp(resp)
    except Exception as e:
        print(f"[tencent-ocr] 识别失败: {e}")
        return None

def _parse_tencent_vat_invoice_resp(resp) -> Dict[str, Any]:
    info_map = {}
    if hasattr(resp, "VatInvoiceInfos") and resp.VatInvoiceInfos:
        for item in resp.VatInvoiceInfos:
            name = getattr(item, "Name", "") or ""
            val = getattr(item, "Value", "") or ""
            if name:
                info_map[name.strip()] = str(val).strip()

    items = []
    if hasattr(resp, "Items") and resp.Items:
        for g in resp.Items:
            g_name = getattr(g, "Name", "") or getattr(g, "Item", "") or ""
            if g_name:
                items.append(str(g_name).strip())

    code = info_map.get("发票代码", "")
    number = info_map.get("发票号码", "")
    date_raw = info_map.get("开票日期", "")
    date_clean = re.sub(r'[年月]', '-', date_raw).replace('日', '').strip()

    def to_float(val):
        if not val:
            return 0.0
        c = re.sub(r'[^\d.]', '', str(val))
        try:
            return float(c)
        except Exception:
            return 0.0

    amount_ex = to_float(info_map.get("合计金额") or info_map.get("不含税金额") or 0)
    tax_amount = to_float(info_map.get("合计税额") or 0)
    total_amount = to_float(info_map.get("价税合计(小写)") or info_map.get("价税合计") or (amount_ex + tax_amount))
    title = info_map.get("发票消费类型") or info_map.get("发票名称") or info_map.get("发票类型") or ""

    return {
        "invoice_title": title,
        "invoice_code": code,
        "invoice_number": number,
        "issue_date": date_clean,
        "buyer_name": info_map.get("购买方名称", ""),
        "buyer_tax_id": info_map.get("购买方纳税人识别号", ""),
        "seller_name": info_map.get("销售方名称", ""),
        "seller_tax_id": info_map.get("销售方纳税人识别号", ""),
        "items": items,
        "amount_ex_tax": amount_ex,
        "tax_amount": tax_amount,
        "total_amount": total_amount,
        "total_words": info_map.get("价税合计(大写)", ""),
        "remarks": info_map.get("备注", ""),
        "issuer": info_map.get("开票人", ""),
        "amount_excluding_tax": amount_ex,
        "total_amount_including_tax": {"in_figures": total_amount},
        "check_code": re.sub(r'\s+', '', info_map.get("校验码", "") or "")
    }

def _call_vlm_prompt(client: OpenAI, model: str, image_path: str, mime: str = "image/png") -> Optional[Dict[str, Any]]:
    b64 = encode_image(image_path)
    prompt = """从发票图片提取完整字段，返回 JSON（所有字段必须存在，无值填空字符串或0）：
{
  "invoice_title": "发票抬头（如 电子发票（普通发票））",
  "invoice_code": "发票代码（无则填空字符串）",
  "invoice_number": "发票号码",
  "issue_date": "开票日期 YYYY-MM-DD",
  "buyer_name": "购买方名称",
  "buyer_tax_id": "购买方税号",
  "seller_name": "销售方名称",
  "seller_tax_id": "销售方税号",
  "items": ["商品或服务名称"],
  "amount_ex_tax": 0.00,
  "tax_amount": 0.00,
  "total_amount": 0.00,
  "total_words": "价税合计大写",
  "remarks": "备注",
  "issuer": "开票人",
  "amount_excluding_tax": 0.00,
  "total_amount_including_tax": {"in_figures": 0.00}
}
注意：金额为数字，去除货币符号和逗号；购买方在发票左侧，销售方在右侧，不要搞反。"""
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": prompt}
    ]}]
    try:
        extra_body = {}
        if "qwen" in model.lower():
            extra_body["enable_thinking"] = False
        resp = client.chat.completions.create(
            model=model, messages=messages,
            response_format={"type": "json_object"}, temperature=0, timeout=90,
            **({"extra_body": extra_body} if extra_body else {})
        )
        raw_content = resp.choices[0].message.content.strip()
        # 清除可能的 markdown json 代码块标记
        clean_json = re.sub(r'^```json\s*', '', raw_content, flags=re.IGNORECASE)
        clean_json = re.sub(r'\s*```$', '', clean_json)
        return json.loads(clean_json)
    except Exception as e:
        print(f"[recognize-vlm] error: {e}")
        return None

def call_unified_invoice_ocr(image_path: str, mime: str = "image/png") -> Optional[Dict[str, Any]]:
    """统一发票识别入口：统一优先采用高精度全能的智谱 GLM-4V-Flash 视觉多模态大模型"""
    if OCR_PROVIDER in ("zhipu", "glm"):
        client, model = get_vlm_client()
        if client:
            return _call_vlm_prompt(client, model, image_path, mime)
        print("[unified-ocr] 智谱大模型客户端未就绪")
        return None

    res = None
    if OCR_PROVIDER == "tencent":
        res = call_tencent_vat_invoice(image_path)
        if res:
            num = str(res.get("invoice_number", "") or "").strip()
            tot = float(res.get("total_amount", 0) or 0)
            ex = float(res.get("amount_ex_tax", 0) or 0)
            title = str(res.get("invoice_title", "") or "")
            is_special_ticket = any(kw in title for kw in ["铁路", "客票", "车票", "航空", "行程单"])
            if num and (tot > 0 or ex > 0) and not is_special_ticket:
                return res
            print(f"[unified-ocr] 传统 OCR 要素不全 (类型:{title}, 号码:{num}, 金额:{tot})，回退到高精度 VLM 识别...")

    client, model = get_vlm_client()
    if client:
        vlm_res = _call_vlm_prompt(client, model, image_path, mime)
        if vlm_res:
            if res:
                for k in ["invoice_number", "invoice_code", "issue_date", "buyer_name", "buyer_tax_id"]:
                    if not vlm_res.get(k) and res.get(k):
                        vlm_res[k] = res.get(k)
            return vlm_res
    return res

IMPORT_BUTTON_SELECTOR  = "#fileCy"
BROWSE_BUTTON_SELECTOR  = "#openBtn"
CONFIRM_IMPORT_SELECTOR = "#fileCyBtn"
CLOSE_DIALOG_SELECTOR   = "#closeDialog"
CAPTCHA_IMG_SELECTOR    = "#yzm_img"
CAPTCHA_INPUT_SELECTOR  = "#yzm"
CHECK_BUTTON_SELECTOR   = "#checkfp"
HINT_SELECTOR           = "#yzminfo"
INVOICE_CODE_SELECTOR   = "#fpdm"
INVOICE_NUMBER_SELECTOR = "#fphm"
INVOICE_DATE_SELECTOR   = "#kprq"
INVOICE_AMOUNT_SELECTOR = "#kjje"
ERROR_POPUP_SELECTOR    = "#popup_container"
ERROR_MESSAGE_SELECTOR  = "#popup_message"
ERROR_OK_BUTTON_SELECTOR= "#popup_ok"
SUCCESS_DIALOG_SELECTOR = "dialog"

MAX_CAPTCHA_RETRIES         = 10
SCREENSHOT_ZOOM_FACTOR      = 0.8
CAPTCHA_RECOGNITION_TIMEOUT = 30

_easyofd_available = True

# =================================================================
#  二维码解析（增值税 / 全电发票）
# =================================================================
# 发票类型代码映射
_QR_INVOICE_TYPE = {
    '01': '增值税专用发票',
    '04': '增值税普通发票',
    '08': '增值税专用发票（电子）',
    '10': '增值税普通发票（电子）',
    '11': '增值税普通发票（卷式）',
    '14': '增值税普通发票（通行费）',
    '31': '全电发票（专用发票）',
    '32': '全电发票（普通发票）',
}

def parse_invoice_qr(raw: str) -> Optional[Dict[str, Any]]:
    """解析增值税/全电发票二维码字符串，返回结构化字段。
    格式：01,类型,发票代码,发票号码,金额,日期,校验码,加密码
    全电发票发票代码字段为空字符串。
    """
    parts = [p.strip() for p in raw.strip().split(',')]
    if len(parts) < 6 or parts[0] != '01':
        return None
    type_code  = parts[1]
    inv_code   = parts[2]   # 全电发票为空
    inv_number = parts[3]
    amount_str = parts[4]
    date_str   = parts[5]   # YYYYMMDD
    verify     = parts[6].strip() if len(parts) > 6 else ''
    encrypt    = parts[7].strip() if len(parts) > 7 else ''

    if not inv_number or not re.match(r'^\d{8,20}$', inv_number):
        return None
    if not re.match(r'^\d{8}$', date_str):
        return None

    issue_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    try:
        total_amount = float(amount_str)
    except (ValueError, TypeError):
        total_amount = 0.0

    return {
        'invoice_type_code': type_code,
        'invoice_title':     _QR_INVOICE_TYPE.get(type_code, f'发票({type_code})'),
        'invoice_code':      inv_code,
        'invoice_number':    inv_number,
        'issue_date':        issue_date,
        'total_amount':      total_amount,
        'verify_code':       verify,
        'encrypt_code':      encrypt,
        'qr_raw':            raw.strip(),
    }


def scan_pdf_page_qr(pdf_path: str, page_index: int = 0) -> Optional[Dict[str, Any]]:
    """从 PDF 指定页扫描二维码并解析发票信息。
    优先高分辨率全页扫描；若失败则裁剪左上角区域重试。
    返回 parse_invoice_qr 的结果，或 None。
    """
    if not _PYZBAR_AVAILABLE:
        return None
    try:
        doc = fitz.open(pdf_path)
        if page_index >= len(doc):
            doc.close()
            return None
        page = doc.load_page(page_index)
        # 300dpi 渲染（3x）
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        img = _PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

        # 全页扫描
        for barcode in qr_decode(img):
            raw = barcode.data.decode('utf-8', errors='replace')
            result = parse_invoice_qr(raw)
            if result:
                return result

        # 左上角裁剪重试（二维码通常在左上 1/3）
        w, h = img.size
        crop = img.crop((0, 0, w // 3, h // 3))
        for barcode in qr_decode(crop):
            raw = barcode.data.decode('utf-8', errors='replace')
            result = parse_invoice_qr(raw)
            if result:
                return result

        return None
    except Exception as e:
        print(f"[qr-scan] page {page_index} error: {e}")
        return None


def scan_image_qr(image_path: str) -> Optional[Dict[str, Any]]:
    """从图片文件扫描发票二维码。"""
    if not _PYZBAR_AVAILABLE:
        return None
    try:
        img = _PILImage.open(image_path).convert('RGB')
        for barcode in qr_decode(img):
            raw = barcode.data.decode('utf-8', errors='replace')
            result = parse_invoice_qr(raw)
            if result:
                return result
        # 左上角裁剪重试
        w, h = img.size
        crop = img.crop((0, 0, w // 3, h // 3))
        for barcode in qr_decode(crop):
            raw = barcode.data.decode('utf-8', errors='replace')
            result = parse_invoice_qr(raw)
            if result:
                return result
        return None
    except Exception as e:
        print(f"[qr-scan] image error: {e}")
        return None


_verify_lock: asyncio.Lock = None

def get_verify_lock() -> asyncio.Lock:
    """在事件循环内懒初始化，避免在 loop 创建前实例化。"""
    global _verify_lock
    if _verify_lock is None:
        _verify_lock = asyncio.Lock()
    return _verify_lock
# =================================================================


def make_step_event(step: str, status: Optional[bool] = None, data: Any = None) -> str:
    ev = {"step": step}
    if status is not None:
        ev["status"] = bool(status)
    if data is not None:
        ev["data"] = data
    status_str = "SUCCESS" if status is True else ("FAIL" if status is False else "RUNNING")
    print(f"[SSE][{step}][{status_str}]", flush=True)
    return f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

def make_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

def log_event(msg: str, level: str = "info", step: str = "execute", status: str = "running") -> str:
    print(f"[{level.upper()}][{step}][{status}] {msg}", flush=True)
    return make_step_event(step=step)



# =================================================================
#  通用工具
# =================================================================
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# =================================================================
#  PDF 相关
# =================================================================
def call_aliyun_vlm(client: Any = None, image_path: str = "") -> Optional[Dict[str, Any]]:
    return call_unified_invoice_ocr(image_path)


def pdf_extract_text(pdf_path: str) -> str:
    """提取 PDF 第一页文本（发票查验用）"""
    try:
        doc = fitz.open(pdf_path)
        text = doc.load_page(0).get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"[pdf-text] 文本提取失败: {e}")
        return ""


def pdf_extract_page_text(pdf_path: str, page_index: int) -> str:
    """提取 PDF 指定页文本"""
    try:
        doc = fitz.open(pdf_path)
        if page_index >= len(doc):
            doc.close()
            return ""
        text = doc.load_page(page_index).get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"[pdf-text] 第{page_index+1}页文本提取失败: {e}")
        return ""


def pdf_page_count(pdf_path: str) -> int:
    """返回 PDF 总页数"""
    try:
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 1


def pdf_parse_regex(text):
    """发票查验四要素提取：发票代码、号码、日期、金额（零token）"""
    if not text or len(text) < 20:
        return None

    num = ""
    m = re.search(r'(?m)^(\d{20})$', text)
    if m: num = m.group(1)
    if not num:
        m = re.search(r'(?m)^(\d{8})$', text)
        if m: num = m.group(1)
    if not num:
        m = re.search(r'发票号码[：:]\s*(\d{8,20})', text)
        if m: num = m.group(1)
    if not num:
        return None

    code = ""
    m = re.search(r'发票代码[：:]\s*(\d{10,12})', text)
    if m and m.group(1) != num:
        code = m.group(1)

    date = ""
    # 优先匹配"开票日期"前缀（铁路客票有乘车日期干扰，必须先锁定开票日期字段）
    m = re.search(r'开票日期[：:]\s*(\d{4})年(\d{1,2})[\u6708\u2F49](\d{1,2})[\u65E5\u2F47]', text)
    if m:
        date = f"{m.group(1)}{m.group(2).zfill(2)}{m.group(3).zfill(2)}"
    if not date:
        # 兼容铁路客票"开票日期:2026 04 28"空格分隔格式
        m = re.search(r'开票日期[：:]\s*(\d{4})[\s/-](\d{1,2})[\s/-](\d{1,2})(?!\d)', text)
        if m:
            date = f"{m.group(1)}{m.group(2).zfill(2)}{m.group(3).zfill(2)}"
    if not date:
        # 兜底：通用年月日格式（兼容 PDF 文字层部首替代字符 ⽉U+2F49 ⽇U+2F47）
        m = re.search(r'(\d{4})年(\d{1,2})[\u6708\u2F49](\d{1,2})[\u65E5\u2F47]', text)
        if m:
            date = f"{m.group(1)}{m.group(2).zfill(2)}{m.group(3).zfill(2)}"
    if not date:
        m = re.search(r'(\d{4})[-/](\d{2})[-/](\d{2})', text)
        if m: date = f"{m.group(1)}{m.group(2)}{m.group(3)}"

    seen, amounts = set(), []
    # 模式1：¥/￥/Y 紧跟数字（同行）
    for m in re.finditer(r'[¥￥]\s*(-?[0-9]+\.[0-9]{2})', text):
        val = m.group(1)
        if val not in seen:
            seen.add(val); amounts.append(float(val))
    for m in re.finditer(r'(?<![A-Z0-9])Y\s*(-?[0-9]+\.[0-9]{2})', text):
        val = m.group(1)
        if val not in seen:
            seen.add(val); amounts.append(float(val))
    # 模式2：数字后紧跟 ¥/￥（同行）
    for m in re.finditer(r'(-?[0-9]+\.[0-9]{2})\s*[¥￥]', text):
        val = m.group(1)
        if val not in seen:
            seen.add(val); amounts.append(float(val))
    # 模式3：¥/￥/Y 换行后接数字（滴滴发票等文字层分行格式）
    for m in re.finditer(r'[¥￥]\s*\n\s*(-?[0-9]+\.[0-9]{2})', text):
        val = m.group(1)
        if val not in seen:
            seen.add(val); amounts.append(float(val))
    # 模式5：数字换行后接 ¥/￥（全电发票金额/符号倒置分行格式）
    for m in re.finditer(r'(-?[0-9]+\.[0-9]{2})\s*\n\s*[¥￥]', text):
        val = m.group(1)
        if val not in seen:
            seen.add(val); amounts.append(float(val))
    # 模式4：专门抓"价税合计"后的裸数字（兜底）
    m_jshj = re.search(r'价税合计[^\n]{0,30}\n\s*(-?[0-9]+\.[0-9]{2})', text)
    if not m_jshj:
        m_jshj = re.search(r'价税合计[（(小写）\s¥￥Y\n]{0,30}(-?[0-9]+\.[0-9]{2})', text)
    if m_jshj:
        val = m_jshj.group(1)
        if val not in seen:
            seen.add(val); amounts.append(float(val))
    # 模式6：裸数字行 + 下行是 (小写)¥ 或单独¥（全电发票价税合计倒置格式）
    for m in re.finditer(r'^\s*(-?[0-9]+\.[0-9]{2})\s*\n\s*(?:\(小写\)\s*)?[¥￥]', text, re.MULTILINE):
        val = m.group(1)
        if val not in seen:
            seen.add(val); amounts.append(float(val))

    amount = 0.0
    if code:
        if amounts: amount = amounts[0]
    else:
        # 无论几个金额，价税合计始终是最大值
        if amounts:
            amount = max(amounts, key=abs)

    print(f'[pdf-text] code={code!r} num={num!r} date={date!r} amounts={amounts} amount={amount}')
    return (code, num, date, amount)


def pdf_to_first_image(pdf_path: str, out_dir: str) -> Optional[str]:
    """将 PDF 第一页渲染为图片（发票查验用）"""
    try:
        doc = fitz.open(pdf_path)
        if not doc: return None
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72), alpha=False)
        out = os.path.join(out_dir, "page_1.png")
        pix.save(out)
        doc.close()
        return out
    except Exception as e:
        print(f"PDF→img error: {e}")
        return None


def pdf_to_page_image(pdf_path: str, page_index: int, out_dir: str) -> Optional[str]:
    """将 PDF 指定页渲染为图片"""
    try:
        doc = fitz.open(pdf_path)
        if page_index >= len(doc):
            doc.close()
            return None
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72), alpha=False)
        out = os.path.join(out_dir, f"page_{page_index + 1}.png")
        pix.save(out)
        doc.close()
        return out
    except Exception as e:
        print(f"PDF→img page {page_index} error: {e}")
        return None


def _parse_vlm_result(data: dict) -> Optional[Tuple[str, str, str, float, str, float, float, float]]:
    code   = str(data.get("invoice_code", "") or "").strip()
    number = str(data.get("invoice_number", "") or "").strip()
    date   = str(data.get("issue_date", "") or "").strip()
    if date and len(date) == 10 and "-" in date:
        date = date.replace("-", "")
    if code and code == number:
        code = ""
    # VLM有时把号码填到code字段：number为空但code不为空时，把code移到number
    if code and not number:
        number = code
        code = ""

    # 全面提取并保留各类金额（不含税金额、价税合计、税额）
    raw_ex = data.get("amount_excluding_tax", 0) or data.get("amount_ex_tax", 0) or 0
    amount_ex_tax = float(raw_ex) if raw_ex else 0.0

    total = data.get("total_amount_including_tax", {}) or {}
    raw_tot = total.get("in_figures", 0) if isinstance(total, dict) else total or 0
    if not raw_tot or float(raw_tot) == 0:
        raw_tot = data.get("total_amount", 0) or 0
    total_amount = float(raw_tot) if raw_tot else (amount_ex_tax if amount_ex_tax else 0.0)

    raw_tax = data.get("tax_amount", 0) or 0
    tax_amount = float(raw_tax) if raw_tax else 0.0

    # 默认兜底金额：若有专票代码优先取不含税，无代码数电票优先取价税合计
    if code:
        amount = amount_ex_tax if amount_ex_tax else total_amount
    else:
        amount = total_amount if total_amount else amount_ex_tax

    check_code = str(data.get("check_code", "") or "").strip()
    # 结构：(code, number, date, default_amount, check_code, amount_ex_tax, total_amount, tax_amount)
    return (code, number, date, amount, check_code, amount_ex_tax, total_amount, tax_amount)


# =================================================================
#  OFD 支持
# =================================================================
def ofd_to_first_image(ofd_path: str, out_dir: str) -> Optional[str]:
    global _easyofd_available
    if not _easyofd_available:
        return None
    try:
        from easyofd.ofd import OFD
        from PIL import Image as PILImage
    except ImportError:
        _easyofd_available = False
        return None
    try:
        import sys, io as _io
        try:
            from loguru import logger as _loguru
            _loguru.disable("easyofd")
        except Exception:
            pass
        with open(ofd_path, "rb") as f:
            b64s = base64.b64encode(f.read()).decode("utf-8")
        _old_stdout = sys.stdout
        sys.stdout = _io.StringIO()
        try:
            ofd = OFD()
            ofd.read(b64s, save_xml=False, xml_name="_tmp_ofd")
            imgs = ofd.to_jpg()
            ofd.del_data()
        finally:
            sys.stdout = _old_stdout
        if not imgs:
            return None
        out = os.path.join(out_dir, "ofd_page_1.jpg")
        PILImage.fromarray(imgs[0]).save(out)
        return out
    except Exception as e:
        _easyofd_available = False
        return None


# =================================================================
#  XML 支持
# =================================================================
_XML_PAT = {
    "code": [
        r"<(?:[^>]*:)?(?:FP_DM|InvoiceCode|InvoiceCodeNum|fpdm|FPDM)[^>]*>([^<]+)<",
        r'Name="发票代码"[^>]*>\s*<[^>]*>([^<]+)<',
    ],
    "num": [
        r"<TaxSupervisionInfo[^>]*>.*?<(?:[^>]*:)?InvoiceNumber[^>]*>([^<]+)<",
        r"<(?:[^>]*:)?(?:FP_HM|InvoiceNo|InvoiceNumber|fphm|FPHM)[^>]*>([^<]+)<",
        r"<(?:[^>]*:)?EIid[^>]*>([^<]+)<",
        r'Name="发票号码"[^>]*>\s*<[^>]*>([^<]+)<',
    ],
    "date": [
        r"<(?:[^>]*:)?(?:KPRQ|IssueDate|IssueTime|kprq|invoiceDate)[^>]*>(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})",
        r"<(?:[^>]*:)?RequestTime[^>]*>(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})",
        r'Name="开票日期"[^>]*>\s*<[^>]*>([^<]+)<',
    ],
    "amex": [
        r"<(?:[^>]*:)?(?:HJBHSJE|TaxExclusiveTotalAmount|TaxExclusiveAmount|TotalAmWithoutTax|AmountWithoutTax|hjbhsje)[^>]*>([0-9]+\.?[0-9]*)<",
        r'Name="合计金额"[^>]*>\s*<[^>]*>([^<]+)<',
        r'Name="不含税金额"[^>]*>\s*<[^>]*>([^<]+)<',
    ],
    "amtot": [
        r"<[^>]*TotalTax[^>-]*-[^>]*includedAmount[^>]*>([0-9]+\.?[0-9]*)<",
        r"<(?:[^>]*:)?(?:JSHJ|TaxInclusiveTotalAmount|TaxInclusiveAmount|TotalAmount|jshj)[^>]*>([0-9]+\.?[0-9]*)<",
        r'Name="价税合计"[^>]*>\s*<[^>]*>([^<]+)<',
    ],
    "title": [
        r"<(?:[^>]*:)?(?:FPLX|InvoiceType|invoiceType)[^>]*>([^<]+)<",
        r'Name="发票种类"[^>]*>\s*<[^>]*>([^<]+)<',
    ],
    "buyer_name": [
        r"<(?:[^>]*:)?(?:GMFMC|BuyerName|buyerName)[^>]*>([^<]+)<",
        r'Name="购买方名称"[^>]*>\s*<[^>]*>([^<]+)<',
    ],
    "buyer_tax": [
        r"<(?:[^>]*:)?(?:GMFNSRSBH|BuyerTaxNo|BuyerTaxID|buyerTaxNo)[^>]*>([^<]+)<",
        r'Name="购买方税号"[^>]*>\s*<[^>]*>([^<]+)<',
    ],
    "seller_name": [
        r"<(?:[^>]*:)?(?:XHFMC|SellerName|sellerName)[^>]*>([^<]+)<",
        r'Name="销售方名称"[^>]*>\s*<[^>]*>([^<]+)<',
    ],
    "seller_tax": [
        r"<(?:[^>]*:)?(?:XHFNSRSBH|SellerTaxNo|SellerTaxID|sellerTaxNo)[^>]*>([^<]+)<",
        r'Name="销售方税号"[^>]*>\s*<[^>]*>([^<]+)<',
    ],
    "tax_amount": [
        r"<(?:[^>]*:)?(?:HJSE|TaxTotalAmount|TaxAmount|totalTaxAmount)[^>]*>([0-9]+\.?[0-9]*)<",
        r'Name="合计税额"[^>]*>\s*<[^>]*>([^<]+)<',
    ],
    "total_words": [
        r"<(?:[^>]*:)?(?:JSHJ_DX|TotalAmountInWords)[^>]*>([^<]+)<",
        r'Name="价税合计大写"[^>]*>\s*<[^>]*>([^<]+)<',
    ],
    "issuer": [
        r"<(?:[^>]*:)?(?:KPR|InvoiceClerk|Drawer|drawer)[^>]*>([^<]+)<",
        r'Name="开票人"[^>]*>\s*<[^>]*>([^<]+)<',
    ],
    "remarks": [
        r"<(?:[^>]*:)?(?:BZ|Note|Remarks|remarks)[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</(?:[^>]*:)?(?:BZ|Note|Remarks|remarks)>",
        r'Name="备注"[^>]*>\s*<[^>]*>([^<]+)<',
    ],
}

_XML_ITEM_PATTERNS = [
    r"<(?:[^>]*:)?(?:SPMC|GoodsName|ItemName|itemName)[^>]*>([^<]+)<",
    r'Name="货物或应税劳务[^"]*"[^>]*>\s*<[^>]*>([^<]+)<',
]


def _norm_date(raw: str) -> str:
    raw = str(raw).strip()
    m = re.match(r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})', raw)
    if m:
        return f"{m.group(1)}{m.group(2).zfill(2)}{m.group(3).zfill(2)}"
    if re.match(r'^\d{8}$', raw):
        return raw
    cleaned = re.sub(r'[年月日\-/\s]', '', raw)
    return cleaned[:8] if len(cleaned) >= 8 else cleaned


def _clean_amount(raw: str) -> float:
    raw = re.sub(r'[￥¥,，\s]', '', str(raw))
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 0.0


def xml_parse_regex(txt: str) -> Optional[Tuple[str, str, str, float]]:
    result = {}
    for field in ["code", "num", "date", "amex", "amtot"]:
        patterns = _XML_PAT.get(field, [])
        for pat in patterns:
            flags = re.DOTALL if "TaxSupervisionInfo" in pat else 0
            m = re.search(pat, txt, flags)
            if m:
                result[field] = m.group(1).strip()
                break
    number = result.get("num", "").strip()
    if not number or not re.match(r'^\d{8,25}$', number):
        return None
    code = result.get("code", "").strip()
    date = _norm_date(result.get("date", ""))
    amount = _clean_amount(result.get("amex" if code else "amtot",
                                      result.get("amtot", result.get("amex", "0"))))
    return (code, number, date, amount)


def xml_parse_full(txt: str) -> Optional[Dict[str, Any]]:
    result = {}
    for field, patterns in _XML_PAT.items():
        for pat in patterns:
            flags = re.DOTALL if "TaxSupervisionInfo" in pat else 0
            m = re.search(pat, txt, flags)
            if m:
                result[field] = m.group(1).strip()
                break

    number = result.get("num", "").strip()
    if not number or not re.match(r'^\d{8,25}$', number):
        return None

    code = result.get("code", "").strip()
    date_raw = result.get("date", "")
    date_formatted = ""
    if date_raw:
        d = _norm_date(date_raw)
        if len(d) == 8:
            date_formatted = f"{d[:4]}-{d[4:6]}-{d[6:]}"

    amex = _clean_amount(result.get("amex", "0"))
    amtot = _clean_amount(result.get("amtot", "0"))
    tax_amount = _clean_amount(result.get("tax_amount", "0"))

    items = []
    for pat in _XML_ITEM_PATTERNS:
        found = re.findall(pat, txt)
        if found:
            items = [x.strip() for x in found if x.strip()]
            break

    return {
        "invoice_title": result.get("title", ""),
        "invoice_code": code,
        "invoice_number": number,
        "issue_date": date_formatted,
        "buyer_name": result.get("buyer_name", ""),
        "buyer_tax_id": result.get("buyer_tax", ""),
        "seller_name": result.get("seller_name", ""),
        "seller_tax_id": result.get("seller_tax", ""),
        "items": items,
        "amount_ex_tax": amex,
        "tax_amount": tax_amount,
        "total_amount": amtot,
        "total_words": result.get("total_words", ""),
        "remarks": result.get("remarks", ""),
        "issuer": result.get("issuer", ""),
        "source": "regex",
    }


def _read_xml_text(path: str) -> str:
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


# =================================================================
#  统一入口：extract_invoice_info（发票查验用）
# =================================================================
def extract_invoice_info(client: OpenAI, invoice_path: str) -> Optional[Tuple[str, str, str, float]]:
    ext = os.path.splitext(invoice_path)[1].lower()
    tmp = tempfile.mkdtemp(prefix="inv_parse_")
    try:
        if ext == ".pdf":
            text = pdf_extract_text(invoice_path)
            if text:
                result = pdf_parse_regex(text)
                if result and result[1] and result[2]:
                    return result
            img = pdf_to_first_image(invoice_path, tmp)
            if not img: return None
            data = call_aliyun_vlm(client, img)
            if not data: return None
            return _parse_vlm_result(data)

        elif ext == ".xml":
            txt = _read_xml_text(invoice_path)
            result = xml_parse_regex(txt)
            if result and result[1]:
                return result
            return None

        elif ext == ".ofd":
            if zipfile.is_zipfile(invoice_path):
                with zipfile.ZipFile(invoice_path, "r") as z:
                    all_xml = [n for n in z.namelist() if n.lower().endswith(".xml")]
                    def xml_priority(name):
                        nl = name.lower()
                        if any(x in nl for x in ("attach", "invoice", "einvoice")): return 0
                        if nl == "ofd.xml": return 1
                        if any(x in nl for x in ("page", "content", "sign", "seal")): return 9
                        return 5
                    all_xml.sort(key=xml_priority)
                    for name in all_xml:
                        try:
                            txt = z.read(name).decode("utf-8", errors="replace")
                            result = xml_parse_regex(txt)
                            if result and result[1]:
                                return result
                        except Exception:
                            continue
            return None

        elif ext in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
            mime_map = {'.jpg':'image/jpeg','.jpeg':'image/jpeg','.png':'image/png','.webp':'image/webp','.bmp':'image/bmp'}
            mime = mime_map.get(ext, 'image/jpeg')
            data = call_unified_invoice_ocr(invoice_path, mime)
            if data:
                return _parse_vlm_result(data)
            return None

        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =================================================================
#  发票识别整理 - 完整字段提取
# =================================================================
def _parse_train_ticket_regex(text: str) -> Optional[Dict[str, Any]]:
    """专项解析铁路电子客票，提取购买方信息和行程详情。"""
    # 发票号码（20位）
    num = ""
    m = re.search(r'发票号码[：:]\s*(\d{20})', text)
    if m: num = m.group(1)
    if not num:
        m = re.search(r'(?m)^(\d{20})$', text)
        if m: num = m.group(1)
    if not num:
        return None

    # 开票日期（格式：开票日期:2026年04月28日）
    date = ""
    m = re.search(r'开票日期[：:]\s*(\d{4})年(\d{1,2})月(\d{1,2})日', text)
    if m:
        date = f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    if not date:
        m = re.search(r'开票日期[：:]\s*(\d{4})[-/](\d{2})[-/](\d{2})', text)
        if m:
            date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # 购买方名称：两种格式
    # 格式A: "购买方名称:江苏坤力..." 值在同行
    # 格式B: "购买方名称:\n江苏坤力..." 值在下行，上行标签后为空
    buyer_name = ""
    m = re.search(r'购买方名称[：:]\s*(.+)', text)
    if m:
        val = m.group(1).strip()
        if val:
            buyer_name = val
        else:
            # 标签行无值，找下一个非空行（公司名单独成行）
            m2 = re.search(r'购买方名称[：:]\s*\n([\u4e00-\u9fa5A-Za-z0-9（）\(\)]{2,})', text)
            if m2:
                buyer_name = m2.group(1).strip()
    # 兜底：在文本中直接匹配公司名（含"有限"等关键词）
    if not buyer_name:
        m = re.search(r'([\u4e00-\u9fa5]{2,}(?:有限|股份|集团|公司|医院|学校|研究)[^\n\s]{0,20}(?:公司|院|校|所)?)', text)
        if m:
            buyer_name = m.group(1).strip()

    # 购买方税号（统一社会信用代码，18位字母数字）
    buyer_tax = ""
    m = re.search(r'统一社会信用代码[：:]\s*([A-Z0-9]{15,20})', text)
    if m:
        buyer_tax = m.group(1).strip()

    # 站名提取：用英文站名的出现顺序确定出发/到达方向，
    # 在带"站"字的中文行里找与英文行顺序一致、总距离最小的匹配对。
    # 完全不依赖对照表，适用于任意站点。
    origin = dest = ""
    _SKIP_STATION_PAT = re.compile(r'[年月日号车座卧票价]')
    _NOT_STATION_WORDS = {'二等座', '一等座', '商务座', '软座', '硬座', '软卧', '硬卧', '无座'}

    def _is_cn_station(s):
        if not re.match(r'^[\u4e00-\u9fa5]{2,8}站?$', s):
            return False
        return s not in _NOT_STATION_WORDS and not _SKIP_STATION_PAT.search(s)

    _lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    _en_idx = [i for i, ln in enumerate(_lines) if re.match(r'^[A-Z][a-z]{2,}$', ln)]
    _cn_all  = [i for i, ln in enumerate(_lines) if _is_cn_station(ln)]
    _cn_zhan = [i for i in _cn_all if _lines[i].endswith('站')]
    _cn_pool = _cn_zhan if len(_cn_zhan) >= 2 else _cn_all

    if len(_en_idx) >= 2 and len(_cn_pool) >= 2:
        # 枚举中文行有序对，选总距离最小且顺序与英文行一致的匹配
        _best_dist, _best_pair = float('inf'), (None, None)
        _en_asc = _en_idx[0] < _en_idx[1]
        for _a in range(len(_cn_pool)):
            for _b in range(len(_cn_pool)):
                if _a == _b:
                    continue
                _ci_a, _ci_b = _cn_pool[_a], _cn_pool[_b]
                if (_ci_a < _ci_b) != _en_asc:
                    continue
                _d = abs(_ci_a - _en_idx[0]) + abs(_ci_b - _en_idx[1])
                if _d < _best_dist:
                    _best_dist = _d
                    _best_pair = (_ci_a, _ci_b)
        if _best_pair[0] is not None:
            origin = _lines[_best_pair[0]].replace('站', '')
            dest   = _lines[_best_pair[1]].replace('站', '')

    # 车次（G/D/C/Z/T/K + 数字）
    train_no = ""
    m = re.search(r'\b([GDCZTKY]\d{1,4}[A-Z]?)\b', text)
    if m:
        train_no = m.group(1)

    # 出行日期（"2026年03月13日"，要排除开票日期）
    travel_date = travel_time = ""
    # 找所有年月日格式，取非开票日期的第一个
    opening_date_str = date.replace("-", "年", 1).replace("-", "月", 1) + "日" if date else ""
    for mm in re.finditer(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text):
        td = f"{mm.group(1)}-{mm.group(2).zfill(2)}-{mm.group(3).zfill(2)}"
        if td != date:
            travel_date = td
            break

    # 出发时间（"16:01开" 或 "16:01"）
    m = re.search(r'(\d{2}:\d{2})开?', text)
    if m:
        travel_time = m.group(1)

    # 车厢座位（"05车05D号" 或 "03车10D号"）
    carriage = seat = ""
    m = re.search(r'(\d{2})车(\d{2,3}[A-Z])号', text)
    if m:
        carriage = m.group(1) + "车"
        seat = m.group(2) + "号"

    # 座位等级
    seat_grade = ""
    m = re.search(r'([一二三]等座|商务座|软座|硬座|软卧|硬卧|无座)', text)
    if m:
        seat_grade = m.group(1)

    # 价税合计金额（"票价:￥303.00" 或 "￥303.00" 单独出现）
    total_amount = 0.0
    m = re.search(r'票价[：:]?\s*[¥￥]\s*([0-9]+\.?[0-9]*)', text)
    if m:
        total_amount = float(m.group(1))
    if not total_amount:
        m = re.search(r'[¥￥]\s*([0-9]+\.[0-9]{2})', text)
        if m:
            total_amount = float(m.group(1))

    # 组装商品详情（单条字符串，便于Excel阅读）
    item_parts = []
    if origin and dest:
        item_parts.append(f"{origin}→{dest}")
    if train_no:
        item_parts.append(f"车次:{train_no}")
    if travel_date:
        item_parts.append(f"日期:{travel_date}")
    if travel_time:
        item_parts.append(f"发车:{travel_time}")
    if carriage and seat:
        item_parts.append(f"{carriage}{seat}")
    elif carriage:
        item_parts.append(carriage)
    if seat_grade:
        item_parts.append(seat_grade)

    items = ["  ".join(item_parts)] if item_parts else []

    return {
        "invoice_title": "电子发票（铁路电子客票）",
        "invoice_code": "",
        "invoice_number": num,
        "issue_date": date,
        "buyer_name": buyer_name,
        "buyer_tax_id": buyer_tax,
        "seller_name": "",
        "seller_tax_id": "",
        "items": items,
        "amount_ex_tax": "",
        "tax_amount": "",
        "total_amount": round(total_amount, 2),
        "total_words": "",
        "remarks": "",
        "issuer": "",
        "source": "regex",
    }


def _pdf_parse_full_regex(text: str) -> Optional[Dict[str, Any]]:
    if not text or len(text) < 20:
        return None

    # 铁路电子客票走专用解析路径
    if "铁路电子客票" in text or "电子客票号" in text:
        return _parse_train_ticket_regex(text)

    title = ""
    for kw in [
        "电子发票（增值税专用发票）", "电子发票（专用发票）",
        "电子发票（普通发票）", "电⼦发票（普通发票）",
        "增值税电子专用发票", "增值税电子普通发票",
        "增值税专用发票", "增值税普通发票",
        "全电专用发票", "全电普通发票", "全电发票",
        "数电票", "电子发票（铁路电子客票）",
    ]:
        if kw in text:
            title = kw.replace("电⼦", "电子")
            break
    # 兜底：部分PDF括号字符丢失（如"电子发票普通发票)"），用模糊匹配
    if not title:
        m_title = re.search(r'电[子�子]发票[^\n]{0,10}(?:专用|普通)发票', text)
        if m_title:
            raw = m_title.group(0)
            if "专用" in raw:
                title = "电子发票（专用发票）"
            else:
                title = "电子发票（普通发票）"

    num = ""
    m = re.search(r'(?m)^(\d{20})$', text)
    if m: num = m.group(1)
    if not num:
        m = re.search(r'发票号码[：:]\s*(\d{20})', text)
        if m: num = m.group(1)
    if not num:
        m = re.search(r'(?m)^(\d{8})$', text)
        if m: num = m.group(1)
    if not num:
        return None

    code = ""
    m = re.search(r'发票代码[：:]\s*(\d{10,12})', text)
    if m and m.group(1) != num:
        code = m.group(1)

    date = ""
    # 兼容 PDF 文字层中的部首替代字符 ⽉(U+2F49) ⽇(U+2F47)
    m = re.search(r'(\d{4})年(\d{1,2})[\u6708\u2F49](\d{1,2})[\u65E5\u2F47]', text)
    if m:
        date = f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    if not date:
        m = re.search(r'开票日期[：:]\s*(\d{4})\s+(\d{2})\s+(\d{2})', text)
        if m: date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    buyer_name = seller_name = buyer_tax = seller_tax = ""

    # ── 购买方/销售方提取（两级策略）──
    # 策略1（优先）：买方锚点在卖方锚点之前时，按区域切分直接提取——最精准
    # 策略2（降级）：区域提取失败时，全局配对算法（税号+公司名联合最优匹配）

    _co_kw  = r'(?:有限|股份|集团|公司|医院|学校|研究院|研究所|门诊|诊所|药店|餐饮店|旅行社|工商户|宾馆|酒店|招待所|饭店|事业单位)'
    _co_pat = re.compile(r'[\u4e00-\u9fa5A-Za-z0-9（）\(\)]{2,50}' + _co_kw + r'[^\n（(]{0,15}(?:[）\)）])?')

    def _is_tax_val(s):
        return (bool(re.match(r'^[A-Z0-9]{15,20}$', s))
                and s != num  # 排除发票号码本身
                and not re.match(r'^\d{20}$', s))  # 排除20位纯数字

    def _valid_name(s):
        return (len(re.findall(r'[\u4e00-\u9fa5]', s)) >= 2
                and not re.match(r'^统一|^纳税|^项目|^规格|^名称$', s))

    def _extract_region(region):
        """在给定区域文本内提取公司名和税号"""
        name = tax = ""
        _m2 = re.search(r'名称[：:]\s*([\u4e00-\u9fa5A-Za-z0-9（）\(\)]{2,50})', region)
        if _m2 and _valid_name(_m2.group(1).strip()): name = _m2.group(1).strip()
        if not name:
            _m2 = _co_pat.search(region)
            if _m2: name = _m2.group(0).strip()
        _m2 = re.search(r'统一社会信用代码[/纳税人识别号]*[：:]\s*([A-Z0-9]{10,20})', region)
        if _m2 and _is_tax_val(_m2.group(1)): tax = _m2.group(1)
        if not tax:
            for _m2 in re.finditer(r'(?m)^([A-Z0-9]{15,20})$', region):
                if _is_tax_val(_m2.group(1)): tax = _m2.group(1); break
        return name, tax

    # 找锚点的开始和结束位置（开始用于切分区域，结束用于全局配对距离参考）
    _buyer_start = _buyer_end = _seller_start = _seller_end = -1
    for _pat in [r'购\n买\n方\n信\n息', r'购\n买\n方', r'购买方信息', r'购买方[：:]']:
        _m2 = re.search(_pat, text)
        if _m2: _buyer_start, _buyer_end = _m2.start(), _m2.end(); break
    for _pat in [r'销\n售\n方\n信\n息', r'销\n售\n方', r'销售方信息', r'销售方[：:]']:
        _m2 = re.search(_pat, text)
        if _m2: _seller_start, _seller_end = _m2.start(), _m2.end(); break

    # ── 策略1：区域切分 ──
    if 0 <= _buyer_end <= _seller_start:
        _items_pos = text.find('项目名称')
        if _items_pos < 0: _items_pos = len(text)
        _br = text[_buyer_end:_seller_start]   # 买方区域：买方锚结束→卖方锚开始
        _sr = text[_seller_end:_items_pos]      # 卖方区域：卖方锚结束→商品表开始
        _bn, _bt = _extract_region(_br)
        _sn, _st = _extract_region(_sr)
        if _bn or _sn:
            buyer_name, buyer_tax, seller_name, seller_tax = _bn, _bt, _sn, _st

    # ── 策略2：全局配对（策略1未能提取完整时降级）──
    if not (buyer_name or seller_name):
        _found_tax = {}
        for _m2 in re.finditer(r'统一社会信用代码[/纳税人识别号]*[：:]\s*([A-Z0-9]{10,20})', text):
            _tv = _m2.group(1)
            if _tv and _is_tax_val(_tv) and _tv not in _found_tax: _found_tax[_tv] = _m2.start()
        for _m2 in re.finditer(r'(?m)^([A-Z0-9]{15,20})$', text):
            _tv = _m2.group(1)
            if _is_tax_val(_tv) and _tv not in _found_tax: _found_tax[_tv] = _m2.start()
        _taxes = sorted(_found_tax.items(), key=lambda x: x[1])

        _found_co = {}
        for _m2 in re.finditer(r'名称[：:]\s*([\u4e00-\u9fa5A-Za-z0-9（）\(\)]{3,50})', text):
            _val = _m2.group(1).strip()
            if _valid_name(_val) and _val not in _found_co: _found_co[_val] = _m2.start()
        for _m2 in _co_pat.finditer(text):
            _val = _m2.group(0).strip()
            if _val not in _found_co: _found_co[_val] = _m2.start()
        _companies = sorted(_found_co.items(), key=lambda x: x[1])

        _bp_ref = _buyer_end if _buyer_end >= 0 else _buyer_start
        _sp_ref = _seller_end if _seller_end >= 0 else _seller_start

        if _taxes:
            if len(_taxes) >= 2:
                _t0, _p0 = _taxes[0]; _t1, _p1 = _taxes[1]
                if _bp_ref >= 0 and _sp_ref >= 0:
                    if abs(_p0-_bp_ref)+abs(_p1-_sp_ref) <= abs(_p0-_sp_ref)+abs(_p1-_bp_ref):
                        buyer_tax, _bp, seller_tax, _sp = _t0, _p0, _t1, _p1
                    else:
                        buyer_tax, _bp, seller_tax, _sp = _t1, _p1, _t0, _p0
                else:
                    buyer_tax, _bp, seller_tax, _sp = _t0, _p0, _t1, _p1
                if len(_companies) >= 2:
                    _best, buyer_name, seller_name = float('inf'), "", ""
                    for _i, (_n0, _cp0) in enumerate(_companies):
                        for _j, (_n1, _cp1) in enumerate(_companies):
                            if _i == _j: continue
                            _sc = abs(_cp0-_bp)+abs(_cp1-_sp)
                            if _sc < _best: _best, buyer_name, seller_name = _sc, _n0, _n1
                elif _companies:
                    _n, _cp = _companies[0]
                    if abs(_cp-_bp) <= abs(_cp-_sp): buyer_name = _n
                    else: seller_name = _n
            else:
                _t0, _p0 = _taxes[0]
                _use_b = _bp_ref >= 0 and (_sp_ref < 0 or abs(_p0-_bp_ref) <= abs(_p0-_sp_ref))
                if _use_b: buyer_tax = _t0
                else: seller_tax = _t0
                if _companies:
                    _n, _ = _companies[0]
                    if _use_b: buyer_name = _n
                    else: seller_name = _n

    # 商品名：*大类*品名 格式（如 *餐饮服务*餐），拼成"大类-品名"
    items = []
    for _cat, _item in re.findall(r'\*([^*\n]+)\*([^\n¥￥*]{0,20})', text):
        _name = (_cat.strip() + ('-' + _item.strip() if _item.strip() else ''))
        if _name:
            items.append(_name)
    # 兜底：无星号格式，找行首非数字/符号的商品描述行
    if not items:
        items = re.findall(r'(?m)^([^\d\s¥￥*#（）\n]{3,20})\s+\d', text)
        items = [x.strip() for x in items if x.strip()][:5]

    seen_amounts, amounts_list = set(), []
    # 模式1：¥/￥/Y 紧跟数字（同行）
    for m2 in re.finditer(r'[¥￥]\s*([0-9]+\.[0-9]{2})', text):
        v = m2.group(1)
        if v not in seen_amounts:
            seen_amounts.add(v); amounts_list.append(float(v))
    for m2 in re.finditer(r'(?<![A-Z0-9])Y\s*([0-9]+\.[0-9]{2})', text):
        v = m2.group(1)
        if v not in seen_amounts:
            seen_amounts.add(v); amounts_list.append(float(v))
    # 模式2：数字后紧跟 ¥/￥
    for m2 in re.finditer(r'([0-9]+\.[0-9]{2})\s*[¥￥]', text):
        v = m2.group(1)
        if v not in seen_amounts:
            seen_amounts.add(v); amounts_list.append(float(v))
    # 模式3：¥/￥/Y 换行后接数字（滴滴发票等文字层分行格式）
    for m2 in re.finditer(r'[¥￥]\s*\n\s*([0-9]+\.[0-9]{2})', text):
        v = m2.group(1)
        if v not in seen_amounts:
            seen_amounts.add(v); amounts_list.append(float(v))
    # 模式5：数字换行后接 ¥/￥（全电发票金额/符号倒置分行格式）
    for m2 in re.finditer(r'([0-9]+\.[0-9]{2})\s*\n\s*[¥￥]', text):
        v = m2.group(1)
        if v not in seen_amounts:
            seen_amounts.add(v); amounts_list.append(float(v))
    # 模式4：专门抓"价税合计"后的裸数字（兜底）
    m_jshj2 = re.search(r'价税合计[^\n]{0,30}\n\s*([0-9]+\.[0-9]{2})', text)
    if not m_jshj2:
        m_jshj2 = re.search(r'价税合计[（(小写）\s¥￥Y\n]{0,30}([0-9]+\.[0-9]{2})', text)
    if m_jshj2:
        v = m_jshj2.group(1)
        if v not in seen_amounts:
            seen_amounts.add(v); amounts_list.append(float(v))
    # 模式6：裸数字行 + 下行是 (小写)¥ 或单独¥（全电发票价税合计倒置格式）
    for m2 in re.finditer(r'^\s*([0-9]+\.[0-9]{2})\s*\n\s*(?:\(小写\)\s*)?[¥￥]', text, re.MULTILINE):
        v = m2.group(1)
        if v not in seen_amounts:
            seen_amounts.add(v); amounts_list.append(float(v))

    amex = tax_amount = amtot = 0.0
    if code:
        if amounts_list: amex = amounts_list[0]
    else:
        # 价税合计 = 最大金额；不含税金额和税额从剩余金额中推算
        if amounts_list:
            amtot = max(amounts_list)
            rest = sorted([a for a in amounts_list if a != amtot])
            if len(rest) >= 2:
                # 找两个加起来等于 amtot 的组合
                found = False
                for i in range(len(rest)):
                    for j in range(i+1, len(rest)):
                        if abs(rest[i] + rest[j] - amtot) < 0.02:
                            amex = rest[i]; tax_amount = rest[j]; found = True; break
                    if found: break
                if not found:
                    amex = rest[-1]  # 取最大的作为不含税金额
                else:
                    # 确保不含税金额 >= 税额
                    if amex < tax_amount:
                        amex, tax_amount = tax_amount, amex
            elif len(rest) == 1:
                amex = rest[0]

    total_words = ""
    # 普通格式：价税合计（大写）紧跟汉字（含部首替代字 ⻆ U+2EC6）
    m2 = re.search(r'(?:价税合计(?:大写)?)[（(（）\s]*([壹贰叁参肆伍陆柒捌玖拾佰仟万亿圆整零角分\u2EC6元]{2,})', text)
    if m2:
        total_words = m2.group(1).strip()
    # 分离格式兜底：全文搜索大写金额（含 ⻆）
    if not total_words:
        m2 = re.search(r'([壹贰叁参肆伍陆柒捌玖拾佰仟万亿圆整零角分\u2EC6元]{2,})', text)
        if m2: total_words = m2.group(1).strip()

    # 发票模板中出现的非人名单字/词，排除掉避免误识别为开票人
    _TMPL_WORDS = {'备', '注', '购', '买', '方', '信', '息', '销', '售', '合', '计',
                   '价', '税', '开', '票', '人', '名', '称', '项', '目', '数', '量',
                   '单', '价', '金', '额', '等', '级'}
    issuer = ""
    _TMPL_WORDS_ISSUER = {
        '备', '注', '购', '买', '方', '信', '息', '销', '售', '合', '计',
        '价', '税', '开', '票', '人', '名', '称', '项', '目', '数', '量',
        '单', '金', '额', '等', '级',
    }
    _NON_NAME_PAT = re.compile(
        r'[名号日地级额率量价行类型信息代码识别统项目规格单位数出等交通工具有效身份证件合计备注开票购买销售'
        r'壹贰叁肆伍陆柒捌玖拾佰仟万亿圆元角分整零]'
    )
    m2 = re.search(r'开票人[：:]\s*(\S+)', text)
    if m2:
        val = m2.group(1).strip()
        # 无论是数字（分离格式）还是模板词（如"备"），都尝试从独立汉字行中提取
        # 非人名黑名单：印章机构名等
        _NOT_ISSUER = {'国家税务总局', '税务总局', '税务局', '北京市税务局',
                       '上海市税务局', '江苏省税务局', '浙江省税务局'}
        need_fallback = (re.match(r'^\d{15,}$', val)          # 纯数字（发票号码）
                         or val in _TMPL_WORDS_ISSUER           # 模板词
                         or val in _NOT_ISSUER                  # 机构印章名
                         or len(val) > 6)                       # 过长（机构名）
        if need_fallback:
            # 真正开票人在数据段末尾：找独立行2-5字人名，允许中间有空格（如"高 洪运"）
            all_names = re.findall(r'(?m)^([\u4e00-\u9fa5][\u4e00-\u9fa5 ]{0,3}[\u4e00-\u9fa5])$', text)
            all_names = [n.strip() for n in all_names
                         if n.strip() and n.strip() not in _NOT_ISSUER
                         and n.strip() not in _TMPL_WORDS_ISSUER
                         and not _NON_NAME_PAT.search(n.strip())]
            if all_names:
                issuer = all_names[-1]
        else:
            # 普通格式：排除单字和模板词
            if val and not re.match(r'^[\u4e00-\u9fa5]$', val):
                issuer = val

    remarks = ""
    m2 = re.search(r'备注[：:]\s*(.+?)(?:\n|$)', text)
    if m2: remarks = m2.group(1).strip()

    return {
        "invoice_title": title,
        "invoice_code": code,
        "invoice_number": num,
        "issue_date": date,
        "buyer_name": buyer_name,
        "buyer_tax_id": buyer_tax,
        "seller_name": seller_name,
        "seller_tax_id": seller_tax,
        "items": items,
        "amount_ex_tax": round(amex, 2),
        "tax_amount": round(tax_amount, 2),
        "total_amount": round(amtot, 2),
        "total_words": total_words,
        "remarks": remarks,
        "issuer": issuer,
        "source": "regex",
    }


def _regex_result_needs_vlm(r: dict) -> bool:
    """
    判断 regex 识别结果是否需要 VLM 补全。
    触发条件（任一满足即触发）：
    1. 购买方名称为空
    2. 销售方名称为空
    3. 买卖方名称相同（识别错乱）
    4. 商品详情为空
    铁路电子客票豁免（本来就没有销售方）。
    """
    if r.get("invoice_title") == "电子发票（铁路电子客票）":
        return False
    buyer  = str(r.get("buyer_name",  "") or "").strip()
    seller = str(r.get("seller_name", "") or "").strip()
    if not buyer or not seller:
        return True
    if buyer == seller:
        return True
    if not r.get("items"):
        return True
    return False


def _merge_vlm_into_regex(regex_r: dict, vlm_r: dict) -> dict:
    """
    用 VLM 结果补全 regex 结果中的空字段。
    原则：regex 已有的字段不覆盖（regex 在号码/金额上更准），
          只补充 regex 识别为空的字段。
    """
    补全字段 = ["buyer_name", "buyer_tax_id", "seller_name", "seller_tax_id",
               "items", "invoice_title", "total_words", "remarks", "issuer"]
    for field in 补全字段:
        if not regex_r.get(field):
            vlm_val = vlm_r.get(field)
            if vlm_val:
                regex_r[field] = vlm_val
    # 标记来源
    regex_r["source"] = regex_r.get("source", "regex") + "+vlm"
    return regex_r


def _call_recognize_vlm(client: Any = None, image_path: str = "", mime: str = "image/png") -> Optional[Dict[str, Any]]:
    return call_unified_invoice_ocr(image_path, mime)


def _fix_vlm_invoice_code(r: dict) -> dict:
    """
    VLM 返回的发票代码/号码字段经常出错，按以下顺序纠正：
    1. code == number → code 置空
    2. code 非空但 number 为空 → 先把 code 移到 number（必须在规则4之前）
    3. 全电票：number 为20位纯数字 → code 强制置空
    4. code 不是10-12位纯数字 → 不是合法代码，置空
    """
    code   = str(r.get("invoice_code",   "") or "").strip()
    number = str(r.get("invoice_number", "") or "").strip()

    # 规则1：code 和 number 相同，code 置空
    if code and code == number:
        code = ""
    # 规则2：code 非空但 number 为空 → 移过去（放在规则4之前，否则20位code会被规则4先清掉）
    if code and not number:
        number = code
        code = ""
    # 规则3：20位纯数字号码 = 全电票，无发票代码
    if re.match(r'^\d{20}$', number):
        code = ""
    # 规则4：代码必须是10-12位纯数字，否则置空
    if code and not re.match(r'^\d{10,12}$', code):
        code = ""

    r["invoice_code"]   = code
    r["invoice_number"] = number
    return r


def _merge_qr_into_record(record: dict, qr: dict) -> dict:
    """用二维码解析结果覆盖/补充识别记录中的高可信字段。
    二维码数据来源于防伪芯片，优先级高于 regex/VLM 结果。
    覆盖字段：invoice_number, invoice_code, issue_date, total_amount, invoice_title
    补充字段：verify_code, encrypt_code, qr_raw（仅在记录中不存在时写入）
    """
    # 高可信字段：直接覆盖
    if qr.get("invoice_number"):
        record["invoice_number"] = qr["invoice_number"]
    if qr.get("invoice_code") is not None:          # 空字符串也要写（全电票无代码）
        record["invoice_code"] = qr["invoice_code"]
    if qr.get("issue_date"):
        record["issue_date"] = qr["issue_date"]
    if qr.get("total_amount"):
        record["total_amount"] = qr["total_amount"]
    if qr.get("invoice_title") and not record.get("invoice_title"):
        record["invoice_title"] = qr["invoice_title"]

    # 附加字段：仅补充
    for key in ("verify_code", "encrypt_code", "qr_raw"):
        if qr.get(key) and not record.get(key):
            record[key] = qr[key]

    # 标记识别方式：在原方式后追加 +qr
    src = record.get("source", "")
    if "+qr" not in src:
        record["source"] = src + "+qr" if src else "qr"
    return record


def recognize_pdf_multipage(invoice_path: str, fname: str, client: OpenAI, tmp: str) -> List[Dict[str, Any]]:
    """
    逐页识别多页 PDF，每页视为一张独立发票。
    优先二维码扫描，其次 regex 文本提取，失败则渲染图片调用 VLM 识别。
    """
    results = []
    page_count = pdf_page_count(invoice_path)

    for i in range(page_count):
        page_label = f"{fname} [第{i+1}页/共{page_count}页]"
        print(f"[multipage-pdf] 识别第 {i+1}/{page_count} 页: {fname}")

        # ① 二维码扫描
        qr = scan_pdf_page_qr(invoice_path, i)

        # ② 文本层 regex
        text = pdf_extract_page_text(invoice_path, i)
        r = None
        if text and len(text) >= 20:
            r = _pdf_parse_full_regex(text)
            if r:
                r["original_file"] = page_label

        # ③ VLM 兜底：regex 完全失败 或 关键字段缺失时调用
        if not r or _regex_result_needs_vlm(r):
            img = pdf_to_page_image(invoice_path, i, tmp)
            if img:
                vlm_r = _call_recognize_vlm(client, img, "image/png")
                if vlm_r:
                    vlm_r = _fix_vlm_invoice_code(vlm_r)
                    if not r:
                        r = vlm_r
                        r["source"] = "vlm"
                        r["original_file"] = page_label
                    else:
                        r = _merge_vlm_into_regex(r, vlm_r)

        # ④ 二维码数据覆盖
        if r and qr:
            r = _merge_qr_into_record(r, qr)

        if r:
            results.append(r)
        else:
            results.append({"original_file": page_label, "error": "识别失败"})

    return results


# =================================================================
#  发票识别整理 - 单文件入口
# =================================================================
def recognize_single_invoice(invoice_path: str, api_key: str = "") -> Dict[str, Any]:
    ext = os.path.splitext(invoice_path)[1].lower()
    fname = os.path.basename(invoice_path)
    client = None
    tmp = tempfile.mkdtemp(prefix="inv_rec_")
    try:
        if ext == ".xml":
            txt = _read_xml_text(invoice_path)
            r = xml_parse_full(txt)
            if r:
                r["original_file"] = fname
                return r

        elif ext == ".ofd":
            if zipfile.is_zipfile(invoice_path):
                with zipfile.ZipFile(invoice_path, "r") as z:
                    all_xml = [n for n in z.namelist() if n.lower().endswith(".xml")]
                    def xml_priority(name):
                        nl = name.lower()
                        if any(x in nl for x in ("attach", "invoice", "einvoice")): return 0
                        if nl == "ofd.xml": return 1
                        if any(x in nl for x in ("page", "content", "sign", "seal")): return 9
                        return 5
                    all_xml.sort(key=xml_priority)
                    for name in all_xml:
                        try:
                            txt = z.read(name).decode("utf-8", errors="replace")
                            r = xml_parse_full(txt)
                            if r:
                                r["original_file"] = fname
                                return r
                        except Exception:
                            continue
            img = ofd_to_first_image(invoice_path, tmp)
            if img:
                r = _call_recognize_vlm(client, img, "image/jpeg")
                if r:
                    r = _fix_vlm_invoice_code(r)
                    r["source"] = "vlm"; r["original_file"] = fname
                    return r
            return {"original_file": fname, "error": "OFD 解析失败"}

        elif ext == ".pdf":
            # 判断页数：多页走逐页识别（返回特殊标记，由调用方展开）
            page_count = pdf_page_count(invoice_path)
            if page_count > 1:
                return {
                    "__multipage__": True,
                    "__path__": invoice_path,
                    "__fname__": fname,
                    "__client__": client,
                    "__tmp__": tmp,
                }
            # 单页：① 二维码扫描
            qr = scan_pdf_page_qr(invoice_path, 0)
            # ② regex 文本层
            r = None
            text = pdf_extract_text(invoice_path)
            if text:
                r = _pdf_parse_full_regex(text)
                if r:
                    r["original_file"] = fname
            # ③ VLM 兜底：regex 完全失败 或 关键字段缺失时调用
            if not r or _regex_result_needs_vlm(r):
                img = pdf_to_page_image(invoice_path, 0, tmp)
                if img:
                    vlm_r = _call_recognize_vlm(client, img, "image/png")
                    if vlm_r:
                        vlm_r = _fix_vlm_invoice_code(vlm_r)
                        if not r:
                            # regex 完全失败，直接用 VLM 结果
                            r = vlm_r
                            r["source"] = "vlm"
                            r["original_file"] = fname
                        else:
                            # regex 部分成功，用 VLM 补全空字段
                            r = _merge_vlm_into_regex(r, vlm_r)
            # ④ 二维码数据覆盖
            if r and qr:
                r = _merge_qr_into_record(r, qr)
            if r:
                return r
            return {"original_file": fname, "error": "PDF 解析失败"}

        elif ext in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
            mime_map = {'.jpg':'image/jpeg','.jpeg':'image/jpeg','.png':'image/png','.webp':'image/webp','.bmp':'image/bmp'}
            mime = mime_map.get(ext, 'image/jpeg')
            # ① 二维码扫描
            qr = scan_image_qr(invoice_path)
            # ② VLM 识别
            r = _call_recognize_vlm(client, invoice_path, mime)
            if r:
                r = _fix_vlm_invoice_code(r)
                r["source"] = "vlm"; r["original_file"] = fname
                # ③ 二维码数据覆盖
                if qr:
                    r = _merge_qr_into_record(r, qr)
                return r
            return {"original_file": fname, "error": "图片识别失败"}

        return {"original_file": fname, "error": f"不支持的格式: {ext}"}
    except Exception as e:
        return {"original_file": fname, "error": str(e)}
    finally:
        # 注意：多页 PDF 时 tmp 由调用方负责清理，此处跳过
        if not (ext == ".pdf" and pdf_page_count(invoice_path) > 1):
            shutil.rmtree(tmp, ignore_errors=True)


# =================================================================
#  验证码识别
# =================================================================
def filter_captcha_by_color(img_bytes: bytes, hint: str = "") -> bytes:
    """根据国税提示词（如“红色文字”、“黑色文字”），使用 HSV 颜色掩膜过滤验证码干扰色"""
    if not hint:
        return img_bytes

    try:
        import cv2
        import numpy as np

        target_color = None
        for c in ["红", "黄", "蓝", "黑"]:
            if c in hint:
                target_color = c
                break

        if not target_color:
            return img_bytes

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return img_bytes

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        color_ranges = {
            "红": [
                (np.array([0, 40, 40]), np.array([14, 255, 255])),
                (np.array([165, 40, 40]), np.array([180, 255, 255]))
            ],
            "黄": [
                (np.array([12, 45, 45]), np.array([40, 255, 255]))
            ],
            "蓝": [
                (np.array([80, 35, 35]), np.array([142, 255, 255]))
            ],
            "黑": [
                (np.array([0, 0, 0]), np.array([180, 255, 90]))
            ]
        }

        if target_color not in color_ranges:
            return img_bytes

        mask = None
        for (lower, upper) in color_ranges[target_color]:
            sub_mask = cv2.inRange(hsv, lower, upper)
            if mask is None:
                mask = sub_mask
            else:
                mask = cv2.bitwise_or(mask, sub_mask)

        # 检查是否命中足够的目标像素
        if np.count_nonzero(mask) < 20:
            return img_bytes

        # 生成纯白背景，目标文字置黑，以获得最高的 OCR 对比度
        res = np.full_like(img, 255)
        res[mask > 0] = [0, 0, 0]

        # 形态学滤波去杂点
        kernel = np.ones((2, 2), np.uint8)
        res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

        success, enc = cv2.imencode(".png", res)
        if success:
            return enc.tobytes()
    except Exception as e:
        print(f"[captcha-color-filter] {e}")

    return img_bytes

async def recognize_captcha(client: OpenAI, img_bytes: bytes, hint: str = "", debug_prefix: str = "") -> Optional[str]:
    # ── 自动持久化验证码至 cache/captcha_debug 供复盘分析 ──
    try:
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "captcha_debug")
        os.makedirs(debug_dir, exist_ok=True)
        t_stamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"{debug_prefix}_{t_stamp}" if debug_prefix else t_stamp
        with open(os.path.join(debug_dir, f"{prefix}_raw.png"), "wb") as f_raw:
            f_raw.write(img_bytes)
    except Exception as _e:
        print(f"[debug-cache] 写入原图失败: {_e}", flush=True)
    # 1. 优先调用高精度视觉大模型 (混元/智谱/OpenAI) + HSV 目标颜色物理图层
    client_vlm, vlm_model_name = get_vlm_client()
    if client_vlm:
        b64_raw = base64.b64encode(img_bytes).decode("utf-8")
        req_hint = hint or "请输入验证码文字"

        # 提取目标颜色物理图层（白底黑字）
        proc_bytes = filter_captcha_by_color(img_bytes, hint)
        try:
            debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "captcha_debug")
            t_stamp = time.strftime("%Y%m%d_%H%M%S")
            prefix = f"{debug_prefix}_{t_stamp}" if debug_prefix else t_stamp
            with open(os.path.join(debug_dir, f"{prefix}_layer.png"), "wb") as f_layer:
                f_layer.write(proc_bytes)
        except Exception as _e:
            pass
        b64_layer = base64.b64encode(proc_bytes).decode("utf-8")

        prompt_text = f"""这是国税发票查验平台的验证码。
题目要求：【{req_hint}】。
图一是原图，图二是物理算法滤除其他杂色后的目标颜色图层（背景为白色，黑色笔画为目标字符）。
请综合两张图，精准提取属于【{req_hint}】的字符。
注意事项：
1. 字符数量不固定（通常仅有 2~4 个字符），绝不要自行脑补凑数；
2. 邻近紧挨着的其他颜色文字已被过滤，请以过滤图层为准，绝不要认入其他颜色的文字；
3. 必须区分大小写与汉字/英数；
4. 严格只输出符合颜色要求的纯字符，严禁任何空格、标点或说明。"""

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_raw}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_layer}"}},
                {"type": "text", "text": prompt_text}
            ]
        }]
        try:
            print(f"[captcha-vlm] 正在请求 {vlm_model_name} 视觉大模型（双图比对）识别验证码...", flush=True)
            resp = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client_vlm.chat.completions.create(
                        model=vlm_model_name, messages=messages, temperature=0.01, max_tokens=100, timeout=20
                    )
                ),
                timeout=25
            )
            vlm_raw_text = resp.choices[0].message.content.strip()
            clean = re.sub(r'[`\s\n\r\*]+', '', vlm_raw_text)
            m = re.search(r'(?:答案是|字符是|结果为|文字是)[：:\s]*[“"「【（\*]*([\u4e00-\u9fa5A-Za-z0-9]{1,6})', vlm_raw_text)
            if m:
                vlm_text = m.group(1)
            else:
                m_quote = re.search(r'[“"「]([\u4e00-\u9fa5A-Za-z0-9]{1,6})[”"」]', vlm_raw_text)
                if m_quote:
                    vlm_text = m_quote.group(1)
                elif len(clean) <= 6:
                    vlm_text = clean
                else:
                    m_end = re.search(r'([\u4e00-\u9fa5A-Za-z0-9]{1,6})$', clean)
                    vlm_text = m_end.group(1) if m_end else clean[:4]

            if vlm_text:
                print(f"[captcha-vlm] {vlm_model_name} 识别成功: {vlm_text} (原始回复: {vlm_raw_text!r})", flush=True)
                return vlm_text
        except Exception as e:
            print(f"[captcha-vlm] {vlm_model_name} 调用异常: {e}", flush=True)

    # 2. 未配置或异常时，回退到本地 ddddocr
    try:
        import ddddocr
        ocr = ddddocr.DdddOcr(show_ad=False)
        proc_bytes = filter_captcha_by_color(img_bytes, hint)
        result = ocr.classification(proc_bytes)
        if result and len(result.strip()) >= 2:
            return result.strip()
        raw_res = ocr.classification(img_bytes)
        if raw_res and len(raw_res.strip()) >= 2:
            return raw_res.strip()
    except Exception as e:
        print(f"[ddddocr] {e}", flush=True)

    return None


async def handle_error_popup(page) -> Optional[str]:
    # 1. 优先捕获国税新版顶部的 .t-message 浮动错误条（如"验证码错误！"）
    try:
        msg = await page.evaluate("""() => {
            const msgs = Array.from(document.querySelectorAll('.t-message, .t-message__content, .t-is-error, .t-message--error'));
            for (const m of msgs) {
                const style = window.getComputedStyle(m);
                if (style && style.display !== 'none' && style.visibility !== 'hidden') {
                    const t = m.innerText.trim();
                    if (t.includes('验证码') || t.includes('错误') || t.includes('有误') || t.includes('不正确') || t.includes('失效')) {
                        return t;
                    }
                }
            }

            // 检查真正的业务报错弹窗（严格排除常驻的首次使用提示）
            const popups = Array.from(document.querySelectorAll('.popup-container, .t-dialog'));
            for (const p of popups) {
                const style = window.getComputedStyle(p);
                if (style && style.display !== 'none' && style.visibility !== 'hidden' && p.offsetWidth > 50) {
                    const msgEl = p.querySelector('.popup-message, .t-dialog__body');
                    const text = msgEl ? msgEl.innerText.trim() : (p.innerText ? p.innerText.trim() : '');
                    if (text && !text.includes('查验结果') && !text.includes('发票详情')) {
                        if (text.includes('首次使用') || text.includes('常见问题') || text.includes('操作说明') || text.includes('12366') || text === '提示 确定' || text === '提示') {
                            continue;
                        }
                        const btn = p.querySelector('button, .t-button');
                        if (btn) btn.click();
                        return text;
                    }
                }
            }
            return null;
        }""")
        if msg:
            await asyncio.sleep(0.5)
            return msg
    except Exception:
        pass

    # 2. 尝试旧版弹窗
    try:
        popup = await page.query_selector(ERROR_POPUP_SELECTOR)
        if popup and await popup.is_visible():
            msg_elem = await page.query_selector(ERROR_MESSAGE_SELECTOR)
            msg = await msg_elem.text_content() if msg_elem else "未知错误"
            ok_btn = await page.query_selector(ERROR_OK_BUTTON_SELECTOR)
            if ok_btn: await ok_btn.click()
            await asyncio.sleep(0.5)
            return msg
    except Exception:
        pass
    return None


async def _verify_one_invoice(client, page, inv_info: tuple, screenshot_path: str, tmp_dir: str, invoice_path: str = None):
    """
    在已打开的 page 上查验单张发票。
    - inv_info: (code, number, date, amount)，若为 None 则尝试文件上传
    - 返回 (success: bool, b64_screenshot: str or None, error_msg: str)
    使用 async generator yield log 事件，最终 return 结果
    这个函数设计为普通 async 函数，日志通过 list 收集后由调用方 yield
    """
    pass  # 占位，实际逻辑在 verify_invoice_stream 内联


async def verify_invoice_stream(api_key: str, invoice_path: str, screenshot_path: str):
    client = None
    ext = os.path.splitext(invoice_path)[1].lower()

    # ── 判断是否多页 PDF ──
    page_count = pdf_page_count(invoice_path) if ext == ".pdf" else 1
    is_multipage = page_count > 1

    inv_info = None

    # ── 单页：解析发票 ──
    if not is_multipage:
        yield make_step_event(step="parse")
        parsed_record = recognize_single_invoice(invoice_path, api_key)
        if parsed_record and not parsed_record.get("error"):
            code = str(parsed_record.get("invoice_code", "") or "").strip()
            number = str(parsed_record.get("invoice_number", "") or "").strip()
            date = str(parsed_record.get("issue_date", "") or "").strip()
            if date and len(date) == 10 and "-" in date:
                date = date.replace("-", "")

            raw_ex = parsed_record.get("amount_excluding_tax", 0) or parsed_record.get("amount_ex_tax", 0) or 0
            amount_ex_tax = float(raw_ex) if raw_ex else 0.0

            total = parsed_record.get("total_amount_including_tax", {}) or {}
            raw_tot = total.get("in_figures", 0) if isinstance(total, dict) else total or 0
            if not raw_tot or float(raw_tot) == 0:
                raw_tot = parsed_record.get("total_amount", 0) or 0
            total_amount = float(raw_tot) if raw_tot else (amount_ex_tax if amount_ex_tax else 0.0)

            raw_tax = parsed_record.get("tax_amount", 0) or 0
            tax_amount = float(raw_tax) if raw_tax else 0.0

            check_code = str(parsed_record.get("check_code", "") or "").strip()
            amount = total_amount if (not code and total_amount) else (amount_ex_tax if amount_ex_tax else total_amount)

            inv_info = (code, number, date, amount, check_code, amount_ex_tax, total_amount, tax_amount)

            clean_invoice_data = {
                "invoice_title": parsed_record.get("invoice_title", "") or "",
                "invoice_code": code,
                "invoice_number": number,
                "issue_date": parsed_record.get("issue_date", "") or "",
                "amount_ex_tax": amount_ex_tax,
                "tax_amount": tax_amount,
                "total_amount": total_amount,
                "check_code": check_code,
                "buyer_name": parsed_record.get("buyer_name", "") or "",
                "buyer_tax_id": parsed_record.get("buyer_tax_id", "") or "",
                "seller_name": parsed_record.get("seller_name", "") or "",
                "seller_tax_id": parsed_record.get("seller_tax_id", "") or "",
                "items": parsed_record.get("items", []) or [],
                "total_words": parsed_record.get("total_words", "") or "",
                "remarks": parsed_record.get("remarks", "") or "",
                "issuer": parsed_record.get("issuer", "") or ""
            }
            yield make_step_event(step="parse", status=True, data=clean_invoice_data)
        else:
            inv_info = extract_invoice_info(client, invoice_path)
            if not inv_info:
                yield make_step_event(step="parse", status=False)
            else:
                code, number, date, amount = inv_info[:4]
                clean_invoice_data = {
                    "invoice_title": "",
                    "invoice_code": code,
                    "invoice_number": number,
                    "issue_date": date,
                    "amount_ex_tax": float(inv_info[5]) if len(inv_info) > 5 else amount,
                    "tax_amount": float(inv_info[7]) if len(inv_info) > 7 else 0.0,
                    "total_amount": float(inv_info[6]) if len(inv_info) > 6 else amount,
                    "check_code": inv_info[4] if len(inv_info) > 4 else "",
                    "buyer_name": "",
                    "buyer_tax_id": "",
                    "seller_name": "",
                    "seller_tax_id": "",
                    "items": [],
                    "total_words": "",
                    "remarks": "",
                    "issuer": ""
                }
                yield make_step_event(step="parse", status=True, data=clean_invoice_data)
    else:
        # 多页 PDF 逐页解析
        page_infos = []
        tmp_pages = tempfile.mkdtemp(prefix="inv_vp_")
        for i in range(page_count):
            label = f"第{i+1}张/共{page_count}张"
            text = pdf_extract_page_text(invoice_path, i)
            info = None
            if text:
                result = pdf_parse_regex(text)
                if result and result[1] and result[2]:
                    info = result
            if not info:
                img = pdf_to_page_image(invoice_path, i, tmp_pages)
                if img:
                    data = call_aliyun_vlm(client, img)
                    if data:
                        info = _parse_vlm_result(data)
            page_infos.append((info, label))
        shutil.rmtree(tmp_pages, ignore_errors=True)

    # ── 打开浏览器（单例锁）──
    lock = get_verify_lock()
    async with lock:
        async with async_playwright() as p:
            exec_path = get_chromium_executable_path()
            browser_args = ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
            if CDP_URL:
                try:
                    browser = await p.chromium.connect_over_cdp(CDP_URL)
                except Exception as _cdp_err:
                    print(f"[playwright] 连接 CDP ({CDP_URL}) 失败: {_cdp_err}，改用本地启动")
                    browser = await p.chromium.launch(
                        executable_path=exec_path,
                        headless=HEADLESS,
                        args=browser_args
                    )
            else:
                browser = await p.chromium.launch(
                    executable_path=exec_path,
                    headless=HEADLESS,
                    args=browser_args
                )

            async def _do_one_verify(inv_info_item, scr_path):
                page = await browser.new_page(viewport={"width": 1280, "height": 900})
                invoice_code, invoice_number, invoice_date, invoice_amount = ("", "", "", 0.0)
                if inv_info_item:
                    invoice_code, invoice_number, invoice_date, invoice_amount = inv_info_item[:4]

                success = False
                b64_result = None

                try:
                    yield make_step_event(step="open_browser")
                    await page.goto("https://inv-veri.chinatax.gov.cn/index.html", wait_until="domcontentloaded", timeout=60000)
                    try:
                        await page.wait_for_selector("input.t-input__inner, #fpdm", timeout=20000)
                    except Exception:
                        pass
                    await asyncio.sleep(1)
                    yield make_step_event(step="open_browser", status=True)

                    yield make_step_event(step="fill_form")
                    if not inv_info_item:
                        yield make_step_event(step="fill_form", status=False)
                        await page.close()
                        yield ("VERIFY_SUMMARY", False, None)
                        return

                    clean_date = invoice_date.replace("-", "").strip()
                    raw_amount = float(invoice_amount) if invoice_amount else 0.0
                    abs_amount = abs(raw_amount)
                    amount_str = f"{abs_amount:.2f}"
                    check_code = inv_info_item[4] if len(inv_info_item) > 4 else ""
                    check_code_last6 = check_code[-6:] if len(check_code) >= 6 else check_code

                    inputs = await page.query_selector_all("input.t-input__inner")
                    is_tdesign = len(inputs) >= 4
                    if is_tdesign:
                        if invoice_code:
                            await inputs[0].fill(invoice_code)
                            await page.keyboard.press("Tab")
                        await inputs[1].fill(invoice_number)
                        await page.keyboard.press("Tab")
                        await inputs[2].fill(clean_date)
                        await page.keyboard.press("Tab")
                        await page.keyboard.press("Escape")
                        await page.click("body", position={"x": 5, "y": 5})
                        await asyncio.sleep(0.5)

                        label_4 = await page.evaluate('''() => {
                            const labels = Array.from(document.querySelectorAll('.t-form__label, label')).map(el => el.innerText.trim()).filter(t => t.length > 0);
                            const u = [];
                            for (const l of labels) { if (!u.length || u[u.length - 1] !== l) u.push(l); }
                            return u.length >= 4 ? u[3] : '';
                        }''')

                        inv_ex_tax = float(inv_info_item[5]) if len(inv_info_item) > 5 and inv_info_item[5] else raw_amount
                        inv_total_tax = float(inv_info_item[6]) if len(inv_info_item) > 6 and inv_info_item[6] else raw_amount

                        if any(k in label_4 for k in ['校验码', '后六位', '后6位']) and check_code_last6:
                            target_val_4 = check_code_last6
                        elif any(k in label_4 for k in ['价税合计', '含税', '票价']):
                            amt_val = abs(inv_total_tax) if inv_total_tax else abs_amount
                            target_val_4 = f"{amt_val:.2f}"
                        elif any(k in label_4 for k in ['不含税', '开具金额', '金额']):
                            amt_val = abs(inv_ex_tax) if inv_ex_tax else abs_amount
                            target_val_4 = f"{amt_val:.2f}"
                        else:
                            target_val_4 = amount_str

                        await inputs[3].fill(target_val_4)
                        await page.keyboard.press("Tab")
                        yield make_step_event(step="fill_form", status=True)
                    else:
                        try:
                            if invoice_code:
                                await page.fill(INVOICE_CODE_SELECTOR, invoice_code)
                            await page.fill(INVOICE_NUMBER_SELECTOR, invoice_number)
                            await page.fill(INVOICE_DATE_SELECTOR, clean_date)
                            await page.fill(INVOICE_AMOUNT_SELECTOR, amount_str)
                            yield make_step_event(step="fill_form", status=True)
                        except Exception as _fe:
                            yield make_step_event(step="fill_form", status=False)

                    # 验证码识别与重试
                    for attempt in range(1, MAX_CAPTCHA_RETRIES + 1):
                        yield make_step_event(step="captcha")
                        await page.evaluate("const c = document.querySelector('#cover'); if(c) c.remove()")

                        if is_tdesign:
                            t_inputs = await page.query_selector_all("input.t-input__inner")
                            if len(t_inputs) >= 5:
                                await t_inputs[4].click()
                            code_img_btn = await page.query_selector('img[src*="code."], img[src^="data:image"]')
                            if code_img_btn:
                                await code_img_btn.click()
                            await asyncio.sleep(2)
                        else:
                            try:
                                await page.click(CAPTCHA_IMG_SELECTOR)
                                await asyncio.sleep(2)
                            except Exception:
                                pass

                        hint = ""
                        try:
                            hint = await page.evaluate('''() => {
                                const el = document.querySelector('.yzm-tips, #yzminfo');
                                if (!el) return '';
                                return el.textContent || el.innerText || '';
                            }''')
                            hint = hint.strip()
                        except Exception:
                            pass

                        captcha_bytes = None
                        if is_tdesign:
                            c_img = await page.query_selector('img[src^="data:image"], img[src*="code."]')
                            if c_img:
                                captcha_bytes = await c_img.screenshot()
                        else:
                            c_img = await page.query_selector(CAPTCHA_IMG_SELECTOR)
                            if c_img:
                                captcha_bytes = await c_img.screenshot()

                        if not captcha_bytes:
                            await asyncio.sleep(3)
                            continue

                        dbg_pfx = f"{invoice_number[-6:] if invoice_number else 'manual'}_att{attempt}"
                        captcha = await recognize_captcha(client, captcha_bytes, hint, debug_prefix=dbg_pfx)
                        if not captcha:
                            await asyncio.sleep(3)
                            continue

                        if is_tdesign:
                            t_inputs = await page.query_selector_all("input.t-input__inner")
                            if len(t_inputs) >= 5:
                                await t_inputs[4].fill(captcha)
                        else:
                            await page.fill(CAPTCHA_INPUT_SELECTOR, captcha)

                        await page.click("body", position={"x": 10, "y": 10})
                        if is_tdesign:
                            await page.evaluate(r'''() => {
                                const btns = Array.from(document.querySelectorAll('button'));
                                const check = btns.find(b => b.innerText && b.innerText.replace(/\s+/g, '') === '查验');
                                if (check) check.click();
                            }''')
                        else:
                            await page.evaluate(f'document.querySelector("{CHECK_BUTTON_SELECTOR}")?.click()')

                        await page.wait_for_timeout(2500)

                        err_msg = await handle_error_popup(page)
                        if err_msg is not None:
                            await asyncio.sleep(3)
                            continue

                        res_info = await page.evaluate('''() => {
                            const body = document.body.innerText || '';
                            const hasTable = !!document.querySelector('table, .invoice-detail, .t-table');
                            const d = document.querySelector('dialog, .popup-container, #content');
                            const hasDialog = d && window.getComputedStyle(d).display !== 'none';
                            const isFinished = hasDialog || hasTable || body.includes('查验结果') || body.includes('发票详情') || body.includes('货物或应税劳务') || body.includes('最多查验') || body.includes('不一致') || body.includes('查无此票') || body.includes('超过');
                            if (!isFinished) return null;

                            let status = 'unknown';
                            if (body.includes('不一致')) {
                                status = 'inconsistent';
                            } else if (body.includes('查无此票')) {
                                status = 'not_found';
                            } else if (body.includes('最多查验') || (body.includes('超过') && body.includes('次数'))) {
                                status = 'over_limit';
                            } else if (hasTable || body.includes('发票详情') || body.includes('货物或应税劳务') || (body.includes('查验结果') && !body.includes('不一致') && !body.includes('查无此票'))) {
                                status = 'valid';
                            }
                            return { finished: true, verification_status: status };
                        }''')

                        if res_info:
                            yield make_step_event(step="captcha", status=True)
                            yield make_step_event(step="fetch_result")
                            await page.wait_for_timeout(1000)
                            await page.screenshot(path=scr_path, full_page=True)
                            with open(scr_path, "rb") as f:
                                b64_result = base64.b64encode(f.read()).decode("utf-8")
                            yield make_step_event(step="fetch_result", status=True)
                            v_status_result = res_info.get("verification_status", "unknown")
                            success = True
                            break
                        else:
                            await asyncio.sleep(3)
                            continue

                    if not success:
                        yield make_step_event(step="captcha", status=False)

                    await page.close()
                    yield ("VERIFY_SUMMARY", success, b64_result, v_status_result if success else "unknown")

                except Exception as e:
                    print(f"[_do_one_verify Error] {e}")
                    await page.close()
                    yield ("VERIFY_SUMMARY", False, None, "unknown")

            # 单页
            if not is_multipage:
                success = False
                b64 = None
                v_status = "unknown"
                async for chunk in _do_one_verify(inv_info, screenshot_path):
                    if isinstance(chunk, tuple) and chunk[0] == "VERIFY_SUMMARY":
                        _, success, b64, v_status = chunk
                    else:
                        yield chunk
                if success and b64:
                    yield make_step_event(
                        step="done",
                        status=True,
                        data={
                            "verification_status": v_status,
                            "screenshot": f"data:image/png;base64,{b64}"
                        }
                    )
                else:
                    yield make_step_event(
                        step="done",
                        status=False,
                        data={
                            "verification_status": "unknown",
                            "screenshot": None
                        }
                    )
            # 多页
            else:
                success_count = 0
                for i, (info, label) in enumerate(page_infos):
                    yield make_step_event(step="page_start", data={"page": i + 1, "total": page_count, "label": label})
                    scr_path = screenshot_path.replace(".png", f"_{i+1}.png")
                    page_success = False
                    page_b64 = None
                    page_v_status = "unknown"
                    async for chunk in _do_one_verify(info, scr_path):
                        if isinstance(chunk, tuple) and chunk[0] == "VERIFY_SUMMARY":
                            _, page_success, page_b64, page_v_status = chunk
                        else:
                            yield chunk
                    if page_success:
                        success_count += 1
                        yield make_step_event(step="page_done", status=True, data={
                            "page": i + 1,
                            "total": page_count,
                            "label": label,
                            "verification_status": page_v_status,
                            "screenshot": f"data:image/png;base64,{page_b64}"
                        })
                    else:
                        yield make_step_event(step="page_done", status=False, data={
                            "page": i + 1,
                            "total": page_count,
                            "label": label,
                            "verification_status": "unknown",
                            "screenshot": None
                        })

                yield make_step_event(step="done", status=(success_count > 0), data=None)

            await browser.close()


# =================================================================
#  Flask 路由
# =================================================================
@app.route('/')
def index():
    return send_from_directory('static', 'invoice_index.html')


@app.route('/api/verify', methods=['POST'])
def verify():
    ok, err_msg = is_ocr_configured()
    if not ok:
        return {"error": err_msg}, 500
    api_key = ""

    file = request.files.get('file')
    if not file:
        return {"error": "未上传文件"}, 400

    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".pdf"
    safe_filename = f"{uuid.uuid4().hex}{ext}"
    tmp_dir = tempfile.mkdtemp(prefix="inv_upload_")
    invoice_path    = os.path.join(tmp_dir, safe_filename)
    screenshot_path = os.path.join(tmp_dir, "result.png")
    file.save(invoice_path)

    def generate():
        import queue
        import threading
        q = queue.Queue()

        def worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def run_flow():
                    async for chunk in verify_invoice_stream(api_key, invoice_path, screenshot_path):
                        q.put(chunk)
                loop.run_until_complete(run_flow())
            except Exception as e:
                print(f"[SSE Error] {e}", flush=True)
                q.put(make_step_event(step="done", status=False, data=None))
            finally:
                q.put(None)
                loop.close()
                shutil.rmtree(tmp_dir, ignore_errors=True)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is None:
                break
            yield item

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/recognize', methods=['POST'])
def recognize():
    ok, err_msg = is_ocr_configured()
    if not ok:
        return jsonify({"error": err_msg}), 500
    api_key = ""

    files = request.files.getlist('files[]')
    if not files:
        return jsonify({"error": "未上传任何文件"}), 400

    tmp_dir = tempfile.mkdtemp(prefix="inv_rec_batch_")

    # 先把所有文件以安全文件名保存到临时目录（防路径穿越）
    saved = []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower() if f.filename else ".pdf"
        safe_filename = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(tmp_dir, safe_filename)
        f.save(save_path)
        original_name = os.path.basename(f.filename) if f.filename else safe_filename
        saved.append((save_path, original_name))

    def _process_one(args):
        save_path, filename = args
        sub_tmps = []
        try:
            record = recognize_single_invoice(save_path, api_key)
            if isinstance(record, dict) and record.get("__multipage__"):
                sub_tmp = tempfile.mkdtemp(prefix="inv_mp_")
                sub_tmps.append(sub_tmp)
                return recognize_pdf_multipage(
                    record["__path__"], record["__fname__"],
                    record["__client__"], sub_tmp,
                )
            else:
                return [record]
        except Exception as e:
            return [{"original_file": filename, "error": str(e)}]
        finally:
            for st in sub_tmps:
                shutil.rmtree(st, ignore_errors=True)

    # 并发处理，最多 8 个线程（VLM 是 IO 等待，线程数可以高一些）
    MAX_WORKERS = min(8, len(saved))
    results = []
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_process_one, args): args for args in saved}
            # 按完成顺序收集，保持原始文件顺序用 dict 暂存
            ordered = {}
            for future in as_completed(futures):
                idx = saved.index(futures[future])
                ordered[idx] = future.result()
            for i in range(len(saved)):
                results.extend(ordered.get(i, []))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return jsonify({"records": results, "total": len(results)})


@app.route('/api/recognize/export', methods=['POST'])
def recognize_export():
    try:
        import pandas as pd
        from io import BytesIO
        data = request.get_json()
        records = data.get("records", [])
        if not records:
            return jsonify({"error": "无数据可导出"}), 400

        rows = []
        for r in records:
            items = r.get("items", [])
            # VLM 有时返回对象数组而非字符串数组，做兜底处理
            safe_items = []
            for it in (items or []):
                if isinstance(it, str):
                    safe_items.append(it)
                elif isinstance(it, dict):
                    safe_items.append(it.get("name") or it.get("item") or str(it))
                else:
                    safe_items.append(str(it))
            items_str = "、".join(safe_items)
            rows.append({
                "原始文件":           r.get("original_file", ""),
                "发票抬头":           r.get("invoice_title", ""),
                "发票代码":           r.get("invoice_code", ""),
                "发票号码":           r.get("invoice_number", ""),
                "开票日期":           r.get("issue_date", ""),
                "购买方名称":         r.get("buyer_name", ""),
                "购买方税号":         r.get("buyer_tax_id", ""),
                "销售方名称":         r.get("seller_name", ""),
                "销售方税号":         r.get("seller_tax_id", ""),
                "商品详情":           items_str,
                "合计金额（不含税）": r.get("amount_ex_tax", "") if r.get("amount_ex_tax", "") != "" else "",
                "合计税额":           r.get("tax_amount", "") if r.get("tax_amount", "") != "" else "",
                "价税合计大写":       r.get("total_words", ""),
                "价税合计小写":       r.get("total_amount", ""),
                "备注":               r.get("remarks", ""),
                "开票人":             r.get("issuer", ""),
                "识别方式":           r.get("source", ""),
                "错误信息":           r.get("error", ""),
                "校验码":             r.get("verify_code", ""),
                "加密字段":           r.get("encrypt_code", ""),
            })

        df = pd.DataFrame(rows)
        buf = BytesIO()
        df.to_excel(buf, index=False, engine='openpyxl')
        buf.seek(0)

        from flask import send_file
        from datetime import datetime
        filename = f"发票识别结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return send_file(
            buf,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    ok, err_msg = is_ocr_configured()
    print("=" * 60)
    print("  发票系统后端服务")
    print(f"  当前识别引擎 (OCR_PROVIDER): {OCR_PROVIDER}")
    print(f"  引擎凭证状态: {'已就绪 ✓' if ok else f'未就绪 ✗ ({err_msg})'}")
    print("  功能一：发票查验  POST /api/verify")
    print("  功能二：发票识别  POST /api/recognize")
    print("  功能三：导出Excel POST /api/recognize/export")
    print("  访问地址：http://localhost:9896")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=9896, threaded=True)
