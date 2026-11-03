"""
Vision Language Model (VLM) & Cloud OCR adapter.
Supports local private endpoints (e.g. Ollama, vLLM on custom ports) to prevent data leakage.
"""

import os
import re
import json
import base64
from typing import Optional, Dict, Any, Tuple
from openai import OpenAI

from config import (
    OFFLINE_ONLY,
    OCR_PROVIDER,
    TENCENTCLOUD_SECRET_ID,
    TENCENTCLOUD_SECRET_KEY,
    TENCENTCLOUD_REGION,
    get_vlm_client
)


def encode_image(path: str) -> str:
    """读取图片文件并进行 Base64 编码"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def _call_vlm_prompt(client: OpenAI, model: str, image_path: str, mime: str = "image/png") -> Optional[Dict[str, Any]]:
    """向视觉大模型发送发票提取 Prompt，支持云端或本地私有模型端口"""
    if OFFLINE_ONLY:
        print("[vlm-privacy] OFFLINE_ONLY=true 严格离线保护已启用，拦截外部多模态调用")
        return None

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
        clean_json = re.sub(r'^```json\s*', '', raw_content, flags=re.IGNORECASE)
        clean_json = re.sub(r'\s*```$', '', clean_json)
        return json.loads(clean_json)
    except Exception as e:
        print(f"[recognize-vlm] error: {e}")
        return None


def call_tencent_vat_invoice(image_path: str) -> Optional[Dict[str, Any]]:
    """调用腾讯云 VatInvoiceOCR 接口识别发票"""
    if OFFLINE_ONLY:
        return None
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


def call_unified_invoice_ocr(image_path: str, mime: str = "image/png") -> Optional[Dict[str, Any]]:
    """统一发票识别入口：支持私有本地端口或云端服务"""
    if OFFLINE_ONLY:
        print("[vlm-privacy] OFFLINE_ONLY=true 拦截图片多模态识别")
        return None

    # 如果配置为 openai 兼容端点（包含用户自建的本地私有端口）
    if OCR_PROVIDER == "openai":
        client, model = get_vlm_client()
        if client:
            return _call_vlm_prompt(client, model, image_path, mime)
        return None

    if OCR_PROVIDER in ("zhipu", "glm"):
        client, model = get_vlm_client()
        if client:
            return _call_vlm_prompt(client, model, image_path, mime)
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


def call_aliyun_vlm(client: Any = None, image_path: str = "") -> Optional[Dict[str, Any]]:
    """兼容旧接口"""
    c, m = get_vlm_client()
    if c:
        return _call_vlm_prompt(c, m, image_path)
    return None
