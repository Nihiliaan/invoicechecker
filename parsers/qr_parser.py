"""
QR Code parser for VAT and fully-digital invoices.
"""

import re
from typing import Optional, Dict, Any

try:
    import pymupdf as fitz
except ImportError:
    import fitz

try:
    from pyzbar.pyzbar import decode as qr_decode
    from PIL import Image as _PILImage
    _PYZBAR_AVAILABLE = True
except ImportError:
    _PYZBAR_AVAILABLE = False


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
    """从 PDF 指定页扫描二维码并解析发票信息。"""
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
