"""
Unified invoice parser dispatcher supporting PDF, XML, OFD, Images and QR.
"""

import os
import re
import shutil
import tempfile
import zipfile
import base64
from typing import Optional, Dict, Any, Tuple, List
from openai import OpenAI

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from parsers.regex_parser import (
    pdf_extract_text,
    pdf_extract_page_text,
    pdf_page_count,
    pdf_parse_regex,
    _pdf_parse_full_regex,
    _regex_result_needs_vlm,
    _merge_vlm_into_regex
)
from parsers.qr_parser import scan_pdf_page_qr, scan_image_qr
from parsers.xml_parser import xml_parse_regex, xml_parse_full, _read_xml_text
from parsers.vlm_parser import (
    call_unified_invoice_ocr,
    call_aliyun_vlm,
    encode_image
)

_easyofd_available = True


def pdf_to_first_image(pdf_path: str, out_dir: str) -> Optional[str]:
    """将 PDF 第一页渲染为图片"""
    try:
        doc = fitz.open(pdf_path)
        if not doc:
            return None
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


def ofd_to_first_image(ofd_path: str, out_dir: str) -> Optional[str]:
    """将 OFD 发票渲染为图片"""
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


def _call_recognize_vlm(client: Any = None, image_path: str = "", mime: str = "image/png") -> Optional[Dict[str, Any]]:
    return call_unified_invoice_ocr(image_path, mime)


def _fix_vlm_invoice_code(r: dict) -> dict:
    code   = str(r.get("invoice_code",   "") or "").strip()
    number = str(r.get("invoice_number", "") or "").strip()

    if code and code == number:
        code = ""
    if code and not number:
        number = code
        code = ""
    if re.match(r'^\d{20}$', number):
        code = ""
    if code and not re.match(r'^\d{10,12}$', code):
        code = ""

    r["invoice_code"]   = code
    r["invoice_number"] = number
    return r


def _merge_qr_into_record(record: dict, qr: dict) -> dict:
    if qr.get("invoice_number"):
        record["invoice_number"] = qr["invoice_number"]
    if qr.get("invoice_code") is not None:
        record["invoice_code"] = qr["invoice_code"]
    if qr.get("issue_date"):
        record["issue_date"] = qr["issue_date"]
    if qr.get("total_amount"):
        record["total_amount"] = qr["total_amount"]
    if qr.get("invoice_title") and not record.get("invoice_title"):
        record["invoice_title"] = qr["invoice_title"]

    for key in ("verify_code", "encrypt_code", "qr_raw"):
        if qr.get(key) and not record.get(key):
            record[key] = qr[key]

    src = record.get("source", "")
    if "+qr" not in src:
        record["source"] = src + "+qr" if src else "qr"
    return record


def recognize_pdf_multipage(invoice_path: str, fname: str, client: OpenAI, tmp: str) -> List[Dict[str, Any]]:
    results = []
    page_count = pdf_page_count(invoice_path)

    for i in range(page_count):
        page_label = f"{fname} [第{i+1}页/共{page_count}页]"
        print(f"[multipage-pdf] 识别第 {i+1}/{page_count} 页: {fname}")

        qr = scan_pdf_page_qr(invoice_path, i)

        text = pdf_extract_page_text(invoice_path, i)
        r = None
        if text and len(text) >= 20:
            r = _pdf_parse_full_regex(text)
            if r:
                r["original_file"] = page_label

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

        if r and qr:
            r = _merge_qr_into_record(r, qr)

        if r:
            results.append(r)
        else:
            results.append({"original_file": page_label, "error": "识别失败"})

    return results


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
                        if any(x in nl for x in ("attach", "invoice", "einvoice")):
                            return 0
                        if nl == "ofd.xml":
                            return 1
                        if any(x in nl for x in ("page", "content", "sign", "seal")):
                            return 9
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
                    r["source"] = "vlm"
                    r["original_file"] = fname
                    return r
            return {"original_file": fname, "error": "OFD 解析失败"}

        elif ext == ".pdf":
            page_count = pdf_page_count(invoice_path)
            if page_count > 1:
                return {
                    "__multipage__": True,
                    "__path__": invoice_path,
                    "__fname__": fname,
                    "__client__": client,
                    "__tmp__": tmp,
                }
            qr = scan_pdf_page_qr(invoice_path, 0)
            r = None
            text = pdf_extract_text(invoice_path)
            if text:
                r = _pdf_parse_full_regex(text)
                if r:
                    r["original_file"] = fname
            if not r or _regex_result_needs_vlm(r):
                img = pdf_to_page_image(invoice_path, 0, tmp)
                if img:
                    vlm_r = _call_recognize_vlm(client, img, "image/png")
                    if vlm_r:
                        vlm_r = _fix_vlm_invoice_code(vlm_r)
                        if not r:
                            r = vlm_r
                            r["source"] = "vlm"
                            r["original_file"] = fname
                        else:
                            r = _merge_vlm_into_regex(r, vlm_r)
            if r and qr:
                r = _merge_qr_into_record(r, qr)
            if r:
                return r
            return {"original_file": fname, "error": "PDF 解析失败"}

        elif ext in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
            mime_map = {'.jpg':'image/jpeg','.jpeg':'image/jpeg','.png':'image/png','.webp':'image/webp','.bmp':'image/bmp'}
            mime = mime_map.get(ext, 'image/jpeg')
            qr = scan_image_qr(invoice_path)
            r = _call_recognize_vlm(client, invoice_path, mime)
            if r:
                r = _fix_vlm_invoice_code(r)
                r["source"] = "vlm"
                r["original_file"] = fname
                if qr:
                    r = _merge_qr_into_record(r, qr)
                return r
            return {"original_file": fname, "error": "图片识别失败"}

        return {"original_file": fname, "error": f"不支持的格式: {ext}"}
    except Exception as e:
        return {"original_file": fname, "error": str(e)}
    finally:
        if not (ext == ".pdf" and pdf_page_count(invoice_path) > 1):
            shutil.rmtree(tmp, ignore_errors=True)


def _parse_vlm_result(data: dict) -> Optional[Tuple[str, str, str, float, str, float, float, float]]:
    code   = str(data.get("invoice_code", "") or "").strip()
    number = str(data.get("invoice_number", "") or "").strip()
    date   = str(data.get("issue_date", "") or "").strip()
    if date and len(date) == 10 and "-" in date:
        date = date.replace("-", "")
    if code and code == number:
        code = ""
    if code and not number:
        number = code
        code = ""

    raw_ex = data.get("amount_excluding_tax", 0) or data.get("amount_ex_tax", 0) or 0
    amount_ex_tax = float(raw_ex) if raw_ex else 0.0

    total = data.get("total_amount_including_tax", {}) or {}
    raw_tot = total.get("in_figures", 0) if isinstance(total, dict) else total or 0
    if not raw_tot or float(raw_tot) == 0:
        raw_tot = data.get("total_amount", 0) or 0
    total_amount = float(raw_tot) if raw_tot else (amount_ex_tax if amount_ex_tax else 0.0)

    raw_tax = data.get("tax_amount", 0) or 0
    tax_amount = float(raw_tax) if raw_tax else 0.0

    if code:
        amount = amount_ex_tax if amount_ex_tax else total_amount
    else:
        amount = total_amount if total_amount else amount_ex_tax

    check_code = str(data.get("check_code", "") or "").strip()
    return (code, number, date, amount, check_code, amount_ex_tax, total_amount, tax_amount)


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
            if not img:
                return None
            data = call_unified_invoice_ocr(img, "image/png")
            if not data:
                return None
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
                        if any(x in nl for x in ("attach", "invoice", "einvoice")):
                            return 0
                        if nl == "ofd.xml":
                            return 1
                        if any(x in nl for x in ("page", "content", "sign", "seal")):
                            return 9
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
