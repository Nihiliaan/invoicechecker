"""
Excel export utility for invoice records.
"""

from io import BytesIO
from typing import List, Dict, Any
import pandas as pd


def export_invoices_to_excel(records: List[Dict[str, Any]]) -> BytesIO:
    """将结构化发票记录转换为 Excel 字节流"""
    rows = []
    for r in records:
        items = r.get("items", [])
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
            "发票号码":           str(r.get("invoice_number", "")),
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
    return buf
