"""
XML and OFD invoice parser.
"""

import re
from typing import Optional, Dict, Any, Tuple
from parsers.regex_parser import _norm_date, _clean_amount

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


def xml_parse_regex(txt: str) -> Optional[Tuple[str, str, str, float]]:
    """提取 XML 中的四要素信息（发票代码、号码、日期、金额）"""
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
    """提取 XML 发票全要素字段"""
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
    """读取 XML 文本并尝试多编码容错"""
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")
