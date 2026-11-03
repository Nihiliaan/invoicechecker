"""
Train ticket regex parser.
"""

import re
from typing import Optional, Dict, Any


def parse_train_ticket_regex(text: str) -> Optional[Dict[str, Any]]:
    """从 PDF 文本中提取铁路电子客票字段"""
    if not text or len(text) < 20:
        return None

    m = re.search(r'发票号码[：:]\s*(\d{20})', text)
    num = m.group(1) if m else ""
    if not num:
        m = re.search(r'(?m)^(\d{20})$', text)
        if m:
            num = m.group(1)
    if not num:
        return None

    # 开票日期
    date = ""
    m = re.search(r'开票日期[：:]\s*(\d{4})年(\d{1,2})月(\d{1,2})日', text)
    if m:
        date = f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    if not date:
        m = re.search(r'开票日期[：:]\s*(\d{4})[-/](\d{2})[-/](\d{2})', text)
        if m:
            date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # 购买方名称
    buyer_name = ""
    m = re.search(r'购买方名称[：:]\s*(.+)', text)
    if m:
        val = m.group(1).strip()
        if val:
            buyer_name = val
        else:
            m2 = re.search(r'购买方名称[：:]\s*\n([\u4e00-\u9fa5A-Za-z0-9（）\(\)]{2,})', text)
            if m2:
                buyer_name = m2.group(1).strip()
    if not buyer_name:
        m = re.search(r'([\u4e00-\u9fa5]{2,}(?:有限|股份|集团|公司|医院|学校|研究)[^\n\s]{0,20}(?:公司|院|校|所)?)', text)
        if m:
            buyer_name = m.group(1).strip()

    # 购买方税号
    buyer_tax = ""
    m = re.search(r'统一社会信用代码[：:]\s*([A-Z0-9]{15,20})', text)
    if m:
        buyer_tax = m.group(1).strip()

    # 站名提取
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

    # 车次
    train_no = ""
    m = re.search(r'\b([GDCZTKY]\d{1,4}[A-Z]?)\b', text)
    if m:
        train_no = m.group(1)

    # 出行日期与时间
    travel_date = travel_time = ""
    for mm in re.finditer(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text):
        td = f"{mm.group(1)}-{mm.group(2).zfill(2)}-{mm.group(3).zfill(2)}"
        if td != date:
            travel_date = td
            break

    m = re.search(r'(\d{2}:\d{2})开?', text)
    if m:
        travel_time = m.group(1)

    carriage = seat = ""
    m = re.search(r'(\d{2})车(\d{2,3}[A-Z])号', text)
    if m:
        carriage = m.group(1) + "车"
        seat = m.group(2) + "号"

    seat_grade = ""
    m = re.search(r'([一二三]等座|商务座|软座|硬座|软卧|硬卧|无座)', text)
    if m:
        seat_grade = m.group(1)

    total_amount = 0.0
    m = re.search(r'票价[：:]?\s*[¥￥]\s*([0-9]+\.?[0-9]*)', text)
    if m:
        total_amount = float(m.group(1))
    if not total_amount:
        m = re.search(r'[¥￥]\s*([0-9]+\.[0-9]{2})', text)
        if m:
            total_amount = float(m.group(1))

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
