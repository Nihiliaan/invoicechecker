"""
PDF text extraction and Regex parser for invoice elements.
"""

import re
from typing import Optional, Dict, Any, Tuple

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from parsers.train_parser import parse_train_ticket_regex


def _norm_date(raw: str) -> str:
    """标准化日期为 YYYYMMDD"""
    raw = str(raw).strip()
    m = re.match(r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})', raw)
    if m:
        return f"{m.group(1)}{m.group(2).zfill(2)}{m.group(3).zfill(2)}"
    if re.match(r'^\d{8}$', raw):
        return raw
    cleaned = re.sub(r'[年月日\-/\s]', '', raw)
    return cleaned[:8] if len(cleaned) >= 8 else cleaned


def _clean_amount(raw: str) -> float:
    """清洗金额字符串并转为浮点数"""
    raw = re.sub(r'[￥¥,，\s]', '', str(raw))
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 0.0


def pdf_extract_text(pdf_path: str) -> str:
    """提取 PDF 全文文本"""
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        for page in doc:
            full_text.append(page.get_text())
        doc.close()
        return "\n".join(full_text)
    except Exception as e:
        print(f"[pdf-text] extract error: {e}")
        return ""


def pdf_extract_page_text(pdf_path: str, page_index: int) -> str:
    """提取 PDF 指定页文本"""
    try:
        doc = fitz.open(pdf_path)
        if page_index >= len(doc):
            doc.close()
            return ""
        text = doc[page_index].get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"[pdf-text] page {page_index} extract error: {e}")
        return ""


def pdf_page_count(pdf_path: str) -> int:
    """获取 PDF 总页数"""
    try:
        doc = fitz.open(pdf_path)
        cnt = len(doc)
        doc.close()
        return cnt
    except Exception:
        return 1


def pdf_parse_regex(text: str) -> Optional[Tuple[str, str, str, float]]:
    """发票查验四要素提取：发票代码、号码、日期、金额（零token）"""
    if not text or len(text) < 20:
        return None

    num = ""
    m = re.search(r'(?m)^(\d{20})$', text)
    if m:
        num = m.group(1)
    if not num:
        m = re.search(r'(?m)^(\d{8})$', text)
        if m:
            num = m.group(1)
    if not num:
        m = re.search(r'发票号码[：:]\s*(\d{8,20})', text)
        if m:
            num = m.group(1)
    if not num:
        return None

    code = ""
    m = re.search(r'发票代码[：:]\s*(\d{10,12})', text)
    if m and m.group(1) != num:
        code = m.group(1)

    date = ""
    # 优先匹配"开票日期"前缀
    m = re.search(r'开票日期[：:]\s*(\d{4})年(\d{1,2})[\u6708\u2F49](\d{1,2})[\u65E5\u2F47]', text)
    if m:
        date = f"{m.group(1)}{m.group(2).zfill(2)}{m.group(3).zfill(2)}"
    if not date:
        m = re.search(r'开票日期[：:]\s*(\d{4})[\s/-](\d{1,2})[\s/-](\d{1,2})(?!\d)', text)
        if m:
            date = f"{m.group(1)}{m.group(2).zfill(2)}{m.group(3).zfill(2)}"
    if not date:
        m = re.search(r'(\d{4})年(\d{1,2})[\u6708\u2F49](\d{1,2})[\u65E5\u2F47]', text)
        if m:
            date = f"{m.group(1)}{m.group(2).zfill(2)}{m.group(3).zfill(2)}"
    if not date:
        m = re.search(r'(\d{4})[-/](\d{2})[-/](\d{2})', text)
        if m:
            date = f"{m.group(1)}{m.group(2)}{m.group(3)}"

    seen, amounts = set(), []
    def _add_amt(s1: str, s2: str):
        is_neg = ('-' in s1) or (s2.startswith('-'))
        clean_num = s2.lstrip('-')
        val_str = f'-{clean_num}' if is_neg else clean_num
        if val_str not in seen:
            seen.add(val_str)
            amounts.append(float(val_str))

    # 模式1：¥/￥/Y 紧跟数字（同行，支持负号在符号前或后，如 -¥100.00 或 ¥-100.00）
    for m in re.finditer(r'(-?)\s*[¥￥]\s*(-?[0-9]+\.[0-9]{2})', text):
        _add_amt(m.group(1), m.group(2))
    for m in re.finditer(r'(-?)\s*(?<![A-Z0-9])Y\s*(-?[0-9]+\.[0-9]{2})', text):
        _add_amt(m.group(1), m.group(2))
    # 模式2：数字后紧跟 ¥/￥（同行）
    for m in re.finditer(r'(-?[0-9]+\.[0-9]{2})\s*[¥￥]', text):
        _add_amt('', m.group(1))
    # 模式3：¥/￥/Y 换行后接数字
    for m in re.finditer(r'(-?)\s*[¥￥]\s*\n\s*(-?[0-9]+\.[0-9]{2})', text):
        _add_amt(m.group(1), m.group(2))
    # 模式5：数字换行后接 ¥/￥
    for m in re.finditer(r'(-?[0-9]+\.[0-9]{2})\s*\n\s*[¥￥]', text):
        _add_amt('', m.group(1))
    # 模式4：专门抓"价税合计"后的裸数字
    m_jshj = re.search(r'价税合计[^\n]{0,30}\n\s*(-?[0-9]+\.[0-9]{2})', text)
    if not m_jshj:
        m_jshj = re.search(r'价税合计[（(小写）\s¥￥Y\n\-]{0,30}(-?[0-9]+\.[0-9]{2})', text)
    if m_jshj:
        _add_amt('', m_jshj.group(1))
    # 模式6：裸数字行 + 下行是 (小写)¥ 或单独¥
    for m in re.finditer(r'^\s*(-?[0-9]+\.[0-9]{2})\s*\n\s*(?:\(小写\)\s*)?[¥￥]', text, re.MULTILINE):
        _add_amt('', m.group(1))

    amount = 0.0
    if code:
        if amounts:
            amount = amounts[0]
    else:
        # 价税合计最大绝对值
        if amounts:
            amount = max(amounts, key=abs)

    print(f'[pdf-text] code={code!r} num={num!r} date={date!r} amounts={amounts} amount={amount}')
    return (code, num, date, amount)


def _pdf_parse_full_regex(text: str) -> Optional[Dict[str, Any]]:
    """从 PDF 文本中提取全部发票字段（全要素）"""
    if not text or len(text) < 20:
        return None

    if "铁路电子客票" in text or "电子客票号" in text:
        return parse_train_ticket_regex(text)

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
    if not title:
        m_title = re.search(r'电[子子]发票[^\n]{0,10}(?:专用|普通)发票', text)
        if m_title:
            raw = m_title.group(0)
            title = "电子发票（专用发票）" if "专用" in raw else "电子发票（普通发票）"

    num = ""
    m = re.search(r'(?m)^(\d{20})$', text)
    if m:
        num = m.group(1)
    if not num:
        m = re.search(r'发票号码[：:]\s*(\d{20})', text)
        if m:
            num = m.group(1)
    if not num:
        m = re.search(r'(?m)^(\d{8})$', text)
        if m:
            num = m.group(1)
    if not num:
        return None

    code = ""
    m = re.search(r'发票代码[：:]\s*(\d{10,12})', text)
    if m and m.group(1) != num:
        code = m.group(1)

    date = ""
    m = re.search(r'(\d{4})年(\d{1,2})[\u6708\u2F49](\d{1,2})[\u65E5\u2F47]', text)
    if m:
        date = f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    if not date:
        m = re.search(r'开票日期[：:]\s*(\d{4})\s+(\d{2})\s+(\d{2})', text)
        if m:
            date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    buyer_name = seller_name = buyer_tax = seller_tax = ""

    _co_kw  = r'(?:有限|股份|集团|公司|医院|学校|研究院|研究所|门诊|诊所|药店|餐饮店|旅行社|工商户|宾馆|酒店|招待所|饭店|事业单位)'
    _co_pat = re.compile(r'[\u4e00-\u9fa5A-Za-z0-9（）\(\)]{2,50}' + _co_kw + r'[^\n（(]{0,15}(?:[）\)）])?')

    def _is_tax_val(s):
        return (bool(re.match(r'^[A-Z0-9]{15,20}$', s))
                and s != num
                and not re.match(r'^\d{20}$', s))

    def _valid_name(s):
        return (len(re.findall(r'[\u4e00-\u9fa5]', s)) >= 2
                and not re.match(r'^统一|^纳税|^项目|^规格|^名称$', s))

    def _extract_region(region):
        name = tax = ""
        _m2 = re.search(r'名称[：:]\s*([\u4e00-\u9fa5A-Za-z0-9（）\(\)]{2,50})', region)
        if _m2 and _valid_name(_m2.group(1).strip()):
            name = _m2.group(1).strip()
        if not name:
            _m2 = _co_pat.search(region)
            if _m2:
                name = _m2.group(0).strip()
        _m2 = re.search(r'统一社会信用代码[/纳税人识别号]*[：:]\s*([A-Z0-9]{10,20})', region)
        if _m2 and _is_tax_val(_m2.group(1)):
            tax = _m2.group(1)
        if not tax:
            for _m2 in re.finditer(r'(?m)^([A-Z0-9]{15,20})$', region):
                if _is_tax_val(_m2.group(1)):
                    tax = _m2.group(1)
                    break
        return name, tax

    _buyer_start = _buyer_end = _seller_start = _seller_end = -1
    for _pat in [r'购\n买\n方\n信\n息', r'购\n买\n方', r'购买方信息', r'购买方[：:]']:
        _m2 = re.search(_pat, text)
        if _m2:
            _buyer_start, _buyer_end = _m2.start(), _m2.end()
            break
    for _pat in [r'销\n售\n方\n信\n息', r'销\n售\n方', r'销售方信息', r'销售方[：:]']:
        _m2 = re.search(_pat, text)
        if _m2:
            _seller_start, _seller_end = _m2.start(), _m2.end()
            break

    if 0 <= _buyer_end <= _seller_start:
        _items_pos = text.find('项目名称')
        if _items_pos < 0:
            _items_pos = len(text)
        _br = text[_buyer_end:_seller_start]
        _sr = text[_seller_end:_items_pos]
        _bn, _bt = _extract_region(_br)
        _sn, _st = _extract_region(_sr)
        if _bn or _sn:
            buyer_name, buyer_tax, seller_name, seller_tax = _bn, _bt, _sn, _st

    if not (buyer_name or seller_name):
        _found_tax = {}
        for _m2 in re.finditer(r'统一社会信用代码[/纳税人识别号]*[：:]\s*([A-Z0-9]{10,20})', text):
            _tv = _m2.group(1)
            if _tv and _is_tax_val(_tv) and _tv not in _found_tax:
                _found_tax[_tv] = _m2.start()
        for _m2 in re.finditer(r'(?m)^([A-Z0-9]{15,20})$', text):
            _tv = _m2.group(1)
            if _is_tax_val(_tv) and _tv not in _found_tax:
                _found_tax[_tv] = _m2.start()
        _taxes = sorted(_found_tax.items(), key=lambda x: x[1])

        _found_co = {}
        for _m2 in re.finditer(r'名称[：:]\s*([\u4e00-\u9fa5A-Za-z0-9（）\(\)]{3,50})', text):
            _val = _m2.group(1).strip()
            if _valid_name(_val) and _val not in _found_co:
                _found_co[_val] = _m2.start()
        for _m2 in _co_pat.finditer(text):
            _val = _m2.group(0).strip()
            if _val not in _found_co:
                _found_co[_val] = _m2.start()
        _companies = sorted(_found_co.items(), key=lambda x: x[1])

        _bp_ref = _buyer_end if _buyer_end >= 0 else _buyer_start
        _sp_ref = _seller_end if _seller_end >= 0 else _seller_start

        if _taxes:
            if len(_taxes) >= 2:
                _t0, _p0 = _taxes[0]
                _t1, _p1 = _taxes[1]
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
                            if _i == _j:
                                continue
                            _sc = abs(_cp0-_bp)+abs(_cp1-_sp)
                            if _sc < _best:
                                _best, buyer_name, seller_name = _sc, _n0, _n1
                elif _companies:
                    _n, _cp = _companies[0]
                    if abs(_cp-_bp) <= abs(_cp-_sp):
                        buyer_name = _n
                    else:
                        seller_name = _n
            else:
                _t0, _p0 = _taxes[0]
                _use_b = _bp_ref >= 0 and (_sp_ref < 0 or abs(_p0-_bp_ref) <= abs(_p0-_sp_ref))
                if _use_b:
                    buyer_tax = _t0
                else:
                    seller_tax = _t0
                if _companies:
                    _n, _ = _companies[0]
                    if _use_b:
                        buyer_name = _n
                    else:
                        seller_name = _n

    items = []
    for _cat, _item in re.findall(r'\*([^*\n]+)\*([^\n¥￥*]{0,20})', text):
        _name = (_cat.strip() + ('-' + _item.strip() if _item.strip() else ''))
        if _name:
            items.append(_name)
    if not items:
        items = re.findall(r'(?m)^([^\d\s¥￥*#（）\n]{3,20})\s+\d', text)
        items = [x.strip() for x in items if x.strip()][:5]

    seen_amounts, amounts_list = set(), []
    for m2 in re.finditer(r'(-?)\s*[¥￥]\s*(-?[0-9]+\.[0-9]{2})', text):
        is_neg = ('-' in m2.group(1)) or (m2.group(2).startswith('-'))
        clean_num = m2.group(2).lstrip('-')
        val_str = f'-{clean_num}' if is_neg else clean_num
        if val_str not in seen_amounts:
            seen_amounts.add(val_str)
            amounts_list.append(float(val_str))

    for m2 in re.finditer(r'(-?[0-9]+\.[0-9]{2})\s*[¥￥]', text):
        v = m2.group(1)
        if v not in seen_amounts:
            seen_amounts.add(v)
            amounts_list.append(float(v))

    for m2 in re.finditer(r'(-?)\s*[¥￥]\s*\n\s*(-?[0-9]+\.[0-9]{2})', text):
        is_neg = ('-' in m2.group(1)) or (m2.group(2).startswith('-'))
        clean_num = m2.group(2).lstrip('-')
        val_str = f'-{clean_num}' if is_neg else clean_num
        if val_str not in seen_amounts:
            seen_amounts.add(val_str)
            amounts_list.append(float(val_str))

    for m2 in re.finditer(r'(-?[0-9]+\.[0-9]{2})\s*\n\s*[¥￥]', text):
        v = m2.group(1)
        if v not in seen_amounts:
            seen_amounts.add(v)
            amounts_list.append(float(v))

    m_jshj2 = re.search(r'价税合计[^\n]{0,30}\n\s*(-?[0-9]+\.[0-9]{2})', text)
    if not m_jshj2:
        m_jshj2 = re.search(r'价税合计[（(小写）\s¥￥Y\n\-]{0,30}(-?[0-9]+\.[0-9]{2})', text)
    if m_jshj2:
        v = m_jshj2.group(1)
        if v not in seen_amounts:
            seen_amounts.add(v)
            amounts_list.append(float(v))

    for m2 in re.finditer(r'^\s*(-?[0-9]+\.[0-9]{2})\s*\n\s*(?:\(小写\)\s*)?[¥￥]', text, re.MULTILINE):
        v = m2.group(1)
        if v not in seen_amounts:
            seen_amounts.add(v)
            amounts_list.append(float(v))

    amex = tax_amount = amtot = 0.0
    if code:
        if amounts_list:
            amex = amounts_list[0]
    else:
        if amounts_list:
            amtot = max(amounts_list, key=abs)
            rest = sorted([a for a in amounts_list if a != amtot], key=abs)
            if len(rest) >= 2:
                found = False
                for i in range(len(rest)):
                    for j in range(i+1, len(rest)):
                        if abs(rest[i] + rest[j] - amtot) < 0.02:
                            amex = rest[i]
                            tax_amount = rest[j]
                            found = True
                            break
                    if found:
                        break
                if not found:
                    amex = rest[-1]
                else:
                    if abs(amex) < abs(tax_amount):
                        amex, tax_amount = tax_amount, amex
            elif len(rest) == 1:
                amex = rest[0]

    total_words = ""
    m2 = re.search(r'(?:价税合计(?:大写)?)[（(（）\s]*([壹贰叁参肆伍陆柒捌玖拾佰仟万亿圆整零角分\u2EC6元]{2,})', text)
    if m2:
        total_words = m2.group(1).strip()
    if not total_words:
        m2 = re.search(r'([壹贰叁参肆伍陆柒捌玖拾佰仟万亿圆整零角分\u2EC6元]{2,})', text)
        if m2:
            total_words = m2.group(1).strip()

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
        _NOT_ISSUER = {'国家税务总局', '税务总局', '税务局', '北京市税务局',
                       '上海市税务局', '江苏省税务局', '浙江省税务局'}
        need_fallback = (re.match(r'^\d{15,}$', val)
                         or val in _TMPL_WORDS_ISSUER
                         or val in _NOT_ISSUER
                         or len(val) > 6)
        if need_fallback:
            all_names = re.findall(r'(?m)^([\u4e00-\u9fa5][\u4e00-\u9fa5 ]{0,3}[\u4e00-\u9fa5])$', text)
            all_names = [n.strip() for n in all_names
                         if n.strip() and n.strip() not in _NOT_ISSUER
                         and n.strip() not in _TMPL_WORDS_ISSUER
                         and not _NON_NAME_PAT.search(n.strip())]
            if all_names:
                issuer = all_names[-1]
        else:
            if val and not re.match(r'^[\u4e00-\u9fa5]$', val):
                issuer = val

    remarks = ""
    m2 = re.search(r'备注[：:]\s*(.+?)(?:\n|$)', text)
    if m2:
        remarks = m2.group(1).strip()

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
    """判断 regex 识别结果是否需要 VLM 补全"""
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
    """用 VLM 结果补全 regex 结果中的空字段"""
    fields_to_fill = ["buyer_name", "buyer_tax_id", "seller_name", "seller_tax_id",
                      "items", "invoice_title", "total_words", "remarks", "issuer"]
    for field in fields_to_fill:
        if not regex_r.get(field):
            vlm_val = vlm_r.get(field)
            if vlm_val:
                regex_r[field] = vlm_val
    return regex_r
