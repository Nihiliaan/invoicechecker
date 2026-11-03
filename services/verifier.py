"""
Playwright-based browser verification service for China National Tax Platform.
"""

import os
import re
import time
import base64
import asyncio
import tempfile
import shutil
from typing import Optional, AsyncGenerator, Any
from playwright.async_api import async_playwright

from config import (
    HEADLESS,
    CDP_URL,
    get_chromium_executable_path,
    get_chromium_args
)
from utils.sse import make_step_event
from parsers.regex_parser import (
    pdf_page_count,
    pdf_extract_page_text,
    pdf_parse_regex
)
from parsers.dispatcher import (
    pdf_to_page_image,
    recognize_single_invoice,
    extract_invoice_info,
    _parse_vlm_result
)
from parsers.vlm_parser import call_aliyun_vlm
from services.captcha import recognize_captcha

# 选择器与常量
INVOICE_CODE_SELECTOR   = "#fpdm"
INVOICE_NUMBER_SELECTOR = "#fphm"
INVOICE_DATE_SELECTOR   = "#kprq"
INVOICE_AMOUNT_SELECTOR = "#kjje"
CAPTCHA_IMG_SELECTOR    = "#yzm_img"
CAPTCHA_INPUT_SELECTOR  = "#yzm"
CHECK_BUTTON_SELECTOR   = "#checkfp"
ERROR_POPUP_SELECTOR    = "#popup_container"
ERROR_MESSAGE_SELECTOR  = "#popup_message"
ERROR_OK_BUTTON_SELECTOR= "#popup_ok"
MAX_CAPTCHA_RETRIES     = 10

_verify_lock: Optional[asyncio.Lock] = None


def get_verify_lock() -> asyncio.Lock:
    """在事件循环内懒加载初始化，避免在 loop 启动前创建 Lock"""
    global _verify_lock
    if _verify_lock is None:
        _verify_lock = asyncio.Lock()
    return _verify_lock


async def handle_error_popup(page) -> Optional[str]:
    """捕获国税平台错误浮动提示或弹窗"""
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

    try:
        popup = await page.query_selector(ERROR_POPUP_SELECTOR)
        if popup and await popup.is_visible():
            msg_elem = await page.query_selector(ERROR_MESSAGE_SELECTOR)
            msg = await msg_elem.text_content() if msg_elem else "未知错误"
            ok_btn = await page.query_selector(ERROR_OK_BUTTON_SELECTOR)
            if ok_btn:
                await ok_btn.click()
            await asyncio.sleep(0.5)
            return msg
    except Exception:
        pass
    return None


async def verify_invoice_stream(api_key: str, invoice_path: str, screenshot_path: str) -> AsyncGenerator[str, None]:
    """核心发票查验 SSE 流式生成器"""
    client = None
    ext = os.path.splitext(invoice_path)[1].lower()

    page_count = pdf_page_count(invoice_path) if ext == ".pdf" else 1
    is_multipage = page_count > 1

    inv_info = None

    # 1. 解析发票四要素
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

    # 2. 启动浏览器与税务查验自动化
    lock = get_verify_lock()
    async with lock:
        async with async_playwright() as p:
            exec_path = get_chromium_executable_path()
            browser_args = get_chromium_args()

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
                        yield ("VERIFY_SUMMARY", False, None, "unknown")
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
                        except Exception:
                            yield make_step_event(step="fill_form", status=False)

                    # 验证码提交与识别重试
                    v_status_result = "unknown"
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
