"""
Captcha recognition service using HSV color layer separation and VLM / ddddocr.
"""

import os
import re
import time
import base64
import asyncio
from typing import Optional
from openai import OpenAI
from config import BASE_DIR, get_vlm_client


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

        if np.count_nonzero(mask) < 20:
            return img_bytes

        res = np.full_like(img, 255)
        res[mask > 0] = [0, 0, 0]

        kernel = np.ones((2, 2), np.uint8)
        res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

        success, enc = cv2.imencode(".png", res)
        if success:
            return enc.tobytes()
    except Exception as e:
        print(f"[captcha-color-filter] {e}")

    return img_bytes


async def recognize_captcha(client: Optional[OpenAI], img_bytes: bytes, hint: str = "", debug_prefix: str = "") -> Optional[str]:
    """识别验证码：优先使用视觉大模型（双图比对）进行高精度识别，回退本地 ddddocr"""
    try:
        debug_dir = os.path.join(BASE_DIR, "cache", "captcha_debug")
        os.makedirs(debug_dir, exist_ok=True)
        t_stamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"{debug_prefix}_{t_stamp}" if debug_prefix else t_stamp
        with open(os.path.join(debug_dir, f"{prefix}_raw.png"), "wb") as f_raw:
            f_raw.write(img_bytes)
    except Exception as _e:
        print(f"[debug-cache] 写入原图失败: {_e}", flush=True)

    client_vlm, vlm_model_name = get_vlm_client()
    if client_vlm:
        b64_raw = base64.b64encode(img_bytes).decode("utf-8")
        req_hint = hint or "请输入验证码文字"
        proc_bytes = filter_captcha_by_color(img_bytes, hint)
        try:
            debug_dir = os.path.join(BASE_DIR, "cache", "captcha_debug")
            t_stamp = time.strftime("%Y%m%d_%H%M%S")
            prefix = f"{debug_prefix}_{t_stamp}" if debug_prefix else t_stamp
            with open(os.path.join(debug_dir, f"{prefix}_layer.png"), "wb") as f_layer:
                f_layer.write(proc_bytes)
        except Exception:
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

    try:
        import ddddocr
        ocr = ddddocr.DdddOcr(show_ad=False)
        proc_bytes = filter_captcha_by_color(img_bytes, hint)
        result = ocr.classification(proc_bytes)
        print(f"[captcha-ddddocr] 识别成功: {result}", flush=True)
        return result
    except Exception as e:
        print(f"[captcha-ddddocr] 调用失败: {e}", flush=True)
        return None
