"""
InvoiceChecker Flask Web Application Entrypoint.
Provides RESTful & SSE APIs for invoice verification, extraction, and reporting.
"""

import os
import uuid
import asyncio
from datetime import datetime
from flask import Flask, request, Response, send_from_directory, jsonify, send_file

from config import is_ocr_configured, BASE_DIR
from utils.sse import make_step_event
from utils.excel import export_invoices_to_excel
from parsers.dispatcher import recognize_single_invoice, recognize_pdf_multipage
from services.verifier import verify_invoice_stream

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30MB 上传限制


@app.route('/')
def index():
    """托管 Web 前端静态界面"""
    return send_from_directory('static', 'invoice_index.html')


@app.route('/api/verify', methods=['POST'])
def verify():
    """发票在线查验接口 (SSE 流式输出)"""
    ok, err_msg = is_ocr_configured()
    if not ok:
        return jsonify({"error": err_msg}), 500

    file = request.files.get('file')
    if not file:
        return jsonify({"error": "未上传文件"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    allowed_exts = {'.pdf', '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.ofd', '.xml'}
    if ext not in allowed_exts:
        return jsonify({"error": f"不支持的文件类型: {ext}"}), 400

    # 安全随机文件名，杜绝路径穿越
    safe_name = f"{uuid.uuid4().hex}{ext}"
    upload_dir = os.path.join(BASE_DIR, 'cache', 'uploads')
    screenshot_dir = os.path.join(BASE_DIR, 'cache', 'screenshots')
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(screenshot_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, safe_name)
    screenshot_path = os.path.join(screenshot_dir, f"result_{uuid.uuid4().hex}.png")
    file.save(file_path)

    def generate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            stream = verify_invoice_stream("", file_path, screenshot_path)
            while True:
                try:
                    event = loop.run_until_complete(stream.__anext__())
                    yield event
                except StopAsyncIteration:
                    break
        except Exception as e:
            print(f"[SSE error] {e}", flush=True)
            yield make_step_event(step="error", status=False, data={"message": str(e)})
        finally:
            loop.close()
            # 查验完成后清理临时上传文件
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """批量发票识别整理接口"""
    files = request.files.getlist('files[]')
    if not files:
        file = request.files.get('file')
        files = [file] if file else []

    if not files:
        return jsonify({"error": "未上传发票文件"}), 400

    temp_dir = os.path.join(BASE_DIR, 'cache', 'temp_rec')
    os.makedirs(temp_dir, exist_ok=True)

    results = []
    saved_paths = []
    try:
        for f in files:
            orig_name = f.filename
            ext = os.path.splitext(orig_name)[1].lower()
            safe_name = f"{uuid.uuid4().hex}{ext}"
            temp_path = os.path.join(temp_dir, safe_name)
            f.save(temp_path)
            saved_paths.append(temp_path)

            res = recognize_single_invoice(temp_path, "")
            if isinstance(res, dict) and res.get("__multipage__"):
                multi_results = recognize_pdf_multipage(
                    invoice_path=res["__path__"],
                    fname=orig_name,
                    client=res["__client__"],
                    tmp=res["__tmp__"]
                )
                results.extend(multi_results)
            else:
                if isinstance(res, dict):
                    res["original_file"] = orig_name
                results.append(res)
    finally:
        for p in saved_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    return jsonify({"results": results})


@app.route('/api/recognize/export', methods=['POST'])
def recognize_export():
    """将批量识别记录导出为 Excel 文件"""
    try:
        data = request.get_json() or {}
        records = data.get("records", [])
        if not records:
            return jsonify({"error": "无数据可导出"}), 400

        buf = export_invoices_to_excel(records)
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
    print("=" * 60)
    print("  InvoiceChecker 服务启动中...")
    print("  功能一：发票查验  POST /api/verify")
    print("  功能二：发票识别  POST /api/recognize")
    print("  功能三：导出Excel POST /api/recognize/export")
    print("  访问地址：http://localhost:9896")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=9896, threaded=True)
