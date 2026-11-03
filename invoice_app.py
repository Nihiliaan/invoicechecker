"""
InvoiceChecker Backward Compatibility Layer.
Re-exports modular components from app, config, parsers, and services.
"""

from app import app
from config import (
    OFFLINE_ONLY,
    OCR_PROVIDER,
    get_vlm_client,
    get_chromium_executable_path,
    get_chromium_args,
    is_ocr_configured
)
from utils.sse import make_step_event, make_event, log_event
from utils.excel import export_invoices_to_excel
from parsers.regex_parser import (
    _norm_date,
    _clean_amount,
    pdf_extract_text,
    pdf_extract_page_text,
    pdf_page_count,
    pdf_parse_regex,
    _pdf_parse_full_regex
)
from parsers.qr_parser import (
    parse_invoice_qr,
    scan_pdf_page_qr,
    scan_image_qr
)
from parsers.train_parser import parse_train_ticket_regex
from parsers.xml_parser import (
    xml_parse_regex,
    xml_parse_full,
    _read_xml_text
)
from parsers.vlm_parser import (
    encode_image,
    call_tencent_vat_invoice,
    call_unified_invoice_ocr,
    call_aliyun_vlm
)
from parsers.dispatcher import (
    pdf_to_first_image,
    pdf_to_page_image,
    ofd_to_first_image,
    extract_invoice_info,
    recognize_single_invoice,
    recognize_pdf_multipage
)
from services.captcha import (
    filter_captcha_by_color,
    recognize_captcha
)
from services.verifier import (
    get_verify_lock,
    handle_error_popup,
    verify_invoice_stream
)

if __name__ == '__main__':
    print("=" * 60)
    print("  InvoiceChecker 服务启动中 (兼容入口)...")
    print("  功能一：发票查验  POST /api/verify")
    print("  功能二：发票识别  POST /api/recognize")
    print("  功能三：导出Excel POST /api/recognize/export")
    print("  访问地址：http://localhost:9896")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=9896, threaded=True)
