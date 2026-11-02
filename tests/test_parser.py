import unittest
from invoice_app import _norm_date, _clean_amount, parse_invoice_qr, pdf_parse_regex


class TestParser(unittest.TestCase):
    """Test suite for date normalization, amount cleaning, QR code and regex parsing."""

    def test_norm_date(self):
        self.assertEqual(_norm_date("2026-11-02"), "20261102")
        self.assertEqual(_norm_date("2026/11/02"), "20261102")
        self.assertEqual(_norm_date("2026年11月02日"), "20261102")
        self.assertEqual(_norm_date("2026年1月5日"), "20260105")
        self.assertEqual(_norm_date("20261102"), "20261102")

    def test_clean_amount(self):
        self.assertEqual(_clean_amount("100.50"), 100.50)
        self.assertEqual(_clean_amount("￥1,234.56"), 1234.56)
        self.assertEqual(_clean_amount("¥99.00"), 99.00)
        self.assertEqual(_clean_amount("-￥500.00"), -500.00)
        self.assertEqual(_clean_amount("invalid"), 0.0)

    def test_parse_invoice_qr_standard(self):
        raw = "01,10,011002000111,12345678,100.00,20261102,123456,ABCDEF"
        res = parse_invoice_qr(raw)
        self.assertIsNotNone(res)
        self.assertEqual(res["invoice_type_code"], "10")
        self.assertEqual(res["invoice_code"], "011002000111")
        self.assertEqual(res["invoice_number"], "12345678")
        self.assertEqual(res["total_amount"], 100.00)
        self.assertEqual(res["issue_date"], "2026-11-02")
        self.assertEqual(res["verify_code"], "123456")

    def test_parse_invoice_qr_fully_digital(self):
        # Digital invoices have an empty invoice code field and 20-digit invoice number
        raw = "01,31,,24112000000000000001,888.88,20261102,,XYZ"
        res = parse_invoice_qr(raw)
        self.assertIsNotNone(res)
        self.assertEqual(res["invoice_type_code"], "31")
        self.assertEqual(res["invoice_code"], "")
        self.assertEqual(res["invoice_number"], "24112000000000000001")
        self.assertEqual(res["total_amount"], 888.88)
        self.assertEqual(res["issue_date"], "2026-11-02")

    def test_parse_invoice_qr_invalid(self):
        self.assertIsNone(parse_invoice_qr("invalid,data"))
        self.assertIsNone(parse_invoice_qr("02,10,123,456"))

    def test_pdf_parse_regex_fully_digital_positive(self):
        text = (
            "电子发票（普通发票）\n"
            "24112000000000000001\n"
            "开票日期：2026年11月02日\n"
            "项目名称 金额 税率 税额\n"
            "服务费  ¥100.00 6% ¥6.00\n"
            "合 计   ¥100.00    ¥6.00\n"
            "价税合计（大写） 壹佰零陆元整 （小写）¥106.00\n"
        )
        res = pdf_parse_regex(text)
        self.assertIsNotNone(res)
        code, num, date, amount = res
        self.assertEqual(code, "")
        self.assertEqual(num, "24112000000000000001")
        self.assertEqual(date, "20261102")
        self.assertEqual(amount, 106.00)

    def test_pdf_parse_regex_fully_digital_negative_red(self):
        # Negative / Red invoice verification
        text = (
            "电子发票（普通发票）\n"
            "24112000000000000002\n"
            "开票日期：2026年11月02日\n"
            "项目名称 金额 税率 税额\n"
            "冲销服务费  -¥100.00 6% -¥6.00\n"
            "合 计       -¥100.00    -¥6.00\n"
            "价税合计（大写） 负壹佰零陆元整 （小写）-¥106.00\n"
        )
        res = pdf_parse_regex(text)
        self.assertIsNotNone(res)
        code, num, date, amount = res
        self.assertEqual(code, "")
        self.assertEqual(num, "24112000000000000002")
        self.assertEqual(date, "20261102")
        self.assertEqual(amount, -106.00)


if __name__ == "__main__":
    unittest.main()
