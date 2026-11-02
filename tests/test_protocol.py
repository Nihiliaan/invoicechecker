import json
import unittest
from invoice_app import make_step_event


class TestProtocol(unittest.TestCase):
    """Test suite for SSE protocol serialization and data schema integrity."""

    def _parse_sse(self, sse_payload: str) -> dict:
        self.assertTrue(sse_payload.startswith("data: "))
        self.assertTrue(sse_payload.endswith("\n\n"))
        json_str = sse_payload[len("data: "):-2]
        return json.loads(json_str)

    def test_running_step_event(self):
        raw = make_step_event(step="browser")
        data = self._parse_sse(raw)
        self.assertEqual(data["step"], "browser")
        self.assertNotIn("status", data)
        self.assertNotIn("data", data)
        # Ensure deprecated legacy fields do not leak
        for legacy_key in ("type", "message", "success", "screenshot_url"):
            self.assertNotIn(legacy_key, data)

    def test_success_step_event_with_data(self):
        invoice_payload = {
            "invoice_code": "011002000111",
            "invoice_number": "12345678",
            "amount": 100.00
        }
        raw = make_step_event(step="parse", status=True, data=invoice_payload)
        data = self._parse_sse(raw)
        self.assertEqual(data["step"], "parse")
        self.assertIs(data["status"], True)
        self.assertEqual(data["data"]["invoice_code"], "011002000111")
        self.assertEqual(data["data"]["amount"], 100.00)

    def test_fail_step_event_with_data(self):
        err_payload = {"message": "Invalid file format"}
        raw = make_step_event(step="upload", status=False, data=err_payload)
        data = self._parse_sse(raw)
        self.assertEqual(data["step"], "upload")
        self.assertIs(data["status"], False)
        self.assertEqual(data["data"]["message"], "Invalid file format")

    def test_done_verification_status_encapsulation(self):
        # Verification outcome must be encapsulated in data.verification_status
        done_payload = {
            "verification_status": "valid",
            "screenshot": "static/screenshots/mock.png"
        }
        raw = make_step_event(step="done", status=True, data=done_payload)
        data = self._parse_sse(raw)
        self.assertEqual(data["step"], "done")
        self.assertIs(data["status"], True)
        self.assertIn("verification_status", data["data"])
        self.assertEqual(data["data"]["verification_status"], "valid")
        self.assertIn("screenshot", data["data"])


if __name__ == "__main__":
    unittest.main()
