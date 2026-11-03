"""
Test suite for configuration, privacy protection and sandbox args.
"""

import unittest
import os
from unittest.mock import patch
import config


class TestConfig(unittest.TestCase):
    def test_chromium_args_default_has_sandbox(self):
        with patch.object(config, 'CHROMIUM_NO_SANDBOX', False):
            with patch('os.geteuid', create=True, return_value=1000):
                args = config.get_chromium_args()
                self.assertIn('--disable-dev-shm-usage', args)
                self.assertNotIn('--no-sandbox', args)

    def test_chromium_args_root_fallback_no_sandbox(self):
        with patch.object(config, 'CHROMIUM_NO_SANDBOX', False):
            with patch('os.geteuid', create=True, return_value=0):
                args = config.get_chromium_args()
                self.assertIn('--no-sandbox', args)
                self.assertIn('--disable-setuid-sandbox', args)

    def test_chromium_args_explicit_no_sandbox(self):
        with patch.object(config, 'CHROMIUM_NO_SANDBOX', True):
            args = config.get_chromium_args()
            self.assertIn('--no-sandbox', args)

    def test_offline_only_privacy_guard(self):
        with patch.object(config, 'OFFLINE_ONLY', True):
            ok, err = config.is_ocr_configured()
            self.assertTrue(ok)
            client, model = config.get_vlm_client()
            self.assertIsNone(client)
            self.assertEqual(model, "")

    def test_local_private_endpoint_configuration(self):
        with patch.object(config, 'OFFLINE_ONLY', False):
            with patch.object(config, 'OCR_PROVIDER', 'openai'):
                with patch.object(config, 'OPENAI_BASE_URL', 'http://127.0.0.1:11434/v1'):
                    with patch.object(config, 'OPENAI_API_KEY', ''):
                        ok, err = config.is_ocr_configured()
                        self.assertTrue(ok)


if __name__ == '__main__':
    unittest.main()
