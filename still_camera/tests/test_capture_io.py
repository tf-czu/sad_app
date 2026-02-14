import unittest
import os
import shutil
import tempfile
import json
from datetime import datetime
from unittest.mock import patch

import numpy as np

# Add project root to path to allow importing rgbd_app
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rgbd_app import capture_io

class TestCaptureIO(unittest.TestCase):
    """Tests for the capture_io module."""

    def setUp(self):
        """Create a temporary directory for test outputs."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the directory after the test."""
        shutil.rmtree(self.test_dir)

    @patch('rgbd_app.capture_io.datetime')
    def test_create_capture_directory(self, mock_datetime):
        """Test the creation of a uniquely named capture directory."""
        # Mock datetime.now() to return a fixed timestamp for a predictable name
        mock_now = datetime(2023, 10, 27, 12, 30, 0)
        mock_datetime.now.return_value = mock_now
        
        serial = "test12345"
        expected_dir_name = "20231027_123000_test12345"
        
        # Call the function
        capture_path = capture_io.create_capture_directory(base_dir=self.test_dir, serial_number=serial)
        
        # Check if the directory was created with the correct name
        expected_path = os.path.join(self.test_dir, expected_dir_name)
        self.assertEqual(capture_path, expected_path)
        self.assertTrue(os.path.isdir(capture_path))

    @patch('rgbd_app.capture_io.cv2.imwrite')
    def test_save_image_functions(self, mock_imwrite):
        """Test that image saving functions call cv2.imwrite with correct paths."""
        # Create dummy image data
        dummy_image_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        dummy_depth_16bit = np.zeros((10, 10), dtype=np.uint16)
        frame_index = 5
        
        # --- Test save_rgb ---
        capture_io.save_rgb(self.test_dir, frame_index, dummy_image_bgr)
        expected_rgb_path = os.path.join(self.test_dir, "frame_0005_rgb.png")
        mock_imwrite.assert_any_call(expected_rgb_path, dummy_image_bgr)
        
        # --- Test save_depth_16bit ---
        capture_io.save_depth_16bit(self.test_dir, frame_index, dummy_depth_16bit)
        expected_depth_path = os.path.join(self.test_dir, "frame_0005_depth.png")
        mock_imwrite.assert_any_call(expected_depth_path, dummy_depth_16bit)
        
        # --- Test save_depth_preview ---
        capture_io.save_depth_preview(self.test_dir, frame_index, dummy_image_bgr)
        expected_preview_path = os.path.join(self.test_dir, "frame_0005_depth_preview.png")
        mock_imwrite.assert_any_call(expected_preview_path, dummy_image_bgr)

    @patch('builtins.open')
    @patch('rgbd_app.capture_io.json.dump')
    def test_save_metadata(self, mock_json_dump, mock_open):
        """Test that save_metadata opens the correct file and saves correct JSON."""
        frame_index = 10
        metadata = {'key': 'value', 'test': True}
        
        # Call the function
        capture_io.save_metadata(self.test_dir, frame_index, metadata)
        
        # Check that open was called with the correct file path and mode
        expected_meta_path = os.path.join(self.test_dir, "frame_0010_meta.json")
        mock_open.assert_called_once_with(expected_meta_path, 'w')
        
        # Check that json.dump was called with the correct data
        file_handle = mock_open.return_value.__enter__.return_value
        mock_json_dump.assert_called_once_with(metadata, file_handle, indent=2)

if __name__ == '__main__':
    unittest.main()
