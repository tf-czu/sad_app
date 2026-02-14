import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import os
import sys

# Add project root to path to allow importing rgbd_app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rgbd_app import capture_service

class TestMetadataGeneration(unittest.TestCase):
    """Tests metadata generation in the capture_service."""

    @patch('rgbd_app.capture_service.capture_io')
    @patch('rgbd_app.capture_service.datetime')
    def test_capture_one_metadata_generation(self, mock_datetime, mock_capture_io):
        """Verify the metadata dictionary created by the capture_one service."""
        
        # --- Setup Mocks ---
        mock_device = MagicMock()
        mock_device.profile = True # Simulate a streaming device
        mock_device.serial_number = "test12345"
        mock_device.get_frames.return_value = ("color", "depth", "colormap")
        mock_intrinsics = {'test': 'intrinsics'}
        mock_device.get_intrinsics.return_value = mock_intrinsics
        mock_device.get_current_exposure.return_value = 8500.0
        mock_device.get_current_gain.return_value = 16.0

        mock_now_utc = datetime(2023, 10, 27, 12, 30, 0)
        mock_datetime.utcnow.return_value = mock_now_utc

        mock_capture_io.create_capture_directory.return_value = "/tmp/capture"
        
        # --- Call Function ---
        capture_service.capture_one(mock_device)
        
        # --- Assertions ---
        # Ensure save_metadata was called once
        mock_capture_io.save_metadata.assert_called_once()
        
        # Extract the metadata dictionary passed to the mocked function
        args, _ = mock_capture_io.save_metadata.call_args
        _, _, metadata = args
        
        # Define the expected metadata structure and content
        expected_metadata = {
            'timestamp_utc': "2023-10-27T12:30:00Z",
            'serial_number': "test12345",
            'intrinsics': mock_intrinsics,
            'exposure': 8500.0,
            'gain': 16.0,
            'frame_index': 0,
        }
        
        self.assertDictEqual(metadata, expected_metadata)

if __name__ == '__main__':
    unittest.main()
