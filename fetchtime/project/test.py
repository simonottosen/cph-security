"""
Unit-tests for app.py

Run with:
    python -m unittest discover fetchtime/project
or simply:
    python fetchtime/project/test.py
"""
import os
import sys
from pathlib import Path
import unittest
import io
from unittest.mock import patch, MagicMock

# Ensure the project directory (where app.py lives) is on sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from app import healthcheck_perform, database_write


class AppTests(unittest.TestCase):
    # ---------- healthcheck_perform() ----------
    @patch("requests.get")
    def test_healthcheck_perform_skip(self, mock_get):
        """If HEALTHCHECK is 'NULL' the function should skip the request."""
        result = healthcheck_perform("NULL")
        self.assertEqual(result, "Skipping. Healthcheck not configured")
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_healthcheck_perform_success(self, mock_get):
        """Should hit the correct URL and return success message."""
        mock_response = MagicMock(status_code=200)
        mock_get.return_value = mock_response
        hc_id = "abc-123"
        expected_url = f"https://hc-ping.com/{hc_id}"

        result = healthcheck_perform(hc_id)

        mock_get.assert_called_once_with(expected_url, timeout=10)
        self.assertEqual(result, "Healthcheck submitted")

    # ---------- database_write() ----------
    @patch("psycopg2.connect")
    def test_database_write_success(self, mock_connect):
        """
        Happy-path: cursor executes and commits without raising.
        Environment variables are patched just for the scope of the test.
        """
        # Mock connection / cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Fake env
        os.environ.update(
            {
                "POSTGRES_PASSWORD": "pwd",
                "POSTGRES_DB": "db",
                "POSTGRES_USER": "user",
                "POSTGRES_PORT": "5432",
                "POSTGRES_HOST": "localhost",
            }
        )

        result = database_write(7, "2025-01-01T00:00:00", "CPH")

        mock_cursor.execute.assert_called_once()  # query executed
        mock_conn.commit.assert_called_once()     # commit called
        mock_conn.close.assert_called_once()      # connection closed
        self.assertEqual(result, "Database write complete")

    # ---------- database_write() -- failure path ----------
    @patch("psycopg2.connect")
    def test_database_write_failure(self, mock_connect):
        """Should propagate an error string if the DB connection fails."""
        mock_connect.side_effect = Exception("unable to connect")
        result = database_write(5, "2025-06-01T00:00:00", "CPH")
        self.assertTrue(result.startswith("An error occurred"))

    # ---------- process_airport_result() ----------
    @patch("app.database_write")
    @patch("app.firebase_write")
    @patch("app.supabase_write")
    @patch("app.healthcheck_perform")
    def test_process_airport_result_happy(
        self, mock_health, mock_supabase, mock_firebase, mock_db
    ):
        """End-to-end helper: should call all writers and print summary."""
        from app import process_airport_result  # local import avoids circulars

        # Arrange
        mock_db.return_value = "Database write complete"
        mock_firebase.return_value = "Firebase write completed"
        mock_supabase.return_value = "Supabase write completed"
        mock_health.return_value = "Skipping. Healthcheck not configured"

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", new=captured):
            process_airport_result(5, "CPH", "NULL", "2025-06-02T12:00:00")

        # Assert each component was invoked correctly
        mock_db.assert_called_once_with(5, "2025-06-02T12:00:00", "CPH")
        mock_firebase.assert_called_once_with("CPH")
        mock_supabase.assert_called_once_with(5, "2025-06-02T12:00:00", "CPH")
        mock_health.assert_called_once_with("NULL")

        # Check printed summary contains key bits
        summary = captured.getvalue()
        self.assertIn("Airport CPH was completed", summary)
        self.assertIn("Queue is 5", summary)
        
if __name__ == "__main__":
    unittest.main()