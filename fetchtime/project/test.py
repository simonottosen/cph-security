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

# Provide dummy Supabase env so app's module-level Supabase client can initialize
os.environ.setdefault("SUPABASE_URL", "https://unit-test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "service-role-unit-test")

from app import healthcheck_perform, database_write, dublin


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

    # ---------- dublin() ----------
    def _dublin_response(self, payload):
        """A requests-like stub returning ``payload`` from .json()."""
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        return response

    @patch("app.process_airport_result")
    @patch("requests.get")
    def test_dublin_reads_terminal_1(self, mock_get, mock_result):
        """The live shape: bare numeric strings for both terminals, T1 wins.

        T2 is deliberately the smaller number, so reading it -- or averaging the
        two -- would show up here rather than as a quiet shift in the series.
        """
        mock_get.return_value = self._dublin_response({"T1": "12", "T2": "3"})

        dublin()

        queue, airport, _ = mock_result.call_args.args
        self.assertEqual(queue, 12)
        self.assertEqual(airport, "DUB")

    @patch("app.process_airport_result")
    @patch("requests.get")
    def test_dublin_accepts_decorated_and_sentinel_values(self, mock_get, mock_result):
        """Comparator prefixes parse; a negative sentinel still clamps to zero."""
        for payload, expected in (
            ({"T1": "= 12"}, 12),
            ({"T1": "< 5"}, 5),
            ({"T1": "-1"}, 0),
        ):
            with self.subTest(payload=payload):
                mock_result.reset_mock()
                mock_get.return_value = self._dublin_response(payload)
                dublin()
                self.assertEqual(mock_result.call_args.args[0], expected)

    @patch("app.process_airport_result")
    @patch("requests.get")
    def test_dublin_refuses_to_invent_a_number(self, mock_get, mock_result):
        """No reading must raise, never write.

        Writing a guess here is worse than writing nothing: the forecaster has
        gap machinery for a missing row and no way at all to recognise a
        fabricated one.
        """
        for payload in ({"T1": "Closed"}, {"T1": "--"}, {"T1": ""}, {"T2": "3"}, {}):
            with self.subTest(payload=payload):
                mock_get.return_value = self._dublin_response(payload)
                with self.assertRaises(RuntimeError):
                    dublin()
                mock_result.assert_not_called()


if __name__ == "__main__":
    unittest.main()