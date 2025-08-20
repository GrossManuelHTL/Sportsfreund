"""
Backend Communication System
Sends session data to the backend
"""
import json
import requests
from typing import Dict, Any, Optional
from datetime import datetime


class BackendClient:
    """Client for communication with the backend"""

    def __init__(self, base_url: str = "http://localhost:3000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def send_session_data(self, session_data: Dict[str, Any]) -> bool:
        """
        Sends session data to the backend

        Args:
            session_data: The session data to send

        Returns:
            bool: True if successfully sent
        """
        try:
            processed_data = self._serialize_timestamps(session_data)

            response = requests.post(
                f"{self.base_url}/api/sessions",
                headers=self.headers,
                json=processed_data,
                timeout=30
            )

            if response.status_code == 200 or response.status_code == 201:
                print(f"Session data successfully sent to backend")
                return True
            else:
                print(f"Backend error: {response.status_code} - {response.text}")
                return False

        except requests.exceptions.ConnectionError:
            print("Connection to backend failed")
            return False
        except requests.exceptions.Timeout:
            print("Backend request timeout")
            return False
        except Exception as e:
            print(f"Unexpected error during backend upload: {e}")
            return False

    def _serialize_timestamps(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Converts datetime objects to ISO strings for JSON"""
        if isinstance(data, dict):
            return {k: self._serialize_timestamps(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_timestamps(item) for item in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data

    def test_connection(self) -> bool:
        """Tests the connection to the backend"""
        try:
            response = requests.get(
                f"{self.base_url}/api/health",
                headers=self.headers,
                timeout=5
            )

            if response.status_code == 200:
                print("✅ Backend connection successful")
                return True
            else:
                print(f"❌ Backend health check failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Backend not reachable: {e}")
            return False

    def save_session_locally(self, session_data: Dict[str, Any], filename: Optional[str] = None) -> bool:
        """
        Saves session data locally as fallback

        Args:
            session_data: The data to save
            filename: Optional filename

        Returns:
            bool: True if successfully saved
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_id = session_data.get('session', {}).get('session_id', 'unknown')[:8]
                filename = f"session_{session_id}_{timestamp}.json"

            from pathlib import Path
            sessions_dir = Path("sessions")
            sessions_dir.mkdir(exist_ok=True)

            processed_data = self._serialize_timestamps(session_data)

            with open(sessions_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)

            print(f"Session saved locally: {filename}")
            return True

        except Exception as e:
            print(f"Error saving locally: {e}")
            return False
