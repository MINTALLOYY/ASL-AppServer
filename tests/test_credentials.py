import unittest
import os
import json
from unittest.mock import patch
from dotenv import load_dotenv
import google.auth
from google.auth import exceptions


class TestGoogleCredentials(unittest.TestCase):

    def test_credentials_env_var_set(self):
        load_dotenv()
        print(os.environ)
        username = os.environ.get("USER_NAME")
        print(f"\n[TEST] USERNAME: {username}")
        """Test if GOOGLE_APPLICATION_CREDENTIALS environment variable is set."""
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        print(f"\n[TEST] GOOGLE_APPLICATION_CREDENTIALS: {creds_path}")
        self.assertIsNotNone(creds_path, "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")

    def test_credentials_file_exists(self):
        """Test if the credentials file exists."""
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path:
            file_exists = os.path.exists(creds_path)
            print(f"[TEST] Credentials file exists: {file_exists}")
            print(f"[TEST] File path: {creds_path}")
            self.assertTrue(file_exists, f"Credentials file does not exist at: {creds_path}")
        else:
            self.skipTest("GOOGLE_APPLICATION_CREDENTIALS not set")

    def test_credentials_file_valid_json(self):
        """Test if the credentials file is valid JSON."""
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path or not os.path.exists(creds_path):
            self.skipTest("Credentials file not found")

        try:
            with open(creds_path, 'r') as f:
                creds_data = json.load(f)
            print(f"[TEST] Credentials file is valid JSON")
            print(f"[TEST] Keys in credentials: {list(creds_data.keys())}")
            
            # Check for required fields
            required_fields = ["type", "project_id", "private_key_id", "private_key", "client_email"]
            for field in required_fields:
                self.assertIn(field, creds_data, f"Missing required field: {field}")
                print(f"[TEST] ✓ Found field: {field}")
            
            print(f"[TEST] Project ID: {creds_data.get('project_id')}")
            print(f"[TEST] Client Email: {creds_data.get('client_email')}")
            
        except json.JSONDecodeError as e:
            self.fail(f"Credentials file is not valid JSON: {e}")

    def test_google_auth_default(self):
        """Test if google.auth.default() can load credentials."""
        try:
            credentials, project = google.auth.default()
            print(f"\n[TEST] Successfully loaded credentials")
            print(f"[TEST] Project: {project}")
            print(f"[TEST] Credentials type: {type(credentials).__name__}")
            print(f"[TEST] Service account email: {getattr(credentials, 'service_account_email', 'N/A')}")
            
            self.assertIsNotNone(credentials, "Credentials are None")
            self.assertIsNotNone(project, "Project is None")
            
        except exceptions.DefaultCredentialsError as e:
            self.fail(f"Failed to load default credentials: {e}")

    def test_credentials_have_required_scopes(self):
        """Test if credentials can be used for Speech API."""
        try:
            credentials, project = google.auth.default()
            
            # Check if credentials support the Speech API scope
            speech_scope = "https://www.googleapis.com/auth/cloud-platform"
            
            # For service account credentials, we can check scopes
            if hasattr(credentials, '_scopes'):
                print(f"[TEST] Credential scopes: {credentials._scopes}")
            
            print(f"[TEST] Credentials should work with scope: {speech_scope}")
            
        except Exception as e:
            self.fail(f"Error checking credentials scopes: {e}")

    def test_speech_client_initialization(self):
        """Test if SpeechClient can be initialized with current credentials."""
        try:
            from google.cloud import speech_v1 as speech
            
            print("\n[TEST] Attempting to initialize SpeechClient...")
            client = speech.SpeechClient()
            print("[TEST] ✓ SpeechClient initialized successfully")
            
            # Try to access client properties
            print(f"[TEST] Client type: {type(client).__name__}")
            
        except Exception as e:
            self.fail(f"Failed to initialize SpeechClient: {e}")


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
