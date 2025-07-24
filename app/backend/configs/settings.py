from pydantic import BaseModel
from typing import Optional
import json
from pathlib import Path


class Settings(BaseModel):
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_sid: Optional[str] = None
    ngrok_authtoken: Optional[str] = None
    google_api_key: Optional[str] = None


class SettingsService:
    def __init__(self, settings_file: str = "./settings.json"):
        self.settings_file_path = Path(settings_file)
        self.settings = self._load()

    def _load(self) -> Settings:
        if self.settings_file_path.exists():
            with open(self.settings_file_path, 'r') as f:
                data = json.loads(f.read())
                return Settings(**data)
        return Settings() # Return an empty settings object if file doesn't exist

    def save(self, settings_data):
        # Merge new data with existing data
        # This allows users to update only one key at a time if they want
        updated_data = self.settings.model_dump(exclude_unset=True)
        updated_data.update(**settings_data)
        
        self.settings = Settings(**updated_data)

        with open(self.settings_file_path, 'w') as f:
            f.write(self.settings.model_dump_json(indent=4))
    
    def get_settings(self) -> Settings:
        # Reload from file each time to ensure it's fresh
        return self._load()
