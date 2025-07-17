from pyngrok import ngrok
from app.backend.settings import SettingsService
from app.backend.convo import update_webhook_url


class NgrokManager:
    """Manager for ngrok tunnels."""
    def __init__(self):
        self.active_tunnel = None
        self.settings_service = SettingsService()
        self.state = {
            "ngrok": "",
            "twilio": "",
            "publicUrl": ""
        }
    
    def start_ngrok_tunnel(self):
        """Start an ngrok tunnel and return the public URL."""
        if self.active_tunnel is not None:
            print("Ngrok tunnel is already running.")
            return self.active_tunnel.public_url
        
        try:
            # Start ngrok tunnel for port 5000 (Flask default)
            self.active_tunnel = ngrok.connect(5000)
            print(f"Ngrok tunnel started: {self.active_tunnel.public_url}")
            self.state['ngrok'] = 'active'
            self.state['publicUrl'] = self.active_tunnel.public_url
            try:
                update_webhook_url(f"{self.active_tunnel.public_url}/voice")
                self.state['twilio'] = 'configured'
            except Exception:
                self.state['twilio'] = 'error'
            return self.active_tunnel.public_url
        except Exception as e:
            self.state['ngrok'] = 'error'
            print(f"Failed to start ngrok tunnel: {e}")
            return None

    def stop_ngrok_tunnel(self):
        """Stop the ngrok tunnel if it is running."""
        if self.active_tunnel is not None:
            ngrok.disconnect(self.active_tunnel.public_url)
            print(f"Ngrok tunnel stopped: {self.active_tunnel.public_url}")
            self.state['ngrok'] = 'inactive'
            self.state['twilio'] = 'unconfigured'
            self.state['publicUrl'] = ''
            self.active_tunnel = None
        else:
            print("No ngrok tunnel to stop.")

    def get_tunnel_status(self):
        """Get the status of the ngrok tunnel."""        
        return self.state


if __name__ == "__main__":
    ngrok_manager = NgrokManager()
    ngrok_manager.start_ngrok_tunnel()