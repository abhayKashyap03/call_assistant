from pyngrok import ngrok
from twilio.rest import Client
from app.backend.configs import services


class NgrokManager:
    """Manager for ngrok tunnels."""
    def __init__(self):
        self.active_tunnel = None
        self.state = {
            "ngrok": "inactive",
            "twilio": "unconfigured",
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
                settings = services.get_settings_service().get_settings()
                account_sid = settings.twilio_account_sid
                auth_token = settings.twilio_auth_token
                phone_sid = settings.twilio_phone_sid

                # Initialize Twilio client
                client = Client(account_sid, auth_token)

                # Update the webhook URL
                client.incoming_phone_numbers(phone_sid).update(
                    voice_url=self.active_tunnel.public_url + '/voice'
                )
                
                print("Webhook URL updated successfully.")
                self.state['twilio'] = 'configured'
            except Exception as e:
                print(f"Failed to update Twilio webhook URL: {e}")
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
