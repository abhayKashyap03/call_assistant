from pyngrok import ngrok


class NgrokManager:
    """Manager for ngrok tunnels."""
    def __init__(self):
        self.active_tunnel = None
    
    def start_ngrok_tunnel(self):
        """Start an ngrok tunnel and return the public URL."""
        if self.active_tunnel is not None:
            print("Ngrok tunnel is already running.")
            return self.active_tunnel.public_url
        
        try:
            # Start ngrok tunnel for port 5000 (Flask default)
            self.active_tunnel = ngrok.connect(5000)
            print(f"Ngrok tunnel started: {self.active_tunnel.public_url}")
            return self.active_tunnel.public_url
        except Exception as e:
            print(f"Failed to start ngrok tunnel: {e}")
            return None

    def stop_ngrok_tunnel(self):
        """Stop the ngrok tunnel if it is running."""
        if self.active_tunnel is not None:
            ngrok.disconnect(self.active_tunnel.public_url)
            print(f"Ngrok tunnel stopped: {self.active_tunnel.public_url}")
            self.active_tunnel = None
        else:
            print("No ngrok tunnel to stop.")

    def get_tunnel_status(self):
        """Get the status of the ngrok tunnel."""
        if self.active_tunnel is not None:
            return {
                'status': 'active',
                'public_url': self.active_tunnel.public_url
            }
        else:
            return {
                'status': 'inactive',
                'public_url': None
            }


if __name__ == "__main__":
    ngrok_manager = NgrokManager()
    ngrok_manager.start_ngrok_tunnel()