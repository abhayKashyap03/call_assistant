import React, { useState, useEffect } from 'react';
import axios from 'axios';

function StartButton() {
  const [isLoading, setIsLoading] = useState(false);

  const handleClick = async () => {
    console.log("Starting ngrok server...");
    setIsLoading(true);

    try {
      const response = await axios.post('http://127.0.0.1:5000/ngrok/start');
      console.log("Ngrok server started:", response.data);
    } catch (error) {
      console.error("Error starting ngrok server:", error);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <button className="my-button" onClick={handleClick}>
      Start ngrok server
    </button>
  );
}

function StopButton() {

  const [isLoading, setIsLoading] = useState(false);

  const handleClick = async () => {
    console.log("Stopping ngrok server...");
    setIsLoading(true);

    try {
      const response = await axios.post('http://127.0.0.1:5000/ngrok/stop');
      console.log("Ngrok server stopped:", response.data);
    } catch (error) {
      console.error("Error stopping ngrok server:", error);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <button className="my-button" onClick={handleClick}>
      Stop ngrok server
    </button>
  );
}

function StatusButton() {

  const [isLoading, setIsLoading] = useState(false);

  const handleClick = async () => {
    console.log("Getting ngrok server status...");
    setIsLoading(true);

    try {
      const response = await axios.get('http://127.0.0.1:5000/ngrok/status');
      console.log("Ngrok server status:", response.data);
    } catch (error) {
      console.error("Error getting ngrok server status:", error);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <button className="my-button" onClick={handleClick}>
      Get ngrok server status
    </button>
  );
}

export default function App() {
  return (
    <div>
      <h1>My App</h1>
      <StartButton />
      <StopButton />
      <StatusButton />
    </div>
  );
}
