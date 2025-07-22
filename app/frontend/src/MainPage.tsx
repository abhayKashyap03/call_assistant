import { useState, useEffect } from 'react';
import axios from 'axios';


interface ActionButtonProps {
  onClick?: () => void;
  disabled: boolean;
  label: string;
  className?: string;
  type?: "button" | "submit";
}

interface AgentState {
  ngrok: 'inactive' | 'active' | 'error';
  twilio: 'unconfigured' | 'configured' | 'error';
  publicUrl: string;
}

export function ActionButton({ onClick, disabled, label, className = '', type = 'button' }: ActionButtonProps) {
    return (
      <button className={`action-button ${className}`} onClick={onClick} disabled={disabled} type={type}>
        {label}
      </button>
    );
}

export default function MainPage() {
    const [agentState, setAgentState] = useState<AgentState>({
        ngrok: 'inactive',
        twilio: 'unconfigured',
        publicUrl: ''
        });
    const [isLoading, setIsLoading] = useState(true);
    const [stage, setStage] = useState('')

    const API_BASE_URL = 'http://localhost:5000';

    const fetchStatus = async () => {
        setIsLoading(true);
        try {
            const response = await axios.get(`${API_BASE_URL}/ngrok/status`);
            setAgentState(response.data);
        } catch (error) {
            setAgentState({ ngrok: 'error', twilio: 'error', publicUrl: '' });
        } finally {
            console.log("Agent state fetched:", agentState);
            setIsLoading(false);
        }
    };

    const handleStart = async () => {
        setIsLoading(true);
        try {
            await axios.post(`${API_BASE_URL}/ngrok/start`);
            setStage('starting');
            setTimeout(fetchStatus, 2000);
            setAgentState({ ...agentState, ngrok: 'active', twilio: 'configured' });
        } catch (error) {
            console.error("Error starting ngrok:", error);
            setAgentState({ ...agentState, ngrok: 'error', twilio: 'unconfigured' });
            setIsLoading(false);
        }
    };
    
    const handleStop = async () => {
        setIsLoading(true);
        try {
            await axios.post(`${API_BASE_URL}/ngrok/stop`);
            setStage('stopping')
            setTimeout(fetchStatus, 2000);
            setAgentState({ ...agentState, ngrok: 'inactive', twilio: 'unconfigured' });
        } catch (error) {
            console.error("Error stopping ngrok:", error);
            setAgentState({ ...agentState, ngrok: 'error', twilio: 'configured' });
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchStatus();
    }, []);

    return (
        <>
            <div className="status-card">
              <h2>Ngrok Status</h2>
              <div className={`status-indicator status-${agentState.ngrok}`}>
                {agentState.ngrok.toUpperCase()}
              </div>
              {agentState.ngrok === 'active' && agentState.publicUrl && (
                <div className="status-details">
                  <p>Public URL (for Twilio):</p>
                  <a href={agentState.publicUrl} target="_blank" rel="noopener noreferrer">
                    {agentState.publicUrl}
                  </a>
                  {agentState.twilio === 'configured' ?
                    (<p>Twilio webhook updated!</p>) : (<p>Twilio webhook not updated; Configure API keys and tokens</p>)
                  }
                </div>
              )}
               {agentState.ngrok === 'error' && (
                <div className="status-details">
                  <p className="error-message">Could not connect to the backend server. Is it running?</p>
                </div>
              )}
            </div>

            <div className="actions-card">
              <h2>Actions</h2>
              <div className="button-group">
                <ActionButton
                  onClick={handleStart}
                  disabled={isLoading || agentState.ngrok === 'active'}
                  label={stage === 'starting' ? 'Started' : 'Start Ngrok'}
                  className="start-button"
                />
                <ActionButton
                  onClick={handleStop}
                  disabled={isLoading || agentState.ngrok !== 'active'}
                  label={stage === 'stopping' ? 'Stopped' : 'Stop Ngrok'}
                  className="stop-button"
                />
                <ActionButton
                  onClick={fetchStatus}
                  disabled={isLoading}
                  label="Refresh Status"
                />
              </div>
            </div>
        </>
    );
}