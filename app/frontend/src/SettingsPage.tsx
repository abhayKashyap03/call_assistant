import { useState, useEffect } from 'react';
import axios from 'axios';
import { ActionButton } from './MainPage';

const API_BASE_URL = '/api';

interface SettingsData {
    twilio_account_sid?: string;
    twilio_auth_token?: string;
    twilio_phone_sid?: string;
    ngrok_auth_token?: string;
    google_api_key?: string;
}

interface SettingsPageProps {
  onSaveSuccess: () => void;
}

export default function SettingsPage({onSaveSuccess}: SettingsPageProps) {
    const [settings, setSettings] = useState<SettingsData>({});
    const [isLoading, setIsLoading] = useState(true);
    const [message, setMessage] = useState('');
    const [showItem, setShowItem] = useState(true);

    useEffect(() => {
        fetchSettings();
    }, []);

    const fetchSettings = async () => {
        setIsLoading(true);
        setMessage('');
        try {
            const response = await axios.get(`${API_BASE_URL}/env`);
            setSettings(response.data);
        } catch (err) {
            console.error("Error fetching settings:", err);
            setMessage("Failed to load settings.");
        } finally {
            setIsLoading(false);
        }
    };

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setSettings(prev => ({ ...prev, [name]: value }));
    };

    const handleSave = async (e: React.FormEvent) => {
        try {
            e.preventDefault();
            await axios.post(`${API_BASE_URL}/env`, settings);
            setMessage("Settings saved successfully! Redirecting to main page...");
            setSettings({}); // Clear the form after saving
            setShowItem(false);
            setTimeout(() => {
                onSaveSuccess();
            }, 3000);
        } catch (err) {
            console.error("Error saving settings:", err);
            setMessage("Failed to save settings.");
            setIsLoading(false);
        }
    };

    if (isLoading && !message) return <div>Loading...</div>;

    return (
        <div className="settings-container">
            <h2>Settings</h2>
            {showItem ? 
            (<p>Configure API keys and auth tokens</p>) : null
            }
            <form onSubmit={handleSave}>
                {Object.entries(settings).map(([key, value]) => (
                    <div key={key} className="form-group">
                        <label>{key.replace(/_/, ' ').toUpperCase()}</label>
                        <input
                            type="password"
                            name={key}
                            value={value || ''}
                            onChange={handleInputChange}
                        />
                    </div>
                ))}
                {showItem ? (
                <ActionButton disabled={isLoading} label={isLoading === true ? 'Saving...' : 'Save Keys'} type='submit' />
                ) : null
                }

                {message && <p className={`message ${message.includes('Error') ? 'error' : 'success'}`}>{message}</p>}
            </form>
        </div>
    );
}