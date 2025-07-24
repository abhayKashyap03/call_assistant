import { useRef, useState } from 'react';
import axios from 'axios';
import { ActionButton } from './MainPage';

const API_BASE_URL = 'http://localhost:5000';

export default function DocIn() {
  const [files, setFiles] = useState<File[]>([]);
  const [url, setUrl] = useState('');
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);


  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  }

  function handleUrlChange(e: React.ChangeEvent<HTMLInputElement>) {
    setUrl(e.target.value);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus('Submitting...');
    setLoading(true);
    try {
      const formData = new FormData();
      files.forEach(file => formData.append('file', file));
      if (url) formData.append('url', url);
      const res = await axios.post(`${API_BASE_URL}/doc_in`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setStatus(res.data.message || 'Success!');
      setFiles([]);
      setUrl('');
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err: any) {
      setStatus(err.response?.data?.message || 'Error uploading document');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="docin-container">
      <h2>Add Documents or URLs</h2>
      <form onSubmit={handleSubmit} className="docin-form">
        <div className="form-group">
          <label>Upload document(s): </label>
          <input type="file" multiple onChange={handleFileChange} placeholder="Select files" ref={fileInputRef} />
        </div>
        <div className="form-group">
          <label>Or enter a web URL: </label>
          <input type="url" value={url} onChange={handleUrlChange} placeholder="https://example.com" />
        </div>
        <ActionButton type="submit" disabled={loading} label={loading ? 'Uploading...' : 'Submit'}>
        </ActionButton>
      </form>
      {status && <div className={`docin-status ${status.includes('error') ? 'error' : 'success'}`}>{status}</div>}
    </div>
  );
}
