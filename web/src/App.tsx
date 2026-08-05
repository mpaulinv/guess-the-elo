import { useCallback, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

interface PredictionResult {
  white_elo: number
  black_elo: number
  white_bucket_range: [number, number]
  black_bucket_range: [number, number]
  white_confidence: number
  black_confidence: number
  plies_used: number
  clock_coverage: number
  warning: string | null
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFile = (f: File) => {
    setFile(f)
    setResult(null)
    setError(null)
  }

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) handleFile(f)
  }, [])

  const onSubmit = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData })
      const data = await res.json()
      if (!res.ok) {
        const detail = Array.isArray(data.detail)
          ? 'Please choose a PGN file.'
          : (data.detail ?? 'Could not read this game. Check the file and try again.')
        throw new Error(detail)
      }
      setResult(data as PredictionResult)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong. Is the API running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <header className="header">
        <h1>Guess the Elo</h1>
        <p className="subtitle">
          Upload a chess game (PGN) and get an Elo rating estimate for both players —
          powered by an 8-expert neural network trained on 3.95M Lichess games.
        </p>
      </header>

      <main className="card">
        <div
          className={`dropzone ${isDragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={onDrop}
          onClick={() => document.getElementById('file-input')?.click()}
          role="button"
          tabIndex={0}
        >
          <input
            id="file-input"
            type="file"
            accept=".pgn"
            style={{ display: 'none' }}
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
          {file ? (
            <>
              <div className="file-icon">&#9812;</div>
              <p className="file-name">{file.name}</p>
              <p className="hint">Click to choose a different file</p>
            </>
          ) : (
            <>
              <div className="file-icon">&#9816;</div>
              <p>Drag &amp; drop a .pgn file here, or click to browse</p>
            </>
          )}
        </div>

        <button className="submit-btn" onClick={onSubmit} disabled={!file || loading}>
          {loading ? 'Analyzing...' : 'Predict Elo'}
        </button>

        {error && <div className="error-box">{error}</div>}

        {result && (
          <div className="results">
            <div className="result-row">
              <div className="result-card result-white">
                <span className="label">White</span>
                <span className="elo">{result.white_elo}</span>
                <span className="range">
                  {result.white_bucket_range[0]}&ndash;{result.white_bucket_range[1]} range
                </span>
                <span className="confidence">{Math.round(result.white_confidence * 100)}% confidence</span>
              </div>
              <div className="result-card result-black">
                <span className="label">Black</span>
                <span className="elo">{result.black_elo}</span>
                <span className="range">
                  {result.black_bucket_range[0]}&ndash;{result.black_bucket_range[1]} range
                </span>
                <span className="confidence">{Math.round(result.black_confidence * 100)}% confidence</span>
              </div>
            </div>
            <p className="plies-used">Based on {result.plies_used} plies of the game</p>
            {result.warning && <div className="warning-box">{result.warning}</div>}
          </div>
        )}
      </main>

      <footer className="footer">
        <p>8-expert bracket MoE model &middot; moves + clock-time only &middot; no engine analysis used</p>
      </footer>
    </div>
  )
}

export default App
