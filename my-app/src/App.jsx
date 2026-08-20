import { useState } from 'react'
import './App.css'

function App() {
  const [inputText, setInputText] = useState('')
  const [translation, setTranslation] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleTranslate = async () => {
    if (!inputText.trim()) return
    setLoading(true)
    setError('')
    setTranslation('')

    try {
      const res = await fetch('http://localhost:5000/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Translation failed')
      setTranslation(data.translation)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleTranslate()
    }
  }

  const outputClass = `output-area${error ? ' error-text' : !translation ? ' empty' : ''}`
  const outputText = error || translation || 'Translation will appear here...'

  return (
    <div className="app">
      <header className="header">
        <div className="header-title">Tarjuma</div>
        <div className="header-sub">French &rarr; English</div>
      </header>

      <main className="main">
        <div className="card">
          <div className="panel">
            <span className="panel-label">French</span>
            <textarea
              className="input-area"
              placeholder="e.g. Bonjour, comment allez-vous?"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={5}
            />
          </div>

          <div className="action-row">
            <button
              className="translate-btn"
              onClick={handleTranslate}
              disabled={loading || !inputText.trim()}
            >
              {loading ? <span className="spinner" /> : null}
              {loading ? 'Translating' : 'Translate'}
            </button>
          </div>

          <div className="panel">
            <span className="panel-label">English</span>
            <div className={outputClass}>{outputText}</div>
          </div>
        </div>
      </main>

      <footer className="footer">Press Enter to translate</footer>
    </div>
  )
}

export default App
