import { useEffect, useMemo, useRef, useState } from 'react'
import { swapInSwapper, health } from './api'

export default function App() {
  const [portraitFile, setPortraitFile] = useState(null)
  const [templateFile, setTemplateFile] = useState(null)
  const [portraitURL, setPortraitURL] = useState('')
  const [templateURL, setTemplateURL] = useState('')
  const [resultURL, setResultURL] = useState('')
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState('Ready')

  // cleanup blob URLs
  useEffect(() => () => { portraitURL && URL.revokeObjectURL(portraitURL); templateURL && URL.revokeObjectURL(templateURL); resultURL && URL.revokeObjectURL(resultURL); }, [portraitURL, templateURL, resultURL])

  // quick health ping on load
  useEffect(() => {
    health().then(() => setStatus('Ready')).catch(() => setStatus('API offline'))
  }, [])

  function onPortraitChange(e) {
    const f = e.target.files?.[0]
    if (!f) return
    setPortraitFile(f)
    setPortraitURL(URL.createObjectURL(f))
  }
  function onTemplateChange(e) {
    const f = e.target.files?.[0]
    if (!f) return
    setTemplateFile(f)
    setTemplateURL(URL.createObjectURL(f))
  }

  async function onSwap() {
    if (!portraitFile || !templateFile) { alert('Pick both images'); return }
    setBusy(true); setProgress(5); setStatus('Uploading…'); setResultURL('')
    const ctrl = new AbortController()
    try {
      const blobURL = await swapInSwapper({
        templateFile, portraitFile, signal: ctrl.signal,
        onProgress: setProgress, download: true
      })
      setResultURL(blobURL)
      setStatus('Done')
    } catch (e) {
      console.error(e)
      setStatus(e.message || 'Error')
      alert(e.message || 'Swap failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="container">
      <h1>Face Template Swap</h1>
      <p className="hint">Take a selfie, choose a template image, then hit Swap. Works with your running API at <code>/infer/infer/swap_inswapper</code>.</p>

      <div className="card row">
        <label className="row">
          <strong>Portrait (camera)</strong>
          <input
            type="file"
            accept="image/*"
            capture="user"
            onChange={onPortraitChange}
          />
          <span className="hint">Use your phone’s camera; JPG/PNG supported.</span>
        </label>

        {portraitURL && <img className="thumb" src={portraitURL} alt="portrait preview" />}

        <label className="row">
          <strong>Template (upload)</strong>
          <input
            type="file"
            accept="image/*"
            onChange={onTemplateChange}
          />
          <span className="hint">Pick the template you want to swap onto.</span>
        </label>

        {templateURL && <img className="thumb" src={templateURL} alt="template preview" />}

        <div className="row">
          <button className="btn primary" onClick={onSwap} disabled={busy || !portraitFile || !templateFile}>
            {busy ? 'Swapping…' : 'Swap'}
          </button>
          <div className="bar"><div style={{ width: `${progress}%` }} /></div>
          <div className="hint">Status: {status}</div>
        </div>
      </div>

      {resultURL && (
        <div className="card row">
          <strong>Result</strong>
          <img className="thumb" src={resultURL} alt="result" />
          <div className="grid-2">
            <a className="btn ghost" href={resultURL} download="swap.png">Download PNG</a>
            <button className="btn ghost" onClick={() => { URL.revokeObjectURL(resultURL); setResultURL('') }}>Clear</button>
          </div>
        </div>
      )}

      <div className="footer">Tip: if the template has multiple faces, we can add a face picker later.</div>
    </div>
  )
}
