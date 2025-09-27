const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://162.168.68.56:8080';

export async function swapInSwapper({ templateFile, portraitFile, signal, onProgress, download = true }) {
  const url = `${DEFAULT_API_BASE}/infer/infer/swap_inswapper${download ? '' : ''}`;
  const form = new FormData();
  form.append('template', templateFile);
  form.append('portrait', portraitFile);
  if (download) form.append('download', '1');

  // progress (only works in Chromium w/ fetch upload streaming; we simulate basic steps)
  onProgress?.(10);
  const res = await fetch(url, { method: 'POST', body: form, signal });
  onProgress?.(90);

  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try { const j = await res.json(); msg = j.detail || msg; } catch (_) {}
    throw new Error(`Swap failed: ${msg}`);
  }

  if (download) {
    const blob = await res.blob();
    onProgress?.(100);
    return URL.createObjectURL(blob);
  } else {
    const json = await res.json();
    onProgress?.(100);
    return json; // { ok, output_path, ... }
  }
}

export async function health() {
  const res = await fetch(`${DEFAULT_API_BASE}/infer/infer/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}
