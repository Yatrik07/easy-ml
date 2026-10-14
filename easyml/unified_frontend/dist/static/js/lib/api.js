// Small fetch helpers. Every function throws ApiError with the server's message on failure.

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

export async function errorMessage(res, fallback) {
  try {
    const body = await res.json();
    return body.error_message || body.message || body.error || fallback;
  } catch (e) {
    return fallback + ' (HTTP ' + res.status + ')';
  }
}

async function request(url, options, fallback) {
  let res;
  try {
    res = await fetch(url, options);
  } catch (e) {
    throw new ApiError('Cannot reach the Easy-ML server. Is it still running?', 0, null);
  }
  if (!res.ok) throw new ApiError(await errorMessage(res, fallback), res.status, null);
  return res;
}

export async function getJSON(url, fallback = 'Request failed') {
  const res = await request(url, {}, fallback);
  return res.json();
}

export async function postJSON(url, body, fallback = 'Request failed') {
  const res = await request(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  }, fallback);
  return res.json();
}

export async function deleteJSON(url, body, fallback = 'Request failed') {
  const res = await request(url, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  }, fallback);
  return res.json();
}

export async function postForm(url, formData, fallback = 'Request failed') {
  const res = await request(url, { method: 'POST', body: formData }, fallback);
  return res.json();
}

// POST and return the response as a Blob (file downloads).
export async function postForBlob(url, body, fallback = 'Download failed') {
  const isForm = body instanceof FormData;
  const res = await request(url, {
    method: 'POST',
    headers: isForm ? undefined : { 'Content-Type': 'application/json' },
    body: isForm ? body : JSON.stringify(body || {}),
  }, fallback);
  return res.blob();
}

export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
