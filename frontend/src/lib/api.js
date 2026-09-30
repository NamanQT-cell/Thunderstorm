// Talks to the FastAPI layer at src/api.py, which wraps the real,
// untouched model/preprocessing pipeline. If that server isn't running
// (e.g. you're only looking at the frontend), every call falls back to
// the offline synthetic generator in offlineSynthetic.js so the UI still
// works -- and the fallback is always flagged in the returned object as
// `offline: true` so the UI can be honest about which mode it's in.

import { computeOfflineFrame, computeOfflineTrend } from "./offlineSynthetic.js";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function safeFetchJson(url) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 4000);
  try {
    const res = await fetch(url, { signal: controller.signal });
    if (!res.ok) throw new Error(`API responded ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timeout);
  }
}

export async function fetchConfig() {
  try {
    const data = await safeFetchJson(`${BASE_URL}/api/config`);
    return { ...data, offline: false };
  } catch {
    return {
      region: "Punjab-Haryana-Delhi",
      leadTimes: [30, 60, 90],
      usingRealModel: false,
      dataMode: "Offline demo (API not reachable)",
      offline: true,
    };
  }
}

export async function fetchFrame(timeStep, lead, grid) {
  try {
    const url = `${BASE_URL}/api/frame?time_step=${timeStep}&lead=${lead}&grid=${grid}`;
    const data = await safeFetchJson(url);
    return { ...data, offline: false };
  } catch {
    return { ...computeOfflineFrame(timeStep, lead, grid), offline: true };
  }
}

export async function fetchTrend(grid) {
  try {
    const data = await safeFetchJson(`${BASE_URL}/api/trend?grid=${grid}`);
    return { ...data, offline: false };
  } catch {
    return { ...computeOfflineTrend(grid), offline: true };
  }
}
