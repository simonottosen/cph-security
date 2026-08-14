// src/lib/liveQueue.ts
//
// Resolves the value shown as an airport's current/"now" queue length.
// When the latest scraped row is fresh we use it directly; when it is stale
// (the upstream airport feed stopped producing rows) we silently substitute an
// ML prediction for "now" so the user keeps seeing a live-looking value.

import axios from 'axios';

import { fetchForecast } from './forecast';

// Latest row older than this is treated as stale and replaced by a prediction.
const STALE_THRESHOLD_MS = 20 * 60 * 1000;

async function predictNow(code: string): Promise<number | null> {
  // 1) XGBoost single-point /predict for "now" (UTC, YYYY-MM-DDTHH:MM).
  try {
    const iso = new Date().toISOString().slice(0, 16);
    const res = await axios.get<{ predicted_queue_length_minutes: number }>(
      `https://waitport.com/api/v1/predict?airport=${code}&timestamp=${iso}`,
    );
    const v = res.data?.predicted_queue_length_minutes;
    if (typeof v === 'number') return Math.round(v);
  } catch {
    /* fall through to forecast */
  }

  // 2) Fallback: earliest still-upcoming AutoGluon /forecast point.
  try {
    const points = await fetchForecast(code);
    return points.length ? Math.round(points[0].mean) : null;
  } catch {
    return null;
  }
}

/**
 * Returns the value to display as the current/"now" queue for an airport.
 * Fresh latest row -> its real queue. Stale -> ML prediction for now.
 * Returns null only when there is no data and no prediction available.
 */
export async function fetchEffectiveQueue(code: string): Promise<number | null> {
  const upper = code.toUpperCase();
  let latest: { queue: number; timestamp: string } | undefined;
  try {
    const res = await axios.get<{ queue: number; timestamp: string }[]>(
      `https://waitport.com/api/v1/all?airport=eq.${upper}&limit=1&select=queue,timestamp&order=id.desc`,
    );
    latest = res.data[0];
  } catch {
    return null;
  }
  if (!latest) return null;

  const ageMs = Date.now() - new Date(latest.timestamp).getTime();
  if (ageMs <= STALE_THRESHOLD_MS) return latest.queue;

  const predicted = await predictNow(code);
  return predicted ?? latest.queue; // both predictors failed -> last known value
}
