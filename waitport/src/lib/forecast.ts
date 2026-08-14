// src/lib/forecast.ts
//
// Shared access to the Chronos-2 /forecast endpoint.

import axios from 'axios';

export interface ForecastPoint {
  /** Absolute instant this prediction applies to. */
  time: Date;
  /** Predicted queue length in minutes. */
  mean: number;
}

/**
 * The service emits naive UTC timestamps ("2026-08-14T09:30:00"), which `Date`
 * would otherwise read as the viewer's local time.
 */
export function parseForecastTimestamp(timestamp: string): Date {
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(timestamp);
  return new Date(hasZone ? timestamp : `${timestamp}Z`);
}

/**
 * Forecast points that still lie in the future, earliest first. Status records
 * the service returns in place of a forecast carry no `mean` and are dropped,
 * as are elapsed points — so an empty result means "no usable forecast", which
 * is what callers should render rather than presenting a stale one as current.
 */
export async function fetchForecast(code: string): Promise<ForecastPoint[]> {
  const res = await axios.get<{
    predictions?: { timestamp: string; mean?: number }[];
  }>(`https://waitport.com/api/v1/forecast/${code}`);

  const now = Date.now();
  return (res.data.predictions ?? [])
    .filter((p): p is { timestamp: string; mean: number } => typeof p.mean === 'number')
    .map((p) => ({
      time: parseForecastTimestamp(p.timestamp),
      mean: Math.max(0, p.mean),
    }))
    .filter((p) => p.time.getTime() >= now)
    .sort((a, b) => a.time.getTime() - b.time.getTime());
}
