// src/lib/airports.ts

/** A union of all supported airport codes */
export type AirportCode =
  | 'cph'
  | 'osl'
  | 'arn'
  | 'dus'
  | 'fra'
  | 'muc'
  | 'lhr'
  | 'ams'
  | 'dub'
  | 'ist';

/** One point in the Autogluon forecast */
export interface ForecastPoint {
  /** Timestamp in ISO format or any string parseable by Date */
  timestamp: string;
  Average: number;
  Low: number;
  High: number;
}

/** Queue length at a specific time */
export interface QueuePoint {
  /** Formatted time string (e.g. "14:30") */
  time: string;
  queue: number;
}

/** Display names (with flag emojis) keyed by code */
export const airportNames: Record<AirportCode, string> = {
  cph: '🇩🇰 Copenhagen Airport',
  osl: '🇳🇴 Oslo Airport',
  arn: '🇸🇪 Stockholm Airport',
  dus: '🇩🇪 Düsseldorf Airport',
  fra: '🇩🇪 Frankfurt Airport',
  muc: '🇩🇪 Munich Airport',
  lhr: '🇬🇧 London Heathrow Airport',
  ams: '🇳🇱 Amsterdam Airport',
  dub: '🇮🇪 Dublin Airport',
  ist: '🇹🇷 Istanbul Airport',
};

/** Plain text names (no emoji) keyed by code */
export const airportNamesText: Record<AirportCode, string> = {
  cph: 'Copenhagen Airport',
  osl: 'Oslo Airport',
  arn: 'Stockholm Airport',
  dus: 'Düsseldorf Airport',
  fra: 'Frankfurt Airport',
  muc: 'Munich Airport',
  lhr: 'London Heathrow Airport',
  ams: 'Amsterdam Airport',
  dub: 'Dublin Airport',
  ist: 'Istanbul Airport',
};