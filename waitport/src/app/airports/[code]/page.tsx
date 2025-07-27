'use client';

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Head from 'next/head';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import Script from 'next/script';
// Tremor Raw components (Tailwind v4)
import { Card } from "@/components/Card";
import { AreaChart } from "@/components/AreaChart";
import 'react-datetime/css/react-datetime.css';
import 'moment/locale/da';

// Dynamically import DateTime to disable SSR for this component
const DateTime = dynamic(() => import('react-datetime').then(mod => mod.default || mod), {
  ssr: false,
  loading: () => <span>Loading date picker...</span>,
});

/* ------------------------------------------------------------------ */
/* Types and constants that we also export so the server wrapper       */
/* can use them.                                                      */
/* ------------------------------------------------------------------ */
import {
  AirportCode,
  ForecastPoint,
  QueuePoint,
  airportNames,
} from '@/lib/airports';

const API_URL = process.env.NEXT_PUBLIC_API_HOST || '/api/v1/predict';

/* ------------------------------------------------------------------ */
/* Client component                                                    */
/* ------------------------------------------------------------------ */
import { useParams } from 'next/navigation';

const ClientPage: React.FC = () => {
  const params = useParams() as { code?: string };
  // Fallback to "cph" so hooks always run in the same order
  const code = (params.code ?? 'cph') as AirportCode;

  const airportName = airportNames[code];

  /* -------------------- STATE -------------------- */
  const [queue, setQueue] = useState<number | null>(null);
  const [averageQueue, setAverageQueue] = useState<number | null>(null);
  const [loadingQueue, setLoadingQueue] = useState(true);
  const [loadingAverage, setLoadingAverage] = useState(true);

  const [selectedDateTime, setSelectedDateTime] = useState<Date>(
    () => new Date(Date.now() + 2 * 60 * 60 * 1000),
  );
  const [predictedQueueLength, setPredictedQueueLength] = useState<number | null>(null);
  const [loadingPredicted, setLoadingPredicted] = useState(true);

  const [forecastData, setForecastData] = useState<ForecastPoint[]>([]);
  const [loadingForecast, setLoadingForecast] = useState(true);

  const [queueSeries, setQueueSeries] = useState<QueuePoint[]>([]);

  /* -------------------- HELPERS -------------------- */
  const formatMinutes = (m: number | null) =>
    m !== null ? `${m} ${m === 1 ? 'minute' : 'minutes'}` : '-';

  /* -------------------- DATA FETCH -------------------- */
  useEffect(() => {
    const fetchQueue = async () => {
      try {
        setLoadingQueue(true);
        const res = await axios.get<{ queue: number }[]>(
          `https://waitport.com/api/v1/all?airport=eq.${code.toUpperCase()}&limit=1&select=queue&order=id.desc`,
        );
        setQueue(res.data[0]?.queue ?? 0);
      } finally {
        setLoadingQueue(false);
      }
    };
    fetchQueue();
  }, [code]);

  useEffect(() => {
    const fetchAverage = async () => {
      try {
        setLoadingAverage(true);
        const res = await axios.get<{ queue: number; timestamp: string }[]>(
          `https://waitport.com/api/v1/all?airport=eq.${code.toUpperCase()}&select=queue,timestamp&limit=24&order=id.desc`,
        );
        const values = res.data.map(d => d.queue);
        const avg = values.length ? Math.round(values.reduce((s, v) => s + v, 0) / values.length) : 0;
        setAverageQueue(avg);
        const formattedSeries = res.data.map(d => ({
          time: new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          queue: d.queue,
        }));
        setQueueSeries(formattedSeries.reverse()); // chronological order
      } finally {
        setLoadingAverage(false);
      }
    };
    fetchAverage();
  }, [code]);

  useEffect(() => {
    const fetchPrediction = async () => {
      try {
        setLoadingPredicted(true);
        const iso = selectedDateTime.toISOString().slice(0, 16);
        const res = await axios.get<{ predicted_queue_length_minutes: number }>(
          `https://waitport.com${API_URL}?timestamp=${iso}&airport=${code}`,
        );
        setPredictedQueueLength(res.data.predicted_queue_length_minutes ?? 0);
      } finally {
        setLoadingPredicted(false);
      }
    };
    fetchPrediction();
  }, [code, selectedDateTime]);

  useEffect(() => {
    const fetchForecast = async () => {
      try {
        setLoadingForecast(true);
        const res = await axios.get<{ predictions: any[] }>(
          `https://waitport.com/api/v1/forecast/${code}`,
        );
        const formatted =
          res.data.predictions?.map(p => {
            const date = new Date(p.timestamp);
            date.setHours(date.getHours() + 2);
            const time = date.toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            });
            return {
              timestamp: time,
              Average: p.mean,
              Low: p.q30,
              High: p.q70,
            };
          }) ?? [];
        setForecastData(formatted);
      } finally {
        setLoadingForecast(false);
      }
    };
    fetchForecast();
  }, [code]);

  /* -------------------- RENDER CALC -------------------- */
  const diffMinutes = Math.round((selectedDateTime.getTime() - Date.now()) / 60000);
  const timeDiffText =
    diffMinutes === 0
      ? 'at this time'
      : diffMinutes < 60
      ? `in ${diffMinutes} minutes`
      : `in ${Math.round(diffMinutes / 60)} hours`;

  /* -------------------- JSX -------------------- */
  return (
    <>
      <Head>
        <title>{`Waitport - Security Queues at ${airportName}`}</title>
        <meta
          name="description"
          content={`Check live and predicted security queue wait times at ${airportName}. Plan your trip effectively with Waitport's real-time data and future estimates.`}
        />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Script
        src="https://umami.waitport.com/script.js"
        data-website-id="25e1973f-f0c8-489c-bb41-6726ad81ca4d"
        strategy="afterInteractive"
      />

      {/* -------------- PAGE LAYOUT -------------- */}
      <div className="min-h-screen flex flex-col bg-linear-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
        {/* Header */}
        <header className="py-6 text-center">
          <Link href="/" className="inline-block">
            <h1 className="text-4xl md:text-5xl font-extrabold text-gray-800 dark:text-gray-100 mb-1 hover:opacity-80">
              Waitport 🛫
            </h1>
          </Link>
          <h2 className="text-lg md:text-xl text-gray-600 dark:text-gray-300">
            Real‑Time &amp; Predicted Airport Security Queues
          </h2>
        </header>

        {/* Main */}
        <main className="flex-1 w-full max-w-6xl mx-auto px-6">
          {/* Queue overview */}
          <section className="mt-4">
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-2">
              {/* Current queue */}
              <Card className="shadow-sm ring-1 ring-gray-200 dark:ring-gray-700 dark:bg-gray-900/50 rounded-lg">
                <h3 className="text-lg md:text-xl font-semibold text-gray-800 dark:text-gray-100 mb-2">
                  Current queue
                </h3>
                {loadingQueue || loadingAverage ? (
                  <p>Loading…</p>
                ) : (
                  <>
                    <div className="flex items-center space-x-2 mt-4">
                      <p className="text-3xl font-bold text-gray-800 dark:text-gray-100">{formatMinutes(queue)}</p>
                      <p className="mt-2 text-gray-500">now</p>
                    </div>
                    <p className="mt-1 text-gray-500">
                      Average in the last&nbsp;2 hours: <span className="font-semibold">{formatMinutes(averageQueue)}</span>
                    </p>
                    {queueSeries.length > 0 && (
                      <AreaChart
                        className="h-48 mt-6"
                        data={queueSeries.map(p => ({
                          time: p.time,
                          Queue: p.queue,
                          Average: averageQueue ?? 0,
                        }))}
                        index="time"
                        categories={['Queue', 'Average']}
                        colors={['blue', 'gray']}
                        showLegend={false}
                        valueFormatter={v => `${v as number} min`}
                      />
                    )}
                  </>
                )}
              </Card>

              {/* Prediction */}
              <Card className="shadow-sm ring-1 ring-gray-200 dark:ring-gray-700 dark:bg-gray-900/50 rounded-lg">
                <h3 className="text-lg md:text-xl font-semibold text-gray-800 dark:text-gray-100 mb-2">
                  Prediction
                </h3>
                {loadingPredicted ? (
                  <p>Loading…</p>
                ) : (
                  <>
                    <p className="text-3xl font-bold text-gray-800 dark:text-gray-100">{formatMinutes(predictedQueueLength)}</p>
                    <p className="mt-2 text-gray-500">{timeDiffText}</p>
                  </>
                )}

                {/* DateTime picker */}
                <div className="mt-6">
                  <label htmlFor="datetime-picker" className="block mb-2 font-medium">
                    Select date &amp; time to personalize your prediction
                  </label>
                  <DateTime
                    locale="da-dk"
                    inputProps={{
                      id: 'datetime-picker',
                      className:
                        'w-full rounded border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2',
                      'aria-label': 'Select date and time',
                    }}
                    dateFormat="DD/MM"
                    timeFormat="HH:mm"
                    closeOnSelect
                    value={selectedDateTime}
                  />
                </div>

                {/* Forecast chart */}
                {loadingForecast ? (
                  <p className="mt-4">Loading forecast…</p>
                ) : (
                  <AreaChart
                    className="h-60 mt-6"
                    data={forecastData}
                    index="timestamp"
                    categories={['High', 'Average']}
                    colors={['blue', 'gray']}
                    showLegend={true}
                    valueFormatter={v => `${Math.round(v as number)} min`}
                  />
                )}
              </Card>
            </div>
          </section>

          {/* Select another airport */}
          <section className="mt-10">
            <h3 className="mb-3 text-lg font-semibold">Select another airport</h3>
            <select
              className="mt-2 block w-full rounded border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2"
              value={code}
              onChange={e =>
                (window.location.href = `/airports/${(e.target as HTMLSelectElement).value}`)
              }
              aria-label="Select Airport"
            >
              {Object.entries(airportNames).map(([c, name]) => (
                <option key={c} value={c}>
                  {name}
                </option>
              ))}
            </select>
          </section>
        </main>

        {/* Footer */}
        <footer className="py-6 text-gray-500 dark:text-gray-400">
          <ul className="flex justify-center border-b border-gray-200 dark:border-gray-700 pb-3 mb-3">
            <li>
              <Link href="https://simonottosen.dk/" className="mx-2 hover:text-gray-700 dark:hover:text-gray-300" target="_blank">
                Other projects
              </Link>
            </li>
            <li>
              <Link
                href="https://waitport.com/api/v1/all?order=id.desc&limit=100"
                className="mx-2 hover:text-gray-700 dark:hover:text-gray-300"
                target="_blank"
              >
                API
              </Link>
            </li>
            <li>
              <Link
                href="https://github.com/simonottosen/cph-security"
                className="mx-2 hover:text-gray-700 dark:hover:text-gray-300"
                target="_blank"
              >
                GitHub
              </Link>
            </li>
          </ul>
          <p className="text-center text-sm">
            Made with <span role="img" aria-label="heart">❤️</span> by Simon Ottosen
          </p>
          <p className="text-center text-xs">&copy; {new Date().getFullYear()} Waitport</p>
        </footer>
      </div>
    </>
  );
};

export default ClientPage;
