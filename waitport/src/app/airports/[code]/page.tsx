'use client';

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Head from 'next/head';
import Link from 'next/link';
import Script from 'next/script';
// Tremor Raw components (Tailwind v4)
import { Card } from "@/components/Card";
import { AreaChart } from "@/components/AreaChart";
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';

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

  const initialDateTime = new Date(Date.now() + 2 * 60 * 60 * 1000);

  const [selectedDateTime, setSelectedDateTime] = useState<Date>(
    initialDateTime,
  );
  const [predictedQueueLength, setPredictedQueueLength] = useState<number | null>(null);
  const [loadingPredicted, setLoadingPredicted] = useState(true);

  const [historical, setHistorical] = useState<{
    yesterday: number | null;
    month: number | null;
    year: number | null;
  }>({
    yesterday: null,
    month: null,
    year: null,
  });
  const [loadingHistorical, setLoadingHistorical] = useState(true);

  const [forecastData, setForecastData] = useState<ForecastPoint[]>([]);
  const [loadingForecast, setLoadingForecast] = useState(true);
  const [forecastHorizon, setForecastHorizon] = useState<number | null>(null);

  const [queueSeries, setQueueSeries] = useState<QueuePoint[]>([]);
  // Combined data for past (actual) and future (predicted) queue values
  const [combinedSeries, setCombinedSeries] = useState<any[]>([]);
  // 0‑1 value that marks where the future prediction starts along the X‑axis
  const [transitionRatio, setTransitionRatio] = useState<number>(0);
  // Predicted average queue length over the next 2 hours
  const [avgNextTwoHours, setAvgNextTwoHours] = useState<number | null>(null);

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
    const fetchHistorical = async () => {
      try {
        setLoadingHistorical(true);
        const now = new Date();
        const targets = [
          { key: 'yesterday', date: new Date(now.getTime() - 24 * 60 * 60 * 1000) },
          { key: 'month', date: new Date(new Date(now).setMonth(now.getMonth() - 1)) },
          { key: 'year', date: new Date(new Date(now).setFullYear(now.getFullYear() - 1)) },
        ] as const;

        const results: Record<string, number | null> = {};
        for (const t of targets) {
          const iso = t.date.toISOString();
          const res = await axios.get<{ queue: number }[]>(
            `https://waitport.com/api/v1/all?airport=eq.${code.toUpperCase()}&timestamp=lte.${iso}&select=queue&order=timestamp.desc&limit=1`,
          );
          results[t.key] = res.data[0]?.queue ?? null;
        }

        setHistorical({
          yesterday: results['yesterday'],
          month: results['month'],
          year: results['year'],
        });
      } finally {
        setLoadingHistorical(false);
      }
    };

    fetchHistorical();
  }, [code]);

useEffect(() => {
    const fetchPrediction = async () => {
      try {
        setLoadingPredicted(true);

        // Try to derive prediction from the richer /forecast endpoint first
        const forecastRes = await axios.get<{ predictions: { timestamp: string; mean: number }[] }>(
          `https://waitport.com/api/v1/forecast/${code}`,
        );

        const targetTime = selectedDateTime.getTime();
        const toleranceMs = 15 * 60 * 1000; // 15‑minute window

        const match = forecastRes.data.predictions?.find(p => {
          const tLocal = new Date(p.timestamp);
          tLocal.setHours(tLocal.getHours() + 2); // shift to CPH
          return Math.abs(tLocal.getTime() - targetTime) <= toleranceMs;
        });

        if (match && typeof match.mean === 'number') {
          // Use the forecast value if it's within the time window
          setPredictedQueueLength(Math.round(match.mean));
        } else {
          // Fall back to the original /predict endpoint
          const iso = selectedDateTime.toISOString().slice(0, 16);
          const res = await axios.get<{ predicted_queue_length_minutes: number }>(
            `https://waitport.com${API_URL}?timestamp=${iso}&airport=${code}`,
          );
          setPredictedQueueLength(res.data.predicted_queue_length_minutes ?? 0);
        }
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
        // Keep only predictions that are in the future (local CPH time)
        const future = res.data.predictions?.filter(p => {
          const d = new Date(p.timestamp);
          d.setHours(d.getHours() + 2); // shift to CPH
          return d.getTime() >= Date.now();
        }) ?? [];

        const formatted = future.map(p => {
          const date = new Date(p.timestamp);
          date.setHours(date.getHours() + 2); // shift to CPH
          const time = date.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          });
          return {
            timestamp: time,
            Average: Math.max(0, p.mean),
            Low: Math.max(0, p.q30),
            High: Math.max(0, p.q70),
          };
        });
        // Derive horizon (last timestamp minus now, in hours)
        if (future.length) {
          const last = new Date(future[future.length - 1].timestamp);
          last.setHours(last.getHours() + 2); // shift to CPH
          const hours = Math.max(
            1,
            Math.round((last.getTime() - Date.now()) / (1000 * 60 * 60)),
          );
          setForecastHorizon(hours);
        } else {
          setForecastHorizon(null);
        }
        setForecastData(formatted);
      } finally {
        setLoadingForecast(false);
      }
    };
    fetchForecast();
  }, [code]);

  // Derive predicted average over the next two hours (first 8 forecast points = 2 h)
  useEffect(() => {
    if (!forecastData.length) {
      setAvgNextTwoHours(null);
      return;
    }
    const window = forecastData.slice(0, 8); // 8 × 15‑min = 2 h
    if (!window.length) {
      setAvgNextTwoHours(null);
      return;
    }
    const avg =
      window.reduce((sum, p) => sum + p.Average, 0) / window.length;
    setAvgNextTwoHours(Math.round(Math.max(0, avg)));
  }, [forecastData]);

  // Merge queueSeries (past) and forecastData (future) into a single series
  useEffect(() => {
    if (!queueSeries.length && !forecastData.length) return;

    const past = queueSeries.map((p) => ({
      time: p.time,
      Past: p.queue,
    }));

    // Easing helper (smoothstep: 3t² − 2t³) for a soft transition
    const smoothstep = (t: number) => 3 * t * t - 2 * t * t * t;

    const lastQueue = queueSeries.length
      ? queueSeries[queueSeries.length - 1].queue
      : null;

    const future = forecastData.map((p, idx) => {
      // If we don't have a last real queue value, fall back to the raw prediction
      if (lastQueue === null) {
        return {
          time: p.timestamp,
          Prediction: Math.max(0, p.Average),
        };
      }

      const progress = (idx + 1) / forecastData.length; // 0‒1
      const blend = smoothstep(progress);
      const blendedValue = Math.max(0, lastQueue * (1 - blend) + p.Average * blend);

      return {
        time: p.timestamp,
        Prediction: blendedValue,
      };
    });

    const merged = [...past, ...future];
    setCombinedSeries(merged);

    // Where does the prediction start (as a % of chart width)?
    const ratio =
      past.length && merged.length > 1
        ? past.length / (merged.length - 1)
        : 0;
    setTransitionRatio(ratio);
  }, [queueSeries, forecastData]);

  /* -------------------- RENDER CALC -------------------- */
  const diffMinutes = Math.round(
    (selectedDateTime.getTime() - Date.now()) / 60000,
  );
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
            Current &amp; future airport security queues
          </h2>
        </header>

        {/* Main */}
        <main className="flex-1 w-full max-w-6xl mx-auto px-6">
          {/* Queue overview */}
          <section className="mt-4">
            <div className="grid gap-6 lg:gap-4 md:grid-cols-2 lg:grid-cols-2">
              {/* Current queue */}
              <Card className="shadow-sm ring-1 ring-gray-200 dark:ring-gray-700 dark:bg-gray-900/50 rounded-lg">
                <h3 className="text-lg md:text-xl font-semibold text-gray-800 dark:text-gray-100 mb-2">
                  Today’s queue
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                  Live security‑queue wait time, updated every few minutes.
                </p>
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
                    <p className="mt-1 text-gray-500">
                      Average over the next&nbsp;2 hours:{' '}
                      <span className="font-semibold">{formatMinutes(avgNextTwoHours)}</span>
                    </p>
                    {/* Forecast chart for today */}
                {loadingAverage || loadingForecast ? (
                  <p className="mt-4">Loading chart…</p>
                ) : (
                  <>
                    <div className="relative mt-6">
                      {/* Past = blue, Prediction = sky‑blue */}
                      <AreaChart
                        className="h-60"
                        data={combinedSeries}
                        index="time"
                        categories={['Past', 'Prediction']}
                        colors={['blue', 'sky']}
                        showLegend={true}
                        curveType="monotone"
                        valueFormatter={(v) =>
                          v === null ? '' : `${Math.round(v as number)} min`
                        }
                      />
                      {/* Marker line where prediction begins */}
                      <div
                        className="absolute bg-yellow-400 dark:bg-yellow-300"
                        style={{
                          top: '4.5rem',
                          bottom: '1.8rem',
                          width: '5px',
                          /* 43 px compensates for chart gutter */
                          left: `calc(${(transitionRatio * 100).toFixed(2)}% + 42px)`,
                        }}
                      />
                      {/* Current queue label */}
                    </div>
                    <style jsx global>{`
                      /* The second area represents future predictions */
                      .recharts-layer .recharts-area:nth-of-type(2) path {
                        stroke-dasharray: 4 4;
                        fill-opacity: 0.15;
                      }
                    `}</style>
                    <p className="mt-4 text-sm text-gray-500 dark:text-gray-400">
                      Expected queue over the next&nbsp;
                      {forecastHorizon ?? 6}
                      &nbsp;hour{forecastHorizon === 1 ? '' : 's'}&nbsp;(forecasts update every few hours)
                    </p>
                  </>
                )}
                  </>
                )}
              </Card>

              {/* Right‑column stack */}
              <div className="flex flex-col gap-6 lg:col-start-2 h-full justify-between">
              {/* Prediction */}
              <Card className="shadow-sm ring-1 ring-gray-200 dark:ring-gray-700 dark:bg-gray-900/50 rounded-lg">
                <h3 className="text-lg md:text-xl font-semibold text-gray-800 dark:text-gray-100 mb-2">
                  Prediction
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                  Estimated wait time for the date&nbsp;&amp;&nbsp;time you pick below.
                </p>


                {/* DateTime picker */}
                <div className="mt-6 mb-8">
                  <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                    Select date&nbsp;&amp;&nbsp;time for prediction
                  </label>

                  <DatePicker
                    selected={selectedDateTime}
                    onChange={(date: Date) => setSelectedDateTime(date as Date)}
                    showTimeSelect
                    timeIntervals={15}
                    dateFormat="Pp"
                    wrapperClassName="block w-full"
                    className="block w-full rounded border-gray-300 dark;border-gray-600 bg-white dark:bg-gray-800 p-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-800 dark:text-gray-100"
                    calendarClassName="bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg shadow-lg p-3"
                    dayClassName={(date) =>
                      date.toDateString() === selectedDateTime.toDateString()
                        ? 'bg-blue-500 text-white rounded-full'
                        : 'hover:bg-gray-200 dark:hover:bg-gray-700 rounded-full'
                    }
                  />

                  <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                  </p>
                </div>
                {loadingPredicted ? (
                  <p>Loading…</p>
                ) : (
                  <>
                    <div className="flex items-center space-x-2 mt-4">
                      <p className="text-3xl font-bold text-gray-800 dark:text-gray-100">{formatMinutes(predictedQueueLength)}</p>
                      <p className="mt-2 text-gray-500">{timeDiffText}</p>
                    </div>
                  </>
                )}

              </Card>

              {/* Historical */}
              <Card className="shadow-sm ring-1 ring-gray-200 dark:ring-gray-700 dark:bg-gray-900/50 rounded-lg">
                <h3 className="text-lg md:text-xl font-semibold text-gray-800 dark:text-gray-100 mb-2">
                  Historical
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                  What the queue looked like at roughly this time yesterday, one&nbsp;month ago, and one&nbsp;year ago.
                </p>
                {loadingHistorical ? (
                  <p>Loading…</p>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {/* Yesterday */}
                    <span className="inline-flex w-48 items-center whitespace-nowrap justify-between gap-2 rounded-md bg-blue-100 dark:bg-blue-900 py-1 pl-2.5 pr-2 text-sm text-gray-800 dark:text-gray-200 ring-1 ring-inset ring-blue-200 dark:ring-blue-800">
                      Yesterday
                      <span className="h-4 w-px bg-blue-300 dark:bg-blue-700" />
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        {historical.yesterday === null ? 'Not available' : formatMinutes(historical.yesterday)}
                      </span>
                    </span>

                    {/* One month ago */}
                    <span className="inline-flex w-48 items-center whitespace-nowrap justify-between gap-2 rounded-md bg-blue-100 dark:bg-blue-900 py-1 pl-2.5 pr-2 text-sm text-gray-800 dark:text-gray-200 ring-1 ring-inset ring-blue-200 dark:ring-blue-800">
                      1&nbsp;month&nbsp;ago
                      <span className="h-4 w-px bg-blue-300 dark:bg-blue-700" />
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        {historical.month === null ? 'Not available' : formatMinutes(historical.month)}
                      </span>
                    </span>

                    {/* One year ago */}
                    <span className="inline-flex w-48 items-center whitespace-nowrap justify-between gap-2 rounded-md bg-blue-100 dark:bg-blue-900 py-1 pl-2.5 pr-2 text-sm text-gray-800 dark:text-gray-200 ring-1 ring-inset ring-blue-200 dark:ring-blue-800">
                      1&nbsp;year&nbsp;ago
                      <span className="h-4 w-px bg-blue-300 dark:bg-blue-700" />
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        {historical.year === null ? 'Not available' : formatMinutes(historical.year)}
                      </span>
                    </span>
                  </div>
                )}
              </Card>
              </div>
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
