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
import { useI18n } from "@/i18n/I18nProvider";
import 'react-datepicker/dist/react-datepicker.css';

/* ------------------------------------------------------------------ */
/* Types and constants that we also export so the server wrapper       */
/* can use them.                                                      */
/* ------------------------------------------------------------------ */
import {
  AirportCode,
  QueuePoint,
  airportNames,
} from '@/lib/airports';
import { fetchEffectiveQueue } from '@/lib/liveQueue';
import { ForecastPoint, fetchForecast } from '@/lib/forecast';

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
  const { t, locale } = useI18n();

  // Compute root path that honours current locale (English uses bare "/")
  const rootPath = locale === 'en' ? '/' : `/${locale}`;

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
    m !== null ? `${m} ${m === 1 ? t('minute') : t('minutes')}` : '-';

  /* -------------------- DATA FETCH -------------------- */
  useEffect(() => {
    const fetchQueue = async () => {
      try {
        setLoadingQueue(true);
        const effective = await fetchEffectiveQueue(code);
        setQueue(effective ?? 0);
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
        const points = await fetchForecast(code);

        const targetTime = selectedDateTime.getTime();
        const toleranceMs = 15 * 60 * 1000; // 15‑minute window

        const match = points.find(
          p => Math.abs(p.time.getTime() - targetTime) <= toleranceMs,
        );

        if (match) {
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
    const loadForecast = async () => {
      try {
        setLoadingForecast(true);
        const points = await fetchForecast(code);
        if (points.length) {
          const last = points[points.length - 1].time;
          setForecastHorizon(
            Math.max(1, Math.round((last.getTime() - Date.now()) / (1000 * 60 * 60))),
          );
        } else {
          setForecastHorizon(null);
        }
        setForecastData(points);
      } finally {
        setLoadingForecast(false);
      }
    };
    loadForecast();
  }, [code]);

  // Derive the predicted average over the next two hours.
  //
  // Select by timestamp rather than by point count. A fixed `slice(0, 8)`
  // silently assumed the service emitted 15-minute steps; on the 5-minute grid
  // it averaged the next 40 minutes while the label promised two hours.
  useEffect(() => {
    if (!forecastData.length) {
      setAvgNextTwoHours(null);
      return;
    }
    const cutoff = Date.now() + 2 * 60 * 60 * 1000;
    // fetchForecast returns future-only points sorted ascending, so this is a
    // prefix. Keep the first point if the horizon starts past the cutoff, so a
    // sparse forecast still reports a number instead of NaN.
    const withinWindow = forecastData.filter((p) => p.time.getTime() <= cutoff);
    const points = withinWindow.length ? withinWindow : forecastData.slice(0, 1);
    const avg = points.reduce((sum, p) => sum + p.mean, 0) / points.length;
    setAvgNextTwoHours(Math.round(Math.max(0, avg)));
  }, [forecastData]);

  // Merge queueSeries (past) and forecastData (future) into a single series
  useEffect(() => {
    if (!queueSeries.length && !forecastData.length) return;

    const lastIdx = queueSeries.length - 1;

    // Align the terminal "now" point with the displayed effective queue value
    // (an ML prediction when live data is stale) so the chart endpoint matches
    // the big "now" number. Earlier history is left as real data.
    const lastQueue =
      queue !== null
        ? queue
        : queueSeries.length
        ? queueSeries[lastIdx].queue
        : null;

    // Copy the last queue value into the Prediction series as well,
    // so the two lines meet without a gap.
    const past = queueSeries.map((p, idx) => {
      const value = idx === lastIdx && lastQueue !== null ? lastQueue : p.queue;
      return {
        time: p.time,
        Past: value,
        Prediction: idx === lastIdx ? value : null,
      };
    });

    // Easing helper (smoothstep: 3t² − 2t³) for a soft transition
    const smoothstep = (t: number) => 3 * t * t - 2 * t * t * t;

    const future = forecastData.map((p, idx) => {
      const time = p.time.toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
      });

      // If we don't have a last real queue value, fall back to the raw prediction
      if (lastQueue === null) {
        return { time, Prediction: p.mean };
      }

      const progress = (idx + 1) / forecastData.length; // 0‒1
      const blend = smoothstep(progress);
      const blendedValue = Math.max(0, lastQueue * (1 - blend) + p.mean * blend);

      return { time, Prediction: blendedValue };
    });

    const merged = [...past, ...future];
    setCombinedSeries(merged);

    // Where does the prediction start (as a % of chart width)?
    const ratio =
      past.length && merged.length > 1
        ? past.length / (merged.length - 1)
        : 0;
    setTransitionRatio(ratio);
  }, [queueSeries, forecastData, queue]);

  /* -------------------- RENDER CALC -------------------- */
  const diffMinutes = Math.round(
    (selectedDateTime.getTime() - Date.now()) / 60000,
  );
  const timeDiffText =
    diffMinutes === 0
      ? t('atThisTime')
      : diffMinutes < 60
      ? t('inMinutes', { minutes: diffMinutes })
      : t('inHours', { hours: Math.round(diffMinutes / 60) });

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
          {/* Preserve current locale when navigating home */}
          <Link href={rootPath} className="inline-block">
            <h1 className="text-4xl md:text-5xl font-extrabold text-gray-800 dark:text-gray-100 mb-1 hover:opacity-80">
              Waitport 🛫
            </h1>
          </Link>
          <h2 className="text-lg md:text-xl text-gray-600 dark:text-gray-300">
            {t('currentFutureQueues')}
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
                  {t('todayQueue')}
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                  {t('todayQueue.description')}
                </p>
                {loadingQueue || loadingAverage ? (
                  <p>{t('loading')}</p>
                ) : (
                  <>
                    <div className="flex items-center space-x-2 mt-4">
                      <p className="text-3xl font-bold text-gray-800 dark:text-gray-100">{formatMinutes(queue)}</p>
                      <p className="mt-2 text-gray-500">{t('now')}</p>
                    </div>
                    <p className="mt-1 text-gray-500">
                      {t('averageLast2Hours')}: <span className="font-semibold">{formatMinutes(averageQueue)}</span>
                    </p>
                    <p className="mt-1 text-gray-500">
                      {t('averageNext2Hours')}: <span className="font-semibold">{formatMinutes(avgNextTwoHours)}</span>
                    </p>
                    {/* Forecast chart for today */}
                {loadingAverage || loadingForecast ? (
                  <p className="mt-4">{t('loadingChart')}</p>
                ) : (
                  <>
                    <div className="relative mt-6">
                      {/* Past = blue, Prediction = sky‑blue */}
                      <AreaChart
                        className="h-60"
                        data={combinedSeries}
                        index="time"
                        categories={[
                          { key: 'Past', label: t('chart.past') },
                          { key: 'Prediction', label: t('chart.prediction') },
                        ]}
                        colors={['blue', 'violet']}
                        showLegend={true}
                        valueFormatter={(v) =>
                          v === null ? '' : `${Math.round(v as number)} min`
                        }
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
                    {forecastHorizon !== null && (
                      <p className="mt-4 text-sm text-gray-500 dark:text-gray-400">
                        {t('expectedQueueHorizon', { hours: forecastHorizon })}
                      </p>
                    )}
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
                  {t('prediction.title')}
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                  {t('prediction.description')}
                </p>


                {/* DateTime picker */}
                <div className="mt-6 mb-8">
                  <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                    {t('picker.label')}
                  </label>

                    <DatePicker
                      selected={selectedDateTime}
                      onChange={(date: Date) => setSelectedDateTime(date as Date)}
                      showTimeSelect
                      timeIntervals={15}
                      dateFormat={locale === 'da' ? 'dd/MM/yyyy, HH:mm' : 'Pp'}
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
                  {t('historical.title')}
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                  {t('historical.description')}
                </p>
                {loadingHistorical ? (
                  <p>{t('loading')}</p>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {/* Yesterday */}
                    <span className="inline-flex w-48 items-center whitespace-nowrap justify-between gap-2 rounded-md bg-blue-100 dark:bg-blue-900 py-1 pl-2.5 pr-2 text-sm text-gray-800 dark:text-gray-200 ring-1 ring-inset ring-blue-200 dark:ring-blue-800">
                      {t('yesterday')}
                      <span className="h-4 w-px bg-blue-300 dark:bg-blue-700" />
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        {historical.yesterday === null ? t('notAvailable') : formatMinutes(historical.yesterday)}
                      </span>
                    </span>

                    {/* One month ago */}
                    <span className="inline-flex w-48 items-center whitespace-nowrap justify-between gap-2 rounded-md bg-blue-100 dark:bg-blue-900 py-1 pl-2.5 pr-2 text-sm text-gray-800 dark:text-gray-200 ring-1 ring-inset ring-blue-200 dark:ring-blue-800">
                      {t('oneMonthAgo')}
                      <span className="h-4 w-px bg-blue-300 dark:bg-blue-700" />
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        {historical.month === null ? t('notAvailable') : formatMinutes(historical.month)}
                      </span>
                    </span>

                    {/* One year ago */}
                    <span className="inline-flex w-48 items-center whitespace-nowrap justify-between gap-2 rounded-md bg-blue-100 dark:bg-blue-900 py-1 pl-2.5 pr-2 text-sm text-gray-800 dark:text-gray-200 ring-1 ring-inset ring-blue-200 dark:ring-blue-800">
                      {t('oneYearAgo')}
                      <span className="h-4 w-px bg-blue-300 dark:bg-blue-700" />
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        {historical.year === null ? t('notAvailable') : formatMinutes(historical.year)}
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
            <h3 className="mb-3 text-lg font-semibold">{t('selectAnotherAirport')}</h3>
            <select
              className="mt-2 block w-full rounded border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2"
              value={code}
              onChange={e =>
                (window.location.href = `/${locale}/airports/${(e.target as HTMLSelectElement).value}`)
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
                {t('otherProjects')}
                </Link>
            </li>
            <li>
                <a
                href="https://waitport.com/api/v1/all?order=id.desc&limit=100"
                className="mx-2 hover:text-gray-700 dark:hover:text-gray-300"
                target="_blank"
                rel="noopener noreferrer"
              >
                {t('api')}
              </a>
            </li>
            <li>
                <Link
                href="https://github.com/simonottosen/cph-security"
                className="mx-2 hover:text-gray-700 dark:hover:text-gray-300"
                target="_blank"
              >
                {t('github')}
              </Link>
            </li>
          </ul>
          <p className="text-center text-sm">
            {t('madeWith')}
          </p>
          <p className="text-center text-xs">{t('copyright', { year: new Date().getFullYear() })}</p>
        </footer>
      </div>
    </>
  );
};

export default ClientPage;
