/* eslint-disable @next/next/no-head-import-in-app-directory */
'use client';

import "./globals.css";
import Head from 'next/head';
import Link from 'next/link';
import Script from 'next/script';
import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { SparkAreaChart } from "@/components/SparkChart";
import { useI18n } from "@/i18n/I18nProvider";

// Define TypeScript type for airport codes
type AirportCode =
  | 'cph'
  | 'edi'
  | 'arn'
  | 'dus'
  | 'fra'
  | 'muc'
  | 'lhr'
  | 'ams'
  | 'dub'
  | 'ist';

// Map airport codes to their display names
const airportNames: Record<AirportCode, string> = {
  cph: '🇩🇰 Copenhagen Airport',
  edi: '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Edinburgh Airport',
  arn: '🇸🇪 Stockholm Airport',
  dus: '🇩🇪 Düsseldorf Airport',
  fra: '🇩🇪 Frankfurt Airport',
  muc: '🇩🇪 Munich Airport',
  lhr: '🇬🇧 London Heathrow Airport',
  ams: '🇳🇱 Amsterdam Airport',
  dub: '🇮🇪 Dublin Airport',
  ist: '🇹🇷 Istanbul Airport',
};
/* ------------------------------------------------------------------ */
/* Small sparkline shown on the front page for each airport           */
/* ------------------------------------------------------------------ */
type SparkPoint = {
  time: string;
  Queue: number;
};

const AirportSparkline: React.FC<{ code: AirportCode }> = ({ code }) => {
  const [data, setData] = useState<SparkPoint[]>([]);
  const [forecastData, setForecastData] = useState<{ time: string; Prediction: number }[]>([]);
  const currentQueue = data.length ? data[data.length - 1].Queue : null;

  // Compare current queue with the final forecast point
  const lastForecast = forecastData.length
    ? forecastData[forecastData.length - 1].Prediction
    : null;

  // Positive → queue expected to rise
  const trend =
    currentQueue !== null && lastForecast !== null
      ? lastForecast - currentQueue
      : 0;

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get<{ queue: number; timestamp: string }[]>(
          `https://waitport.com/api/v1/all?airport=eq.${code.toUpperCase()}&select=queue,timestamp&limit=24&order=id.desc`,
        );
        const series = res.data.map((d) => ({
          time: new Date(d.timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          }),
          Queue: d.queue,
        }));
        setData(series.reverse());          // chronological order
      } catch {
        setData([]);                        // fallback → empty spark
      }
    };

    fetchData();
  }, [code]);

  useEffect(() => {
    const fetchForecast = async () => {
      try {
        const res = await axios.get<{ predictions: { timestamp: string; mean: number }[] }>(
          `https://waitport.com/api/v1/forecast/${code}`,
        );
        const future = res.data.predictions ?? [];
        const formatted = future.slice(0, 8).map((p) => {
          const d = new Date(p.timestamp);
          d.setHours(d.getHours() + 2); // shift to CPH local time
          return {
            time: d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            Prediction: Math.max(0, p.mean),
          };
        });
        setForecastData(formatted);
      } catch {
        setForecastData([]);
      }
    };
    fetchForecast();
  }, [code]);

  const chartSeries = useMemo(() => {
    // Smooth transition (3t² − 2t³) between last real point and forecast
    const smoothstep = (t: number) => 3 * t * t - 2 * t * t * t;

    const lastQueueValue = data.length ? data[data.length - 1].Queue : null;

    const blendedFuture = forecastData.map((p, idx) => {
      if (lastQueueValue === null) {
        return { time: p.time, Prediction: p.Prediction };
      }
      const blend = smoothstep((idx + 1) / forecastData.length); // 0‒1
      const value = Math.max(
        0,
        lastQueueValue * (1 - blend) + p.Prediction * blend,
      );
      return { time: p.time, Prediction: value };
    });

    return [
      ...data.map((d) => ({ time: d.time, Queue: d.Queue })),
      ...blendedFuture,
    ];
  }, [data, forecastData]);

  // Memoized: dot vertical position aligned to chart zero-baseline
  const dotTopPct = useMemo(() => {
    if (currentQueue === null) return 50;
    // Recharts SparkAreaChart uses 0 as the baseline, so the Y‑domain is [0, max]
    const max = Math.max(
      currentQueue,
      ...chartSeries.flatMap((d) =>
        Object.keys(d)
          .filter((k) => k !== 'time')
          .map((k) => (d as any)[k] as number | null)
          .filter((v): v is number => v !== null && !Number.isNaN(v)),
      ),
    );
    if (max === 0) return 95; // flat-at‑zero ⇒ dot at bottom
    const min = 0; // baseline
    // If the series is essentially flat (max ≈ currentQueue with < 1 % variance):
    // • For a flat non‑zero line, dot should sit on the line (top of the chart) → ~0 %.
    // • For a flat zero line we already handled above (max === 0).
    if (Math.abs(max - currentQueue) < 0.01 * max) {
      return 37; // offset aligns dot with flat non‑zero line
    }
    const yRatio = (currentQueue - min) / (max - min); // 0 → bottom, 1 → top
    return (1 - yRatio) * 100;
  }, [chartSeries, currentQueue]);

  // Memoized: ratio for the transition between real and forecast
  const transitionRatio = useMemo(
    () =>
      data.length > 1 && chartSeries.length > 1
        ? (data.length - 1) / (chartSeries.length - 1)
        : 0,
    [data, chartSeries],
  );

  if (!data.length) {
    return <span className="text-xs text-gray-400">—</span>;
  }

  return (
    <div className="flex items-center space-x-1">
      <div className="relative">
        <SparkAreaChart
          data={chartSeries}
          categories={['Queue', 'Prediction']}
          index="time"
          colors={['blue', 'violet']}
          className="h-8 w-20 sm:h-8 sm:w-28"
        />
        <span
          className="absolute block h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-blue-600 dark:bg-blue-400"
          style={{
            top: `${dotTopPct}%`,
            left: `${(transitionRatio * 100).toFixed(2)}%`,
          }}
        />
      </div>
      <span className="inline-flex items-center gap-0.5 rounded-md bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-800 dark:bg-blue-900 dark:text-blue-200">
        {currentQueue} min
        {lastForecast !== null && (
          <span
            className={
              trend > 0.5
                ? 'text-red-600 dark:text-red-400'
                : trend < -0.5
                ? 'text-green-600 dark:text-green-400'
                : 'text-gray-500 dark:text-gray-400'
            }
          >
            {trend > 0.5 ? '↑' : trend < -0.5 ? '↓' : '→'}
          </span>
        )}
      </span>
    </div>
  );
};

const Home: React.FC = () => {
  const { t, locale } = useI18n();
  return (
    <>
      {/* Head for SEO optimization */}
      <Head>
        <title>Waitport - Real-time &amp; Predicted Airport Security Queues</title>
        <meta
          name="description"
          content="Check live and predicted security queue wait times at major European airports. Plan your trip effectively with Waitport's real-time data and future estimates."
        />
        <meta
          property="og:title"
          content="Waitport - Real-time &amp; Predicted Airport Security Queues"
        />
        <meta
          property="og:description"
          content="Check live and predicted security queue wait times at major European airports. Plan your trip effectively with Waitport's real-time data and future estimates."
        />
        <meta property="og:url" content="https://waitport.com" />
        <meta property="og:type" content="website" />
        <link rel="canonical" href="https://waitport.com" />
        {/* Structured Data for SEO */}
        <link rel="icon" href="/favicon.ico" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              '@context': 'https://schema.org',
              '@type': 'WebSite',
              name: 'Waitport',
              url: 'https://waitport.com',
              description:
                'Check live and predicted security queue wait times at major European airports. Plan your trip effectively with Waitport\'s real-time data and future estimates.',
              potentialAction: {
                '@type': 'SearchAction',
                target: 'https://waitport.com/search?query={search_term_string}',
                'query-input': 'required name=search_term_string',
              },
            }),
          }}
        />
      </Head>

      <Script
        src="https://umami.waitport.com/script.js"
        data-website-id="25e1973f-f0c8-489c-bb41-6726ad81ca4d"
        strategy="afterInteractive"
      />

      <div className="min-h-screen flex flex-col bg-white dark:bg-gray-900">

        <section className="w-full bg-white dark:bg-gray-900 py-8 text-center">
          <p className="text-sm font-semibold uppercase tracking-widest text-indigo-600 dark:text-indigo-400 mb-3">
            {t('builtForTravellers')}
          </p>
          <h1 className="text-5xl md:text-6xl font-extrabold text-gray-900 dark:text-gray-100 tracking-tight mb-6">
            Waitport&nbsp;🛫
          </h1>
          <p className="mx-auto px-5 sm:px-0 max-w-[42ch] sm:max-w-2xl text-lg sm:text-xl md:text-2xl text-gray-600 dark:text-gray-300">
            {t('home.description')}
          </p>
        </section>

        <div className="h-px w-full bg-gray-200 dark:bg-gray-800" />

        <section className="w-full bg-gray-50 dark:bg-gray-800 py-8">
          <div className="flex justify-center px-4">
            <div className="w-full max-w-3xl bg-white dark:bg-gray-900/40 rounded-xl shadow-sm ring-1 ring-gray-200 dark:ring-gray-800 border border-gray-100 dark:border-gray-700 p-6">
              <h2 className="text-2xl md:text-3xl font-semibold text-gray-800 dark:text-gray-100 mb-2">{t('about.title')}</h2>
              <p className="text-lg">
                {t('about.description')}
              </p>
            </div>
          </div>
        </section>

        <section className="w-full bg-white dark:bg-gray-900 py-8">
          <div className="flex justify-center px-4">
            <div className="w-full max-w-3xl bg-white dark:bg-gray-900/40 rounded-xl shadow-sm ring-1 ring-gray-200 dark:ring-gray-800 border border-gray-100 dark:border-gray-700 p-6">
              <h2 className="text-2xl md:text-3xl font-semibold text-gray-800 dark:text-gray-100 mb-2">{t('selectAirport')}</h2>
              <ul className="divide-y divide-gray-200 dark:divide-gray-700">
                {Object.entries(airportNames).map(([code, name]) => (
                <li
                  key={code}
                  className="flex items-center justify-between p-4 hover:bg-indigo-50 dark:hover:bg-gray-700 transition-colors"
                >
                  <Link href={`/${locale}/airports/${code}`} className="flex-1 text-left">
                    {name}
                  </Link>
                  <AirportSparkline code={code as AirportCode} />
                </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        <div className="mt-8 h-px w-full bg-gray-200 dark:bg-gray-800" />

        {/* Footer Section */}
        <div className="max-w-5xl mx-auto">
          <footer className="py-3 my-4 text-gray-500 dark:text-gray-400">
            <ul className="flex justify-center border-b border-gray-200 dark:border-gray-700 pb-3 mb-3">
              <li className="nav-item">
                <a
                  href="https://simonottosen.dk/"
                  className="mx-2 text-gray-600 hover:text-gray-800"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {t('otherProjects')}
                </a>
              </li>
              <li className="nav-item">
                <a
                  href="https://waitport.com/api/v1/all?order=id.desc&limit=100"
                  className="mx-2 text-gray-600 hover:text-gray-800"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {t('api')}
                </a>
              </li>
              <li className="nav-item">
                <a
                  href="https://github.com/simonottosen/cph-security"
                  className="mx-2 text-gray-600 hover:text-gray-800"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {t('github')}
                </a>
              </li>
            </ul>
            <p className="text-center text-sm text-gray-500 dark:text-gray-400">
              {t('madeWith')}
            </p>
          </footer>
        </div>
      </div>
    </>
  );
};

export default Home;
