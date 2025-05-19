/* eslint-disable @next/next/no-head-import-in-app-directory */
'use client';

import "./globals.css";
import Head from 'next/head';
import Link from 'next/link';
import Script from 'next/script';

// Define TypeScript type for airport codes
type AirportCode =
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

// Map airport codes to their display names
const airportNames: Record<AirportCode, string> = {
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

const Home: React.FC = () => {
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
            Built for Travellers
          </p>
          <h1 className="text-5xl md:text-6xl font-extrabold text-gray-900 dark:text-gray-100 tracking-tight mb-6">
            Waitport&nbsp;🛫
          </h1>
          <p className="max-w-2xl mx-auto text-xl md:text-2xl text-gray-600 dark:text-gray-300">
            Real‑time &amp; predicted security queue times at Europe’s busiest airports
          </p>
        </section>

        <div className="h-px w-full bg-gray-200 dark:bg-gray-800" />

        <section className="w-full bg-gray-50 dark:bg-gray-800 py-8">
          <div className="flex justify-center px-4">
            <div className="w-full max-w-3xl bg-white dark:bg-gray-900/40 rounded-xl shadow-sm ring-1 ring-gray-200 dark:ring-gray-800 border border-gray-100 dark:border-gray-700 p-6">
              <h2 className="text-2xl md:text-3xl font-semibold text-gray-800 dark:text-gray-100 mb-2">About Waitport</h2>
              <p className="text-lg">
                Welcome to <strong>Waitport</strong>! Here, you can track security
                waiting times across major European airports in real-time. We also
                provide conservative queue <strong>predictions</strong> for future
                dates and times. This helps you plan your trip more effectively and
                avoid unexpected delays.
                <br />
                <br />
                Select an airport below to see the current and predicted security
                queue times.{' '}
                queue times. <span role="img" aria-label="globe">🌏</span>
                <br />
                <br />
                Safe travels!
              </p>
            </div>
          </div>
        </section>

        <section className="w-full bg-white dark:bg-gray-900 py-8">
          <div className="flex justify-center px-4">
            <div className="w-full max-w-3xl bg-white dark:bg-gray-900/40 rounded-xl shadow-sm ring-1 ring-gray-200 dark:ring-gray-800 border border-gray-100 dark:border-gray-700 p-6">
              <h2 className="text-2xl md:text-3xl font-semibold text-gray-800 dark:text-gray-100 mb-2">Select airport</h2>
              <ul className="divide-y divide-gray-200 dark:divide-gray-700">
                {Object.entries(airportNames).map(([code, name]) => (
                  <li key={code} className="p-4 hover:bg-indigo-50 dark:hover:bg-gray-700 transition-colors">
                    <Link href={`/airports/${code}`} className="text-decoration-none">
                      {name}
                    </Link>
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
                  Other projects
                </a>
              </li>
              <li className="nav-item">
                <a
                  href="https://waitport.com/api/v1/all?order=id.desc&limit=100"
                  className="mx-2 text-gray-600 hover:text-gray-800"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  API
                </a>
              </li>
              <li className="nav-item">
                <a
                  href="https://github.com/simonottosen/cph-security"
                  className="mx-2 text-gray-600 hover:text-gray-800"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  GitHub
                </a>
              </li>
            </ul>
            <p className="text-center text-sm text-gray-500 dark:text-gray-400">
              Made with <span role="img" aria-label="heart">❤️</span> by Simon Ottosen
            </p>
          </footer>
        </div>
      </div>
    </>
  );
};

export default Home;