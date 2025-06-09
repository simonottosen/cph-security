// pages/airports/[code].js

import { useState, useEffect } from "react";
import axios from "axios";
import Head from "next/head";
import dynamic from "next/dynamic";
import Link from "next/link";
import "react-datetime/css/react-datetime.css";
import "moment/locale/da";
import Script from 'next/script';
import { AreaChart } from "@tremor/react";   


// Dynamically import DateTime to disable SSR for this component
const DateTime = dynamic(
  () =>
    import("react-datetime")
      .then((mod) => mod.default || mod),
  {
    ssr: false,
    loading: () => <span>Loading date picker...</span>,
  }
);

const API_URL = process.env.NEXT_PUBLIC_API_HOST || "/api/v1/predict";

// Define your airports with their codes and names
const airportNames = {
  cph: "🇩🇰 Copenhagen Airport",
  arn: "🇸🇪 Stockholm Airport",
  dus: "🇩🇪 Düsseldorf Airport",
  fra: "🇩🇪 Frankfurt Airport",
  muc: "🇩🇪 Munich Airport",
  lhr: "🇬🇧 London Heathrow Airport",
  ams: "🇳🇱 Amsterdam Airport",
  dub: "🇮🇪 Dublin Airport",
  ist: "🇹🇷 Istanbul Airport",
};

const airportNamesText = {
  cph: "Copenhagen Airport",
  arn: "Stockholm Airport",
  dus: "Düsseldorf Airport",
  fra: "Frankfurt Airport",
  muc: "Munich Airport",
  lhr: "London Heathrow Airport",
  ams: "Amsterdam Airport",
  dub: "Dublin Airport",
  ist: "Istanbul Airport",
};

export async function getStaticPaths() {
  const paths = Object.keys(airportNames).map((code) => ({
    params: { code },
  }));

  return { paths, fallback: false };
}

export async function getStaticProps({ params }) {
  const { code } = params;
  const airportName = airportNames[code] || "Unknown Airport";

  return {
    props: {
      code,
      airportName,
    },
  };
}

export default function AirportPage({ code, airportName }) {
  const displayName = airportNamesText[code] || airportName;

  const [queue, setQueue] = useState(null);
  const [averageQueue, setAverageQueue] = useState(null);
  const [loadingQueue, setLoadingQueue] = useState(true);
  const [loadingAverage, setLoadingAverage] = useState(true);
  const [selectedDateTime, setSelectedDateTime] = useState(() => new Date(Date.now() + 2 * 60 * 60 * 1000));
  const [predictedQueueLength, setPredictedQueueLength] = useState(null);
  const [loadingPredicted, setLoadingPredicted] = useState(true);

  // 6‑hour forecast data for the Tremor chart
  const [forecastData, setForecastData] = useState([]);
  const [loadingForecast, setLoadingForecast] = useState(true);

  const handleDateTimeChange = (momentObj) => {
    setSelectedDateTime(momentObj.toDate());
  };

  useEffect(() => {
    const fetchQueueInformation = async () => {
      try {
        setLoadingQueue(true);
        const response = await axios.get(
          `https://waitport.com/api/v1/all?airport=eq.${code.toUpperCase()}&limit=1&select=queue&order=id.desc`
        );
        setQueue(response.data[0]?.queue || "0");
      } catch (error) {
        console.error(error);
      } finally {
        setLoadingQueue(false);
      }
    };
    fetchQueueInformation();
  }, [code]);

  useEffect(() => {
    const fetchQueueAverageInformation = async () => {
      try {
        setLoadingAverage(true);
        const response = await axios.get(
          `https://waitport.com/api/v1/all?airport=eq.${code.toUpperCase()}&select=queue&limit=24&order=id.desc`
        );
        const queueValues = response.data.map((data) => data.queue);
        const average = queueValues.length
          ? Math.round(
              queueValues.reduce((total, value) => total + value, 0) /
                queueValues.length
            )
          : "0";
        setAverageQueue(average);
      } catch (error) {
        console.error(error);
      } finally {
        setLoadingAverage(false);
      }
    };
    fetchQueueAverageInformation();
  }, [code]);

  useEffect(() => {
    const fetchPredictedQueueLength = async () => {
      try {
        setLoadingPredicted(true);
        const dateTimeString = selectedDateTime
          .toISOString()
          .slice(0, 16)
          .replace("T", "T");
        const response = await axios.get(
          `https://waitport.com${API_URL}?timestamp=${dateTimeString}&airport=${code.toLowerCase()}`
        );
        setPredictedQueueLength(
          response.data.predicted_queue_length_minutes || "0"
        );
      } catch (error) {
        console.error(error);
      } finally {
        setLoadingPredicted(false);
      }
    };
    fetchPredictedQueueLength();
  }, [code, selectedDateTime]);

  // Fetch full forecast series (mean, q30, q70) for the line/area chart
  useEffect(() => {
    const fetchForecast = async () => {
      try {
        setLoadingForecast(true);
        const response = await axios.get(
          `http://127.0.0.1:5000/forecast/${code.toLowerCase()}`
        );
        const preds = response.data?.predictions ?? [];
        const formatted = preds.map((p) => ({
          timestamp: new Date(p.timestamp).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
          mean: p.mean,
          q30: p.q30,
          q70: p.q70,
        }));
        setForecastData(formatted);
      } catch (error) {
        console.error(error);
      } finally {
        setLoadingForecast(false);
      }
    };
    fetchForecast();
  }, [code]);

  // Helper function for pluralization
  const formatMinutes = (minutes) => {
    return `${minutes} ${minutes === 1 ? "minute" : "minutes"}`;
  };

  // Formatting the date/time output
  const day = selectedDateTime.getDate();
  const month = selectedDateTime.toLocaleString("default", { month: "long" });
  const hour = selectedDateTime.getHours();
  const minute = selectedDateTime.getMinutes().toString().padStart(2, "0");

  const formattedDate = `${day}${getOrdinalSuffix(day)} of ${month} at ${hour}:${minute}`;
  const now = new Date();
  const diffMs = selectedDateTime - now;
  const diffMinutes = Math.round(diffMs / 60000);
  let timeDiffText = "";
  if (diffMinutes === 0) {
    timeDiffText = "at this time";
  } else if (diffMinutes > 0) {
    if (diffMinutes < 60) {
      timeDiffText = `in ${diffMinutes} minutes`;
    } else {
      const diffHours = Math.round(diffMinutes / 60);
      timeDiffText = `in ${diffHours} hours`;
    }
  }

  function getOrdinalSuffix(day) {
    const suffixes = ["th", "st", "nd", "rd"];
    const lastDigit = day % 10;
    if (day % 100 >= 11 && day % 100 <= 13) {
      return "th";
    }
    return suffixes[lastDigit] || "th";
  }

  return (
    <>
      <Head>
        <link rel="icon" href="/favicon.ico" />
        <title>{`Waitport - Security Queues at ${airportName}`}</title>
        <meta
          name="description"
          content={`Check live and predicted security queue wait times at ${airportName}. Plan your trip effectively with Waitport's real-time data and future estimates.`}
        />
        <meta
          name="keywords"
          content={`airport security queue, real-time wait times, ${airportName}, travel planning, Waitport, security wait predictions`}
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta
          property="og:title"
          content={`Waitport - Security Queues at ${airportName}`}
        />
        <meta
          property="og:description"
          content={`Stay updated with the latest security queue times at ${airportName}. Avoid long waits and plan your journey efficiently.`}
        />
        <meta property="og:type" content="website" />
        <meta property="og:url" content={`https://waitport.com/airports/${code}`} />
        <meta property="og:image" content={`https://waitport.com/images/${code}.jpg`} />
        <link rel="canonical" href={`https://waitport.com/airports/${code}`} />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "WebPage",
              "name": `Waitport - Security Queues at ${airportName}`,
              "description": `Real-time and predicted security queue wait times at ${airportName}. Plan your trip effectively with our data.`,
              "url": `https://waitport.com/airports/${code}`,
              "mainEntity": {
                "@type": "Service",
                "serviceType": "Security Queue Information",
                "provider": {
                  "@type": "Organization",
                  "name": "Waitport",
                  "url": "https://waitport.com",
                },
                "areaServed": {
                  "@type": "Airport",
                  "name": airportName,
                  "iataCode": code.toUpperCase(),
                },
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

      <div className="bg-gray-50 p-5">
        <header>
          <h1 className="text-center">Waitport 🛫</h1>
          <h4 className="text-center mb-5">
            Real-Time &amp; Predicted Airport Security Queues
          </h4>
        </header>

        <div className="max-w-5xl mx-auto">

          <section aria-labelledby="queue-overview" className="mt-4">
            <div className="grid lg:grid-cols-2 gap-6">
              <div>
                <div className="bg-white shadow rounded-lg p-6 mb-4" aria-live="polite">
                  <h2 className="text-xl font-semibold mb-4">Current Security Queue</h2>
                  {(loadingQueue || loadingAverage) ? (
                    <p>Loading...</p>
                  ) : (
                    <p className="text-lg">
                      The wait time at {displayName} is currently <strong>{formatMinutes(queue)}</strong>.
                      <br />
                      <small className="text-muted">
                        Over the last 2 hours, the <strong>average</strong> queue has been <strong>{formatMinutes(averageQueue)}</strong>.
                      </small>
                    </p>
                  )}
                </div>
              </div>
              <div>
                <div className="bg-white shadow rounded-lg p-6 mb-4" aria-live="polite">
                  <h2 className="text-xl font-semibold mb-4">Predicted Security Queue</h2>
                  {loadingPredicted ? (
                    <p>Loading...</p>
                  ) : (
                    <p className="text-lg">
                      We estimate <strong>{formatMinutes(predictedQueueLength)}</strong> of waiting at {displayName} {timeDiffText}.
                    </p>
                  )}
                  <div className="mt-3">
                    <h5 className="mb-2">Select Date &amp; Time</h5>
                    <div className="flex items-center space-x-2">
                      <span id="datetime-addon" aria-label="calendar icon">📅</span>
                      <DateTime
                        locale="da-dk"
                        inputProps={{
                          id: "datetime-picker",
                          "aria-label": "Select Date and Time",
                          className: "w-full border rounded px-3 py-2"
                        }}
                        dateFormat="MM/DD"
                        timeFormat="HH:mm"
                        closeOnSelect={true}
                        value={selectedDateTime}
                        onChange={handleDateTimeChange}
                      />
                    </div>
                  </div>
                  {loadingForecast ? (
                    <p className="mt-4">Loading forecast...</p>
                  ) : (
                    <AreaChart
                      className="h-72 mt-4"
                      data={forecastData}
                      index="timestamp"
                      categories={["q70", "q30", "mean"]}
                      colors={["sky", "sky", "indigo"]}
                      stack={false}
                      showLegend={false}
                      valueFormatter={(v) => `${Math.round(v)} min`}
                    />
                  )}
                </div>
              </div>
            </div>
          </section>

          <hr className="my-4" />

          <section aria-labelledby="about-waitport">
            <div>
              <h2 id="about-waitport" className="mb-3">
                About Waitport
              </h2>
              <p className="text-lg">
                Welcome to <strong>Waitport</strong>! Here, you can track
                security waiting times across major European airports in
                real-time. We also provide conservative queue{" "}
                <strong>predictions</strong> for future dates and times. This
                helps you plan your trip more effectively and avoid unexpected
                delays.
                <br />
                <br />
                Start by selecting an airport below, then choose a date and
                time for your expected travel to see our predicted queue.{" "}
                <span role="img" aria-label="globe">
                  🌏
                </span>
                <br />
                <br />
                Safe travels!
              </p>
            </div>
          </section>
          <hr className="my-4" />

          <section aria-labelledby="select-airport">
            <div className="grid lg:grid-cols-2 gap-6">
              <div>
                <h2 id="select-airport" className="mb-3">
                  Select another airport
                </h2>
                <select
                  className="mt-2 block w-full rounded border-gray-300 p-2"
                  value={code}
                  onChange={(e) => {
                    window.location.href = `/airports/${e.target.value}`;
                  }}
                  aria-label="Select Airport"
                >
                  {Object.entries(airportNames).map(([c, name]) => (
                    <option key={c} value={c}>
                      {name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </section>

          <div className="b-example-divider"></div>

          <footer className="py-3 my-4">
            <ul className="flex justify-center border-b pb-3 mb-3">
              <li>
                <a
                  href="https://simonottosen.dk/"
                  className="mx-2 text-gray-600 hover:text-gray-800"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Other Projects
                </a>
              </li>
              <li>
                <a
                  href="https://waitport.com/api/v1/all?order=id.desc&limit=100"
                  className="mx-2 text-gray-600 hover:text-gray-800"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  API Documentation
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/simonottosen/cph-security"
                  className="mx-2 text-gray-600 hover:text-gray-800"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  GitHub Repository
                </a>
              </li>
            </ul>
            <p className="text-center text-muted">
              Made with{" "}
              <span role="img" aria-label="heart">
                ❤️
              </span>{" "}
              by Simon Ottosen
            </p>
            <p className="text-center text-muted">
              &copy; {new Date().getFullYear()} Waitport. All rights reserved.
            </p>
          </footer>
        </div>
      </div>
    </>
  );
}
