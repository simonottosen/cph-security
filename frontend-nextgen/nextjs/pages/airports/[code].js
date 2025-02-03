// pages/airports/[code].js

import { useState, useEffect } from "react";
import axios from "axios";
import Head from "next/head";
import DateTime from "react-datetime";
import Container from "react-bootstrap/Container";
import Dropdown from "react-bootstrap/Dropdown";
import DropdownButton from "react-bootstrap/DropdownButton";
import Link from 'next/link';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'react-datetime/css/react-datetime.css';
import 'moment/locale/da';

const API_URL = process.env.NEXT_PUBLIC_API_HOST || "/api/v1/predict";

// Define your airports with their codes and names
const airportNames = {
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

export async function getStaticPaths() {
  const paths = Object.keys(airportNames).map(code => ({
    params: { code },
  }));

  return { paths, fallback: false };
}

export async function getStaticProps({ params }) {
  const { code } = params;
  const airportName = airportNames[code] || 'Unknown Airport';

  return {
    props: {
      code,
      airportName,
    },
  };
}

export default function AirportPage({ code, airportName }) {
  const [queue, setQueue] = useState(null);
  const [averageQueue, setAverageQueue] = useState(null);
  const [selectedDateTime, setSelectedDateTime] = useState(new Date());
  const [predictedQueueLength, setPredictedQueueLength] = useState(null);

  const handleDateTimeChange = (moment) => {
    setSelectedDateTime(moment.toDate());
  };

  useEffect(() => {
    const fetchQueueInformation = async () => {
      try {
        const response = await axios.get(
          `https://waitport.com/api/v1/all?airport=eq.${code.toUpperCase()}&limit=1&select=queue&order=id.desc`
        );
        setQueue(response.data[0]?.queue || '0');
      } catch (error) {
        console.error(error);
      }
    };
    fetchQueueInformation();
  }, [code]);

  useEffect(() => {
    const fetchQueueAverageInformation = async () => {
      try {
        const response = await axios.get(
          `https://waitport.com/api/v1/all?airport=eq.${code.toUpperCase()}&select=queue&limit=24&order=id.desc`
        );
        const queueValues = response.data.map((data) => data.queue);
        const averageQueue = queueValues.length
          ? Math.round(
              queueValues.reduce((total, value) => total + value, 0) /
                queueValues.length
            )
          : '0';
        setAverageQueue(averageQueue);
      } catch (error) {
        console.error(error);
      }
    };
    fetchQueueAverageInformation();
  }, [code]);

  useEffect(() => {
    const fetchPredictedQueueLength = async () => {
      try {
        const dateTimeString = selectedDateTime
          .toISOString()
          .slice(0, 16)
          .replace("T", "T");
        const response = await axios.get(
          `https://waitport.com${API_URL}?timestamp=${dateTimeString}&airport=${code.toLowerCase()}`
        );
        setPredictedQueueLength(
          response.data.predicted_queue_length_minutes || '0'
        );
      } catch (error) {
        console.error(error);
      }
    };
    fetchPredictedQueueLength();
  }, [code, selectedDateTime]);

  // Formatting the date/time output
  const day = selectedDateTime.getDate();
  const month = selectedDateTime.toLocaleString('default', { month: 'long' });
  const hour = selectedDateTime.getHours();
  const minute = selectedDateTime.getMinutes().toString().padStart(2, '0');

  const formattedDate = `${day}${getOrdinalSuffix(day)} of ${month} at ${hour}:${minute}`;

  function getOrdinalSuffix(day) {
    const suffixes = ['th', 'st', 'nd', 'rd'];
    const lastDigit = day % 10;
    if (day % 100 >= 11 && day % 100 <= 13) {
      return 'th';
    }
    return suffixes[lastDigit] || 'th';
  }

  return (
    <>
      <Head>
        <title>{`Waitport - Security Queues at ${airportName}`}</title>
        <meta
          name="description"
          content={`Check live and predicted security queue wait times at ${airportName}. Plan your trip effectively with Waitport's real-time data and future estimates.`}
        />
        <meta property="og:title" content={`Waitport - Security Queues at ${airportName}`} />
        <meta property="og:description" content={`Stay updated with the latest security queue times at ${airportName}. Avoid long waits and plan your journey efficiently.`} />
        <meta property="og:url" content={`https://waitport.com/airports/${code}`} />
        <link rel="canonical" href={`https://waitport.com/airports/${code}`} />
        {/* Structured Data for SEO */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "WebPage",
              "name": `Waitport - Security Queues at ${airportName}`,
              "description": `Real-time and predicted security queue wait times at ${airportName}. Plan your trip effectively with our data.`,
              "url": `https://waitport.com/airports/${code}`,
            }),
          }}
        />
      </Head>

      <Container fluid="sm" className="bg-light p-5">
        <h1 className="text-center">Waitport 🛫</h1>
        <h4 className="text-center mb-5">Real-Time & Predicted Airport Security Queues</h4>

        <div className="container">
          {/* About Section */}
          <div className="row justify-content-start">
            <div className="col-12">
              <h2 className="mb-3">About Waitport</h2>
              <p className="lead">
                Welcome to <strong>Waitport</strong>! Here, you can track security waiting times across major European airports in real-time.
                We also provide conservative queue <strong>predictions</strong> for future dates and times. This helps you plan your trip more effectively and avoid unexpected delays.
                <br /><br />
                Start by selecting an airport below, then choose a date and time for your expected travel to see our predicted queue. <span role="img" aria-label="globe">🌏</span>
                <br /><br />
                Safe travels!
              </p>
              <hr />
            </div>

            {/* Airport Selection Dropdown */}
            <div className="col-lg-4 col-sm-6">
              <h2 className="mb-3">Select Airport</h2>
              <DropdownButton
                id="airport-select"
                title={airportName}
                onSelect={(eventKey) => {
                  // Redirect to the selected airport's page
                  window.location.href = `/airports/${eventKey}`;
                }}
              >
                {Object.entries(airportNames).map(([code, name]) => (
                  <Dropdown.Item key={code} eventKey={code}>
                    {name}
                  </Dropdown.Item>
                ))}
              </DropdownButton>
            </div>

            {/* Current Queue Information */}
            <div className="col-lg-8 col-md-12">
              {queue !== null && (
                <div className="mt-4">
                  <p className="lead">
                    <strong>Current Queue</strong>: The wait time at {airportName} is currently <strong>{queue}</strong> minutes.
                    <br />
                    <small className="text-muted">
                      Over the last several entries (approx. 2 hours), the <strong>average</strong> queue has been <strong>{averageQueue}</strong> minutes.
                    </small>
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Predicted Queue Section */}
          <div className="row mt-5">
            <div className="col-lg-4 col-sm-6">
              <h2 className="mb-3">Select Date &amp; Time</h2>
              <DateTime
                locale="da-dk"
                inputProps={{ id: "datetime-picker" }}
                dateFormat="MM/DD"
                initialValue={selectedDateTime}
                initialViewDate={selectedDateTime}
                initialViewMode="time"
                onChange={handleDateTimeChange}
              />
            </div>

            <div className="col-lg-8 col-md-12">
              {predictedQueueLength !== null && (
                <div className="mt-4">
                  <p className="lead">
                    <strong>Predicted Queue</strong>: We estimate <strong>{predictedQueueLength}</strong> minutes of waiting at {airportName} on {formattedDate}.
                  </p>
                </div>
              )}
            </div>
          </div>

          <div className="b-example-divider"></div>

          {/* Footer Section */}
          <div className="container">
            <footer className="py-3 my-4">
              <ul className="nav justify-content-center border-bottom pb-3 mb-3">
                <li className="nav-item">
                  <a
                    href="https://simonottosen.dk/"
                    className="nav-link px-2 text-muted"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Other projects
                  </a>
                </li>
                <li className="nav-item">
                  <a
                    href="https://waitport.com/api/v1/all?order=id.desc&limit=100"
                    className="nav-link px-2 text-muted"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    API
                  </a>
                </li>
                <li className="nav-item">
                  <a
                    href="https://github.com/simonottosen/cph-security"
                    className="nav-link px-2 text-muted"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    GitHub
                  </a>
                </li>
              </ul>
              <p className="text-center text-muted">
                Made with <span role="img" aria-label="heart">❤️</span> by Simon Ottosen
              </p>
            </footer>
          </div>
        </div>
      </Container>
    </>
  );
}