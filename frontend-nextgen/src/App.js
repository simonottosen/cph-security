import React, { useState, useEffect } from "react";
import axios from "axios";
import { Helmet } from "react-helmet";
import DateTime from "react-datetime";
import "react-datetime/css/react-datetime.css";
import Container from "react-bootstrap/Container";
import Dropdown from "react-bootstrap/Dropdown";
import DropdownButton from "react-bootstrap/DropdownButton";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";
import 'moment/locale/da';
import { Link } from 'react-router-dom';

const API_URL = process.env.REACT_APP_API_HOST || "/api/v1/predict";

function App() {
  const [selectedAirport, setSelectedAirport] = useState("cph");
  const [queue, setQueue] = useState(null);
  const [averageQueue, setAverageQueue] = useState(null);
  const [selectedDateTime, setSelectedDateTime] = useState(new Date());
  const [predictedQueueLength, setPredictedQueueLength] = useState(null);

  // Update the document title when the component mounts
  useEffect(() => {
    document.title = 'Waitport - Airport Security Queue';
  }, []);

  const handleAirportChange = (eventKey) => {
    setSelectedAirport(eventKey);
  };

  const handleDateTimeChange = (moment) => {
    setSelectedDateTime(moment.toDate());
  };

  // Fetch the most recent queue information
  useEffect(() => {
    const fetchQueueInformation = async () => {
      try {
        const response = await axios.get(
          `https://waitport.com/api/v1/all?airport=eq.${selectedAirport.toUpperCase()}&limit=1&select=queue&order=id.desc`
        );
        setQueue(response.data[0].queue);
      } catch (error) {
        console.error(error);
      }
    };
    fetchQueueInformation();
  }, [selectedAirport]);

  // Fetch the average queue information over the last 24 entries (approx. 2 hours if updated every 5 min)
  useEffect(() => {
    const fetchQueueAverageInformation = async () => {
      try {
        const response = await axios.get(
          `https://waitport.com/api/v1/all?airport=eq.${selectedAirport.toUpperCase()}&select=queue&limit=24&order=id.desc`
        );
        const queueValues = response.data.map((data) => data.queue);
        const averageQueue = Math.round(
          queueValues.reduce((total, value) => total + value, 0) / queueValues.length
        );
        setAverageQueue(averageQueue);
      } catch (error) {
        console.error(error);
      }
    };
    fetchQueueAverageInformation();
  }, [selectedAirport]);

  // Fetch predicted queue length for the selected date/time
  useEffect(() => {
    const fetchPredictedQueueLength = async () => {
      try {
        const dateTimeString = selectedDateTime
          .toISOString()
          .slice(0, 16)
          .replace("T", "T");
        const response = await axios.get(
          `https://waitport.com${API_URL}?timestamp=${dateTimeString}&airport=${selectedAirport.toLowerCase()}`
        );
        setPredictedQueueLength(response.data.predicted_queue_length_minutes);
      } catch (error) {
        console.error(error);
      }
    };
    fetchPredictedQueueLength();
  }, [selectedAirport, selectedDateTime]);

  // Determine airport name
  let airportName;
  if (selectedAirport === 'cph') {
    airportName = 'Copenhagen Airport';
  }  else if (selectedAirport === 'arn') {
    airportName = 'Stockholm Airport';
  } else if (selectedAirport === 'dus') {
    airportName = 'Düsseldorf Airport';
  } else if (selectedAirport === 'fra') {
    airportName = 'Frankfurt Airport';
  } else if (selectedAirport === 'muc') {
    airportName = 'Munich Airport';
  } else if (selectedAirport === 'lhr') {
    airportName = 'London Heathrow Airport';
  } else if (selectedAirport === 'ams') {
    airportName = 'Amsterdam Airport';
  } else if (selectedAirport === 'dub') {
    airportName = 'Dublin Airport';
  } else if (selectedAirport === 'ist') {
    airportName = 'Istanbul Airport';
  }

  // Formatting the date/time output
  const day = selectedDateTime.getDate();
  const month = selectedDateTime.toLocaleString('default', { month: 'long' });
  const hour = selectedDateTime.getHours();
  const minute = selectedDateTime.getMinutes().toString().padStart(2, '0');

  const formattedDate = `${day}${getOrdinalSuffix(day)} of ${month} at ${hour}:${minute}`;

  function getOrdinalSuffix(day) {
    const suffixes = ['th', 'st', 'nd', 'rd'];
    const lastDigit = day % 10;
    // Handle teens (11th, 12th, 13th)
    if (day % 100 >= 11 && day % 100 <= 13) {
      return 'th';
    }
    // Use suffixes array for 1st, 2nd, 3rd, etc.
    return suffixes[lastDigit] || 'th';
  }

  return (
    <>
      {/* Helmet for dynamic SEO tags */}
      <Helmet>
        <title>Waitport - Real-time & Predicted Airport Security Queues</title>
        <meta
          name="description"
          content={`Check live and predicted security queue wait times at ${airportName}. Plan your trip effectively with Waitport's real-time data and future estimates.`}
        />
        <link rel="canonical" href="https://waitport.com" />
      </Helmet>

      <Container fluid="sm" className="bg-light p-5 md">
        <h1 className="text-center">
          Waitport 🛫
        </h1>
        <h4 className="text-center mb-5">Real-Time & Predicted Airport Security Queues</h4>
        <div className="container">
          <div className="row justify-content-start">
            <div className="col-12">
              <h2 className="mb-3">About Waitport</h2>
              <p className="lead">
                Welcome to <strong>Waitport</strong>! Here, you can track security waiting times 
                across major European airports in real-time. We also provide 
                conservative queue <strong>predictions</strong> for future dates and times. 
                This helps you plan your trip more effectively and avoid unexpected delays. 
                <br />
                <br />
                Start by selecting an airport below, then choose a date and time for your expected travel 
                to see our predicted queue. <span role="img" aria-label="globe">🌏</span> 
                <br />
                <br />
                Safe travels!
              </p>
              <hr />
            </div>

            <div className="col-lg-4 col-sm-6">
              <h2 className="mb-3">Select Airport</h2>
              <DropdownButton
                id="airport-select"
                title={airportName}
                onSelect={handleAirportChange}
              >
                <Dropdown.Item eventKey="cph">🇩🇰 Copenhagen Airport</Dropdown.Item>
                <Dropdown.Item eventKey="arn">🇸🇪 Stockholm Arlanda Airport</Dropdown.Item>
                <Dropdown.Item eventKey="dus">🇩🇪 Düsseldorf International Airport</Dropdown.Item>
                <Dropdown.Item eventKey="fra">🇩🇪 Frankfurt Airport</Dropdown.Item>
                <Dropdown.Item eventKey="muc">🇩🇪 Munich Airport</Dropdown.Item>
                <Dropdown.Item eventKey="lhr">🇬🇧 London Heathrow Airport</Dropdown.Item>
                <Dropdown.Item eventKey="ams">🇳🇱 Amsterdam Schipol Airport</Dropdown.Item>
                <Dropdown.Item eventKey="dub">🇮🇪 Dublin Airport</Dropdown.Item>
                <Dropdown.Item eventKey="ist">🇹🇷 Istanbul Airport</Dropdown.Item>
              </DropdownButton>
            </div>

            <div className="col-lg-8 col-md-12">
              {queue !== null && (
                <div className="mt-4">
                  <p className="lead">
                    <strong>Current Queue</strong>: The wait time at {airportName} 
                    is currently <strong>{queue}</strong> minutes.
                    <br />
                    <small className="text-muted">
                      Over the last several entries (approx. 2 hours), 
                      the <strong>average</strong> queue has been <strong>{averageQueue}</strong> minutes.
                    </small>
                  </p>
                </div>
              )}
            </div>
          </div>

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
                    <strong>Predicted Queue</strong>: We estimate 
                    <strong> {predictedQueueLength}</strong> minutes 
                    of waiting at {airportName} on {formattedDate}.
                  </p>
                </div>
              )}
            </div>
          </div>

          <div className="b-example-divider"></div>

          <div className="container">
            <footer className="py-3 my-4">
              <ul className="nav justify-content-center border-bottom pb-3 mb-3">
                <li className="nav-item">
                  <a
                    href="https://simonottosen.dk/"
                    className="nav-link px-2 text-muted"
                  >
                    Other projects
                  </a>
                </li>
                <li className="nav-item">
                  <a
                    href="https://waitport.com/api/v1/all?order=id.desc&limit=100"
                    className="nav-link px-2 text-muted"
                  >
                    API
                  </a>
                </li>
                <li className="nav-item">
                  <a
                    href="https://github.com/simonottosen/cph-security"
                    className="nav-link px-2 text-muted"
                  >
                    GitHub
                  </a>
                </li>
              </ul>
              <p className="text-center text-muted">
                Made with{" "}
                <Link to="/pant" style={{ textDecoration: "none" }}>
                  ❤️
                </Link>{" "}
                by Simon Ottosen
              </p>
            </footer>
          </div>
        </div>
      </Container>
    </>
  );
}

export default App;