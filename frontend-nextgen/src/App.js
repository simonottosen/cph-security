import React, { useState, useEffect } from "react";
import axios from "axios";
import DateTime from "react-datetime";
import "react-datetime/css/react-datetime.css";
import Container from "react-bootstrap/Container";
import Dropdown from "react-bootstrap/Dropdown";
import DropdownButton from "react-bootstrap/DropdownButton";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";
import 'moment/locale/da';



function App() {
  const [selectedAirport, setSelectedAirport] = useState("cph");
  const [queue, setQueue] = useState(null);
  const [averageQueue, setAverageQueue] = useState(null);
  const [selectedDateTime, setSelectedDateTime] = useState(new Date());
  const [predictedQueueLength, setPredictedQueueLength] = useState(null);

  useEffect(() => {
    document.title = 'Waitport - Airport Security Queue';
  }, []);

  const handleAirportChange = (eventKey) => {
    setSelectedAirport(eventKey);
  };

  const handleDateTimeChange = (moment) => {
    setSelectedDateTime(moment.toDate());
  };

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


  useEffect(() => {
    const fetchQueueAverageInformation = async () => {
      try {
        const response = await axios.get(
          `https://waitport.com/api/v1/all?airport=eq.${selectedAirport.toUpperCase()}&select=queue&limit=24&order=id.desc`
        );
        const queueValues = response.data.map(data => data.queue);
        const averageQueue = Math.round(queueValues.reduce((total, value) => total + value, 0) / queueValues.length);
        setAverageQueue(averageQueue);
      } catch (error) {
        console.error(error);
      }
    };
    fetchQueueAverageInformation();
  }, [selectedAirport]);


  useEffect(() => {
    const fetchPredictedQueueLength = async () => {
      try {
        const dateTimeString = selectedDateTime
          .toISOString()
          .slice(0, 16)
          .replace("T", "T");
        const response = await axios.get(
          `https://waitport.com/api/v1/predict?timestamp=${dateTimeString}&airport=${selectedAirport.toLowerCase()}`
        );
        setPredictedQueueLength(response.data.predicted_queue_length_minutes);
      } catch (error) {
        console.error(error);
      }
    };
  
    fetchPredictedQueueLength();
  }, [selectedAirport, selectedDateTime]);
  

  let airportName;

  if (selectedAirport === 'osl') {
    airportName = 'Oslo Airport';
  }

  if (selectedAirport === 'cph') {
    airportName = 'Copenhagen Airport';
  }

  if (selectedAirport === 'dus') {
    airportName = 'Düsseldorf Airport';
  }

  if (selectedAirport === 'arn') {
    airportName = 'Stockholm Airport';
  }

  if (selectedAirport === 'ber') {
    airportName = 'Berlin Airport';
  }

  if (selectedAirport === 'ams') {
    airportName = 'Amsterdam Airport';
  }

  if (selectedAirport === 'dub') {
    airportName = 'Dublin Airport';
  }

  const day = selectedDateTime.getDate();
  const month = selectedDateTime.toLocaleString('default', { month: 'long' });
  const hour = selectedDateTime.getHours();
  const minute = selectedDateTime.getMinutes().toString().padStart(2, '0');

  const formattedDate = `${day}${getOrdinalSuffix(day)} of ${month} at ${hour}:${minute}`;

  function getOrdinalSuffix(day) {
    const suffixes = ['th', 'st', 'nd', 'rd'];
    const lastDigit = day % 10;
    return suffixes[(day % 100 - 10) in [11, 12, 13] ? 0 :
      (lastDigit > 3) ? 0 : lastDigit];
  }



  return (

    <Container fluid="sm" className="bg-light p-5 md">
      <h1 className="text-center mb-5"><b>Wait</b>port 🛫</h1>
      <div className="container">
        <div className="row justify-content-start">
          <div className="col-12">
            <p className="lead">Hey, welcome to <b><b>Wait</b>port!</b> <br></br><br></br>You can track the waiting-time in the
              security across various European airports - we're constantly adding new airports to the
              page. The data is collected in real-time while the prediction model is re-trained continiously and offers a conservative estimate of the queue at a given date and time.  <br></br><br></br>Start by selecting the airport you're interested in below! 🌏
              <br></br><br></br>
              If you also select a time and date for when you expect to be in the airport, I'll do my best to estimate the queue in the future 🔮
            </p>
            <br></br>
            <hr></hr>
            <br></br>
          </div>
          <div className="col-lg-4 col-sm-6">

            <DropdownButton
              id="airport-select"
              title={airportName}
              onSelect={handleAirportChange}>
              <Dropdown.Item eventKey="cph">🇩🇰 Copenhagen Airport</Dropdown.Item>
              <Dropdown.Item eventKey="osl">🇳🇴 Oslo Gardermoen Airport</Dropdown.Item>
              <Dropdown.Item eventKey="arn">🇸🇪 Stockholm Arlanda Airport</Dropdown.Item>
              <Dropdown.Item eventKey="dus">🇩🇪 Düsseldorf International Airport</Dropdown.Item>
              <Dropdown.Item eventKey="ber">🇩🇪 Berlin Brandenburg Airport</Dropdown.Item>
              <Dropdown.Item eventKey="ams">🇳🇱 Amsterdam Schipol Airport</Dropdown.Item>
              <Dropdown.Item eventKey="dub">🇮🇪 Dublin Airport</Dropdown.Item>
            </DropdownButton>

          </div>
          <div className="col-lg-8 col-md-12">
            <br class="d-md-none" />
            {queue !== null && (
              <p className="lead">
                The <b>current</b> queue at {airportName} is{" "}
                <strong>{queue}</strong> minutes. <br></br><i>In the last two hours, the average queue has been <strong>{averageQueue}</strong> minutes.</i>
              </p>
            )}

          </div>
        </div>


        <div className="row">

          <div className="col-lg-4 col-sm-6">
            <br></br>

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
            <br></br>

            {predictedQueueLength !== null && (
              <p className="lead">
                The <b>predicted</b> queue length is{" "}
                <strong>{predictedQueueLength}</strong> minutes at{" "}
                {formattedDate}.
              </p>
            )}
          </div>
        </div>

        <div class="b-example-divider"></div>

        <div class="container">
          <footer class="py-3 my-4">
            <ul class="nav justify-content-center border-bottom pb-3 mb-3">
              <li class="nav-item"><a href="https://simonottosen.dk/" class="nav-link px-2 text-muted">Other projects</a></li>
              <li class="nav-item"><a href="https://waitport.com/api/v1/all?order=id.desc&limit=100" class="nav-link px-2 text-muted">API</a></li>
              <li class="nav-item"><a href="https://github.com/simonottosen/cph-security" class="nav-link px-2 text-muted">GitHub</a></li>
            </ul>
            <p class="text-center text-muted">Made with ❤️ by Simon Ottosen</p>
          </footer>
        </div>

      </div>




    </Container>

  );
}

export default App;
