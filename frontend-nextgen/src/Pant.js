import React, { useState, useEffect } from "react";
import DateTime from "react-datetime";
import "react-datetime/css/react-datetime.css";
import Container from "react-bootstrap/Container";
import Dropdown from "react-bootstrap/Dropdown";
import DropdownButton from "react-bootstrap/DropdownButton";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";
import 'moment/locale/da';
import { Link } from 'react-router-dom';



function App() {
  const [selectedAirport, setSelectedAirport] = useState("fields");
  const [queue, setQueue] = useState(null);
  const [averageQueue, setAverageQueue] = useState(null);
  const [selectedDateTime, setSelectedDateTime] = useState(new Date());

  useEffect(() => {
    document.title = 'Pantport - Fields Pant Security Queue';
  }, []);

  const handleAirportChange = (eventKey) => {
    setSelectedAirport(eventKey);
  };

  const handleDateChange = (date) => {
    setSelectedDateTime(date.toDate());
  };


  const getTimeStatement = () => {
    const hours = selectedDateTime.getHours();
    if (hours >= 10 && hours < 11) return "It's early in the day but probably still bad";
    else if (hours >= 11 && hours < 18) return "Very likely the worst experience of your life";
    else return "It's closed at this time";
  };
  



  useEffect(() => {
    const date = new Date();
    const currentHour = date.getHours();
    console.log(`Current Hour: ${currentHour}`);
    if (currentHour >= 10 && currentHour < 11) {
      setAverageQueue(".. to early to say it's not even been open for an hour");
      setQueue("already way to long");
    } 
    else if (currentHour >= 11 && currentHour < 12) {
      setAverageQueue(".. to early to say it's only been open for an hour");
      setQueue("it looks too long. Worst experience ever likely");
    }
    else if (currentHour >= 12 && currentHour <= 18) {
      setAverageQueue("quite long as well");
      setQueue("way too long. Worst experience ever");
    }else {
      setAverageQueue("well, it's been closed");
      setQueue("not existing, since it's closed");

    }
  }, [selectedAirport]);
  

        

  let airportName;

  if (selectedAirport === 'fields') {
    airportName = 'Fields Pantstation';
  }




  return (

    <Container fluid="sm" className="bg-light p-5 md">
      <h1 className="text-center mb-5"><b>Pant</b>port 🥤🗑️</h1>
      <div className="container">
        <div className="row justify-content-start">
          <div className="col-12">
            <p className="lead">Hey, welcome to <b><b>Pant</b>port!</b> <br></br><br></br>You can track the waiting-time in the
              pantstation only at Fields - we're very likely not adding new pant stations to the
              page. The data is collected in real-time while the prediction model is re-trained continiously and offers a conservative estimate of the queue at a given date and time.  <br></br><br></br>Start by selecting the Fields pantstation, as that's what you're interested in below! 🏬
              <br></br><br></br>
              If you also select a time and date for when you expect to be at Fields, I'll do my best to estimate the queue in the future 🔮
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
              <Dropdown.Item eventKey="fields">🇩🇰🗑️ Fields Pantstation</Dropdown.Item>
            </DropdownButton>

          </div>
          <div className="col-lg-8 col-md-12">
            <br class="d-md-none" />
            {queue !== null && (
              <p className="lead">
                The <b>current</b> queue at {airportName} is{" "}
                <strong>{queue}</strong>. <br></br><i>In the last two hours, the average queue has been <strong>{averageQueue}</strong>.</i>
              </p>
            )}

          </div>
        </div>


        <div className="row">

        <div className="col-lg-4 col-sm-6">
           <br/>
           <DateTime
             locale="da-dk"
             inputProps={{ id: "datetime-picker" }}
             dateFormat="MM/DD"
             value={selectedDateTime}
             onChange={handleDateChange}
             initialViewMode="time"
           />
         </div>
         <div className="col-lg-8 col-md-12">
          <br/>
            <p className="lead">
              <strong>{getTimeStatement()}.</strong> 
            </p>
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
            <p class="text-center text-muted">Made with <Link to="/pant">❤️</Link> by Simon Ottosen</p>
          </footer>
        </div>

      </div>




    </Container>

  );
}

export default App;
