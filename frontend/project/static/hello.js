const airportSelect = document.getElementById('airportSelect');
const dateSelect = document.getElementById('dateSelect');
const calculateBtn = document.getElementById('calculateBtn');

airportSelect.addEventListener('change', fetchAndUpdateEstimate);
dateSelect.addEventListener('change', fetchAndUpdateEstimate);


async function fetchAndUpdateEstimate() {
    // Get the selected values from the selects
    const selectedAirport = airportSelect.value;
    const selectedDateTime = dateSelect.value;

    // Fetch the estimated waiting time based on the selected data
    const url = `https://waitport.com/api/v1/predict?airport=${selectedAirport}&timestamp=${selectedDateTime}`;
    const response = await fetch(url, {
        method: 'GET',
        headers: {'Content-Type': 'application/json'}
    });
    if (response.status === 200) {
        const data = await response.json();
        const estimate = data.predicted_queue_length_minutes;
        updateEstimate(estimate);
    } else {
        console.log(response.statusText);
    }
}

function updateEstimate(estimate) {
    const estimateEl = document.querySelector('#displayText b');
    estimateEl.textContent = estimate;
}

calculateBtn.addEventListener('click', (event) => {
    event.preventDefault();
    fetchAndUpdateEstimate();
});
