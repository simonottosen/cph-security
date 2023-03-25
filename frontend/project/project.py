from flask import Flask, render_template, request
import requests

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    airport_options = ['cph', 'osl', 'dus']
    selected_airport = 'cph'

    if request.method == 'POST':
        # Get the form data
        selected_airport = request.form.get('airport')
        date = request.form.get('date_time')

        # Fetch the queue data from API
        url = f'https://waitport.com/api/v1/{selected_airport}?limit=1&select=queue'
        print(url)
        response = requests.get(url)
        if response.status_code == 200:
            queue_data = response.json()[0]
            current_queue = queue_data['queue']
            print(current_queue)
        else:
            current_queue = "N/A"


        # Fetch the estimated waiting time based on the selected data
        url = f'https://waitport.com/api/v1/predict?airport={selected_airport}&timestamp={date}'
        estimate_request = requests.get(url)

        if estimate_request.status_code == 200:
            estimate_data = estimate_request.json()
            estimate = estimate_data['predicted_queue_length_minutes']
        else:
            estimate = "N/A"
    else:
        current_queue = "N/A"
        estimate = "N/A"

    return render_template('index.html', current_queue=current_queue, airport_options=airport_options, estimate=estimate, selected_airport=selected_airport)


if __name__ == '__main__':
    app.run(debug=True)
