# from selenium.webdriver import Chrome
# from selenium.webdriver.chrome.service import Service as ChromeService

# url = 'https://www.swedavia.com/arlanda/security-control/'
# service = ChromeService() 
# with Chrome(service=service) as driver:
#     driver.get(url)

#     element = driver.find_element(by='css selector', value='#waitingtimes > div > div > div:nth-child(3) > div > div > div > div:nth-child(3) > div.terminalQueueTime')
#     waiting_time = element.text.strip()

# print(waiting_time)


from selenium.webdriver import Firefox
from selenium.webdriver.firefox.service import Service as FirefoxService
# Will also need standalone Selenium Container - docker run -d -p 4444:4444 -p 7900:7900 --shm-size="2g" selenium/standalone-firefox:4.8.1-20230306


def select_first_char(string):
    if string[0].isdigit():
        return int(string[0])
    else:
        raise ValueError("First character is not an integer.")

url = 'http://localhost:4444/wd/hub' # Replace <container_ip> with actual IP of your Docker container running selenium Firefox
service = FirefoxService(executable_path='/usr/bin/geckodriver', port=4444) # Update executable path based on where geckodriver is installed in your docker container & matching port number
with Firefox(service=service) as driver:
    driver.get('https://www.swedavia.com/arlanda/security-control/')

    element = driver.find_element(by='css selector', value='#waitingtimes > div > div > div:nth-child(3) > div > div > div > div:nth-child(3) > div.terminalQueueTime')
    waiting_time = element.text.strip()

try:
    result = select_first_char(waiting_time)
except ValueError as e:
    print(f"Error: {e}")

print(result)
