import time
from distutils.log import Log

import selenium
from selenium import webdriver
from selenium.webdriver.common.bidi.console import Console
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import trio

d = DesiredCapabilities.CHROME
d['goog:loggingPrefs'] = { 'browser':'ALL' }


async def printConsoleLogs():
    chrome_options = webdriver.ChromeOptions()
    #chrome_options.add_experimental_option("detach", True)
    chrome_options.add_experimental_option('debuggerAddress', 'localhost:9515')
    driver = webdriver.Chrome(options=chrome_options, desired_capabilities=d)
    #driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver', service_args=['--verbose'], options=chrome_options, desired_capabilities=dc)
    #driver = Chrome(executable_path='chromedriver')
    driver.get("https://www.tiktok.com/@evalicious_8910/video/7029269165189975301?_t=8TsbUUQw3XH&_r=1")
    action_list = driver.find_elements(By.CLASS_NAME, "tiktok-1pqxj4k-ButtonActionItem")
    cmt_btn = action_list[1]
    cmt_btn.click()
    time.sleep(1)
    print("Clicked!")
    time.sleep(1)
    print("Executing script...")
    for element in driver.find_elements(By.CLASS_NAME, "tiktok-q9aj5z-PCommentText"):
        print(element.text)



    '''csv = driver.execute_script(open("./ScrapeTikTokComments.js").read())
    print(csv)
    time.sleep(20)
    for entry in driver.get_log('browser'):
        print(entry)'''
'''
def get_browser_log_entries(driver):
    """get log entreies from selenium and add to python logger before returning"""
    loglevels = {'NOTSET': 0, 'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'SEVERE': 40, 'CRITICAL': 50}

    # initialise a logger
    browserlog = logging.getLogger("chrome")
    # get browser logs
    slurped_logs = driver.get_log('browser')
    for entry in slurped_logs:
        # convert broswer log to python log format
        rec = browserlog.makeRecord("%s.%s" % (browserlog.name, entry['source']), loglevels.get(entry['level']),
                                    '.', 0, entry['message'], None, None)
        rec.created = entry['timestamp'] / 1000  # log using original timestamp.. us -> ms
        try:
            # add browser log to python log
            browserlog.handle(rec)
        except:
            print(entry)
    # and return logs incase you want them
    return slurped_logs

def demo():
    caps = webdriver.DesiredCapabilities.CHROME.copy()
    caps['goog:loggingPrefs'] = {'browser': 'ALL'}
    driver = webdriver.Chrome(desired_capabilities=caps)

    driver.get("http://localhost")

    consolemsgs = get_browser_log_entries(driver)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)7s:%(message)s')
    logging.info("start")
    demo()
    logging.info("end")```'''
trio.run(printConsoleLogs)