
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--window-size=1920,1080')
options.add_experimental_option('excludeSwitches', ['enable-automation'])

driver = webdriver.Chrome(options=options)
driver.get('https://www.charusat.ac.in/course-single')
WebDriverWait(driver, 20).until(
    lambda d: d.find_elements(By.CSS_SELECTOR, 'h4')
)
time.sleep(2)

h4s = driver.find_elements(By.CSS_SELECTOR, 'h4')
h4 = h4s[2]  # skip first two nav h4s, grab a real course one
print('H4 text:', h4.text)

# Print the tag names of the next 5 siblings
result = driver.execute_script('''
    var h = arguments[0];
    var sib = h.nextElementSibling;
    var tags = [];
    for (var i = 0; i < 5 && sib; i++) {
        tags.push(sib.tagName + \" class=\" + sib.className);
        sib = sib.nextElementSibling;
    }
    return tags;
''', h4)
print('Siblings:', result)

# Also print the parent tag
parent = driver.execute_script('return arguments[0].parentElement.tagName + \" > \" + arguments[0].parentElement.parentElement.tagName', h4)
print('Parent chain:', parent)

driver.quit()
