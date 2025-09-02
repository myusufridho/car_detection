from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time, os, random

opt = Options()
opt.add_argument("--headless=new")
driver = webdriver.Chrome(options=opt)

driver.get("https://euphonious-concha-ab5c5d.netlify.app/")
time.sleep(3)

xpaths = {
    "front_left":  "//button[contains(text(),'Front Left Door')]",
    "front_right": "//button[contains(text(),'Front Right Door')]",
    "rear_left":   "//button[contains(text(),'Rear Left Door')]",
    "rear_right":  "//button[contains(text(),'Rear Right Door')]",
    "hood":        "//button[contains(text(),'Hood')]"
}

os.makedirs("dataset", exist_ok=True)

def random_drag():
    canvas = driver.find_element(By.TAG_NAME, "canvas")
    action = ActionChains(driver)
    dx = random.randint(-500, 150)
    dy = random.randint(-250, 100)
    action.move_to_element(canvas).click_and_hold().move_by_offset(dx, dy).release().perform()
    time.sleep(0.5)

def snap(label):
    driver.execute_script("""
        let p = document.querySelector('div[style*="position: absolute"]');
        if (p) p.style.display = 'none';
    """)
    time.sleep(0.2)
    fname = f"dataset/{label}_{int(time.time()*1000)}.png"
    driver.save_screenshot(fname)
    print("->", fname)
    driver.execute_script("""
        let p = document.querySelector('div[style*="position: absolute"]');
        if (p) p.style.display = 'block';
    """)
    time.sleep(0.2)

TOTAL = 500
N = TOTAL // (len(xpaths) * 2)

for k, xp in xpaths.items():
    try:
        el = driver.find_element(By.XPATH, xp)
        for _ in range(N):
            random_drag()
            snap(f"{k}_closed")
            el.click(); time.sleep(1.2)
            random_drag()
            snap(f"{k}_open")
            el.click(); time.sleep(1.2)
    except Exception as e:
        print("error", k, e)

driver.quit()
