from playwright.async_api import async_playwright

browser = None
page = None

async def start_browser(headless: bool = False):
    global browser, page
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=headless)
    page = await browser.new_page()
    print("Browser started!")

async def login(url: str, username: str, password: str):
    global page
    await page.goto(url)
    await page.wait_for_load_state("domcontentloaded")
    await page.get_by_placeholder("Enter your email").fill(username)
    await page.locator("input[type='password']").fill(password)
    await page.get_by_role("button", name="Login").click()

    # Wait for URL to change away from login page
    #await page.wait_for_url(lambda url: "login" not in url, timeout=10000)
    await page.wait_for_load_state("networkidle")
    print("Logged in successfully!")

async def take_screenshot_dom():
    global page
    screenshot_bytes = await page.screenshot(full_page=True)
    dom = []
    elements = await page.query_selector_all("button, a, input, select, textarea")
    for el in elements:
        dom.append({
            "type": el.evaluate("e => e.tagName.toLowerCase()"),
            "text": el.inner_text() or el.get_attribute("value") or "",
            "id": el.get_attribute("id"),
            "classes": el.get_attribute("class")
        })
    return screenshot_bytes, dom

async def get_radio_inputs():
    global page
    radios = await page.locator("input[type='radio']").all()
    for i, radio in enumerate(radios):
        # Get the exact next h6 after each radio
        label = await radio.locator("xpath=following::h6[1]").inner_text()
        print(f"Radio {i}: value={await radio.get_attribute('value')} | label={label}")
    return radios

async def click_radio_by_label(label_text: str):
    global page
    radios = await page.locator("input[type='radio']").all()
    for i, radio in enumerate(radios):
        label = await radio.locator("xpath=following::h6[1]").inner_text()
        if label_text.lower() in label.lower():
            await radio.click()
            print(f"Clicked radio with label: {label}")
            break

    # Find and click Select & Continue button
    await page.get_by_role("button", name="Select & Continue").click()
    await page.wait_for_load_state("networkidle")
    print("Clicked Select & Continue!")

async def get_sidebar_links():
    global page
    await page.wait_for_load_state("networkidle")

    # Wait for sidebar to appear
    await page.wait_for_selector("nav, aside, [class*='sidebar'], [class*='side-bar'], [id*='sidebar']")

    # Get all links inside sidebar
    links = await page.locator("nav a, aside a, [class*='sidebar'] a, [class*='side-bar'] a, [id*='sidebar'] a").all()
    
    sidebar_links = []
    for i, link in enumerate(links):
        text = await link.inner_text()
        href = await link.get_attribute("href")
        if text.strip():  # skip empty links
            sidebar_links.append({"text": text.strip(), "href": href})
            print(f"Link {i}: text={text.strip()} | href={href}")

    return sidebar_links

async def close_browser():
    global browser
    await browser.close()
    print("Browser closed!")

