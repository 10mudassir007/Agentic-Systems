import asyncio
from browser import (start_browser, login, take_screenshot_dom, close_browser, get_radio_inputs,click_radio_by_label,get_sidebar_links)
from utils import process_image
from ai import ask_question


async def main():
    await start_browser()

    await login(
        url="https://api.imis.com.pk:9001",
        username="admin_imis@yopmail.com",
        password="Intra$12345"
    )

    await asyncio.sleep(5)

    screen_shot, dom = await take_screenshot_dom()
    print(screen_shot[:100])
    print("\n\n")
    screenshot_b64 = process_image(screen_shot)
    question = "How do I view all the programs?"
    print("Question: ", question)
    print("")
    response = ask_question(question, screenshot_b64, dom)
    print("Agent Response: ", response.content)
    print("\n\n")
    
    await click_radio_by_label("MUMTA")
    
    side_bar_links = await get_sidebar_links()
    print("Sidebar Links: ", side_bar_links)
    await close_browser()
    

asyncio.run(main())