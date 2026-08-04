import asyncio
import logging
from typing import Any, Dict, Optional

from playwright.async_api import async_playwright, Browser, Page, Playwright

from run_registry import registry

logger = logging.getLogger(__name__)


class BrowserManager:
    _instance: Optional["BrowserManager"] = None
    _playwright: Optional[Playwright] = None
    _browser: Optional[Browser] = None
    _pages: Dict[str, Page] = {}
    _cdp_sessions: Dict[str, Any] = {}

    def __new__(cls) -> "BrowserManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self) -> None:
        if self._browser is not None:
            return
        p = await async_playwright().start()
        try:
            b = await p.chromium.launch(headless=False, args=["--start-maximized"])
        except Exception:
            await p.stop()
            raise
        self._playwright = p
        self._browser = b
        logger.info("Browser launched")

    async def get_page(self, run_id: str) -> Page:
        await self.initialize()
        if run_id not in self._pages:
            ctx = await self._browser.new_context(viewport={"width": 1280, "height": 720})
            page = await ctx.new_page()
            self._pages[run_id] = page
            logger.info("Created page for run %s", run_id)
        return self._pages[run_id]

    async def start_screencast(self, run_id: str) -> None:
        page = self._pages.get(run_id)
        if not page:
            return
        if run_id in self._cdp_sessions:
            return

        try:
            cdp = await page.context.new_cdp_session(page)

            def make_handler(rid: str):
                def handler(params: Dict[str, Any]) -> None:
                    data = params.get("data", "")
                    session_id = params.get("sessionId")
                    asyncio.create_task(self._publish_and_ack(rid, data, session_id, cdp))
                return handler

            cdp.on("Page.screencastFrame", make_handler(run_id))
            await cdp.send("Page.startScreencast", {
                "format": "jpeg",
                "quality": 70,
                "everyNthFrame": 1,
            })
            self._cdp_sessions[run_id] = cdp
            logger.info("Screencast started for run %s", run_id)
        except Exception as exc:
            logger.warning("Failed to start screencast for run %s: %s", run_id, exc)

    async def _publish_and_ack(self, run_id: str, data: str, session_id: Any, cdp: Any) -> None:
        try:
            await registry.publish_frame(run_id, data)
            await cdp.send("Page.screencastFrameAck", {"sessionId": session_id})
        except Exception:
            pass

    async def stop_screencast(self, run_id: str) -> None:
        cdp = self._cdp_sessions.pop(run_id, None)
        if cdp:
            try:
                await cdp.send("Page.stopScreencast")
                await cdp.detach()
            except Exception:
                pass
            logger.info("Screencast stopped for run %s", run_id)

    async def close_page(self, run_id: str) -> None:
        await self.stop_screencast(run_id)
        page = self._pages.pop(run_id, None)
        if page is not None:
            ctx = page.context
            await page.close()
            await ctx.close()
            logger.info("Closed page for run %s", run_id)

    async def close_all(self) -> None:
        for rid in list(self._cdp_sessions.keys()):
            await self.stop_screencast(rid)
        for p in self._pages.values():
            await p.close()
        self._pages.clear()
        self._cdp_sessions.clear()
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("All browser resources released")
