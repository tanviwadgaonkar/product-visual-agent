"""Capture README screenshots of the running app with Playwright (used in CI).

    python scripts/screenshots.py http://127.0.0.1:8000/app/ docs/screenshots
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

base, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 900})

    page.goto(base + "#/")
    page.fill("#q", "leather handbag")
    page.press("#q", "Enter")
    page.wait_for_selector("article img")
    page.wait_for_load_state("networkidle")
    page.screenshot(path=out / "search.png")

    first = page.locator("article a").first.get_attribute("href")
    for box in page.locator("article input[type=checkbox]").all()[:3]:
        box.check()

    page.goto(base + first)
    page.wait_for_selector("text=Visually similar")
    page.wait_for_load_state("networkidle")
    page.screenshot(path=out / "product.png", full_page=True)

    page.goto(base + "#/compare")
    page.wait_for_selector("table")
    page.wait_for_load_state("networkidle")
    page.screenshot(path=out / "compare.png")

    browser.close()
print("saved", sorted(p.name for p in out.iterdir()))
