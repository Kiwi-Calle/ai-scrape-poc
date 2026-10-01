import os
import re
import sys
import uuid
import datetime
import asyncio
import logging
from typing import List, Optional, Literal
from urllib.parse import quote_plus, urlparse

from flask import Flask, jsonify
from google.cloud import bigquery
from google import genai
from google.genai import types
from pydantic import BaseModel
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# --- Configure Python Logging for Cloud Run ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ai_overview_poc")

app = Flask(__name__)

# --- Hardcoded Prompts for POC ---
PROMPTS = [
    {
        "client_id": "client_alpha",
        "keyword": "best enterprise crm software",
        "target_brand": "Salesforce",
    },
    {
        "client_id": "client_beta",
        "keyword": "top rank tracking tools",
        "target_brand": "Ahrefs",
    },
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

# Google's AI Overview markup is not a stable API; these selectors WILL break eventually.
AIO_SELECTOR = 'div[data-attrid="wa_overview"], div.s8bR9d'


# --- Pydantic Data Schemas ---
class BrandMention(BaseModel):
    brand_name: str
    rank_position: int
    sentiment: Literal["POSITIVE", "NEUTRAL", "NEGATIVE"]


class BrandExtraction(BaseModel):
    """What the LLM is expected to extract and return."""
    brand_mentions: List[BrandMention]


class ExtractedAIOverview(BaseModel):
    """What the parser hands back to the endpoint."""
    ai_overview_present: bool
    raw_text: str
    brand_mentions: List[BrandMention]
    target_brand_found: bool = False
    target_brand_rank: Optional[int] = None


# --- Helpers ---
def domain_of(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def normalize_mentions(mentions: List[BrandMention]) -> List[BrandMention]:
    """Dedupe by name and renumber ranks in code, so we never trust the LLM's numbering."""
    seen = set()
    cleaned: List[BrandMention] = []
    for m in mentions:
        key = m.brand_name.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(
            BrandMention(
                brand_name=m.brand_name.strip(),
                rank_position=len(cleaned) + 1,
                sentiment=m.sentiment,
            )
        )
    return cleaned


def evaluate_target_brand(mentions: List[BrandMention], target_brand: str):
    """Whole-word, case-insensitive match, so 'Salesforce Sales Cloud' counts as 'Salesforce'."""
    if not target_brand:
        return False, None
    pattern = re.compile(rf"\b{re.escape(target_brand.strip())}\b", re.IGNORECASE)
    for mention in mentions:  # already in rank order
        if pattern.search(mention.brand_name):
            return True, mention.rank_position
    return False, None


# --- Method 1: Playwright Scraper ---
async def dismiss_consent(page) -> None:
    """EU IPs are often redirected to a Google consent page before any results."""
    if "consent.google" not in page.url:
        return
    logger.info("[Playwright] Consent page detected, dismissing...")
    for label in ("Reject all", "Accept all"):
        button = page.get_by_role("button", name=label)
        if await button.count() > 0:
            await button.first.click()
            await page.wait_for_load_state("domcontentloaded")
            return
    logger.warning("[Playwright] Consent page found but no known button matched")


async def scrape_google_aio(keyword: str) -> dict:
    logger.info(f"[Playwright] Launching browser for keyword: '{keyword}'")
    aio_text = ""
    citations = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            try:
                context = await browser.new_context(
                    user_agent=USER_AGENT,
                    locale="en-US",
                    viewport={"width": 1366, "height": 900},
                )
                page = await context.new_page()

                search_url = f"https://www.google.com/search?q={quote_plus(keyword)}&hl=en&gl=us"
                logger.info(f"[Playwright] Navigating to: {search_url}")
                await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
                await dismiss_consent(page)

                if "/sorry/" in page.url:
                    logger.error(f"[Playwright] Google CAPTCHA/block page for '{keyword}': {page.url}")
                    return {"raw_text": "", "citations": []}

                try:
                    await page.wait_for_selector(AIO_SELECTOR, timeout=10000)
                    container = page.locator(AIO_SELECTOR).first
                    aio_text = (await container.inner_text()).strip()
                    logger.info(f"[Playwright] AI Overview found! Length: {len(aio_text)} chars")

                    for link in await container.locator('a[href^="http"]').all():
                        href = await link.get_attribute("href")
                        if not href:
                            continue
                        domain = domain_of(href)
                        if domain and not domain.endswith("google.com"):
                            citations.append({"domain": domain, "url": href})
                    logger.info(f"[Playwright] Extracted {len(citations)} citations")
                except PlaywrightTimeoutError:
                    logger.warning(
                        f"[Playwright] No AI Overview for '{keyword}' (not shown, or selector outdated). "
                        f"URL: {page.url}"
                    )
            finally:
                await browser.close()
    except Exception as e:
        logger.error(f"[Playwright] Scraper error for '{keyword}': {e}", exc_info=True)
        return {"raw_text": "", "citations": []}

    unique_citations = list({c["url"]: c for c in citations}.values())
    return {"raw_text": aio_text, "citations": unique_citations}


# --- Method 2: Direct Gemini Grounded Search API ---
def fetch_gemini_grounded_api(keyword: str) -> dict:
    logger.info(f"[Gemini API] Executing Grounded Search for: '{keyword}'")
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"What are the best options for: {keyword}?",
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
        )

        raw_text = response.text or ""
        logger.info(f"[Gemini API] Grounded response received ({len(raw_text)} chars)")

        citations = []
        if response.candidates and response.candidates[0].grounding_metadata:
            chunks = response.candidates[0].grounding_metadata.grounding_chunks or []
            for chunk in chunks:
                if chunk.web and chunk.web.uri:
                    # chunk.web.uri is a vertexaisearch redirect link, so every URL would
                    # share the same host. The real site name is in chunk.web.title.
                    domain = (chunk.web.title or "").strip() or domain_of(chunk.web.uri)
                    citations.append({"domain": domain, "url": chunk.web.uri})

        return {
            "raw_text": raw_text,
            "citations": list({c["url"]: c for c in citations}.values()),
        }
    except Exception as e:
        logger.error(f"[Gemini API] Grounded Search failed for '{keyword}': {e}", exc_info=True)
        return {"raw_text": "", "citations": []}


# --- Entity Parser ---
def parse_structured_entities(keyword: str, raw_text: str, target_brand: str) -> ExtractedAIOverview:
    if not raw_text:
        logger.info(f"[Parser] Raw text is empty for '{keyword}'. Skipping parsing.")
        return ExtractedAIOverview(ai_overview_present=False, raw_text="", brand_mentions=[])

    logger.info(f"[Parser] Extracting entities for '{keyword}'...")
    try:
        client = genai.Client()
        prompt = f"""Search query: "{keyword}

Extract every distinct brand or product name recommended or mentioned in the text below, in order of first mention.
- rank_position: order of first mention (first = 1)
- sentiment: POSITIVE, NEUTRAL or NEGATIVE, based on how the text describes that brand

<text>
{raw_text}
</text>"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=BrandExtraction,
                temperature=0.0,
            ),
        )

        parsed = response.parsed
        if parsed is None:
            logger.warning(f"[Parser] response.parsed is None. Raw response: {response.text!r}")
            parsed = BrandExtraction.model_validate_json(response.text)

        mentions = normalize_mentions(parsed.brand_mentions)
        logger.info(f"[Parser] Parsed {len(mentions)} brand mentions.")

        found, rank = evaluate_target_brand(mentions, target_brand)
        return ExtractedAIOverview(
            ai_overview_present=True,  # decided by code: we have text
            raw_text=raw_text,
            brand_mentions=mentions,
            target_brand_found=found,
            target_brand_rank=rank,
        )

    except Exception as e:
        logger.error(f"[Parser] Entity parsing failed for '{keyword}': {e}", exc_info=True)
        # The text exists; only the extraction failed
        return ExtractedAIOverview(ai_overview_present=True, raw_text=raw_text, brand_mentions=[])


def build_row(item: dict, platform: str, parsed: ExtractedAIOverview, citations: list) -> dict:
    return {
        "run_id": str(uuid.uuid4()),
        "client_id": item["client_id"],
        "target_brand": item.get("target_brand", ""),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "keyword": item["keyword"],
        "platform": platform,
        "ai_overview_present": parsed.ai_overview_present,
        "target_brand_found": parsed.target_brand_found,
        "target_brand_rank": parsed.target_brand_rank,
        "brand_mentions": [bm.model_dump() for bm in parsed.brand_mentions],
        "citations": citations,
        "raw_text": parsed.raw_text,
    }


# --- Core Execution Endpoint ---
@app.route("/run-check", methods=["GET", "POST"])
def run_check():
    logger.info("=== Starting new run-check job ===")
    dataset_id = os.environ.get("BQ_DATASET", "ai_insights")
    table_id = os.environ.get("BQ_TABLE", "raw_prompt_runs")
    bq_client = bigquery.Client()
    rows_to_insert = []

    for idx, item in enumerate(PROMPTS, 1):
        keyword = item["keyword"]
        target_brand = item.get("target_brand", "")
        logger.info(f"--- Prompt {idx}/{len(PROMPTS)}: '{keyword}' (Target: {target_brand}) ---")

        # 1. Playwright scrape
        scrape_res = asyncio.run(scrape_google_aio(keyword))
        parsed_scrape = parse_structured_entities(keyword, scrape_res["raw_text"], target_brand)
        rows_to_insert.append(
            build_row(item, "google_ai_overview", parsed_scrape, scrape_res["citations"])
        )

        # 2. Gemini grounded search
        api_res = fetch_gemini_grounded_api(keyword)
        parsed_api = parse_structured_entities(keyword, api_res["raw_text"], target_brand)
        rows_to_insert.append(
            build_row(item, "gemini_grounded_api", parsed_api, api_res["citations"])
        )

    table_ref = f"{bq_client.project}.{dataset_id}.{table_id}"
    logger.info(f"[BigQuery] Streaming {len(rows_to_insert)} rows to '{table_ref}'...")

    try:
        errors = bq_client.insert_rows_json(table_ref, rows_to_insert)
        if errors:
            logger.error(f"[BigQuery] Insert failed with errors: {errors}")
            return jsonify({"status": "error", "details": errors}), 500

        logger.info("[BigQuery] Insertion successful!")
        return jsonify({
            "status": "success",
            "total_rows_inserted": len(rows_to_insert),
            "processed_prompts": len(PROMPTS),
        }), 200

    except Exception as bq_err:
        logger.error(f"[BigQuery] Fatal error streaming to BigQuery: {bq_err}", exc_info=True)
        return jsonify({"status": "error", "message": str(bq_err)}), 500


if __name__ == "__main__":
    logger.info("Starting Flask Server...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
