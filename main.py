import os
import uuid
import datetime
import asyncio
from flask import Flask, jsonify
from google.cloud import bigquery
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List
from playwright.async_api import async_playwright

app = Flask(__name__)

# --- Hardcoded Prompts for POC ---
PROMPTS = [
    {
        "client_id": "client_alpha",
        "keyword": "best enterprise crm software",
        "target_brand": "Salesforce"
    },
    {
        "client_id": "client_beta",
        "keyword": "top rank tracking tools",
        "target_brand": "Ahrefs"
    }
]

# --- Pydantic Data Schemas ---
class BrandMention(BaseModel):
    brand_name: str
    rank_position: int
    sentiment: str

class ExtractedAIOverview(BaseModel):
    ai_overview_present: bool
    raw_text: str
    brand_mentions: List[BrandMention]


# --- Method 1: Playwright Scraper ---
async def scrape_google_aio(keyword: str) -> dict:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0.0.0"
        )
        page = await context.new_page()

        search_url = f"https://www.google.com/search?q={keyword.replace(' ', '+')}&hl=en"
        await page.goto(search_url, wait_until="domcontentloaded")

        aio_text = ""
        citations = []

        try:
            # Wait for Google AI Overview container
            await page.wait_for_selector('div[data-attrid="wa_overview"], div.MjjYud', timeout=5000)
            container = page.locator('div[data-attrid="wa_overview"]').first
            
            if await container.count() > 0:
                aio_text = await container.inner_text()
                links = await container.locator('a[href^="http"]').all()
                for link in links:
                    href = await link.get_attribute('href')
                    if href and 'google.com' not in href:
                        domain = href.split('/')[2] if '/' in href else href
                        citations.append({"domain": domain, "url": href})
        except Exception:
            aio_text = ""

        await browser.close()
        unique_citations = list({c['url']: c for c in citations}.values())
        return {"raw_text": aio_text, "citations": unique_citations}


# --- Method 2: Direct Gemini Grounded Search API ---
def fetch_gemini_grounded_api(keyword: str) -> dict:
    client = genai.Client()
    
    # Force Gemini to run a live Google Search
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"What are the best options for: {keyword}?",
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    )
    
    citations = []
    if response.candidates and response.candidates[0].grounding_metadata:
        chunks = response.candidates[0].grounding_metadata.grounding_chunks
        if chunks:
            for chunk in chunks:
                if chunk.web:
                    url = chunk.web.uri
                    domain = url.split('/')[2] if '/' in url else url
                    citations.append({"domain": domain, "url": url})

    return {
        "raw_text": response.text or "",
        "citations": list({c['url']: c for c in citations}.values())
    }


# --- Entity Parser: Uses Gemini Flash with Pydantic ---
def parse_structured_entities(keyword: str, raw_text: str) -> ExtractedAIOverview:
    if not raw_text:
        return ExtractedAIOverview(ai_overview_present=False, raw_text="", brand_mentions=[])

    client = genai.Client()
    prompt = f"""
    Analyze the following text for the search query: "{keyword}".
    Extract all brand or product names in order of mention rank (first mentioned = 1), 
    along with sentiment (POSITIVE, NEUTRAL, NEGATIVE).
    Text: {raw_text}
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=ExtractedAIOverview
        )
    )
    return ExtractedAIOverview.model_validate_json(response.text)


# --- Core Execution Endpoint ---
@app.route("/run-check", methods=["GET", "POST"])
def run_check():
    dataset_id = os.environ.get("BQ_DATASET", "ai_insights")
    table_id = os.environ.get("BQ_TABLE", "raw_prompt_runs")
    bq_client = bigquery.Client()
    rows_to_insert = []

    for item in PROMPTS:
        keyword = item["keyword"]
        client_id = item["client_id"]

        # -------------------------------------------------------------
        # RUN CHECK 1: Playwright Web Scraper (Google AI Overview)
        # -------------------------------------------------------------
        scrape_res = asyncio.run(scrape_google_aio(keyword))
        parsed_scrape = parse_structured_entities(keyword, scrape_res["raw_text"])

        rows_to_insert.append({
            "run_id": str(uuid.uuid4()),
            "client_id": client_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "keyword": keyword,
            "platform": "google_ai_overview",
            "ai_overview_present": parsed_scrape.ai_overview_present,
            "brand_mentions": [bm.model_dump() for bm in parsed_scrape.brand_mentions],
            "citations": scrape_res["citations"]
        })

        # -------------------------------------------------------------
        # RUN CHECK 2: Direct Gemini Grounded Search API
        # -------------------------------------------------------------
        api_res = fetch_gemini_grounded_api(keyword)
        parsed_api = parse_structured_entities(keyword, api_res["raw_text"])

        rows_to_insert.append({
            "run_id": str(uuid.uuid4()),
            "client_id": client_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "keyword": keyword,
            "platform": "gemini_grounded_api",
            "ai_overview_present": True if api_res["raw_text"] else False,
            "brand_mentions": [bm.model_dump() for bm in parsed_api.brand_mentions],
            "citations": api_res["citations"]
        })

    # --- Stream Both Checks directly into BigQuery ---
    table_ref = f"{bq_client.project}.{dataset_id}.{table_id}"
    errors = bq_client.insert_rows_json(table_ref, rows_to_insert)

    if errors:
        return jsonify({"status": "error", "details": errors}), 500

    return jsonify({
        "status": "success", 
        "total_rows_inserted": len(rows_to_insert),
        "processed_prompts": len(PROMPTS)
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))