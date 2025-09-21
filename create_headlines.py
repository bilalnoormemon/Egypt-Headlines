import os
import os
import time
import random
from PyPDF2 import PdfReader, PdfWriter
from google import genai
import pandas as pd

# Keep working directory local to the project
os.chdir("/Users/bilalmemon/Desktop/Egypt-Headlines")
KEY = "AIzaSyAGgY922nyPObSw6gTJx7FD96EFvS3fQ1g"


def Main():
    client = genai.Client(api_key=KEY)

    docs = ["Ansar al-Sunna/al-Hadi al-Nabawi/الهدي النبوي 17.pdf"]

    # ensure runtime directories exist
    os.makedirs("temp", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("datastore", exist_ok=True)

    for doc in docs:
        CreateHeadlines(doc, client)


def CreateHeadlines(doc, client):
    """Split the PDF and find headlines for each page."""
    SplitPDF(doc)
    FindHeadlines(doc, client)


def SplitPDF(input_path):
    reader = PdfReader(f"datastore/{input_path}")
    for i, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)

        output_path = f"temp/{i}.pdf"
        with open(output_path, "wb") as f:
            writer.write(f)


def FindHeadlines(input_path, client):
    # determine number of pages in temp folder
    try:
        x = max(int(f.removesuffix('.pdf')) for f in os.listdir("temp") if f.endswith('.pdf'))
    except ValueError:
        print("No pages found in temp/ - did SplitPDF run?")
        return

    page_list = [i for i in range(1, x + 1)]
    # optional: override for testing
    # page_list = [1]
    headlines_all = []

    for i in page_list:
        headline_i = FindHeadlinePage(i, client)
        headlines_all.append(headline_i)

    df = pd.DataFrame({
        "source": input_path,
        "page": page_list,
        "headline": headlines_all
    })

    long_df = df.explode('headline').reset_index(drop=True)
    long_df.to_csv("output/output.csv", encoding='utf-8-sig', index=False)


def _retry_with_backoff(fn, max_retries=5, base_delay=1.0, max_delay=30.0, jitter=0.3):
    """Helper to retry a callable with exponential backoff and jitter.

    fn: callable that raises on failure.
    Returns the callable's result or raises after exhausting retries.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            # exponential backoff with full jitter
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            # apply jitter
            jitter_factor = random.uniform(1 - jitter, 1 + jitter)
            sleep_time = delay * jitter_factor
            print(f"Attempt {attempt} failed: {e}. Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)


def FindHeadlinePage(i, client):
    """Upload a page PDF and ask Gemini for headlines, with retries/backoff.

    Returns a list of headlines (may be empty). On permanent failure, returns an empty list.
    """
    file_path = f"temp/{i}.pdf"

    def upload():
        return client.files.upload(file=file_path)

    try:
        myfile = _retry_with_backoff(upload)
    except Exception as e:
        print(f"Failed to upload {file_path} after retries: {e}")
        return []

    prompt = (
        "List all the headlines on this page. Separate each headline from each other by '\n'. "
        "If there are no headlines, return an empty list. Do not include any other text in your response."
    )

    def gen_call():
        return client.models.generate_content(
            model="gemini-2.5-flash", contents=[prompt, myfile]
        )

    try:
        response = _retry_with_backoff(gen_call)
    except Exception as e:
        print(f"Failed to call Gemini generate_content for {file_path} after retries: {e}")
        return []

    # Some client libs put result text in .text or .content; guard for both
    text = getattr(response, 'text', None) or getattr(response, 'content', None) or ''
    headlines = [h for h in text.split("\n") if h.strip()]
    return headlines


if __name__ == '__main__':
    Main()

