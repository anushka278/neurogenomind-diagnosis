# Download full text of articles from PubMed using DOI and Playwright for institutional access

import pandas as pd
import time
import os
from bs4 import BeautifulSoup
from Bio import Entrez
from playwright.sync_api import sync_playwright
from urllib.parse import quote_plus

Entrez.email = "lukezhao@stanford.edu"
Entrez.api_key = "3691e0c9d13eb488c45319264143fd821f08"

# Directories
os.makedirs("fulltext2", exist_ok=True)

# Load PMID list
df = pd.read_csv("pmid-list-3.csv")
pmids = [int(x) for x in df.iloc[:, 0].dropna().tolist()]

# PMID → DOI mapping
def get_doi_from_pmid(pmid):
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
        records = Entrez.read(handle)
        article = records["PubmedArticle"][0]["MedlineCitation"]["Article"]
        for id in article["ELocationID"]:
            if id.attributes["EIdType"] == "doi":
                return id.title()
    except Exception as e:
        print(f"[!] Could not get DOI for PMID {pmid}: {e}")
    return None

# Save DOI list to csv
doi_list = [get_doi_from_pmid(pmid) for pmid in pmids]
doi_df = pd.DataFrame({"PMID": pmids, "DOI": doi_list})
doi_df.to_csv("doi-list3.csv", index=False)

# Playwright function to fetch full text via institutional login
def fetch_doi_text_with_playwright(doi, filename):
    proxied_url = f"https://laneproxy.stanford.edu/login?url=https://doi.org/{doi}"
    # url = f"https://doi.org/{quote_plus(doi)}"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(proxied_url)

        print(f"[*] Please log in manually if needed. Article page: {proxied_url}")
        input("[*] Press ENTER after full text loads...")

        try:
            content = page.locator("body").inner_text()
            with open(f"fulltext/{filename}.txt", "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[+] Saved full text to fulltext/{filename}.txt")
        except Exception as e:
            print(f"[!] Failed to extract full text for DOI {doi}: {e}")
        
        context.storage_state(path="auth_cookies.json")
        browser.close()

# Load in DOI csv file
doi_df = pd.read_csv("doi-list5.csv")
doi_list = doi_df.iloc[:, 1].dropna().tolist()

i = 0
for doi in doi_list:
    fetch_doi_text_with_playwright(doi, str(pmids[i]))
    i += 1

##### TESTING #####

# Test if doi can be fetched for all pmids
# for pmid in pmids:
#     doi = get_doi_from_pmid(pmid)
#     if doi:
#         print(f"[+] DOI for PMID {pmid}: {doi}")
#     else:
#         print(f"[!] No DOI found for PMID {pmid}")

# handle = Entrez.efetch(db="pubmed", id=12784278, retmode="xml")
# records = Entrez.read(handle)
# article = records["PubmedArticle"][0]["MedlineCitation"]["Article"]
# for id in article["ELocationID"]:
#     if id.attributes["EIdType"] == "doi":
#         print(id.title())
#####

# if __name__ == "__main__":
#     run()
