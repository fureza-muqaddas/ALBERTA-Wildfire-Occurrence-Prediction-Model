import requests
import os

url = "https://static.ags.aer.ca/files/document/DIG/DIG_2021_0019.zip"
output_path = "data/raw/AlbertaLandCover2020.zip"

os.makedirs("data/raw", exist_ok=True)

# Check URL is accessible before downloading
check = requests.head(url)
if check.status_code != 200:
    raise SystemExit(
        f"URL returned {check.status_code} — cannot download land cover data.\n"
        f"Update the URL in climate.py with a valid link to DIG_2021_0019.zip."
    )

size_mb = int(check.headers.get("content-length", 0)) / 1e6
print(f"URL OK ({check.status_code}). File size: {size_mb:.1f} MB")

response = requests.get(url, stream=True)
total = int(response.headers.get('content-length', 0))

with open(output_path, 'wb') as f:
    downloaded = 0
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
        downloaded += len(chunk)
        print(f"Downloaded: {downloaded/1e6:.1f} MB / {total/1e6:.1f} MB", end='\r')

print("Done!")