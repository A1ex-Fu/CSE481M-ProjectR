import json
import os
import requests
from tqdm import tqdm

# Create papers directory if it doesn't exist
papers_dir = "paperDataset"
os.makedirs(papers_dir, exist_ok=True)

# Read the annotations file
with open("leaderboard-generation/tdm_annotations.json", "r") as f:
    annotations = json.load(f)

# Download each paper
for paper_id, paper_info in tqdm(annotations.items()):
    paper_url = paper_info["PaperURL"]
    output_path = os.path.join(papers_dir, paper_id)
    
    # Skip if already downloaded
    if os.path.exists(output_path):
        continue
    
    try:
        # Download the paper
        response = requests.get(paper_url)
        response.raise_for_status()
        
        # Save the paper
        with open(output_path, "wb") as f:
            f.write(response.content)
            
    except Exception as e:
        print(f"Error downloading {paper_id}: {str(e)}") 