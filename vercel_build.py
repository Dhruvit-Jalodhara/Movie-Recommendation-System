import os
import urllib.request

print("Vercel Build Step: Fetching large ML models from DagsHub DVC remote...")

# Ensure the local artifacts folder exists inside the compilation container
os.makedirs("artifacts", exist_ok=True)

# DagsHub Repository Target Details
DAGSHUB_USER = os.environ.get("DAGSHUB_USER", "Dhruvit-Jalodhara")
DAGSHUB_REPO = os.environ.get("DAGSHUB_REPO", "Movie-Recommendation-System")
DAGSHUB_TOKEN = os.environ.get("DAGSHUB_TOKEN", "") # Required only if your DagsHub repo is private

# List of files managed by DVC that your pipeline needs
files = [
    "movies_processed.joblib",
    "movie_features_svd.joblib",
    "tfidf_matrix.joblib",
    "movie_mapping.joblib"
]

base_url = f"https://dagshub.com/api/v1/repos/{DAGSHUB_USER}/{DAGSHUB_REPO}/raw/main/artifacts"

for file in files:
    url = f"{base_url}/{file}"
    target_path = os.path.join("artifacts", file)
    print(f"Downloading {file} from DagsHub DVC storage...")
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Vercel-Build-Agent'})
    if DAGSHUB_TOKEN:
        req.add_header("Authorization", f"Bearer {DAGSHUB_TOKEN}")
        
    try:
        with urllib.request.urlopen(req) as response, open(target_path, "wb") as out_file:
            out_file.write(response.read())
        print(f"successfully downloaded {file}")
    except Exception as e:
        print(f"Failed to download {file}: {str(e)}")
        raise e
        
print("🎉 All production DVC models loaded natively into Vercel build bundle!")