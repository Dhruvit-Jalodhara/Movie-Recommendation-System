from flask import Flask, render_template, request
import os
import pandas as pd
# Import the custom pipeline module directly matching your directory structure
from src.pipeline import predict_pipeline

app = Flask(__name__)

# --- 1. INITIALIZE INFRASTRUCTURE & LOAD ARTIFACTS ON STARTUP ---
try:
    # Instantiate the pipeline class object to fire up vector spaces once when server boots
    pipeline = predict_pipeline.PredictPipeline()
    
    # Extract the full list of ~16,000 movie titles directly from the mapping index Series
    # Sorting alphabetically ensures the dropdown tray looks professional and organized
    FULL_MOVIE_CATALOG = sorted(pipeline.movie_to_idx.index.tolist())
    
    print(f"✅ Recommender Engine Connected! Loaded {len(FULL_MOVIE_CATALOG)} titles into memory.")
except Exception as init_error:
    FULL_MOVIE_CATALOG = []
    print(f"⚠️ Initialization Failure: Could not load machine learning artifacts. Details: {str(init_error)}")

@app.route('/')
def home():
    """Renders the Project Showcase landing page introduction."""
    return render_template('home.html')

@app.route('/configure')
def configure():
    """Serves the configuration panel packed with the complete 16,000+ movie index dataset."""
    return render_template('predict.html', movies=FULL_MOVIE_CATALOG)

@app.route('/results', methods=['POST'])
def results():
    """Executes the machine learning model pipeline using the live user preferences."""
    selected_movie = request.form.get('movie_title')
    
    # Extract alpha configuration slider balance scale (ranges cleanly from 0.0 to 1.0)
    alpha_value = float(request.form.get('alpha', 0.5))
    
    # Strict fallback validation check against the database index universe
    if not selected_movie or selected_movie not in FULL_MOVIE_CATALOG:
        return render_template(
            'predict.html', 
            movies=FULL_MOVIE_CATALOG, 
            error="Please select a valid movie title directly from our verified database catalog."
        )
    
    try:
        # --- 2. EXECUTE YOUR INFERENCE ENGINE ---
        # Pass parameters straight into your model instance method
        raw_model_outputs = pipeline.get_hybrid_recommendations(
            movie_title=selected_movie, 
            alpha=alpha_value, 
            top_n=5
        )
        
        # Validation guardrail in case matrix slice suggestions yield empty frames
        if not raw_model_outputs:
            return render_template(
                'predict.html',
                movies=FULL_MOVIE_CATALOG,
                error="The matrix processing layer was unable to compute similarity slices for this title."
            )
        
        # --- 3. STRUCTURE MODEL RECORDS CLEANLY FOR UI SLIDERS ---
        # The pipeline already scales metrics by 100. We cast them to clean integers for the frontend meters.
        formatted_recommendations = []
        for row in raw_model_outputs:
            formatted_recommendations.append({
                "rank": row.get('rank'),
                "title": row.get('title'),
                "combined_score": int(round(row.get('combined_score'))),
                "tfidf_score": int(round(row.get('tfidf_score'))),
                "svd_score": int(round(row.get('svd_score')))
            })
                
        return render_template(
            'results.html', 
            movie_title=selected_movie, 
            recommendations=formatted_recommendations
        )
        
    except Exception as server_error:
        # Fallback tracking if matrix math or sparse vectors mismatch at runtime
        return render_template(
            'predict.html', 
            movies=FULL_MOVIE_CATALOG, 
            error=f"Recommender Engine Error: {str(server_error)}"
        )

if __name__ == '__main__':
    # Hugging Face Spaces environment requires host 0.0.0.0 and port 7860
    app.run(host='0.0.0.0', port=7860, debug=False)