from flask import Flask, render_template, request
import os
import pandas as pd
from src.pipeline import predict_pipeline

app = Flask(__name__)

# Global state tracking for catalog data and pipeline errors
INIT_ERROR_MSG = None
FULL_MOVIE_CATALOG = []
VAL_LOOKUP_MAP = {}

try:
    # Initialize ML pipeline on server boot
    pipeline = predict_pipeline.PredictPipeline()
    
    # Extract keys safely regardless of object type (Dict, Series, or Index)
    if hasattr(pipeline, 'movie_dict'):
        FULL_MOVIE_CATALOG = sorted(list(pipeline.movie_dict.keys()))
    elif hasattr(pipeline.movie_to_idx, 'index'):
        FULL_MOVIE_CATALOG = sorted(pipeline.movie_to_idx.index.tolist())
    else:
        FULL_MOVIE_CATALOG = sorted(list(pipeline.movie_to_idx.keys()))
    
    # Lowercase mapping for case-insensitive validation
    VAL_LOOKUP_MAP = {str(title).lower().strip(): title for title in FULL_MOVIE_CATALOG}
    print(f"✅ Engine Connected! Loaded {len(FULL_MOVIE_CATALOG)} titles.")
except Exception as init_error:
    INIT_ERROR_MSG = str(init_error)
    print(f"⚠️ Initialization Failure: {INIT_ERROR_MSG}")

@app.route('/')
def home():
    """Renders the main page. Surfaces startup errors if data loading fails."""
    if INIT_ERROR_MSG:
        return render_template(
            'predict.html', 
            movies=[], 
            error=f"Backend Load Failure: {INIT_ERROR_MSG}. Check your file paths or data URLs."
        )
    return render_template('predict.html', movies=FULL_MOVIE_CATALOG)

@app.route('/configure')
def configure():
    if INIT_ERROR_MSG:
        return render_template(
            'predict.html', 
            movies=[], 
            error=f"Backend Load Failure: {INIT_ERROR_MSG}."
        )
    return render_template('predict.html', movies=FULL_MOVIE_CATALOG)

@app.route('/results', methods=['POST'])
def results():
    raw_user_input = request.form.get('movie_title', '').strip()
    alpha_value = float(request.form.get('alpha', 0.5))
    
    # Match user string case-insensitively against catalog map keys
    selected_movie = VAL_LOOKUP_MAP.get(raw_user_input.lower())
    
    if not selected_movie:
        return render_template(
            'predict.html', 
            movies=FULL_MOVIE_CATALOG, 
            error="Please select a valid movie title directly from our verified database catalog."
        )
    
    try:
        # Run inference matrix engine
        raw_model_outputs = pipeline.get_hybrid_recommendations(
            movie_title=selected_movie, 
            alpha=alpha_value, 
            top_n=5
        )
        
        if not raw_model_outputs:
            return render_template(
                'predict.html',
                movies=FULL_MOVIE_CATALOG,
                error="The matrix processing layer was unable to compute similarity slices for this title."
            )
        
        # Format metric scores cleanly into integers for frontend meters
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
        return render_template(
            'predict.html', 
            movies=FULL_MOVIE_CATALOG, 
            error=f"Recommender Engine Error: {str(server_error)}"
        )

if __name__ == '__main__':
    # Read container port assigned dynamically by the cloud host environment
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)