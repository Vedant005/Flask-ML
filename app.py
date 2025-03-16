from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

df = pd.read_csv('data/freelance.csv')

features = ['amount_amount', 'hourly_rate', 'client_total_reviews', 'client_total_spent', 'total_freelancers_to_hire']

df.fillna(0, inplace=True)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

similarity_matrix = cosine_similarity(df_scaled)

def recommend_gigs(gig_index, num_recommendations=5):
    if gig_index >= len(df):
        return {"error": "Invalid gig index"}

    similar_gigs = list(enumerate(similarity_matrix[gig_index]))

    similar_gigs = sorted(similar_gigs, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]

    recommended_gigs = df.iloc[[idx for idx, score in similar_gigs]].to_dict(orient="records")

    return recommended_gigs

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        gig_index = int(request.args.get('gig_index', 0))
        num_recommendations = int(request.args.get('num_recommendations', 5))

        recommendations = recommend_gigs(gig_index, num_recommendations)
        return jsonify(recommendations)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
