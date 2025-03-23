import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from flask import Flask, request, jsonify
from pymongo import MongoClient
import json

app = Flask(__name__)

# ✅ MongoDB connection
MONGODB_URI= "mongodb+srv://vedant:mongovedant101@cluster0.7m05iz5.mongodb.net"
  # Replace with your connection string
DB_NAME = "skillbridge"
COLLECTION_NAME = "clients"

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# ✅ Function to load and process gigs data
def load_data_from_mongodb():
    gigs = list(collection.aggregate([
        { "$unwind": "$gigs" },  # Flatten gigs array
        { 
            "$project": {
                "_id": 0,
                "clientId": "$_id",
                "email": 1,
                "location": 1,
                "gigs.gigId": 1,
                "gigs.title": 1,
                "gigs.amount_amount": 1,
                "gigs.hourly_rate": 1,
                "gigs.duration": 1,
                "gigs.type": 1,
                "gigs.Description": 1,
                "gigs.Status": 1,
                "gigs.created_on": 1,
                "gigs.engagement": 1,
                "gigs.proposals_tier": 1,
                "gigs.published_on": 1,
                "gigs.client_total_reviews": 1,
                "gigs.client_total_spent": 1,
                "gigs.client_location_country": 1,
                "gigs.occupations_category_pref_label": 1,
                "gigs.occupations_oservice_pref_label": 1,
            }
        }
    ]))

    if not gigs:
        return pd.DataFrame()

    # Convert MongoDB response to DataFrame
    df = pd.DataFrame(gigs)

    # Flatten gig structure
    df = pd.concat([df.drop(columns=["gigs"]), df["gigs"].apply(pd.Series)], axis=1)

    # Fill missing values
    num_features = ["amount_amount", "hourly_rate", "client_total_spent"]
    cat_features = ["title", "type", "duration", "client_location_country"]

    df[num_features] = df[num_features].fillna(0)
    df[cat_features] = df[cat_features].fillna("Unknown")

    return df

# 🔥 Recommendation function with serialization
def recommend_gigs_by_id(gig_id, num_recommendations=5):
    df = load_data_from_mongodb()
    
    if df.empty:
        return {"error": "No gigs found in MongoDB"}

    if gig_id not in df["gigId"].values:
        return {"error": "Gig ID not found"}

    # Feature selection
    num_features = ["amount_amount", "hourly_rate", "client_total_spent"]
    cat_features = ["title", "type", "duration", "client_location_country","occupations_oservice_pref_label","occupations_category_pref_label"]

    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_cats = encoder.fit_transform(df[cat_features])

    # Scale numerical features
    scaler = StandardScaler()
    scaled_nums = scaler.fit_transform(df[num_features])

    # Combine numerical and categorical features
    X = np.hstack((scaled_nums, encoded_cats))

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(X)

    # Find the index of the selected gig
    gig_index = df[df["gigId"] == gig_id].index[0]

    # Get similarity scores
    similar_gigs = list(enumerate(similarity_matrix[gig_index]))

    # Sort by similarity (excluding itself)
    similar_gigs = sorted(similar_gigs, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]

    # Retrieve the recommended gigs with the required format
    recommended_gigs = []
    
    for idx, _ in similar_gigs:
        gig = df.iloc[idx]
        
        # Convert int64 and float64 types to native Python types for JSON serialization
        recommended_gigs.append({
            "email": str(gig["email"]),
            "location": str(gig["client_location_country"]),
            "gigs": {
                "gigId": str(gig["gigId"]),
                "title": str(gig["title"]),
                "amount_amount": int(gig["amount_amount"]) if pd.notna(gig["amount_amount"]) else 0,
                "hourly_rate": float(gig["hourly_rate"]) if pd.notna(gig["hourly_rate"]) else 0.0,
                "duration": str(gig["duration"]),
                "type": str(gig["type"]),
                "Description": str(gig["Description"]),
                "Status": str(gig["Status"]),
                "created_on": str(gig["created_on"]),
                "engagement": str(gig["engagement"]),
                "proposals_tier": str(gig["proposals_tier"]),
                "published_on": str(gig["published_on"]),
                "client_total_reviews": int(gig["client_total_reviews"]) if pd.notna(gig["client_total_reviews"]) else 0,
                "client_total_spent": int(gig["client_total_spent"]) if pd.notna(gig["client_total_spent"]) else 0,
                "client_location_country": str(gig["client_location_country"]),
                "occupations_category_pref_label":str(gig["occupations_category_pref_label"]),
                "occupations_oservice_pref_label":str(gig["occupations_oservice_pref_label"])

            },
            "clientId": str(gig["clientId"])
        })

    return recommended_gigs

# 🚀 API Route: Get Recommended Gigs
@app.route("/recommend", methods=["GET"])
def getRecommendedGigs():
    gig_id = request.args.get("gig_id")

    if not gig_id:
        return jsonify({"error": "Gig ID is required"}), 400

    recommendations = recommend_gigs_by_id(gig_id)

    # ✅ Use `json.dumps()` to serialize the dictionary before returning it
    return app.response_class(
        response=json.dumps(recommendations, default=str),
        status=200,
        mimetype="application/json"
    )

@app.route('/gigs_with_sentiment', methods=['GET'])
def get_gigs_with_sentiment():
    gigs = list(collection.find({}))

    gigs_with_sentiment = []

    for client in gigs:
        for gig in client['gigs']:
            
            # ✅ Check for review and feedback values
            reviews = gig.get("client_total_reviews", 0)
            feedback = gig.get("client_total_feedback", 0)

            # ✅ Determine sentiment based on feedback
            if feedback >= 4.0:
                sentiment = "Positive"
            elif 2.5 <= feedback < 4.0:
                sentiment = "Neutral"
            else:
                sentiment = "Negative"

            gigs_with_sentiment.append({
                "email": client['email'],
                "location": client['location'],
                "clientId": client['_id'],
                "gig": {
                    **gig,
                    "sentiment": sentiment,    # ✅ Add sentiment field
                    "total_reviews": reviews,
                    "feedback_score": feedback
                }
            })

    return jsonify(gigs_with_sentiment)

if __name__ == "__main__":
    app.run(debug=True)
