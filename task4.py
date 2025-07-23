import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from surprise import Dataset, SVD, accuracy
from surprise.model_selection import train_test_split

# Load MovieLens 100k dataset
data = Dataset.load_builtin('ml-100k')

# Train-test split
train_set, test_set = train_test_split(data, test_size=0.25, random_state=42)

# Initialize and train SVD model
model = SVD()
model.fit(train_set)

# Make predictions
predictions = model.test(test_set)

# Evaluate model
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print("\n📊 Model Evaluation:")
print(f" - RMSE: {rmse:.4f}")
print(f" - MAE : {mae:.4f}")

# Get Top-N Recommendations for each user
def get_top_n(predictions, n=4):
    top_n = defaultdict(list)
    for pred in predictions:
        top_n[pred.uid].append((pred.iid, pred.est))
    for uid, user_ratings in top_n.items():
        top_n[uid] = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:n]
    return top_n

top_recommendations = get_top_n(predictions, n=4)

# Show recommendations for first 3 users
print("\n🎬 Top 4 Recommendations for 3 Users:")
for user_id, recs in list(top_recommendations.items())[:3]:
    print(f"\nUser {user_id}:")
    for movie_id, est_rating in recs:
        print(f" - Movie ID: {movie_id}, Predicted Rating: {round(est_rating, 2)}")

# Visualization 1: Predicted Ratings Distribution
pred_ratings = [round(pred.est, 2) for pred in predictions]
plt.figure(figsize=(8, 5))
sns.histplot(pred_ratings, bins=20, kde=True, color='cornflowerblue')
plt.title("Distribution of Predicted Ratings")
plt.xlabel("Predicted Rating")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Visualization 2: True vs Predicted Ratings
true_ratings = [pred.r_ui for pred in predictions]
plt.figure(figsize=(8, 5))
sns.scatterplot(x=true_ratings, y=pred_ratings, alpha=0.5, color='darkgreen')
sns.regplot(x=true_ratings, y=pred_ratings, scatter=False, color='red', ci=None)
plt.title("True vs Predicted Ratings")
plt.xlabel("True Rating")
plt.ylabel("Predicted Rating")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Visualization 3: Prediction Errors
errors = [abs(pred.r_ui - pred.est) for pred in predictions]
plt.figure(figsize=(8, 4))
sns.histplot(errors, bins=20, kde=True, color='tomato')
plt.title("Prediction Error Distribution")
plt.xlabel("Absolute Error |True - Predicted|")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
