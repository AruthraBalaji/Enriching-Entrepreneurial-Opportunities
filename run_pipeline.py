from datetime import datetime
from collections import defaultdict, Counter
from typing import List

from config.database import db
from nlp_engine.sentiment import analyze_sentiment
from nlp_engine.topic_model import run_topic_modeling
from nlp_engine.trend_analysis import analyze_trends
from nlp_engine.scoring import compute_opportunity_scores


POSTS_COLLECTION = db["posts"]


def load_preprocessed_posts(limit: int = 500) -> List[dict]:
    cursor = POSTS_COLLECTION.find(
        {
            "preprocessed": True,
            "is_candidate": True,
            "processed_text": {"$exists": True, "$ne": ""}
        }
    ).limit(limit)

    posts = list(cursor)
    print(f"✅ Loaded {len(posts)} preprocessed posts")
    return posts


def main():
    print("\n🚀 Starting NLP Opportunity Pipeline\n")

    # 1️⃣ Load data
    posts = load_preprocessed_posts()

    # 🔍 SAMPLE CHECK
    print("\n🔍 SAMPLE PREPROCESSED POSTS\n")
    for i, p in enumerate(posts[:5]):
        print(f"Post {i+1}")
        print("Original title:", p.get("title"))
        print("Original text:", p.get("selftext", "")[:200])
        print("Processed text:", p.get("processed_text", "")[:200])
        print("-" * 60)

    texts = [p["processed_text"] for p in posts]
    timestamps = [p.get("created_utc", datetime.utcnow()) for p in posts]

    # 2️⃣ Sentiment Analysis
    print("🔹 Running sentiment analysis...")
    sentiments = [analyze_sentiment(t) for t in texts]

    # 3️⃣ Topic Modeling
    print("🔹 Running topic modeling (BERTopic)...")
    topics, topic_keywords = run_topic_modeling(texts)

    # 4️⃣ Trend Analysis
    print("🔹 Analyzing topic trends...")
    trend_scores = analyze_trends(topics, timestamps)

    # 5️⃣ Aggregate per-topic stats
    topic_agg = defaultdict(lambda: {
        "count": 0,
        "sentiment_sum": 0.0,
        "sentiment_labels": []
    })

    for topic, sent in zip(topics, sentiments):
        if topic == -1:
            continue
        topic_agg[topic]["count"] += 1
        topic_agg[topic]["sentiment_sum"] += sent["compound"]
        topic_agg[topic]["sentiment_labels"].append(sent["label"])

    # 6️⃣ Build topic_stats
    topic_stats = {}

    for topic, stats in topic_agg.items():
        avg_sentiment = stats["sentiment_sum"] / stats["count"]
        dominant_sentiment = Counter(stats["sentiment_labels"]).most_common(1)[0][0]

        topic_stats[topic] = {
            "demand": stats["count"],
            "sentiment": avg_sentiment,
            "sentiment_label": dominant_sentiment,
            "trend": trend_scores.get(topic, 0.0),
            "competition": 0.5
        }

    # 7️⃣ Compute Opportunity Scores
    print("🔹 Computing opportunity scores...")
    scores = compute_opportunity_scores(topic_stats)

    # 8️⃣ Output
    opportunities = []

    for topic, score in scores.items():
        opportunities.append({
            "topic": topic,
            "score": score,
            "volume": topic_stats[topic]["demand"],
            "trend": topic_stats[topic]["trend"],
            "sentiment": topic_stats[topic]["sentiment_label"],
            "keywords": topic_keywords.get(topic, [])
        })

    opportunities.sort(key=lambda x: x["score"], reverse=True)

    print("\n🎯 TOP OPPORTUNITIES\n")
    for opp in opportunities:
        print(f"Topic ID: {opp['topic']}")
        print(f"Score: {opp['score']}")
        print(f"Volume: {opp['volume']}")
        print(f"Trend: {opp['trend']:.2f}")
        print(f"Sentiment: {opp['sentiment']}")
        print(f"Keywords: {opp['keywords']}")
        print("-" * 40)

    # 9️⃣ UPDATE MongoDB with results
    print("💾 Updating MongoDB with sentiment, topic, trend, score...")

    for post, sent, topic in zip(posts, sentiments, topics):
        if topic == -1:
            continue

        POSTS_COLLECTION.update_one(
            {"_id": post["_id"]},
            {
                "$set": {
                    "sentiment": {
                        "label": (
                            "positive" if sent["compound"] > 0.05
                            else "negative" if sent["compound"] < -0.05
                            else "neutral"
                        ),
                        "compound": sent["compound"]
                    },
                    "topic_id": topic,
                    "trend": trend_scores.get(topic, 0.0),
                    "score": scores.get(topic, 0.0),
                    "pipeline_version": "v1",
                    "updated_at": datetime.utcnow()
                }
            }
        )

    print("✅ MongoDB updated successfully")
    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
