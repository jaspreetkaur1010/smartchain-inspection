def generate_recommendations(df):
    recs = []
    if df["Shipping costs"].mean() > 400:
        recs.append("🔸 Shipping costs are high. Optimize logistics contracts.")
    if df["Lead time"].mean() > 30:
        recs.append("🔸 Lead times exceed 30 days. Consider faster suppliers.")
    if df["Defect rates"].mean() > 5:
        recs.append("🔸 Defect rate > 5%. Re-evaluate manufacturing QA processes.")
    # Correlation-based suggestions
    if df["Shipping times"].mean() > 20:
        recs.append("🚚 Consider switching to faster carriers or optimizing shipping routes.")
    if df["Order quantities"].mean() > 700:
        recs.append("📦 Large order sizes (>700) correlate with higher inspection failure. Break them into smaller batches.")
    if df["Lead time"].mean() > 25:
        recs.append("⏱️ Long supplier lead times reduce inspection performance. Try engaging more reliable suppliers.")
    if df["Availability"].mean() < 50:
        recs.append("📉 Low availability correlates with inspection failures. Improve inventory forecasting.")
    return recs 