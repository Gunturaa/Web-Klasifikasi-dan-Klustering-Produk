from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd 
from collections import Counter

app = Flask(__name__)

# --- Load Model Klasifikasi dan Encoder ---
try:
    model = pickle.load(open("best_rf_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    le_category = pickle.load(open("le_category.pkl", "rb"))
    le_sitename = pickle.load(open("le_sitename.pkl", "rb"))
    print("Model Klasifikasi dan Encoders berhasil dimuat.")
except FileNotFoundError as e:
    print(f"Error memuat file model klasifikasi atau encoder: {e}")
    model = None
    scaler = None
    le_category = None
    le_sitename = None
except Exception as e:
    print(f"Terjadi kesalahan lain saat memuat model klasifikasi atau encoder: {e}")
    model = None
    scaler = None
    le_category = None
    le_sitename = None

# --- Load Model K-Means dan Scaler Clustering ---
try:
    kmeans_model = pickle.load(open("kmeans_model.pkl", "rb"))
    scaler_clustering = pickle.load(open("scaler_clustering.pkl", "rb"))
    print("Model K-Means dan Scaler Clustering berhasil dimuat.")
except FileNotFoundError as e:
    print(f"Error memuat file model K-Means atau scaler clustering: {e}")
    kmeans_model = None
    scaler_clustering = None
except Exception as e:
    print(f"Terjadi kesalahan lain saat memuat model K-Means atau scaler clustering: {e}")
    kmeans_model = None
    scaler_clustering = None


# Riwayat Klasifikasi sementara (RAM)
history = []

# Helper aman untuk transform label
def safe_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        # Tambah label baru jika perlu
        encoder.classes_ = np.append(encoder.classes_, value)
        return encoder.transform([value])[0]

def generate_advice(price_ori, price_actual, item_rating, total_rating, favorite, prediction):
    tips = []
    # Saran berdasarkan rating
    if item_rating < 3:
        tips.append("⭐ Tingkatkan pelayanan dan kualitas produk untuk mendapatkan rating lebih baik.")
    # Saran berdasarkan harga
    if price_actual > price_ori * 0.9:
        tips.append("💸 Pertimbangkan untuk memberikan diskon agar produk lebih menarik.")
    # Saran berdasarkan total rating
    if total_rating < 10:
        tips.append("📝 Dorong pembeli untuk memberikan ulasan agar meningkatkan kepercayaan.")
    # Saran berdasarkan favorite
    if favorite < 5:
        tips.append("❤️ Promosikan produk di media sosial untuk menambah jumlah favorite.")

    # Saran umum berdasarkan hasil klasifikasi
    if prediction == "LARIS":
        tips.insert(0, "✅ Produk Anda sudah laris! Pertahankan strategi saat ini.")
    else:
        tips.insert(0, "⚠️ Produk belum laris. Coba beberapa strategi berikut:")

    return tips

@app.route("/", methods=["GET"])
def index():
    # Hanya tampilkan landing page
    return render_template("index.html")

@app.route("/klasifikasi", methods=["GET", "POST"])
def klasifikasi():
    prediction = None
    advice_list = None
    categories = sorted(list(le_category.classes_))
    sites = sorted(list(le_sitename.classes_))

    if request.method == "POST":
        try:
            price_ori = float(request.form["price_ori"])
            price_actual = float(request.form["price_actual"])
            item_category_detail = request.form["item_category_detail"]
            item_rating = float(request.form["item_rating"])
            total_rating = int(request.form["total_rating"])
            favorite = int(request.form["favorite"])
            sitename = request.form["sitename"]

            cat_encoded = safe_transform(le_category, item_category_detail)
            site_encoded = safe_transform(le_sitename, sitename)

            X = np.array([[price_ori, price_actual, cat_encoded, item_rating, total_rating, favorite, site_encoded]])
            X_scaled = scaler.transform(X)
            result = model.predict(X_scaled)[0]
            prediction = "LARIS" if result == 1 else "TIDAK LARIS"
            advice_list = generate_advice(price_ori, price_actual, item_rating, total_rating, favorite, prediction)

            history.append({
                "harga_ori": price_ori,
                "harga_aktual": price_actual,
                "kategori": item_category_detail,
                "rating": item_rating,
                "total_rating": total_rating,
                "favorite": favorite,
                "sitename": sitename,
                "hasil": prediction
            })

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("klasifikasi.html", prediction=prediction, categories=categories, sites=sites, history=history, advice_list=advice_list)

@app.route("/cluster", methods=["GET", "POST"])
def cluster():
    cluster_label = None
    cluster_info = ""
    error = None

    # Global riwayat detail
    if "cluster_history_detail" not in globals():
        global cluster_history_detail
        cluster_history_detail = []

    cluster_desc = {
        0: "Produk dengan penjualan rendah",
        1: "Produk dengan penjualan sedang",
        2: "Produk dengan penjualan tinggi"
    }

    if request.method == "POST":
        try:
            price_actual = float(request.form["price_actual"])
            item_rating = float(request.form["item_rating"])
            total_rating = int(request.form["total_rating"])
            favorite = int(request.form["favorite"])

            input_data = np.array([[price_actual, item_rating, total_rating, favorite]])
            input_scaled = scaler_clustering.transform(input_data)
            cluster_label = int(kmeans_model.predict(input_scaled)[0])
            cluster_info = cluster_desc.get(cluster_label, "Segmentasi tidak diketahui")

            # Simpan detail riwayat
            cluster_history_detail.append({
                "price_actual": price_actual,
                "item_rating": item_rating,
                "total_rating": total_rating,
                "favorite": favorite,
                "cluster": cluster_label,
                "desc": cluster_info
            })
        except Exception as e:
            error = f"Terjadi kesalahan: {e}"

    # Hitung jumlah data di tiap cluster untuk visualisasi
    from collections import Counter
    cluster_counts = dict(Counter([d["cluster"] for d in cluster_history_detail]))
    for i in range(kmeans_model.n_clusters):
        if i not in cluster_counts:
            cluster_counts[i] = 0

    return render_template(
        "cluster.html",
        cluster_label=cluster_label,
        cluster_info=cluster_info,
        error=error,
        cluster_counts=cluster_counts,
        cluster_history_detail=cluster_history_detail
    )

if __name__ == "__main__":
    app.run(debug=True, port=5001)


