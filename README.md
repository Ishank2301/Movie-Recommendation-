# 🎬 Movie Recommendation System

A content-based movie recommendation system that suggests similar movies based on user input using **cosine similarity**. The system leverages the IMDb dataset and provides an interactive UI built with Streamlit.

---

## 🚀 Features

* 🔍 Recommend movies based on similarity
* 🎭 Content-based filtering using movie metadata
* ⚡ Fast similarity computation using cosine similarity
* 🖥️ Interactive UI with Streamlit
* 📊 Clean and simple user experience

---

## 🧠 How It Works

1. **Data Collection**

   * Uses the IMDb dataset containing movie metadata (title, genres, keywords, etc.)

2. **Preprocessing**

   * Combine important features (genres, overview, cast, etc.)
   * Text cleaning and transformation

3. **Feature Extraction**

   * Convert text data into numerical vectors using techniques like CountVectorizer / TF-IDF

4. **Similarity Computation**

   * Apply cosine similarity to compute similarity scores between movies

5. **Recommendation**

   * Based on user input, the system retrieves top N most similar movies

---

## 🛠️ Tech Stack

* Python
* Pandas & NumPy
* Scikit-learn
* Streamlit

---

## 📂 Project Structure

```
movie-recommendation/
│
├── data/
│   └── imdb_dataset.csv
│
├── model/
│   └── similarity.pkl
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/movie-recommendation.git
cd movie-recommendation
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the app:

```
streamlit run app.py
```

---

## 💡 Example

* Input: *Inception*
* Output:

  * Interstellar
  * The Prestige
  * Shutter Island
  * The Matrix

---

## 📊 Results

* Efficient recommendation using cosine similarity
* Handles large datasets with fast response time
* Provides relevant and meaningful movie suggestions

---

## 🔮 Future Improvements

* 🎯 Hybrid recommendation system (content + collaborative filtering)
* 🌐 Deployment on cloud (AWS / Render / HuggingFace Spaces)
* 🎥 Add movie posters using external APIs
* ⭐ User-based personalization

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* IMDb Dataset
* Scikit-learn documentation
* Streamlit community

