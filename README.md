# 2026 FIFA World Cup Match Predictor

An end-to-end Machine Learning pipeline and interactive Streamlit dashboard designed to predict match outcomes, scorelines, and knockout stage advancements for the 2026 FIFA World Cup. 

The project uses an ensemble of **XGBoost** and **LightGBM** models, optimized via **Optuna**, to compute dynamic Poisson scoring distributions and compound 90-minute regulation probabilities with conditional penalty shootout histories.

---

## Key Features

* **Dynamic Scoring Grids:** Generates Poisson scoring matrices for every fixture to calculate explicit win, draw, and loss probabilities.
* **Conditional Knockout Modeling:** Compounds 90-minute dominance matrix grids with historical penalty shootout tendencies to accurately forecast tournament progression.
* **Isolated Penalty Metrics:** Separates regulation expectancy from penalty shootout tie-breakers to provide a clear, intuitive UX.
* **Interactive Dashboard:** A dark-themed Streamlit web app allowing users to explore matches across the Group Stage and Knockout Rounds.

---

## Model Performance

| Metric | Accuracy |
| :--- | :--- |
| **Group Stage Outcomes** (Win / Draw / Loss) | **60%** |
| **Knockout Stage Advancements** | **78%** |
| **Exact Scoreline Predictions** | **12 Matches** |

*Note: While the model achieved high accuracy on knockout advancements, human intuition still edged out the algorithm in select edge-case fixtures!*

---

## Tech Stack & Tools

* **Language:** Python
* **Machine Learning:** XGBoost, LightGBM, Optuna, Scikit-learn
* **Data Processing & Analytics:** Pandas, NumPy, SciPy
* **Dashboard / UI:** Streamlit
* **Data Visualization:** Matplotlib / Seaborn

---

## Project Structure

```text
├── data/
│   ├── raw/                   # Raw match history and FIFA rankings
│   └── processed/             # Generated prediction CSV files
├── src/
│   ├── feature_engineering.py # Rolling form, Elo ratings, and feature scripts
│   ├── train_model.py         # Model training and Optuna hyperparameter tuning
│   └── streamlit_app.py       # Streamlit web application dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Getting Started
Prerequisites
- Ensure you have Python 3.9+ installed on your system.

Installation
### Clone the repository:
- git clone [https://github.com/ahanakar/2026WCPredictor.git](https://github.com/ahanakar/2026WCPredictor.git)
- cd 2026WCPredictor
### Create a virtual environment (optional but recommended):
- python -m venv venv
- source venv/bin/activate  # On Windows: venv\Scripts\activate
### Install dependencies:
- pip install -r requirements.txt
### Run the Streamlit App:
- streamlit run src/streamlit_app.py

## How the Knockout Logic Works
In knockout football, a match cannot end in a draw, but standard regulation (90 mins) still heavily dictates advancement. The system calculates total advancement using conditional probability:
Total Advancement=P(Win in 90m)+[P(Draw in 90m)×P(Win Shootout)]
This ensures that teams with strong 90-minute dominance are rewarded properly while treating penalty shootouts strictly as a weighted tie-breaker for draw scenarios.

## Future Improvements

- [ ] Incorporate player-level injury data and tactical formation features.
- [ ] Add live API integration to automatically update group standings during the tournament.
- [ ] Expand penalty shootout modeling to include individual goalkeeper shot-stopping stats.


## Acknowledgments

* FIFA for match dataset accessibility and historical rankings data.
* The **Streamlit** team for providing a seamless deployment platform.
* The open-source maintainers of **XGBoost**, **LightGBM**, and **Optuna**.

## Author

**Ahana Kar**
* **GitHub:** [@ahanakar](https://github.com/ahanakar)
* **LinkedIn:** [Ahana Kar](https://www.linkedin.com/in/ahana-kar-757407380/)


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
