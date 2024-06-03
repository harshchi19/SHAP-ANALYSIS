# SHAP-ANALYSIS
# Travel Budget Estimation

This is a Streamlit app for estimating travel budgets based on various parameters such as destination, accommodation type, travel mode, duration, and attractions. The application also provides insights into the dataset's distribution, budget analysis, frequency analysis, and distance to the nearest airport for each destination.

## Features

- **Input Parameters:** Users can select destination, accommodation type, travel mode, duration, and attractions using the sidebar input form.
- **Data Distribution:** Visualizes the distribution of duration (days) and attractions visited through histograms.
- **Budget Analysis:** Analyzes the budget variation based on destination, accommodation type, and travel mode using box plots.
- **Frequency Analysis:** Displays the frequency of destinations, accommodation types, and travel modes using count plots.
- **Distance to Nearest Airport:** Calculates and displays the distance to the nearest airport for each destination.
- **Model Evaluation:** Trains a Gradient Boosting Regressor model and evaluates its performance using Mean Squared Error.
- **Estimated Budget:** Provides an estimated budget based on user input parameters.
- **SHAP Summary Plot:** Explains model predictions using the SHAP (SHapley Additive exPlanations) summary plot.

## How to Use

1. **Installation:** Clone the repository to your local machine.

```bash
git clone https://github.com/harshchi19/travel-budget-estimation.git
```

2. **Setup Environment:** Navigate to the project directory and install the required dependencies.

```bash
cd travel-budget-estimation
pip install -r requirements.txt
```

3. **Run the App:** Execute the following command to start the Streamlit app.

```bash
streamlit run app.py
```

4. **Input Parameters:** Adjust the input parameters using the sidebar form to estimate the travel budget.

5. **Explore Insights:** Explore the data distribution, budget analysis, frequency analysis, and distance to the nearest airport.

6. **View Results:** Check the estimated budget and SHAP summary plot to understand the factors influencing the budget prediction.

## Technologies Used

- Python
- Streamlit
- NumPy
- Pandas
- scikit-learn
- SHAP
- Matplotlib
- Seaborn
- Geopy

## License

This project is licensed under the [MIT License](LICENSE).
