# Stock prediction

This project retrieves stock prices automatically through an API and fits a LSTM model to predict future prices .
The entire pipeline will be incorporated in Apache Airflow.

1. Install dependencies 
pip install -r requirements.txt

2. Retrieve stock data, run hyper-parameter tuning and plot results
python3 train_stock_prediction.py --ticker AMZN --api_alpha_vantage API_KEY
(Please generate your own API key through https://www.alphavantage.co/support/#api-key)
