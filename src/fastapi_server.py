from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests
import yfinance as yf
import openai
from dotenv import load_dotenv
import os
from fastapi_pagination import Page, add_pagination, paginate
from fastapi_pagination.utils import disable_installed_extensions_check
from fastapi.responses import JSONResponse
import pprint

app = FastAPI()

# Disable pagination extensions check warning
disable_installed_extensions_check()

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Get Alpha Vantage API keys from environment variables
alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
alpha_vantage_stock_data_api_key = os.getenv('ALPHA_VANTAGE_STOCK_DATA_API_KEY')


class StockRequest(BaseModel):
    symbol: str


def fetch_from_alpha_vantage_stock_data(symbol: str):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={alpha_vantage_stock_data_api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.RequestException as e:
        return {"error": str(e)}


def fetch_from_alpha_vantage(symbol: str):
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.RequestException as e:
        return {"error": str(e)}


def fetch_daily_time_series(symbol: str, outputsize: str = "compact"):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={outputsize}&apikey={alpha_vantage_stock_data_api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.RequestException as e:
        return {"error": str(e)}


def fetch_from_yahoo_finance(symbol: str):
    ticker = yf.Ticker(symbol)
    try:
        hist = ticker.history(period="1mo")
        info = ticker.info
        return {"info": info, "history": hist.to_dict()}
    except Exception as e:
        return {"error": str(e)}


@app.middleware("http")
def add_pagination_headers(request: Request, call_next):
    response = call_next(request)
    if isinstance(response, JSONResponse) and isinstance(response.body, dict):
        data = response.body
        if 'items' in data and isinstance(data['items'], list):
            response.headers['X-Total-Count'] = str(len(data['items']))
            response.headers['X-Page-Count'] = str((len(data['items']) // 10) + 1)
    return response


@app.post("/predict", response_model=Page[dict])
def predict(stock_request: StockRequest, request: Request):
    symbol = stock_request.symbol

    # Fetch data from Alpha Vantage
    alpha_vantage_stock_data = fetch_from_alpha_vantage_stock_data(symbol)
    alpha_vantage_data = fetch_from_alpha_vantage(symbol)
    alpha_vantage_daily_data = fetch_daily_time_series(symbol)

    # Check if Alpha Vantage is rate limited
    if (alpha_vantage_stock_data and 'Information' in alpha_vantage_stock_data and
            'Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day.' in
            alpha_vantage_stock_data['Information']):
        yahoo_data = fetch_from_yahoo_finance(symbol)
        if not yahoo_data or "error" in yahoo_data:
            raise HTTPException(status_code=404, detail="Stock data not found")
        combined_data = yahoo_data
    else:
        if (not alpha_vantage_stock_data or "Time Series (5min)" not in alpha_vantage_stock_data or
                not alpha_vantage_data or "Symbol" not in alpha_vantage_data or
                not alpha_vantage_daily_data or "Time Series (Daily)" not in alpha_vantage_daily_data):
            raise HTTPException(
                status_code=404,
                detail={
                    "symbol": symbol,
                    "alpha_vantage_stock_data_response": alpha_vantage_stock_data,
                    "alpha_vantage_overview_response": alpha_vantage_data,
                    "alpha_vantage_daily_data_response": alpha_vantage_daily_data
                }
            )

        # Combine data (this is a simple example, you may want to implement a more sophisticated method)
        combined_data = {
            "alpha_vantage_stock_data": alpha_vantage_stock_data,
            "alpha_vantage": alpha_vantage_data,
            "alpha_vantage_daily_data": alpha_vantage_daily_data
        }

    # Format the data for the OpenAI prompt
    prompt = f"Predict the next movement for the stock {symbol} based on the following data: {pprint.pformat(combined_data)}"

    models = ["gpt-4-turbo"]

    predictions = {}

    for model in models:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.5
            )
            prediction = response.choices[0].message['content'].strip()
            predictions[model] = prediction
        except openai.APIConnectionError as e:
            # Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            return {"error": f"Failed to connect to OpenAI API: {e}"}
        except openai.APIError as e:
            error_details = e.error
            print(f"OpenAI API returned an API Error: {error_details}")
            return paginate([{"data": {"error": "OpenAI API returned an API Error", "details": error_details}}])
        except openai.RateLimitError as e:
            error_details = e.error
            print(f"OpenAI API request exceeded rate limit: {error_details}")
            return paginate([{"data": {"error": "OpenAI API request exceeded rate limit", "details": error_details}}])

    combined_data["predictions"] = predictions
    return paginate([{"data": combined_data}])


add_pagination(app)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
