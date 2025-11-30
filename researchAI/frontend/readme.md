to run locally : 

change the url in app.py to this :

api_url = os.environ.get("API_URL", "http://localhost:8000")  # Changed from localhost

the terminal command is this : 

streamlit run ./app.py

