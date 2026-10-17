FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV RADAR_DB=/data/radar.db
VOLUME ["/data"]
EXPOSE 8501
# Web app by default; for scheduled monitoring run: python -m jobs.run_watchlists
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
