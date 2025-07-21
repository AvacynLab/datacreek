FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt requirements-ci.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ci.txt
COPY datacreek ./datacreek
COPY configs ./configs
COPY README.md ./

EXPOSE 8000

CMD ["uvicorn", "datacreek.api:app", "--host", "0.0.0.0", "--port", "8000"]
