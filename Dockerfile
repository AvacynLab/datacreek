FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY datacreek ./datacreek
COPY configs ./configs
COPY README.md ./
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "datacreek.api:app", "--host", "0.0.0.0", "--port", "8000"]
