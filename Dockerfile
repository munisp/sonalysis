FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
COPY deploy ./deploy
RUN pip install --no-cache-dir -e .
RUN python -m sonalysis.training --samples 12000 --epochs 4 --output artifacts
EXPOSE 8080
CMD ["uvicorn","sonalysis.api:app","--app-dir","src","--host","0.0.0.0","--port","8080"]
