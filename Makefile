# Iris Classifier Makefile
# Provides convenient commands for development and deployment

.PHONY: help install train serve test docker-build docker-run clean

# Default target
help:
	@echo "Iris Classifier - Available Commands:"
	@echo "  install      - Install all dependencies"
	@echo "  train        - Train the model and generate visualizations"
	@echo "  serve        - Start the FastAPI server"
	@echo "  test         - Run all tests"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  clean        - Remove generated files"
	@echo "  all          - Install, train, and test"

# Install dependencies
install: python -m pip install -r requirements.txt
train: python main.py
serve: uvicorn inference.app:app --reload --port 8000
test: pytest -q
docker-build: docker build -t iris_classifier:latest .
docker-run: docker run -p 8000:8000 iris_classifier:latest
