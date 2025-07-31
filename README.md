# Iris-Classifier :sparkles:

[![CI/CD Pipeline](https://github.com/yourusername/iris-classifier/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/yourusername/iris-classifier/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Elevator Pitch (60s):** "Construí un prototipo de ML clásico que demuestra dominio completo del stack de Data Science. El proyecto incluye EDA con visualizaciones, entrenamiento con GridSearchCV, API REST con FastAPI, tests automatizados, Docker containerization, y CI/CD con GitHub Actions. Es un showcase profesional que combina pandas, scikit-learn, FastAPI, y DevOps - perfecto para entrevistas técnicas."

## :sparkles: Problema • Datos • Modelo • Métrica • Resultado

### **Problema**
Clasificación multiclase de flores Iris en tres especies: setosa, versicolor, y virginica.

### **Datos**
- **Fuente:** `sklearn.datasets.load_iris()`
- **Features:** 4 medidas morfológicas (longitud/ancho de sépalo y pétalo)
- **Target:** 3 especies (50 muestras cada una)
- **Split:** 80% train, 20% test (stratified)

### **Modelo**
- **Pipeline:** `StandardScaler` → `LogisticRegression(max_iter=1000)`
- **Optimización:** `GridSearchCV` con `C: [0.1, 1, 10]`
- **Validación:** 5-fold cross-validation

### **Métrica**
- **Principal:** Accuracy
- **Secundarias:** Precision, Recall, F1-score por clase
- **Visualización:** Matriz de confusión y curva de validación

### **Resultado**
- **Accuracy CV:** ~96-98%
- **Accuracy Test:** ~95-97%
- **API Response:** < 100ms

<img src="validation_curve.png" width="400" alt="Validation Curve">

## :building_construction: Estructura del Repo

```
iris_classifier/
├── main.py                 # Training pipeline
├── data_exploration.ipynb  # EDA with visualizations
├── cmatrix.png            # Confusion matrix (generated)
├── validation_curve.png   # Validation curve (generated)
├── inference/
│   ├── app.py            # FastAPI application
│   └── Dockerfile        # Container configuration
├── tests/
│   └── test_api.py       # API tests
├── .github/workflows/
│   └── ci.yml           # CI/CD pipeline
├── README.md
├── requirements.txt
└── Makefile
```

## :rocket: Quick Start

### 1. Install Dependencies
```bash
make install
# or
pip install -r requirements.txt
```

### 2. Train Model
```bash
make train
# or
python main.py
```

### 3. Serve API
```bash
make serve
# or
cd iris_classifier && python -c "import uvicorn; uvicorn.run('inference.app:app', host='127.0.0.1', port=8000)"
```

### 4. Test API
```bash
make test
# or
pytest tests/ -v
```

### 5. Test API Manually
```bash
python -c "import requests; resp = requests.post('http://127.0.0.1:8000/predict', json={'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2}); print(f'Status: {resp.status_code}'); print(f'Response: {resp.json()}')"
```

### 6. Docker (Optional)
```bash
make docker-build
make docker-run
```

## :bar_chart: API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 5.1,
       "sepal_width": 3.5,
       "petal_length": 1.4,
       "petal_width": 0.2
     }'
```

### Response
```json
{
  "class_id": 0,
  "class_name": "setosa"
}
```

## :chart_with_upwards_trend: Key Findings from EDA

1. **Petal Length is Most Discriminative:** Setosa has distinctly smaller petals (1.0-1.9 cm) vs others (3.0-6.9 cm)
2. **Sepal Width Shows Overlap:** Versicolor and virginica have similar sepal widths, making it less reliable
3. **High Petal Correlation:** Petal length and width are highly correlated (0.96), suggesting redundancy

## :wrench: Development

### Available Commands
```bash
make install       # Install dependencies
make train         # Train model
make serve         # Start API server
make test          # Run tests
make docker-build  # Build Docker image
make docker-run    # Run Docker container
```

### Code Quality
```bash
make format        # Format with black/isort
make lint          # Lint with flake8
make security      # Security checks
```

## :rocket: Próximos Pasos

- [ ] **Modelo más avanzado:** Random Forest, SVM, Neural Networks
- [ ] **Feature Engineering:** Ratios, áreas, derivadas
- [ ] **MLOps:** Model versioning, A/B testing, monitoring
- [ ] **Frontend:** Web interface con visualizaciones
- [ ] **Deployment:** AWS/GCP deployment con auto-scaling
- [ ] **Monitoring:** Prometheus + Grafana dashboards

## :bulb: Lecciones Aprendidas

### **Technical Skills Demonstrated**
- **Data Science:** pandas, NumPy, scikit-learn, Matplotlib
- **Web Development:** FastAPI, Pydantic, async/await
- **DevOps:** Docker, GitHub Actions, CI/CD
- **Testing:** pytest, coverage, integration tests
- **Best Practices:** Type hints, error handling, documentation

### **Professional Practices**
- **Code Quality:** PEP-8, linting, formatting
- **Documentation:** Comprehensive README, docstrings
- **Testing:** Unit tests, integration tests, edge cases
- **Containerization:** Multi-stage Docker builds
- **CI/CD:** Automated testing, security scanning

### **Interview-Ready Features**
- **Scalable Architecture:** Modular design, clear separation of concerns
- **Production-Ready:** Error handling, logging, health checks
- **Developer Experience:** Makefile, comprehensive documentation
- **Modern Stack:** FastAPI, type hints, async programming

## :memo: Additional Resources

- **[Historias de experiencia y checklist para entrevistas](EXTRA.md)**

## :page_facing_up: License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## :handshake: Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Built with :heart: for ML interviews and professional development**
