# 🤖 AI Text Detection

An advanced machine learning system to classify text as human-written or AI-generated. Built with Flask backend and an interactive web interface.

## Features

- **Multiple ML Models**: Logistic Regression, Random Forest, Gradient Boosting, Linear SVM, and Neural Networks
- **Stylometric Analysis**: Analyzes 10+ linguistic features including:
  - Average word/sentence length
  - Punctuation density
  - Contraction usage
  - First-person pronoun frequency
  - Passive voice indicators
  - Hedge word density
  - Type-token ratio (lexical diversity)
- **TF-IDF N-gram Features**: Captures lexical patterns
- **Interactive Web UI**: Real-time predictions with confidence scores
- **Comprehensive EDA**: Automatic exploratory data analysis and visualization

## Project Structure

```
.
├── main.py                 # Training pipeline
├── requirements.txt        # Python dependencies
├── src/
│   ├── app.py             # Flask backend API
│   ├── index.html         # Frontend UI
│   ├── load_data.py       # Data loading utilities
│   ├── preprocess.py      # Text preprocessing
│   ├── features.py        # Feature engineering
│   ├── eda.py             # Exploratory data analysis
│   ├── model.py           # Classical ML models
│   └── deep_model.py      # Neural network model
├── saved_models/          # Trained models (generated after training)
└── eda_plots/             # Visualizations (generated after training)
```

## Installation

### Prerequisites
- Python 3.8+
- macOS/Linux/Windows

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/abhay8507beaift24-web/AI-Text-Detection.git
   cd AI-Text-Detection
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Model

Run the complete training pipeline:

```bash
python main.py
```

This will:
- Generate synthetic training data
- Preprocess the text
- Run EDA and save visualizations
- Train all models
- Select and save the best model
- Save feature pipeline artifacts

**Output:**
- Trained models in `saved_models/`
- EDA plots in `eda_plots/`
- Model performance metrics printed to console

### 2. Start the Web Server

```bash
# Default: port 8000
python src/app.py

# Or specify a different port
PORT=5000 python src/app.py
```

Access the interface at: **http://localhost:8000**

### 3. Use the API

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Get Example Texts
```bash
curl http://localhost:8000/examples
```

#### Classify Text
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here..."}'
```

**Response:**
```json
{
  "prediction": 0,
  "confidence": 0.95,
  "probabilities": {"human": 0.95, "ai": 0.05},
  "stylometric_features": {...},
  "signal_indicators": [...],
  "processing_time_ms": 4.3
}
```

## Model Performance

All models achieve excellent performance on the synthetic dataset:

| Model | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|---------|
| Logistic Regression | 100% | 100% | 100% |
| Linear SVM | 100% | 100% | 100% |
| Random Forest | 100% | 100% | 100% |
| Gradient Boosting | 100% | 100% | 100% |
| MLP Neural Network | 100% | 100% | 100% |

*Note: Performance on real-world data may vary*

## Key Technologies

- **Backend**: Flask, Flask-CORS
- **ML/Data**: scikit-learn, pandas, numpy, scipy
- **Visualization**: matplotlib
- **Frontend**: HTML5, CSS3, vanilla JavaScript
- **Architecture**: Modern async/await with responsive design

## Features Analyzed

### Stylometric Features
1. **Average Sentence Length**: Human and AI text differ in sentence structure
2. **Average Word Length**: Character patterns vary by authorship
3. **Punctuation Density**: Punctuation usage differs between humans and AI
4. **Type-Token Ratio**: Measures vocabulary diversity
5. **Contraction Density**: Humans use more contractions (I'm, don't, etc.)
6. **First-Person Density**: Personal pronoun usage patterns
7. **Passive Voice Density**: AI tends toward passive constructions
8. **Hedge Word Density**: AI often uses hedging language
9. **Exclamation/Question Ratio**: Emotional language markers
10. **Sentence Length Variance**: Stylistic variation

### N-gram Features
- TF-IDF weighted uni-, bi-, and tri-grams
- Captures lexical and linguistic patterns

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve frontend |
| GET | `/health` | API health check |
| GET | `/examples` | Get example texts |
| POST | `/predict` | Classify text |
| GET | `/stats` | Get training statistics |

## Configuration

### Environment Variables
- `PORT`: Server port (default: 8000)
- `DEBUG`: Debug mode (default: False)

### Model Parameters

Edit `src/model.py` and `src/deep_model.py` to adjust:
- Model hyperparameters
- Regularization strength
- Training iterations
- Activation functions

## Dataset

The project uses **synthetic data** for training:
- 2000 samples (1000 human, 1000 AI-generated)
- Balanced class distribution
- Can be replaced with real CSV data

### Using Custom Data

Place your CSV file with columns `text` and `label` (0=human, 1=AI) and modify `main.py`:

```python
df = load_from_csv("your_data.csv")
```

## Improvements Made

- ✅ Fixed import paths for modular structure
- ✅ Enhanced neural network architecture (3 hidden layers, early stopping)
- ✅ Added comprehensive stylometric feature extraction
- ✅ Implemented Flask API with proper error handling
- ✅ Created responsive, modern web interface
- ✅ Added CORS support for cross-origin requests
- ✅ Dynamic API endpoint detection in frontend
- ✅ Comprehensive EDA and visualization pipeline

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

MIT License - See LICENSE file for details

## Author

Abhay Gandotra

## Support

For issues or questions, please create an issue on GitHub.

---

**Made with ❤️ for AI transparency and text analysis**
