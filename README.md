# 🤖 AI Dataset Generator

> **Generate synthetic ML datasets using open-source LLMs — no API keys required.**
> GitHub Actions runs everything automatically. GitHub Pages shows live charts.

[![Generate AI Datasets](https://github.com/PranayMahendrakar/ai-dataset-generator/actions/workflows/generate.yml/badge.svg)](https://github.com/PranayMahendrakar/ai-dataset-generator/actions/workflows/generate.yml)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://PranayMahendrakar.github.io/ai-dataset-generator/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **LLM-Guided Schemas** | DistilGPT2 / GPT-2 / Phi-2 define realistic feature distributions |
| 📊 **3 Built-in Datasets** | Hospital Patients, E-Commerce Transactions, Employee Attrition |
| ⚙️ **Fully Configurable** | Features, size, noise level, seed — all via `config.yml` |
| 🤗 **No API Keys** | 100% open-source models via HuggingFace Transformers |
| ⚡ **GitHub Actions CI** | Auto-generates datasets on every push |
| 📈 **Live Visualizations** | GitHub Pages with interactive Chart.js charts |
| 🎲 **Realistic Noise** | Configurable noise injection for realistic data imperfections |
| 💾 **CSV + JSON Output** | Download datasets as CSV; statistics saved as JSON |

---

## 🚀 Quick Start

### Option 1: Use GitHub Actions (Recommended)

1. **Fork or clone** this repository
2. Go to **Actions** → **Generate AI Datasets & Deploy Pages**
3. Click **Run workflow** and fill in parameters:

```
dataset:  hospital_patients   (or: ecommerce_transactions, employee_attrition, all)
size:     10000               (number of rows)
features: 15                  (number of features, blank = all)
noise:    0.2                 (noise level 0.0–1.0)
```

4. Wait ~3 minutes → **GitHub Pages** auto-deploys with charts!

### Option 2: Run Locally

```bash
# Clone the repo
git clone https://github.com/PranayMahendrakar/ai-dataset-generator.git
cd ai-dataset-generator

# Install dependencies (CPU-only torch for lighter install)
pip install numpy transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Generate hospital patient dataset (500 rows, 15 features, noise=0.2)
python generate_dataset.py \
  --dataset hospital_patients \
  --features 15 \
  --size 500000 \
  --noise 0.2 \
  --output output

# Build GitHub Pages locally
OUTPUT_DIR=output DOCS_DIR=docs python build_pages.py

# Open docs/index.html in your browser
```

---

## 📦 Example Input / Output

### Input
```yaml
dataset: hospital_patients
features: 15
size: 500000
noise: 0.2
```

### Output: Dataset (`output/hospital_patients.csv`)
```
age,gender,blood_pressure_systolic,bmi,cholesterol,glucose,...,readmitted
52,F,118.3,26.1,198.4,94.2,...,0
71,M,142.7,31.5,245.1,128.9,...,1
...
```

### Output: Statistics (`output/hospital_patients_stats.json`)
```json
{
  "age": {
    "type": "numeric",
    "mean": 55.1,
    "std": 18.3,
    "histogram": { "counts": [...], "edges": [...] }
  },
  "readmitted": {
    "type": "binary",
    "positive_rate": 0.152
  }
}
```

### Output: Visualizations (GitHub Pages)
- 📊 Histogram for each numeric feature
- 🍩 Doughnut charts for categorical features
- 📉 Bar charts for binary targets
- 📱 Fully responsive, dark-themed dashboard

---

## 🗂️ Available Datasets

### 🏥 Hospital Patients
**17 features** | Medical domain | Target: `readmitted` (15% positive rate)

| Feature | Type | Distribution |
|---|---|---|
| age | int | Normal(55, 18) |
| gender | categorical | M/F/Other |
| blood_pressure_systolic | float | Normal(120, 20) |
| bmi | float | Normal(27, 5.5) |
| diagnosis | categorical | 6 categories |
| readmitted | binary | 15% positive |
| ... | ... | ... |

### 🛒 E-Commerce Transactions
**10 features** | Retail domain | Target: `is_fraud` (2% positive rate)

### 👥 Employee Attrition
**10 features** | HR domain | Target: `attrition` (16% positive rate)

---

## 🛠️ Configuration (`config.yml`)

```yaml
generation:
  dataset: all          # all | hospital_patients | ecommerce_transactions | employee_attrition
  size: 10000           # rows per dataset
  features: null        # null = all features, or specify number
  noise: 0.15           # 0.0 (clean) to 1.0 (very noisy)
  seed: 42              # for reproducibility

model:
  name: distilgpt2      # distilgpt2 | gpt2 | microsoft/phi-1
  device: cpu           # cpu (required for GitHub Actions free tier)
```

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────┐
                    │     config.yml / CLI args        │
                    │   dataset, features, size, noise │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │       generate_dataset.py        │
                    │                                  │
                    │  LLM Schema (DistilGPT2/Phi-2)   │
                    │  + NumPy Statistical Generator   │
                    │  + Noise Injection               │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │     output/                      │
                    │  ├── dataset.csv                 │
                    │  ├── dataset_stats.json          │
                    │  └── dataset_metadata.json       │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │       build_pages.py             │
                    │  Chart.js HTML Generator         │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │     docs/ (GitHub Pages)         │
                    │  ├── index.html                  │
                    │  ├── hospital_patients.html      │
                    │  ├── ecommerce_transactions.html │
                    │  └── employee_attrition.html     │
                    └─────────────────────────────────┘
```

---

## 🤖 Why Small Models?

GitHub Actions free tier has memory and time limits (~7GB RAM, 6h timeout). Large models like LLaMA-7B or Mistral-7B exceed these limits. This project uses a **hybrid approach**:

| Approach | Tool | Purpose |
|---|---|---|
| **Schema Design** | DistilGPT2 (82M) | Define realistic feature distributions |
| **Value Generation** | NumPy | Fast statistical sampling at scale |
| **Noise Injection** | NumPy | Realistic data imperfections |

For heavier models locally:
```bash
# Use Ollama locally for schema generation
ollama run mistral "Generate a hospital patient dataset schema with 15 features..."

# Use llama.cpp for Phi-2
./llama.cpp/main -m phi-2.gguf -p "Hospital patient features..."
```

---

## 📁 Project Structure

```
ai-dataset-generator/
├── .github/
│   └── workflows/
│       └── generate.yml      # CI/CD pipeline
├── docs/                     # GitHub Pages output (auto-generated)
│   ├── index.html
│   └── *.html
├── output/                   # Dataset files (auto-generated)
│   ├── *.csv
│   ├── *_stats.json
│   └── *_metadata.json
├── generate_dataset.py       # Core dataset generator
├── build_pages.py            # GitHub Pages builder
├── config.yml                # User configuration
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 📊 Live Demo

🌐 **[View Live Visualizations](https://PranayMahendrakar.github.io/ai-dataset-generator/)**

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Credits

Built with:
- [HuggingFace Transformers](https://huggingface.co/transformers) — DistilGPT2, GPT-2
- [Chart.js](https://www.chartjs.org/) — Interactive visualizations
- [NumPy](https://numpy.org/) — Statistical distributions
- [GitHub Actions](https://github.com/features/actions) — CI/CD automation
- [GitHub Pages](https://pages.github.com/) — Free hosting
