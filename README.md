# 🧬 White Blood Cells Classification

An AI-powered system for **classifying white blood cells** 🩸  

- ⚡ **PyTorch (ResNet50)** – Deep learning backbone  
- 🌐 **FastAPI** – Backend API for predictions  
- 🎨 **Lovable** – Frontend UI  
- 📊 **Dataset** – From Kaggle

---

## 🚀 Features
- 🔍 Detects and classifies multiple WBC types  
- 📈 Handles **class imbalance** with dynamic weights  
- 🛑 Early stopping with best-model saving  
- 🖼️ Confusion matrix & training curves  
- 🌍 Full-stack integration (Frontend + Backend)  

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/white-blood-cells-classification.git
cd white-blood-cells-classification

cd backend
pip install -r requirements.txt

Run the backend:
python -m uvicorn App.backend:app --reload --host 0.0.0.0 --port 8000



📊 Results

✅ Final Train Accuracy: 99.99%
✅ Final Test Accuracy: 98.84%

🧑‍💻 Contributors

👨‍🔬 [Your Name] – ML Engineer
🎨 Lovable – Frontend
⚡ FastAPI – Backend
