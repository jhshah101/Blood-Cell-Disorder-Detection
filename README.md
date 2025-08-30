# 🧬 Software Front and Back End White Blood Cells Classification

An AI-powered system for **classifying white blood cells** 🩸  

- ⚡ **PyTorch** – Deep learning backbone  
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
- ☁️ **Reusable Notebook (`Software_code.ipynb`)** for **Google Colab**  

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/white-blood-cells-classification.git
cd white-blood-cells-classification
```

### 2️⃣ Run Backend (Optional – for API deployment)
```bash
cd backend
pip install -r requirements.txt

# Run the backend
python -m uvicorn App.backend:app --reload --host 0.0.0.0 --port 8000
```

---

## 📒 Reusable Code on Google Colab

If you don’t want to set up the backend/frontend and just need the **core model training and evaluation**, use the notebook:

➡️ Open (Software_code.ipynb) in **Google Colab**.  

Steps inside Colab:
1. Upload the dataset (or mount Google Drive).  
2. Install required libraries (PyTorch, torchvision, scikit-learn, matplotlib).  
3. Run cells step by step:
   - **Data loading & preprocessing**  
   - **Model definition (CNN + Transformer hybrid)**  
   - **Weighted loss for imbalance**  
   - **Training with early stopping**  
   - **Evaluation: confusion matrix, accuracy, per-class report**  
4. Save trained model (`best_model.pth`) for later use.  

This way, the notebook works as a **standalone, reusable training pipeline** without needing backend or frontend setup.  

---

## 🧑‍💻 Contributors
👨‍🔬 [jamal] – ML Engineer  
🎨 Lovable – Frontend  
⚡ FastAPI – Backend  
