# 🏥 MedMap AI - Intelligent Medical Management System

MedMap AI is a Flask-based healthcare platform that leverages local Large Language Models (LLMs) to automate medical workflows. It helps doctors manage consultations, extract medicine data from documents, and generate AI-verified prescriptions.

---

## 🌟 Key Features

* **🤖 AI Symptom Triage:** Analyzes patient symptoms to determine the likely medical situation using `Llama 3.2`.
* **📄 Medical OCR (PDF/Image):** Automatically extracts structured medicine data (Brand, Generic, Strength, Form) from uploaded prescriptions or medical files.
* **💊 Smart Auto-Fill:** Suggests appropriate medications from the database based on the patient's diagnosed condition.
* **🛡️ AI Safety Verification:** Validates if a chosen medicine is safe and relevant for the specific symptoms provided.
* **📚 RAG (Knowledge Training):** Allows doctors to upload medical PDFs to "train" the system, providing the AI with localized or specialized medical context.
* **📊 Patient Instructions:** Generates detailed reports including dosage calculations, advantages, and side-effect warnings.

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend** | Python / Flask |
| **Database** | SQLite3 (Relational & Knowledge storage) |
| **AI Engine** | Ollama (Running `llama3.2:1b`) |
| **OCR** | Tesseract OCR & Pytesseract |
| **Document Parsing** | PyPDF2 |

---

## 🚀 Getting Started

### 1. Prerequisites
* **Python 3.8+**
* **Ollama:** [Download here](https://ollama.com/)
* **Tesseract OCR:** Install on your OS ([Guide](https://tesseract-ocr.github.io/tessdoc/Installation.html))

### 2. Install Dependencies
```bash
pip install flask requests PyPDF2 Pillow pytesseract
