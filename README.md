# PyTorch-Based-PPE-Detection
This project detects **Personal Protective Equipment (PPE)** using **PyTorch** and **YOLO**.

---

## Requirements
- **Python 3.8+**
- **Virtual environment (recommended)**

---

## Installation and Setup

**Clone the repository:**
```bash
git clone https://github.com/Carl-Erick/PyTorch-Based-PPE-Detection.git
cd PyTorch-Based-PPE-Detection
```

**Create and activate a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Install dependencies:**
```bash
pip install -r Codebase_Sprint/PPE_detection_YOLO/requirements_final.txt
```

**Running the Application**

Navigate to the app directory:
```bash
cd Codebase_Sprint/PPE_detection_YOLO
```

Run the application:
```bash
python run_app.py
```

---

## Project Structure
```
Codebase_Sprint/
├── PPE_detection_YOLO/          # Main application folder
│   ├── app.py                   # Flask web app
│   ├── run_app.py              # Application launcher
│   ├── ppe_detector.py         # Core detection logic
│   ├── config.py               # Configuration settings
│   ├── requirements_final.txt   # Python dependencies
│   ├── templates/              # HTML templates
│   ├── static/                 # Static assets (CSS, JS, images)
│   └── Videos/                 # Video samples
├── PROJECT-SPRINT/             # Sprint documentation
└── static/                      # Static files
```
