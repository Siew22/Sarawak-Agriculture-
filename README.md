# 🌿 Sarawak Agri-Advisor

**AI-powered plant disease diagnosis system for farmers in Sarawak.**

This project is a web-based application designed to assist farmers in Sarawak by providing instant, AI-driven diagnosis of crop diseases, starting with pepper plants. It leverages computer vision and soft computing techniques to deliver accurate, localized, and easy-to-understand advice.

---

## ✨ Features (MVP)

-   **AI Disease Diagnosis:** Upload a photo of a plant leaf and get an instant diagnosis powered by a custom-trained deep learning model.
-   **Automated Environmental Risk Analysis:** The system automatically fetches local weather data (temperature and humidity) based on your GPS location to provide a contextual disease risk score.
-   **Dynamic Management Suggestions:** Receive clear, actionable advice in multiple languages, generated from an embedded knowledge base.
-   **Multi-language Support:** The user interface and diagnosis reports are available in English, Bahasa Malaysia, and Chinese.

---

## 🚀 Getting Started

Follow these instructions to get the project up and running on your local machine for development and testing purposes.

### 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Git:** For version control. [Download Git](https://git-scm.com/downloads)
2.  **Python:** Version 3.8 - 3.10 is recommended. [Download Python](https://www.python.org/downloads/)
3.  **(Optional but Recommended for Training) NVIDIA GPU:** With updated drivers and CUDA support for training the AI model.

### ⚙️ Installation & Setup

1.  **Clone the repository:**
    Open your terminal or Git Bash and run the following command:
    ```bash
    git clone https://github.com/Siew22/sarawak_agri.git
    cd sarawak_agri
    ```

2.  **Create and activate a virtual environment:**
    It is crucial to use a virtual environment to manage project dependencies.
    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate it (Windows)
    .\venv\Scripts\activate

    # Activate it (macOS/Linux)
    # source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    All dependencies are listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This step may take some time, especially for PyTorch.*

4.  **Download the Dataset (Required for Training):**
    The AI model is trained on the PlantVillage dataset. You need to download it into the project's root directory.
    ```bash
    git clone https://github.com/spMohanty/PlantVillage-Dataset.git
    ```
    *This is a large download. The `.gitignore` file is configured to prevent this folder from being uploaded back to GitHub.*

5.  **Obtain the AI Model File (Crucial for Running the Backend):**
    The trained model file (`.pth`) is too large to be included in this repository. You have two options:
    *   **Option A (Recommended):** Get the pre-trained `.pth` file from a project partner and place it inside the `models_store/` folder.
    *   **Option B (From Scratch):** Train the model yourself by following the instructions in the "Training the Model" section below.

---

## 🔬 Training the Model (Optional, if you don't have the `.pth` file)

If you need to train the model from scratch, you can use the provided training scripts located in the `train/` directory.

1.  **Generate the label file:**
    First, ensure your dataset is ready. Then, from the **project root directory** (`sarawak_agri`), run:
    ```bash
    python train/create_labels.py
    ```
    This will create a `disease_labels.json` file in the `models_store/` directory.

2.  **Run the training script:**
    Choose the script based on your available GPU VRAM.
    *   **For 12GB VRAM (Recommended):**
        ```bash
        python train/train_model_2.py
        ```
    *   **For 6GB VRAM:**
        ```bash
        python train/train_model.py
        ```
    This process will take a significant amount of time and will generate a new model file (e.g., `scratch_model_b0_6gb_v2.pth`) in the `models_store/` directory.

3.  **Configure the backend to use your new model:**
    *   Open the `app/models/disease_classifier.py` file.
    *   Update the `MODEL_PATH` variable at the bottom of the file to point to your newly trained model file.
    *   Ensure the `MODEL_ARCH` variable (`'b0'` or `'b2'`) matches the model you trained.

---

## ▶️ Running the Application

The project consists of a backend server and a frontend web page. Both need to be running.

### 1. Run the Backend Server

1.  Make sure your virtual environment is activated and you are in the **project root directory** (`sarawak_agri`).
2.  Run the following command to start the Uvicorn server:
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
3.  You should see output indicating that the server is running on `http://0.0.0.0:8000`. The backend is now ready.
4.  You can access the interactive API documentation at `http://127.0.0.1:8000/docs`.

### 2. Run the Frontend

1.  Navigate to the `frontend/` directory in your file explorer.
2.  **Simply double-click the `index.html` file.**
3.  This will open the application in your default web browser. You can now use the app to upload images and get a diagnosis.

🚀 Getting Started (Automated Setup)
Follow these instructions to automatically set up the project environment on a Windows 10/11 machine using PowerShell.
📋 Prerequisites
You need to run these commands in Windows PowerShell as an Administrator.
An active internet connection.
⚙️ Automated Installation & Setup
Open Windows PowerShell as Administrator (Right-click the Start Menu -> select "PowerShell (Admin)" or "Terminal (Admin)") and execute the following commands step-by-step.
Install Chocolatey (The Windows Package Manager):
Chocolatey is a command-line package manager for Windows that simplifies software installation. We'll use it to install Git and Python.
First, check if Chocolatey is installed:
code
Powershell
choco --version
If it returns a version number, skip to the next step. If you get an error, run the following command to install it (copy and paste the entire block):
code
Powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
After installation, close and reopen PowerShell as Administrator to ensure choco is in your PATH.
Install Git and Python using Chocolatey:
This single command will install the latest stable versions of Git and Python 3.10.
code
Powershell
choco install git python --version=3.10 -y
After installation, close and reopen PowerShell as Administrator again to ensure git and python are recognized.
Clone the project repository:
Navigate to your desired projects directory (e.g., Documents) and clone the repository.
code
Powershell
cd ~\Documents
git clone https://github.com/Siew22/sarawak_agri.git
cd sarawak_agri
Create and activate a Python virtual environment:
This isolates the project's dependencies from your system.
code
Powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
You should now see (venv) at the beginning of your command prompt.
Install all required Python packages:
code
Powershell
pip install -r requirements.txt
Download the Dataset (Required for Training):
code
Powershell
git clone https://github.com/spMohanty/PlantVillage-Dataset.git
Obtain the AI Model File:
The trained model (.pth) is not included in the repository.
Option A (Recommended): Manually get the pre-trained .pth file from a project partner and place it inside the models_store/ folder.
Option B (From Scratch): Follow the "Training the Model" section below.
✅ Setup Complete! Your environment is now fully configured. Proceed to the "Running the Application" section.

---

## 🛠️ Project Structure
├── app/ # Main backend application source code
├── frontend/ # All frontend files (HTML, CSS, JS)
├── knowledge_base/ # YAML files containing the multi-language knowledge
├── models_store/ # Stores the trained model (.pth) and label files
├── train/ # Scripts for training the AI model
├── .gitignore # Specifies files for Git to ignore
├── README.md # This file
├── requirements.txt # Python dependencies
