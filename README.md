<img width="629" height="833" alt="Screenshot 2025-10-18 102937" src="https://github.com/user-attachments/assets/e2812c27-5626-4833-8de8-f8350851f56a" /># 🌿 Sarawak Agri-Advisor

**AI-powered plant disease diagnosis system for farmers in Sarawak.**

This project is a web-based application designed to assist farmers in Sarawak by providing instant, AI-driven diagnosis of crop diseases, starting with pepper plants. It leverages computer vision and soft computing techniques to deliver accurate, localized, and easy-to-understand advice.

---

## ✨ Features (MVP)

-   **AI Disease Diagnosis:** Upload a photo of a plant leaf and get an instant diagnosis.
-   **Automated Environmental Risk Analysis:** Automatically fetches local weather data (temperature & humidity) via GPS to provide a contextual disease risk score.
-   **Dynamic Management Suggestions:** Receive clear, actionable advice in multiple languages, generated from an embedded knowledge base.
-   **Multi-language Support:** Reports are available in English, Bahasa Malaysia, and Chinese.

---

## 🚀 Getting Started (Automated Setup for Windows)

Follow these instructions to automatically set up the project environment on a **Windows 10/11** machine using PowerShell.

### 📋 Prerequisites

-   You must run these commands in **Windows PowerShell as an Administrator**.
-   An active internet connection is required.

### ⚙️ Installation & Setup Steps

**Step 1: Open PowerShell as Administrator**
*   Right-click the Windows Start Menu.
*   Select **"Terminal (Admin)"** or **"Windows PowerShell (Admin)"**.

**Step 2: Install Chocolatey (The Windows Package Manager)**
*   First, check if Chocolatey is already installed:
    ```powershell
    choco --version
    ```
*   If you see a version number, proceed to Step 3. If you get an error, run the following command to install Chocolatey (copy the entire block and paste it):
    ```powershell
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    ```
*   After installation, **CLOSE and REOPEN PowerShell as Administrator** to continue.

**Step 3: Install Git and Python**
*   This single command installs Git and a specific version of Python (3.10) for consistency.
    ```powershell
    choco install git python --version=3.10 -y
    ```
*   After installation, **CLOSE and REOPEN PowerShell as Administrator** one more time to ensure the new commands are recognized.

**Step 4: Clone Repositories and Set Up Project**
*   Navigate to your `Documents` folder and execute the following commands one by one:
    ```powershell
    # Navigate to your Documents folder
    cd ~\Documents

    # Clone this project repository
    git clone https://github.com/Siew22/sarawak_agri.git
    
    # Enter the project directory
    cd sarawak_agri

    # Create the Python virtual environment
    python -m venv venv

    # Activate the virtual environment
    .\venv\Scripts\Activate.ps1
    ```
    *(You should now see `(venv)` at the beginning of your command prompt.)*

**Step 5: Install Python Dependencies and Download Dataset**
*   With the virtual environment active, run the following:
    ```powershell
    # Install all required Python packages
    pip install -r requirements.txt

    # Download the PlantVillage dataset (this is large and will take time)
    git clone https://github.com/spMohanty/PlantVillage-Dataset.git
    ```

**Step 6: Obtain the AI Model File**
*   The trained model file (`.pth`) is essential for the backend to run but is not included in the repository due to its large size.
    *   **Option A (Recommended):** Manually get the pre-trained `.pth` file from a project partner and place it inside the `models_store/` folder.
    *   **Option B (From Scratch):** Follow the instructions in the "Training the Model" section below.

**✅ Setup Complete!** Your environment is now fully configured. Proceed to the "Running the Application" section.

---

## 🔬 Training the Model (Optional)

If you need to train the model from scratch, you can use the provided scripts in the `train/` directory.

1.  **Generate the label file:**
    *   From the **project root directory** (`sarawak_agri`) with your virtual environment active, run:
        ```powershell
        python train/create_labels.py
        ```

2.  **Run the training script:**
    *   Choose the script based on your GPU VRAM.
        *   **For 12GB+ VRAM (Recommended for best results):**
            ```powershell
            python train/train_model_2.py
            ```
        *   **For 6GB VRAM:**
            ```powershell
            python train/train_model.py
            ```

3.  **Configure the backend to use your new model:**
    *   Open `app/models/disease_classifier.py`.
    *   Update the `MODEL_PATH` and `MODEL_ARCH` variables at the bottom of the file to match your newly trained model.

---

## ▶️ Running the Application

### 1. Run the Backend Server

1.  Ensure your virtual environment is active and you are in the **project root directory**.
2.  Start the Uvicorn server:
    ```powershell
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
3.  The backend is now running and accessible. You can view the API docs at `http://127.0.0.1:8000/docs`.

### 2. Run the Frontend

1.  Navigate to the `frontend/` directory in your file explorer.
2.  Simply **double-click the `index.html`** file.
3.  This will open the application in your default web browser. You can now use the app to get a diagnosis.

---

## 🛠️ Project Structure

-   `app/`: Main backend application source code.
-   `frontend/`: All frontend files (HTML, CSS, JS).
-   `knowledge_base/`: YAML files containing the multi-language knowledge.
-   `models_store/`: Stores the trained model (`.pth`) and label files.
-   `train/`: Scripts for training the AI model.
-   `.gitignore`: Specifies files for Git to ignore.
-   `README.md`: This file.
-   `requirements.txt`: Python dependencies.

# Prototype Look Like Phase 1
- 
