# 🐍 Python Automation & System Scripting
### Advanced Utilities for System Maintenance and Process Automation

This repository contains a collection of Python-based automation tools designed to handle system-level tasks, directory management, and automated reporting.

---

## 🚀 Featured Project: Duplicate File Cleaner & Log Automation
*An intelligent system utility to maintain storage integrity by detecting and removing redundant files.*

### Key Features:
- **Accuracy:** Utilizes **MD5 Checksum-based** detection (via `hashlib`) to identify duplicate files by content rather than just name.
- **Automation:** Integrated with the **Schedule** library to perform periodic cleanups automatically.
- **Reporting:** Features an automated **Audit Log** system that records all operations with timestamps.
- **Remote Monitoring:** Uses **smtplib** to automatically email execution logs to the administrator for remote auditing.

---

## 🛠️ Technical Stack
- **Languages:** Python
- **Core Modules:** `os`, `hashlib`, `shutil`
- **Automation:** `schedule`
- **Networking:** `smtplib` (Email Automation)

---

## 📂 Project List
- **Duplicate File Detector:** Scans directories and identifies duplicates using hashing.
- **Automated Email Logger:** Sends system logs to a specified email address.
- **Periodic Task Scheduler:** Runs scripts at specific intervals (hourly, daily, or weekly).
- **Log Generator:** Creates structured text files for all system operations.

---

## 📈 How to Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/Bhavesh112004/Marvellous-Python-Assignments.git](https://github.com/Bhavesh112004/Marvellous-Python-Assignments.git)'''
   # Create the environment
2. Set Up a Virtual Environment (Recommended)
 
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate

3. Install Required Libraries
   
# pip install -r requirements.txt

4. Run the Project
For Machine Learning scripts: python [filename].py

For Automation tasks: python test.py
   
   
