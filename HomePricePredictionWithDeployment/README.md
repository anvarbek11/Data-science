![amazondeployment](https://github.com/user-attachments/assets/5bd8b9a8-5592-444e-92d3-168bb9d01de2)
# 🏡 Real Estate Price Prediction Website

This is a full-stack **Data Science Project Series** that walks you through building and deploying a **Real Estate Price Prediction** website from scratch. 

We'll go step-by-step through:

- 🧠 Building a machine learning model to predict home prices in Bangalore.
- 🌐 Creating a Flask API to serve the model.
- 🎨 Designing a front-end UI with HTML/CSS/JavaScript.
- ☁️ Deploying the app on AWS EC2 with Nginx as a reverse proxy.

---

## 📊 Project Overview

### 🔹 Step 1: Build the ML Model

We use **Linear Regression** with the **Bangalore Home Prices** dataset from [Kaggle](https://www.kaggle.com/). During this step, we cover:

- Data loading & cleaning
- Outlier detection & removal
- Feature engineering
- Dimensionality reduction
- GridSearchCV for hyperparameter tuning
- K-Fold cross-validation

### 🔹 Step 2: Create Flask API

- Build a Python Flask server
- Load and use the trained model
- Handle HTTP requests from the UI and return predictions

### 🔹 Step 3: Frontend Web App

- Built using **HTML**, **CSS**, and **JavaScript**
- Takes user inputs like square footage, number of bedrooms, etc.
- Sends input data to Flask API and displays predicted price

---

## 🛠️ Tech Stack

| Category          | Tools Used                        |
|-------------------|-----------------------------------|
| Programming       | Python                            |
| Data Processing   | Pandas, NumPy                     |
| Visualization     | Matplotlib                        |
| ML Model          | Scikit-learn (Linear Regression)  |
| IDEs              | Jupyter Notebook, VS Code, PyCharm|
| API Server        | Python Flask                      |
| Frontend          | HTML, CSS, JavaScript             |
| Deployment        | AWS EC2, Nginx                    |

---

## ☁️ Cloud Deployment Guide (AWS EC2 + Nginx)

### 🔸 1. Create EC2 Instance

- Use AWS Console
- Add a security group rule to allow **HTTP (port 80)** traffic

### 🔸 2. Connect to EC2 via SSH
ssh -i "C:\Users\YourUser\.ssh\BHP.pem" ubuntu@your-ec2-public-dns

### 🔸 3. Install Nginx

sudo apt-get update
sudo apt-get install nginx
sudo service nginx status  # Check status
Start/Stop/Restart commands:

### ▶️ Nginx Control
sudo service nginx start
sudo service nginx stop
sudo service nginx restart
Now visiting your public EC2 URL should show "Welcome to Nginx".

### 🚀 Deploy Your App
### 🔸 4. Upload Your Code
Use WinSCP to copy all project files to:

/home/ubuntu/BangloreHomePrices
### 🔸 5. Configure Nginx
Create a new file:


sudo nano /etc/nginx/sites-available/bhp.conf
Paste this config:

server {
    listen 80;
    server_name bhp;

    root /home/ubuntu/BangloreHomePrices/client;
    index app.html;

    location /api/ {
        rewrite ^/api(.*) $1 break;
        proxy_pass http://127.0.0.1:5000;
    }
}
Create symlink and remove default:

sudo ln -s /etc/nginx/sites-available/bhp.conf /etc/nginx/sites-enabled/
sudo unlink /etc/nginx/sites-enabled/default
sudo service nginx restart
### 🧪 Install Dependencies & Run Flask Server

sudo apt-get install python3-pip
sudo pip3 install -r /home/ubuntu/BangloreHomePrices/server/requirements.txt

# Start the Flask server
python3 /home/ubuntu/BangloreHomePrices/server/server.py
### 🌐 Final Step
Visit your EC2 public URL in the browser: In my case link was like this 
http://ec2-56-228-13-222.eu-north-1.compute.amazonaws.com

