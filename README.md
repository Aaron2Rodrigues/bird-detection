# Predictive Modeling of Bird Species using a Custom-Built Neural Network

This project classifies bird species using a custom-built neural network implemented from scratch with **NumPy**, avoiding frameworks like TensorFlow, PyTorch, or Keras. It also integrates machine learning techniques with a Flask-based web application to provide real-time predictions.

---

## Features
- **Custom-Built Neural Network**: Implemented using NumPy for efficient matrix operations.
- **Data Preprocessing**: Image resizing, normalization, and flattening for input compatibility.
- **Machine Learning Integration**: Used scikit-learn for data splitting and evaluation.
- **Model Serialization**: Persisted the trained model and label encoder with Pickle.
- **Flask Web Application**: Enables users to upload bird images and obtain predictions in real-time.

---

## Installation and Setup

### Prerequisites
Ensure you have Python installed (version 3.7 or higher). You’ll also need pip for dependency installation.

### Steps to Set Up

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/bird-species-prediction.git
   cd bird-species-prediction
2. **Installation**
   ```
   pip install -r requirements.txt
   ```
3. **Run the Flask Application Start the web application by executing**
   ```
   python app.py
   ```
4. **Project Structure**
   
  ```├── data/
│   ├── train/                   # Training dataset
│   └── test/                    # Testing dataset
│
├── model/
│   ├── neural_network.py        # Custom-built neural network implementation
│   ├── train_model.py           # Training script
│   ├── model.pkl                # Serialized trained model
│   └── label_encoder.pkl        # Serialized label encoder
│
├── static/                      # Static files for the web app (CSS, JS)
├── templates/                   # HTML templates for the web app
│
├── app.py                       # Flask application entry point
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
   ```
