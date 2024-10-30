# Delhi Housing Price Prediction

This project provides an interactive tool for predicting housing prices in Delhi, leveraging machine learning and data science techniques. Using various features, including area, location, number of rooms, and property amenities, the model delivers precise price predictions, empowering users to make informed real estate decisions.

---

## 🚀 Project Overview

The Delhi Housing Price Prediction project utilizes the **XGBoost Regressor**, a powerful and highly accurate machine learning model, to forecast housing prices based on historical and geographical data. The app’s backend is built on **Flask**, and the application is deployed on **Render** for broad accessibility. 

### Key Features

- **Accurate Price Predictions**: Predicts housing prices based on essential property features.
- **User-Friendly Interface**: A web-based front end with HTML, CSS, and JavaScript.
- **Containerized Environment**: Dockerized application ensures easy deployment and scalability.

---

## 🛠️ Technology Stack

- **Machine Learning**: XGBoost Regressor (trained using Scikit-Learn)
- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Containerization**: Docker
- **Deployment**: Render

---

## 📂 Project Structure

```plaintext
DelhiHousingPricePrediction/
├── src/
│   ├── app.py                 # Main Flask app
│   ├── model_app.py           # Handles model loading and prediction
│   └── setup.py               # Setup configuration
├── templates/                 # HTML templates for web pages
├── static/                    # CSS and JavaScript files
├── artifacts/
│   └── model.pkl              # Trained model file (tracked with Git LFS)
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

---

## 📊 Dataset

The dataset used for training includes features like property area, location, room counts, parking availability, and more, sourced from [Kaggle](https://lnkd.in/grGZU3cw).

---

## 🖥️ Getting Started

### Prerequisites

Ensure Python 3.8+ is installed, along with Docker (if you want to use containerization). Install required packages with:

```bash
pip install -r requirements.txt
```

### Running the Project Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/DelhiHousingPricePrediction.git
   ```

2. Navigate to the project folder:

   ```bash
   cd DelhiHousingPricePrediction
   ```

3. Run the Flask application:

   ```bash
   flask run
   ```

The app will be accessible at `http://localhost:5000`.

### Docker Setup

1. Build the Docker image:

   ```bash
   docker build -t delhi-housing-prediction .
   ```

2. Run the Docker container:

   ```bash
   docker run -p 5000:5000 delhi-housing-prediction
   ```

---

## 🎯 Usage Guide

1. **Input Property Details**: Enter information such as area, number of rooms, parking, and location.
2. **Get Price Prediction**: Click the predict button to see the estimated price for the property.

---

## 🧠 Future Enhancements

- **Additional Feature Integration**: Add location-based amenities like schools or hospitals.
- **Enhanced UI/UX**: Improve frontend with responsive design.
- **Cloud Hosting**: Migrate to cloud-based database and hosting.

---

## 🔗 Links

- **GitHub Repository**: [Delhi Housing Price Prediction](https://lnkd.in/gZ4KrxBR)
- **Deployed Web App**: [Explore the App](https://lnkd.in/gV4Js7PZ)

---

## 🤝 Contributing

Contributions are welcome! 

---

## 🛡️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

Thanks to **Render** for hosting support and **Kaggle** for the dataset.
