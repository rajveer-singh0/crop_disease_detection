# Crop Disease Detection Web Application

This is a production-ready web application for crop disease detection using a pre-trained deep learning model. The application allows users to upload images of plant leaves and get predictions about potential diseases along with treatment recommendations.

## Features

- User-friendly web interface for image upload
- Real-time disease detection using a pre-trained TensorFlow model
- Detailed results including disease name, confidence score, description, and treatment recommendations
- Responsive design that works on desktop and mobile devices
- Error handling and validation for production use

## Prerequisites

- Python 3.8 or higher
- Pip package manager
- Pre-trained model file (`Crop_Disease_Detection.keras`)

## Installation

1. Clone this repository or download the source code

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure your model file (`Crop_Disease_Detection.keras`) is in the root directory of the project

## Running the Application Locally

1. Start the Flask development server:

```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

## Deployment Options

### Deploying to Heroku

1. Create a Heroku account and install the Heroku CLI

2. Create a new Heroku app:

```bash
heroku create your-app-name
```

3. Add a Procfile to the root directory with the following content:

```
web: gunicorn app:app
```

4. Deploy to Heroku:

```bash
git init
git add .
git commit -m "Initial commit"
git push heroku master
```

### Deploying with Docker

1. Create a Dockerfile in the root directory with the following content:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

2. Build and run the Docker container:

```bash
docker build -t crop-disease-detection .
docker run -p 5000:5000 crop-disease-detection
```

## Project Structure

```
crop_disease_detection/
├── app.py                  # Main Flask application
├── Crop_Disease_Detection.keras  # Pre-trained model file
├── requirements.txt        # Python dependencies
├── static/                 # Static files
│   ├── css/
│   │   └── style.css      # Custom CSS styles
│   └── js/
│       └── main.js        # JavaScript for the frontend
├── templates/              # HTML templates
│   └── index.html         # Main page template
└── uploads/               # Directory for uploaded images (created automatically)
```

## Customization

- To add more disease information, update the `disease_info` dictionary in `app.py`
- To modify the UI, edit the files in the `templates` and `static` directories
- To change the model, replace the `Crop_Disease_Detection.keras` file and update the preprocessing function if needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.