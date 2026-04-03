# Comparative Analysis Engine for Super Resolution

A comprehensive web application for comparing different super-resolution algorithms side-by-side. This project features a FastAPI backend with multiple SR models and a modern React frontend for intuitive image upload and comparison.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project provides a comparative analysis platform for super-resolution (SR) algorithms. Super-resolution is the process of enhancing the resolution of an image or video. This engine allows users to:

- Upload low-resolution images
- Process them through multiple SR models simultaneously
- Compare results visually and analytically
- Understand the strengths and weaknesses of different approaches

## ✨ Features

- **Multiple SR Models**: Compare Bicubic interpolation and SRCNN
- **Web Interface**: Clean, user-friendly React interface for image upload
- **RESTful API**: FastAPI backend for robust and scalable processing
- **Real-time Processing**: Instant comparison of multiple algorithms
- **CORS Support**: Easy integration with frontend clients
- **Pre-trained Models**: Includes RealESR weights for enhanced results
- **Image Utilities**: Comprehensive preprocessing and utility functions

## 📁 Project Structure

```
├── backend/                          # FastAPI backend server
│   ├── app/
│   │   ├── api.py                   # Main API endpoints
│   │   ├── main.py                  # Standalone execution example
│   │   ├── main_api.py              # Alternative API configuration
│   │   ├── engine/
│   │   │   └── comparator.py        # ComparatorEngine for model comparison
│   │   ├── models/
│   │   │   ├── base.py              # Base model class
│   │   │   ├── bicubic.py           # Bicubic interpolation model
│   │   │   ├── srcnn.py             # SRCNN model
│   │   │   └── weights/             # Pre-trained model weights
│   │   │       └── realesr-general-x4v3.pth
│   │   └── utils/
│   │       ├── image_utils.py       # Image loading and conversion utilities
│   │       └── preprocessing.py     # Image preprocessing functions
│   ├── requirements.txt             # Python dependencies
│   └── README.md
│
├── frontend/                         # React + Vite frontend
│   ├── public/                       # Static assets
│   ├── src/
│   │   ├── App.jsx                  # Main App component
│   │   ├── App.css                  # App styling
│   │   ├── main.jsx                 # Entry point
│   │   ├── index.css                # Global styles
│   │   └── assets/                  # Additional assets
│   ├── package.json
│   ├── vite.config.js               # Vite configuration
│   ├── eslint.config.js             # ESLint rules
│   ├── index.html
│   └── README.md
│
├── LICENSE
├── README.md                         # This file
```

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI
- **Image Processing**: OpenCV (cv2)
- **Deep Learning**: PyTorch
- **HTTP Server**: Uvicorn

### Frontend
- **Library**: React 19.2
- **Build Tool**: Vite with Rolldown
- **Styling**: CSS3
- **Linting**: ESLint

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install any missing dependencies (if requirements.txt is incomplete):
   ```bash
   pip install fastapi uvicorn opencv-python numpy torch torchvision
   ```

4. Verify the models directory contains required weights:
   - Check `app/models/weights/realesr-general-x4v3.pth`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## 🚀 Usage

### Running the Backend

```bash
# From the backend directory
cd backend

# Option 1: Using FastAPI directly
python -m uvicorn app.api:app --reload --port 8000

# Option 2: Using main.py for standalone processing
python app/main.py
```

The API will be available at `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Running the Frontend

```bash
# From the frontend directory
cd frontend

# Development mode with HMR
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run linting
npm run lint
```

The frontend will be available at `http://localhost:5173`

### Complete Setup (Both Services)

```bash
# Terminal 1: Start backend
cd backend
python -m uvicorn app.api:app --reload --port 8000

# Terminal 2: Start frontend
cd frontend
npm run dev
```

## 📡 API Documentation

### Endpoints

#### POST `/compare`
Compares all available SR models on the uploaded image.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Parameter: `file` (image file)

**Response:**
```json
{
  "outputs": {
    "bicubic": "bicubic_api_out.jpg",
    "srcnn": "srcnn_api_out.jpg"
  }
}
```

#### GET `/`
Health check endpoint.

**Response:**
```json
{
  "status": "API is running"
}
```

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## 🧠 Models

### Bicubic Interpolation
- **Type**: Classical interpolation method
- **Advantage**: Fast, no training required
- **Limitation**: Limited perceptual quality
- **Use Case**: Baseline comparison

### SRCNN (Super-Resolution Convolutional Neural Network)
- **Type**: Deep learning-based SR
- **Advantage**: Better perceptual quality than bicubic
- **Architecture**: 3-layer CNN
- **Use Case**: Practical SR applications

### Planned Models
- ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks (partially implemented)
- RealESR: Real-world super-resolution
- SwinIR: Swin Transformer-based SR

## 📝 Notes

- The project includes pre-trained weights for RealESR in `backend/app/models/weights/`
- CORS is configured to accept requests from `http://localhost:5173`
- Images are processed and returned in RGB format
- Output images are saved in the working directory with model name prefixes

## 🔧 Configuration

### CORS Settings
Edit `backend/app/api.py` to modify allowed origins:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Modify as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
```

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. Code follows PEP 8 style guidelines
2. Add appropriate docstrings
3. Test new models before submitting
4. Update documentation as needed

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

**Last Updated**: April 2026
**Version**: 1.0.0
