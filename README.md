# Age & Gender Detection Mini Project

A comprehensive web application that detects faces in images and predicts age range and gender using deep learning models. This project demonstrates computer vision, machine learning, and web development concepts in a single, production-ready application.

## 🌟 Features

### Core Functionality
- **Multi-Face Detection**: Detects and analyzes multiple faces in a single image
- **Age Range Prediction**: Classifies age into 8 ranges: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)
- **Gender Classification**: Predicts Male/Female with confidence scores
- **Real-time Processing**: Live camera feed with real-time age and gender detection
- **Image Upload**: Support for various image formats with preview functionality

### Technical Features
- **Automatic Model Download**: Downloads pre-trained Caffe models automatically on first use (~70MB total)
- **Multiple Fallback Sources**: Robust model downloading with multiple mirror URLs
- **Demo Mode**: Automatically switches to demo mode with sample images when no camera is available
- **Professional UI**: Modern dark theme with responsive design
- **Confidence Scores**: Displays prediction confidence percentages for each detection
- **Annotated Results**: Returns processed images with bounding boxes and labels

## 🛠 Technology Stack

### Backend
- **Python 3.13+** - Core programming language
- **Flask 3.0.3** - Web framework
- **OpenCV 4.10** - Computer vision library
- **NumPy 1.26** - Numerical computations
- **Requests 2.32** - HTTP client for model downloads

### Frontend
- **HTML5/CSS3** - Modern web standards
- **JavaScript (ES6+)** - Interactive functionality
- **Responsive Design** - Works on desktop and mobile devices

### Deep Learning Models
- **Haar Cascade Classifier** - Face detection
- **Pre-trained Caffe Models** - Age and gender prediction
  - Age model: Based on IMDB-WIKI dataset
  - Gender model: Binary classification with high accuracy

## 📁 Project Structure

```
workspace/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── static/               # Frontend assets
│   ├── style.css        # Modern dark theme styles
│   ├── script.js        # Image upload functionality
│   └── camera.js        # Real-time camera interface
├── templates/            # HTML templates
│   ├── index.html       # Main page with image upload
│   └── camera.html      # Real-time detection page
├── models/              # Auto-downloaded deep learning models
│   ├── age_deploy.prototxt      # Age model architecture
│   ├── age_net.caffemodel       # Age model weights (45.6MB)
│   ├── gender_deploy.prototxt   # Gender model architecture
│   └── gender_net.caffemodel    # Gender model weights (45.6MB)
└── samples/             # Demo images (auto-downloaded)
```

## 🚀 Setup & Installation

### Prerequisites
- Python 3.13 or higher
- pip package manager
- 70MB free disk space for models
- Webcam (optional, demo mode available without)

### Installation Steps

1. **Clone/Navigate to Project Directory**
   ```bash
   cd /workspace
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Application**
   ```bash
   python app.py
   ```

5. **Access Application**
   - Open browser to: `http://localhost:5000`
   - Application will automatically download models on first use

## 📖 Usage Guide

### Image Upload Mode
1. Navigate to `http://localhost:5000`
2. Click "Choose an image" and select a photo with faces
3. Click "Analyze" to process the image
4. View results with:
   - Annotated image showing detected faces
   - Individual face analysis cards with age/gender predictions
   - Confidence scores for each prediction

### Real-time Camera Mode
1. Click "📹 Live Camera Detection" on the main page
2. Click "Start Camera" to begin real-time detection
3. View live feed with automatic face detection and labeling
4. Performance metrics displayed (FPS counter)
5. Falls back to demo mode if no camera is detected

### API Endpoints

#### Health Check
```
GET /healthz
Response: {"status": "ok"}
```

#### Image Analysis
```
POST /analyze
Body: multipart/form-data with 'image' file
Response: {
  "faces": [
    {
      "box": {"x": 100, "y": 100, "w": 200, "h": 200},
      "gender": "Male",
      "gender_confidence": 0.8542,
      "age_range": "(25-32)",
      "age_confidence": 0.7234
    }
  ],
  "annotated_image": "data:image/png;base64,..."
}
```

#### Camera Status
```
GET /camera_status
Response: {
  "camera_available": false,
  "demo_mode": true,
  "message": "Demo mode with sample images"
}
```

#### Video Feed
```
GET /video_feed
Response: multipart/x-mixed-replace stream
```

## 🧠 Technical Implementation

### Face Detection Pipeline
1. **Image Preprocessing**: Resize and normalize input images
2. **Face Detection**: Haar cascade classifier locates faces
3. **Feature Extraction**: Extract face regions for analysis
4. **Age/Gender Prediction**: Deep neural networks classify each face
5. **Post-processing**: Annotate image with results and confidence scores

### Model Architecture
- **Age Model**: CNN trained on IMDB-WIKI dataset with 8-class output
- **Gender Model**: Binary classification CNN with high accuracy
- **Input Size**: 227x227 pixels per face
- **Preprocessing**: Mean subtraction with values (78.4, 87.7, 114.9)

### Performance Optimizations
- **Lazy Loading**: Models loaded only when needed
- **Caching**: Efficient model reuse across requests
- **Error Handling**: Graceful fallbacks for model download failures
- **Memory Management**: Optimized image processing pipeline

## 🎨 User Interface

### Design Features
- **Dark Theme**: Modern aesthetic with excellent contrast
- **Glass Morphism**: Subtle transparency effects
- **Responsive Layout**: Adapts to different screen sizes
- **Smooth Animations**: Hover effects and transitions
- **Progressive Enhancement**: Works without JavaScript for basic functionality

### Accessibility
- **Semantic HTML**: Proper heading structure and landmarks
- **High Contrast**: WCAG-compliant color schemes
- **Keyboard Navigation**: Full keyboard accessibility
- **Screen Reader Support**: Descriptive alt text and labels

## 🔧 Configuration

### Environment Variables
- `PORT`: Server port (default: 5000)

### Customization Options
- **Age Ranges**: Modify `AGE_BUCKETS` in `app.py`
- **Model Sources**: Update URLs in model download configuration
- **UI Theme**: Customize CSS variables in `style.css`
- **Sample Images**: Modify `SAMPLE_IMAGE_URLS` for demo mode

## 📊 Model Performance

### Age Detection
- **Accuracy**: ~60-70% for exact age group
- **Tolerance**: Higher accuracy within ±1 age group
- **Bias**: Optimized for clear, front-facing images

### Gender Detection
- **Accuracy**: ~90%+ for clear images
- **Robustness**: Handles various lighting conditions
- **Speed**: Real-time processing capability

## 🛡 Security & Privacy

### Data Handling
- **No Storage**: Images are processed in memory only
- **No Logging**: Personal data is not logged or stored
- **Local Processing**: All computation happens locally
- **No Tracking**: No analytics or tracking implemented

### Best Practices
- **Input Validation**: Secure file upload handling
- **Error Handling**: Graceful error responses
- **Resource Limits**: Protection against DoS attacks
- **CORS Configuration**: Controlled cross-origin access

## 🐛 Troubleshooting

### Common Issues

**Models Not Downloading**
- Check internet connection
- Verify firewall allows downloads
- Try restarting the application

**Camera Not Working**
- Grant browser camera permissions
- Check if camera is used by another application
- Demo mode will activate automatically as fallback

**Poor Detection Accuracy**
- Use clear, well-lit, front-facing photos
- Ensure faces are clearly visible
- Try different image formats or sizes

**Performance Issues**
- Close other resource-intensive applications
- Use smaller image files
- Check available system memory

## 🎓 Educational Value

This project demonstrates key concepts in:

### Computer Vision
- Image preprocessing and normalization
- Face detection algorithms
- Feature extraction techniques
- Real-time video processing

### Machine Learning
- Pre-trained model usage
- Classification tasks
- Confidence scoring
- Model evaluation and bias

### Web Development
- RESTful API design
- Real-time data streaming
- Progressive web applications
- Responsive user interfaces

### Software Engineering
- Error handling and graceful degradation
- Modular code architecture
- Documentation and testing
- Production deployment considerations

## 📈 Future Enhancements

### Potential Improvements
- **Emotion Detection**: Add facial expression analysis
- **Multiple Model Support**: Integrate different detection algorithms
- **Batch Processing**: Handle multiple images simultaneously
- **Export Features**: Save results in various formats
- **Advanced Analytics**: Age/gender distribution charts
- **Mobile App**: Native mobile application version

### Research Opportunities
- **Model Optimization**: Reduce model size and improve speed
- **Bias Mitigation**: Address demographic biases in predictions
- **Dataset Expansion**: Train on more diverse datasets
- **Edge Deployment**: Optimize for edge computing devices

## 📜 License & Attribution

### Model Credits
- **Age/Gender Models**: Based on research by Gil Levi and Tal Hassner
- **Haar Cascades**: OpenCV community contributions
- **Sample Images**: Unsplash (demo mode only)

### Dependencies
- All dependencies listed in `requirements.txt` with their respective licenses
- OpenCV: Apache License 2.0
- Flask: BSD License
- NumPy: BSD License

## 👥 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Implement improvements with tests
4. Submit pull request with detailed description

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings for functions
- Include error handling

---

**Note**: This project is designed for educational purposes and demonstrates practical applications of computer vision and machine learning. For production use, consider additional security measures, performance optimizations, and comprehensive testing.

**Accuracy Disclaimer**: Age and gender predictions are estimates based on facial features and may not always be accurate. Results should not be used for critical decision-making without additional validation.
