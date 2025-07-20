# PSGAN Web Application

🚀 **AI-Powered Makeup Transfer Web App** built with Flask and PSGAN

## ✨ Features

- **Modern UI**: Beautiful, responsive interface with gradient backgrounds and smooth animations
- **Drag & Drop Upload**: Easy image upload with drag-and-drop support
- **Dual Reference Options**: 
  - Choose from pre-loaded makeup gallery
  - Upload custom reference makeup photos
- **Real-time Processing**: Live feedback with loading indicators
- **Download Results**: Direct download of processed images
- **Mobile Responsive**: Works perfectly on all device sizes

## 🖼️ Screenshots

### Main Interface
- Clean, modern design with gradient backgrounds
- Intuitive upload areas with visual feedback
- Gallery of pre-loaded makeup styles

### Processing
- Real-time loading indicators
- Progress feedback during AI processing
- Error handling with user-friendly messages

### Results
- Side-by-side comparison view
- Download functionality for results
- Responsive image grid

## 🛠️ Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Assets
Make sure you have:
- Model file: `assets/models/G.pth`
- Reference makeup images in: `assets/images/makeup/`
- Configuration file: `configs/base.yaml`

### 3. Run the Application
```bash
python app.py
```

### 4. Access the App
Open your browser and go to: `http://localhost:5000`

## 📁 Project Structure

```
PSGAN/
├── app.py                 # Flask web application
├── templates/
│   └── makeup_app.html   # Modern HTML template
├── static/
│   └── images/           # Static assets
├── uploads/              # Temporary upload directory
├── results/              # Processed results directory
├── assets/
│   ├── models/           # PSGAN model files
│   └── images/
│       └── makeup/       # Reference makeup images
└── requirements.txt      # Python dependencies
```

## 🎨 UI/UX Features

### Design Elements
- **Gradient Backgrounds**: Beautiful purple-blue gradients
- **Glass Morphism**: Frosted glass effects with backdrop blur
- **Smooth Animations**: Hover effects and transitions
- **Responsive Grid**: Adaptive layouts for all screen sizes

### Interactive Components
- **File Upload Areas**: Visual drag-and-drop zones
- **Image Gallery**: Clickable reference makeup selection
- **Progress Indicators**: Animated loading spinners
- **Alert System**: Success/error message notifications

### Accessibility
- **Keyboard Navigation**: Full keyboard support
- **Screen Reader Friendly**: Proper ARIA labels
- **High Contrast**: Clear visual hierarchy
- **Mobile Optimized**: Touch-friendly interface

## 🔧 Technical Features

### Backend (Flask)
- **File Upload Handling**: Secure file processing with size limits
- **Image Processing**: Integration with PSGAN models
- **Error Handling**: Comprehensive error management
- **API Endpoints**: RESTful design for frontend communication

### Frontend (HTML/CSS/JS)
- **Modern CSS**: Flexbox/Grid layouts with CSS variables
- **Vanilla JavaScript**: No external JS dependencies
- **Responsive Design**: Mobile-first approach
- **Progressive Enhancement**: Works without JavaScript for basic functionality

### Security Features
- **File Type Validation**: Only allow image files
- **Size Limitations**: 16MB maximum file size
- **Secure Filenames**: Sanitized file naming
- **Path Traversal Protection**: Secure file handling

## 🚀 Usage Guide

### For Users
1. **Upload Your Photo**: Click or drag-drop your source image
2. **Choose Makeup Style**: 
   - Select from gallery of pre-loaded styles
   - Or upload your own reference photo
3. **Process**: Click "Apply Makeup Transfer" button
4. **Download**: Save your transformed image

### For Developers
- **Customize UI**: Edit `templates/makeup_app.html` and CSS
- **Add Features**: Extend Flask routes in `app.py`
- **Model Integration**: Modify PSGAN integration as needed
- **Deploy**: Use production WSGI server (gunicorn, uwsgi)

## 📱 Browser Compatibility

- ✅ Chrome (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge
- ✅ Mobile browsers

## 🐛 Troubleshooting

### Common Issues
1. **Models not loading**: Check model file paths in `assets/models/`
2. **Reference images not showing**: Verify `assets/images/makeup/` contains images
3. **Upload fails**: Check file size (max 16MB) and format (PNG/JPG)
4. **Processing errors**: Ensure face is clearly visible in source image

### Debug Mode
Run with debug enabled:
```bash
export FLASK_DEBUG=1
python app.py
```

## 🔮 Future Enhancements

- [ ] User account system
- [ ] Image history and favorites
- [ ] Batch processing
- [ ] Advanced makeup controls
- [ ] Social sharing features
- [ ] Real-time camera integration
- [ ] Mobile app version

## 📄 License

This project follows the same license as the original PSGAN research work.

## 🙏 Acknowledgments

- Original PSGAN research team
- Flask web framework
- Modern CSS design patterns
- Open source community
