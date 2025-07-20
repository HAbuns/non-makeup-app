# PSGAN Project Structure

## Cấu trúc thư mục đã được tổ chức lại:

```
PSGAN/
├── app.py                          # Main entry point
├── requirements.txt                # Dependencies
├── setup.py                       # Setup script
├── README.md                      # Project documentation
├── WEB_APP_README.md              # Web app specific documentation
│
├── web_app/                       # Web Application
│   ├── backend/                   # Flask backend files
│   │   ├── app.py                # Main Flask app (original)
│   │   ├── app_backup.py         # Backup version
│   │   └── app_new.py            # New version
│   ├── templates/                 # HTML templates
│   │   ├── makeup_app.html       # Main app template
│   │   └── gallery.html          # Gallery template
│   └── static/                    # Static web assets
│
├── models/                        # ML Models & Core Logic
│   ├── psgan/                    # PSGAN model implementation
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── inference.py
│   │   ├── net.py
│   │   ├── solver.py
│   │   └── ...
│   ├── faceutils/                # Face processing utilities
│   │   ├── faceplusplus.py
│   │   ├── dlibutils/
│   │   └── mask/
│   ├── data_loaders/             # Data loading utilities
│   │   └── makeup_dataloader.py
│   └── concern/                  # Additional model utilities
│       ├── image.py
│       ├── track.py
│       └── visualize.py
│
├── datasets/                      # Data & Results
│   ├── data/                     # Training/validation data (if any)
│   ├── reference_faces/          # Reference face images for makeup transfer
│   └── results/                  # Output results
│       ├── results/              # Generated results
│       ├── saved_results/        # Saved results
│       ├── saved_uploads/        # Saved uploaded files
│       └── uploads/              # Temporary uploads
│
├── utils/                         # Utilities & Tools
│   ├── scripts/                  # Utility scripts
│   │   ├── get_face.py
│   │   ├── get_lms.py
│   │   └── ...
│   ├── tools/                    # Development tools
│   │   ├── data_reader.py
│   │   ├── inception_score.py
│   │   └── plot.py
│   ├── configs/                  # Configuration files
│   │   └── base.yaml
│   └── ops/                      # Custom operations
│       ├── histogram_loss.py
│       ├── spectral_norm.py
│       └── ...
│
├── tests/                         # Test files
│   ├── test_api.py
│   └── test_webapp.py
│
├── logs/                          # Log files
│   ├── beauty_transform_app.log
│   └── psgan_webapp.log
│
└── docs/                          # Documentation & Assets
    ├── images/                    # Documentation images
    │   ├── MT-results.png
    │   ├── MWild-results.png
    │   ├── Video_MT.png
    │   └── psgan_framework.png
    └── assets/                    # Additional assets
        ├── assets/                # Original assets folder
        └── backup_reference_faces/ # Backup reference faces
```

## Chạy ứng dụng:

```bash
python app.py
```

Hoặc:

```bash
python web_app/backend/app.py
```
