import os
import uuid
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
from deepface import DeepFace
import numpy as np

# Import PSGAN components
from psgan import Inference, PostProcess
from setup import setup_config, setup_argparser
import argparse

app = Flask(__name__)

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('psgan_webapp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'  # Temporary uploads
app.config['RESULT_FOLDER'] = 'results'  # Temporary results
app.config['SAVED_UPLOADS_FOLDER'] = 'saved_uploads'  # Permanent uploads
app.config['SAVED_RESULTS_FOLDER'] = 'saved_results'  # Permanent results
app.config['REFERENCE_FACES_FOLDER'] = 'reference_faces'  # Auto-selected reference faces

# Ensure all directories exist
for folder in ['UPLOAD_FOLDER', 'RESULT_FOLDER', 'SAVED_UPLOADS_FOLDER', 'SAVED_RESULTS_FOLDER', 'REFERENCE_FACES_FOLDER']:
    os.makedirs(app.config[folder], exist_ok=True)
    logger.info(f"Tạo thư mục: {app.config[folder]}")

logger.info("Ứng dụng sử dụng chế độ tự động chọn reference face bằng DeepFace")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global variables for PSGAN
inference_model = None
postprocess_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_psgan():
    """Initialize PSGAN models"""
    global inference_model, postprocess_model
    
    try:
        logger.info("Bắt đầu khởi tạo PSGAN models...")
        
        # Setup configuration
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_file", default="configs/base.yaml")
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--model_path", default="assets/models/G.pth")
        parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
        
        args = parser.parse_args([])
        config = setup_config(args)
        
        # Initialize models
        inference_model = Inference(config, args.device, args.model_path)
        postprocess_model = PostProcess(config)
        
        logger.info("✅ PSGAN models đã được khởi tạo thành công!")
        return True
    except Exception as e:
        logger.error(f"❌ Lỗi khi khởi tạo PSGAN: {str(e)}")
        return False

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def find_most_similar_face(source_image_path):
    """Tìm khuôn mặt giống nhất với source image bằng DeepFace"""
    try:
        logger.info("Bắt đầu tìm kiếm khuôn mặt tương tự...")
        
        reference_faces_dir = app.config['REFERENCE_FACES_FOLDER']
        reference_faces = []
        
        # Get all reference face images
        for filename in os.listdir(reference_faces_dir):
            if allowed_file(filename):
                face_path = os.path.join(reference_faces_dir, filename)
                reference_faces.append(face_path)
        
        if not reference_faces:
            logger.error("Không tìm thấy reference faces")
            return None
        
        logger.info(f"Tìm thấy {len(reference_faces)} reference faces")
        
        best_match = None
        min_distance = float('inf')
        
        for ref_face_path in reference_faces:
            try:
                # So sánh khuôn mặt
                result = DeepFace.verify(
                    img1_path=source_image_path, 
                    img2_path=ref_face_path,
                    model_name='VGG-Face',  # Có thể thay đổi model
                    distance_metric='cosine'
                )
                
                distance = result['distance']
                logger.info(f"Distance với {os.path.basename(ref_face_path)}: {distance}")
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = ref_face_path
                    
            except Exception as e:
                logger.warning(f"Không thể so sánh với {ref_face_path}: {str(e)}")
                continue
        
        if best_match:
            logger.info(f"Khuôn mặt giống nhất: {os.path.basename(best_match)} (distance: {min_distance})")
            return best_match
        else:
            logger.warning("Không tìm thấy khuôn mặt phù hợp, sử dụng reference face đầu tiên")
            return reference_faces[0] if reference_faces else None
            
    except Exception as e:
        logger.error(f"Lỗi khi tìm kiếm khuôn mặt tương tự: {str(e)}")
        # Fallback: trả về reference face đầu tiên
        reference_faces_dir = app.config['REFERENCE_FACES_FOLDER']
        for filename in os.listdir(reference_faces_dir):
            if allowed_file(filename):
                return os.path.join(reference_faces_dir, filename)
        return None

@app.route('/')
def index():
    """Main page - không cần reference images nữa"""
    logger.info("Truy cập trang chính")
    return render_template('makeup_app.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and makeup transfer với auto reference selection"""
    try:
        logger.info("Bắt đầu xử lý upload file và makeup transfer")
        
        if 'source_image' not in request.files:
            logger.warning("Không có source image trong request")
            return jsonify({'error': 'No source image provided'}), 400
        
        source_file = request.files['source_image']
        
        logger.info(f"Source file: {source_file.filename}")
        
        if source_file.filename == '':
            logger.warning("Không có file được chọn")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(source_file.filename):
            logger.warning(f"File type không được hỗ trợ: {source_file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG files.'}), 400

        # Create unique session ID for this processing
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Tạo session ID: {session_id}, Timestamp: {timestamp}")
        
        # Save source image to both temp and permanent folders
        source_filename = secure_filename(f"{timestamp}_{session_id}_{source_file.filename}")
        source_temp_path = os.path.join(app.config['UPLOAD_FOLDER'], source_filename)
        source_saved_path = os.path.join(app.config['SAVED_UPLOADS_FOLDER'], source_filename)
        
        source_file.save(source_temp_path)
        source_file.seek(0)  # Reset file pointer
        source_file.save(source_saved_path)
        
        logger.info(f"Lưu source image: {source_saved_path}")
        
        # Tự động tìm reference face phù hợp nhất
        reference_path = find_most_similar_face(source_temp_path)
        if not reference_path:
            logger.error("Không tìm thấy reference face phù hợp")
            return jsonify({'error': 'No suitable reference face found'}), 500
        
        reference_filename = os.path.basename(reference_path)
        logger.info(f"Sử dụng auto-selected reference: {reference_filename}")
        
        # Check if models are initialized
        if inference_model is None or postprocess_model is None:
            logger.error("PSGAN models chưa được khởi tạo")
            return jsonify({'error': 'PSGAN models not initialized. Please restart the server.'}), 500

        # Perform makeup transfer
        logger.info("Bắt đầu quá trình makeup transfer...")
        source_image = Image.open(source_temp_path).convert("RGB")
        reference_image = Image.open(reference_path).convert("RGB")
        
        # Transfer makeup
        result_image, face = inference_model.transfer(source_image, reference_image, with_face=True)
        
        if result_image is None:
            logger.error("Không phát hiện được khuôn mặt trong ảnh source")
            return jsonify({'error': 'Face not detected in source image. Please try another image.'}), 400
        
        # Post-process the result
        logger.info("Post-processing kết quả...")
        source_crop = source_image.crop((face.left(), face.top(), face.right(), face.bottom()))
        final_result = postprocess_model(source_crop, result_image)
        
        # Save result to both temp and permanent folders
        result_filename = f"result_{timestamp}_{session_id}.png"
        result_temp_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        result_saved_path = os.path.join(app.config['SAVED_RESULTS_FOLDER'], result_filename)
        
        final_result.save(result_temp_path)
        final_result.save(result_saved_path)
        
        logger.info(f"Lưu kết quả: {result_saved_path}")
        
        # Save metadata as JSON
        metadata = {
            'session_id': session_id,
            'timestamp': timestamp,
            'source_filename': source_filename,
            'reference_filename': reference_filename,
            'reference_type': 'auto_selected',
            'result_filename': result_filename,
            'processing_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'face_similarity_used': True
        }
        
        metadata_filename = f"metadata_{timestamp}_{session_id}.json"
        metadata_path = os.path.join(app.config['SAVED_RESULTS_FOLDER'], metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Lưu metadata: {metadata_path}")
        
        # Convert images to base64 for display
        source_b64 = image_to_base64(source_image)
        reference_b64 = image_to_base64(reference_image)
        result_b64 = image_to_base64(final_result)
        
        # Clean up temporary files (keep saved ones)
        try:
            os.remove(source_temp_path)
            logger.info("Dọn dẹp temporary files thành công")
        except Exception as e:
            logger.warning(f"Không thể dọn dẹp temporary files: {e}")
        
        logger.info(f"Hoàn thành makeup transfer cho session: {session_id}")
        
        return jsonify({
            'success': True,
            'source_image': source_b64,
            'reference_image': reference_b64,
            'result_image': result_b64,
            'result_filename': result_filename,
            'session_id': session_id,
            'reference_filename': reference_filename,
            'saved_paths': {
                'source': source_saved_path,
                'reference': reference_path,
                'result': result_saved_path,
                'metadata': metadata_path
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'auto_reference_used': True
        })
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình upload và processing: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_result(filename):
    """Download result image"""
    try:
        logger.info(f"Request download file: {filename}")
        
        # Try temp results first
        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        if os.path.exists(result_path):
            logger.info(f"Download từ temp folder: {result_path}")
            return send_file(result_path, as_attachment=True)
        
        # Try saved results
        saved_result_path = os.path.join(app.config['SAVED_RESULTS_FOLDER'], filename)
        if os.path.exists(saved_result_path):
            logger.info(f"Download từ saved folder: {saved_result_path}")
            return send_file(saved_result_path, as_attachment=True)
        
        logger.warning(f"File không tồn tại: {filename}")
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Lỗi download file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/gallery')
def view_gallery():
    """View all saved results"""
    try:
        logger.info("Truy cập trang gallery")
        
        saved_results = []
        saved_results_dir = app.config['SAVED_RESULTS_FOLDER']
        
        # Get all metadata files
        metadata_count = 0
        for filename in os.listdir(saved_results_dir):
            if filename.startswith('metadata_') and filename.endswith('.json'):
                metadata_count += 1
                metadata_path = os.path.join(saved_results_dir, filename)
                
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if result file exists
                    result_path = os.path.join(saved_results_dir, metadata['result_filename'])
                    if os.path.exists(result_path):
                        saved_results.append(metadata)
                
                except Exception as e:
                    logger.warning(f"Không thể đọc metadata file {filename}: {e}")
                    continue
        
        # Sort by timestamp, newest first
        saved_results.sort(key=lambda x: x['timestamp'], reverse=True)
        
        logger.info(f"Tìm thấy {len(saved_results)}/{metadata_count} kết quả hợp lệ")
        
        return render_template('gallery.html', results=saved_results)
        
    except Exception as e:
        logger.error(f"Lỗi khi load gallery: {str(e)}")
        return jsonify({'error': f'Failed to load gallery: {str(e)}'}), 500

@app.route('/view_result/<session_id>')
def view_result(session_id):
    """View a specific result with metadata"""
    try:
        logger.info(f"Request xem result cho session: {session_id}")
        
        # Find metadata file for this session
        saved_results_dir = app.config['SAVED_RESULTS_FOLDER']
        metadata_files = [f for f in os.listdir(saved_results_dir) 
                         if f.startswith('metadata_') and session_id in f]
        
        if not metadata_files:
            logger.warning(f"Không tìm thấy metadata cho session: {session_id}")
            return jsonify({'error': 'Result not found'}), 404
        
        metadata_path = os.path.join(saved_results_dir, metadata_files[0])
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get image paths
        result_path = os.path.join(saved_results_dir, metadata['result_filename'])
        source_path = os.path.join(app.config['SAVED_UPLOADS_FOLDER'], metadata['source_filename'])
        
        # Reference path là từ reference_faces folder
        reference_path = os.path.join(app.config['REFERENCE_FACES_FOLDER'], metadata['reference_filename'])
        
        # Convert to base64 for display
        result_b64 = None
        source_b64 = None
        reference_b64 = None
        
        if os.path.exists(result_path):
            result_image = Image.open(result_path)
            result_b64 = image_to_base64(result_image)
            logger.info("Load result image thành công")
        
        if os.path.exists(source_path):
            source_image = Image.open(source_path)
            source_b64 = image_to_base64(source_image)
            logger.info("Load source image thành công")
        
        if os.path.exists(reference_path):
            reference_image = Image.open(reference_path)
            reference_b64 = image_to_base64(reference_image)
            logger.info("Load reference image thành công")
        
        logger.info(f"Trả về kết quả cho session: {session_id}")
        
        return jsonify({
            'success': True,
            'metadata': metadata,
            'images': {
                'source': source_b64,
                'reference': reference_b64,
                'result': result_b64
            }
        })
        
    except Exception as e:
        logger.error(f"Lỗi khi load result: {str(e)}")
        return jsonify({'error': f'Failed to load result: {str(e)}'}), 500

@app.route('/delete_result/<session_id>', methods=['DELETE'])
def delete_result(session_id):
    """Delete a saved result and its files"""
    try:
        logger.info(f"Request xóa result cho session: {session_id}")
        
        # Find and load metadata
        saved_results_dir = app.config['SAVED_RESULTS_FOLDER']
        metadata_files = [f for f in os.listdir(saved_results_dir) 
                         if f.startswith('metadata_') and session_id in f]
        
        if not metadata_files:
            logger.warning(f"Không tìm thấy metadata cho session: {session_id}")
            return jsonify({'error': 'Result not found'}), 404
        
        metadata_path = os.path.join(saved_results_dir, metadata_files[0])
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Delete files (không xóa reference vì nó được dùng chung)
        files_to_delete = [
            os.path.join(saved_results_dir, metadata['result_filename']),
            os.path.join(app.config['SAVED_UPLOADS_FOLDER'], metadata['source_filename']),
            metadata_path
        ]
        
        deleted_files = []
        for file_path in files_to_delete:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    logger.info(f"Xóa file: {file_path}")
            except Exception as e:
                logger.warning(f"Không thể xóa file {file_path}: {e}")
        
        logger.info(f"Xóa thành công {len(deleted_files)} files cho session: {session_id}")
        
        return jsonify({
            'success': True,
            'message': f'Deleted {len(deleted_files)} files',
            'deleted_files': deleted_files
        })
        
    except Exception as e:
        logger.error(f"Lỗi khi xóa result: {str(e)}")
        return jsonify({'error': f'Failed to delete result: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    logger.info("Health check request")
    model_status = "initialized" if (inference_model and postprocess_model) else "not initialized"
    return jsonify({
        'status': 'running',
        'models': model_status,
        'auto_reference': True,
        'reference_faces_count': len([f for f in os.listdir(app.config['REFERENCE_FACES_FOLDER']) if allowed_file(f)]),
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(413)
def too_large(e):
    logger.warning("File upload quá lớn")
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    logger.warning(f"Endpoint không tồn tại: {request.url}")
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Đang khởi động PSGAN Web Application với Auto Face Matching...")
    logger.info("Khởi động ứng dụng web PSGAN với DeepFace")
    
    # Initialize PSGAN models
    if init_psgan():
        logger.info("🌟 PSGAN Web App với DeepFace đã sẵn sàng!")
        print("🌟 PSGAN Web App với DeepFace đã sẵn sàng!")
        print("📝 Truy cập ứng dụng tại: http://localhost:5000")
        print("📄 Log được lưu vào file: psgan_webapp.log")
        print("🎭 Ứng dụng sẽ tự động chọn reference face phù hợp nhất!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("❌ Không thể khởi tạo PSGAN models")
        print("❌ Không thể khởi tạo PSGAN models. Vui lòng kiểm tra setup.")
