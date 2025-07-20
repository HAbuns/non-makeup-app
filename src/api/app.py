import os
import sys
import uuid
import json
import logging
import pickle
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
from deepface import DeepFace
import numpy as np

# Add paths for the clean project structure
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'src', 'models'))
sys.path.append(os.path.join(project_root, 'src', 'utils'))

# Import PSGAN components
from psgan import Inference, PostProcess
from setup import setup_config, setup_argparser
import argparse

app = Flask(__name__, 
           template_folder=os.path.join(project_root, 'src', 'core', 'templates'),
           static_folder=os.path.join(project_root, 'src', 'core', 'static'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'beauty_transform_app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(project_root, 'temp', 'datasets', 'results', 'uploads')  # Temporary uploads
app.config['RESULT_FOLDER'] = os.path.join(project_root, 'temp', 'datasets', 'results', 'results')  # Temporary results
app.config['SAVED_UPLOADS_FOLDER'] = os.path.join(project_root, 'temp', 'datasets', 'results', 'saved_uploads')  # Permanent uploads
app.config['SAVED_RESULTS_FOLDER'] = os.path.join(project_root, 'temp', 'datasets', 'results', 'saved_results')  # Permanent results
app.config['REFERENCE_FACES_FOLDER'] = os.path.join(project_root, 'temp', 'datasets', 'reference_faces')  # Auto-selected reference faces

# Ensure all directories exist
for folder in ['UPLOAD_FOLDER', 'RESULT_FOLDER', 'SAVED_UPLOADS_FOLDER', 'SAVED_RESULTS_FOLDER', 'REFERENCE_FACES_FOLDER']:
    os.makedirs(app.config[folder], exist_ok=True)
    logger.info(f"Created directory: {app.config[folder]}")

logger.info("Application using automatic face matching with DeepFace")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global variables for PSGAN
inference_model = None
postprocess_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_models():
    """Initialize PSGAN models"""
    global inference_model, postprocess_model
    
    try:
        logger.info("Starting model initialization...")
        
        # Setup configuration
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_file", default=os.path.join(project_root, "src", "utils", "configs", "base.yaml"))
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--model_path", default=os.path.join(project_root, "docs", "assets", "models", "G.pth"))
        parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
        
        args = parser.parse_args([])
        config = setup_config(args)
        
        # Initialize models
        inference_model = Inference(config, args.device, args.model_path)
        postprocess_model = PostProcess(config)
        
        logger.info("✅ Models initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Error initializing models: {str(e)}")
        return False

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def get_embeddings_cache_path():
    """Get the path for embeddings cache file"""
    return os.path.join(app.config['REFERENCE_FACES_FOLDER'], 'embeddings_cache.pkl')

def compute_face_embedding(image_path, model_name='VGG-Face'):
    """Compute face embedding for a single image"""
    try:
        result = DeepFace.represent(
            img_path=image_path, 
            model_name=model_name,
            enforce_detection=False  # More flexible face detection
        )
        
        # Debug: Print the actual output to understand format
        # logger.info(f"DeepFace output type for {os.path.basename(image_path)}: {type(result)}")
        # if isinstance(result, list) and len(result) > 0:
        #     logger.info(f"First item type: {type(result[0])}")
        #     if isinstance(result[0], dict):
        #         logger.info(f"Dict keys: {list(result[0].keys())}")
        # elif isinstance(result, dict):
        #     logger.info(f"Dict keys: {list(result.keys())}")
        
        # Handle different result formats
        if isinstance(result, list):
            # Check if it's a list of floats (direct embedding)
            if len(result) > 0 and isinstance(result[0], float):
                return np.array(result)
            # Or a list of dictionaries
            elif len(result) > 0 and isinstance(result[0], dict) and 'embedding' in result[0]:
                return np.array(result[0]['embedding'])
            else:
                logger.warning(f"No face detected or unexpected format in {image_path}")
                return None
        elif isinstance(result, dict):
            # Old format: single dictionary
            if 'embedding' in result:
                return np.array(result['embedding'])
            else:
                logger.warning(f"No 'embedding' key in result dict")
                return None
        elif isinstance(result, np.ndarray):
            # Direct embedding array
            return np.array(result)
        else:
            logger.warning(f"Unknown result format for {image_path}: {type(result)}")
            return None
            
    except Exception as e:
        logger.warning(f"Could not compute embedding for {image_path}: {str(e)}")
        return None

def load_or_create_embeddings_cache():
    """Load existing embeddings or create new ones"""
    reference_faces_dir = app.config['REFERENCE_FACES_FOLDER']
    cache_path = get_embeddings_cache_path()
    
    # Get current reference face files
    current_files = {}
    for filename in os.listdir(reference_faces_dir):
        if allowed_file(filename):
            file_path = os.path.join(reference_faces_dir, filename)
            # Use file modification time as version check
            current_files[filename] = os.path.getmtime(file_path)
    
    # Try to load existing cache
    embeddings_cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Check if cache is still valid
            cached_files = cache_data.get('files', {})
            if cached_files == current_files:
                logger.info("Using cached embeddings")
                return cache_data.get('embeddings', {})
            else:
                logger.info("Cache outdated, recomputing embeddings")
        except Exception as e:
            logger.warning(f"Could not load embeddings cache: {str(e)}")
    
    # Compute new embeddings
    logger.info("Computing new embeddings for reference faces...")
    embeddings_cache = {}
    
    for filename, mtime in current_files.items():
        file_path = os.path.join(reference_faces_dir, filename)
        logger.info(f"Computing embedding for {filename}")
        
        embedding = compute_face_embedding(file_path)
        if embedding is not None:
            embeddings_cache[filename] = {
                'embedding': embedding,
                'path': file_path
            }
    
    # Save cache
    try:
        cache_data = {
            'files': current_files,
            'embeddings': embeddings_cache
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"Saved embeddings cache with {len(embeddings_cache)} faces")
    except Exception as e:
        logger.warning(f"Could not save embeddings cache: {str(e)}")
    
    return embeddings_cache

def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    # Normalize vectors
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return np.dot(embedding1, embedding2) / (norm1 * norm2)

def find_most_similar_face_fast(source_image_path):
    """Find the most similar face using pre-computed embeddings (FAST VERSION)"""
    try:
        logger.info("Starting fast face similarity search with embeddings...")
        
        # Load or create embeddings cache
        embeddings_cache = load_or_create_embeddings_cache()
        
        if not embeddings_cache:
            logger.error("No reference face embeddings available")
            return None
        
        # Compute embedding for source image
        logger.info("Computing embedding for source image")
        source_embedding = compute_face_embedding(source_image_path)
        if source_embedding is None:
            logger.error("Could not compute embedding for source image")
            return None
        
        # Find best match using cosine similarity
        best_match = None
        best_similarity = -1  # Cosine similarity ranges from -1 to 1
        
        for filename, data in embeddings_cache.items():
            reference_embedding = data['embedding']
            similarity = cosine_similarity(source_embedding, reference_embedding)
            
            logger.info(f"Similarity with {filename}: {similarity:.4f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = data['path']
        
        if best_match:
            logger.info(f"Best matching face: {os.path.basename(best_match)} (similarity: {best_similarity:.4f})")
            return best_match
        else:
            logger.warning("No suitable face found, using first reference face")
            # Return first available reference face
            first_face = list(embeddings_cache.values())[0]['path']
            return first_face
            
    except Exception as e:
        logger.error(f"Error in fast face similarity search: {str(e)}")
        # Fallback to old method
        return find_most_similar_face_fallback(source_image_path)

def find_most_similar_face_fallback(source_image_path):
    """Fallback method using direct DeepFace comparison"""
    try:
        logger.info("Using fallback face similarity search...")
        reference_faces_dir = app.config['REFERENCE_FACES_FOLDER']
        reference_faces = []
        
        # Get all reference face images
        for filename in os.listdir(reference_faces_dir):
            if allowed_file(filename):
                face_path = os.path.join(reference_faces_dir, filename)
                reference_faces.append(face_path)
        
        if not reference_faces:
            logger.error("No reference faces found")
            return None
        
        best_match = None
        min_distance = float('inf')
        
        for ref_face_path in reference_faces:
            try:
                # Compare faces
                result = DeepFace.verify(
                    img1_path=source_image_path, 
                    img2_path=ref_face_path,
                    model_name='VGG-Face',
                    distance_metric='cosine'
                )
                
                distance = result['distance']
                logger.info(f"Distance with {os.path.basename(ref_face_path)}: {distance}")
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = ref_face_path
                    
            except Exception as e:
                logger.warning(f"Cannot compare with {ref_face_path}: {str(e)}")
                continue
        
        if best_match:
            logger.info(f"Best matching face: {os.path.basename(best_match)} (distance: {min_distance})")
            return best_match
        else:
            return reference_faces[0] if reference_faces else None
            
    except Exception as e:
        logger.error(f"Error in fallback face search: {str(e)}")
        return None

# Keep the old function as alias for compatibility
def find_most_similar_face(source_image_path):
    """Main function - uses fast embedding-based comparison"""
    return find_most_similar_face_fast(source_image_path)

@app.route('/')
def index():
    """Main page - no reference images selection needed"""
    logger.info("Accessing main page")
    return render_template('makeup_app.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and makeup transfer with auto reference selection"""
    try:
        logger.info("Starting file upload and makeup transfer processing")
        
        if 'source_image' not in request.files:
            logger.warning("No source image in request")
            return jsonify({'error': 'No source image provided'}), 400
        
        source_file = request.files['source_image']
        
        logger.info(f"Source file: {source_file.filename}")
        
        if source_file.filename == '':
            logger.warning("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(source_file.filename):
            logger.warning(f"Unsupported file type: {source_file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG files.'}), 400

        # Create unique session ID for this processing
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Created session ID: {session_id}, Timestamp: {timestamp}")
        
        # Save source image to both temp and permanent folders
        source_filename = secure_filename(f"{timestamp}_{session_id}_{source_file.filename}")
        source_temp_path = os.path.join(app.config['UPLOAD_FOLDER'], source_filename)
        source_saved_path = os.path.join(app.config['SAVED_UPLOADS_FOLDER'], source_filename)
        
        source_file.save(source_temp_path)
        source_file.seek(0)  # Reset file pointer
        source_file.save(source_saved_path)
        
        logger.info(f"Saved source image: {source_saved_path}")
        
        # Automatically find the most suitable reference face
        reference_path = find_most_similar_face(source_temp_path)
        if not reference_path:
            logger.error("No suitable reference face found")
            return jsonify({'error': 'No suitable reference face found'}), 500
        
        reference_filename = os.path.basename(reference_path)
        logger.info(f"Using auto-selected reference: {reference_filename}")
        
        # Check if models are initialized
        if inference_model is None or postprocess_model is None:
            logger.error("PSGAN models not initialized")
            return jsonify({'error': 'PSGAN models not initialized. Please restart the server.'}), 500

        # Perform makeup transfer
        logger.info("Starting makeup transfer process...")
        source_image = Image.open(source_temp_path).convert("RGB")
        reference_image = Image.open(reference_path).convert("RGB")
        
        # Transfer makeup
        result_image, face = inference_model.transfer(source_image, reference_image, with_face=True)
        
        if result_image is None:
            logger.error("No face detected in source image")
            return jsonify({'error': 'Face not detected in source image. Please try another image.'}), 400
        
        # Post-process the result
        logger.info("Post-processing result...")
        source_crop = source_image.crop((face.left(), face.top(), face.right(), face.bottom()))
        final_result = postprocess_model(source_crop, result_image)
        
        # Save result to both temp and permanent folders
        result_filename = f"result_{timestamp}_{session_id}.png"
        result_temp_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        result_saved_path = os.path.join(app.config['SAVED_RESULTS_FOLDER'], result_filename)
        
        final_result.save(result_temp_path)
        final_result.save(result_saved_path)
        
        logger.info(f"Result saved: {result_saved_path}")
        
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
            
        logger.info(f"Metadata saved: {metadata_path}")
        
        # Convert images to base64 for display
        source_b64 = image_to_base64(source_image)
        reference_b64 = image_to_base64(reference_image)
        result_b64 = image_to_base64(final_result)
        
        # Clean up temporary files (keep saved ones)
        try:
            os.remove(source_temp_path)
            logger.info("Cleaned up temporary files successfully")
        except Exception as e:
            logger.warning(f"Cannot clean up temporary files: {e}")
        
        logger.info(f"Completed makeup transfer for session: {session_id}")
        
        return jsonify({
            'success': True,
            'source_image': source_b64,
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
        logger.error(f"Error in upload and processing: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_result(filename):
    """Download result image"""
    try:
        logger.info(f"Download request for file: {filename}")
        
        # Try temp results first
        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        if os.path.exists(result_path):
            logger.info(f"Download from temp folder: {result_path}")
            return send_file(result_path, as_attachment=True)
        
        # Try saved results
        saved_result_path = os.path.join(app.config['SAVED_RESULTS_FOLDER'], filename)
        if os.path.exists(saved_result_path):
            logger.info(f"Download from saved folder: {saved_result_path}")
            return send_file(saved_result_path, as_attachment=True)
        
        logger.warning(f"File not found: {filename}")
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/gallery')
def view_gallery():
    """View all saved results"""
    try:
        logger.info("Accessing gallery page")
        
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
                    logger.warning(f"Cannot read metadata file {filename}: {e}")
                    continue
        
        # Sort by timestamp, newest first
        saved_results.sort(key=lambda x: x['timestamp'], reverse=True)
        
        logger.info(f"Found {len(saved_results)}/{metadata_count} valid results")
        
        return render_template('gallery.html', results=saved_results)
        
    except Exception as e:
        logger.error(f"Error loading gallery: {str(e)}")
        return jsonify({'error': f'Failed to load gallery: {str(e)}'}), 500

@app.route('/view_result/<session_id>')
def view_result(session_id):
    """View a specific result with metadata"""
    try:
        logger.info(f"Request to view result for session: {session_id}")
        
        # Find metadata file for this session
        saved_results_dir = app.config['SAVED_RESULTS_FOLDER']
        metadata_files = [f for f in os.listdir(saved_results_dir) 
                         if f.startswith('metadata_') and session_id in f]
        
        if not metadata_files:
            logger.warning(f"No metadata found for session: {session_id}")
            return jsonify({'error': 'Result not found'}), 404
        
        metadata_path = os.path.join(saved_results_dir, metadata_files[0])
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get image paths
        result_path = os.path.join(saved_results_dir, metadata['result_filename'])
        source_path = os.path.join(app.config['SAVED_UPLOADS_FOLDER'], metadata['source_filename'])
        
        # Reference path from reference_faces folder
        reference_path = os.path.join(app.config['REFERENCE_FACES_FOLDER'], metadata['reference_filename'])
        
        # Convert to base64 for display
        result_b64 = None
        source_b64 = None
        reference_b64 = None
        
        if os.path.exists(result_path):
            result_image = Image.open(result_path)
            result_b64 = image_to_base64(result_image)
            logger.info("Result image loaded successfully")
        
        if os.path.exists(source_path):
            source_image = Image.open(source_path)
            source_b64 = image_to_base64(source_image)
            logger.info("Source image loaded successfully")
        
        if os.path.exists(reference_path):
            reference_image = Image.open(reference_path)
            reference_b64 = image_to_base64(reference_image)
            logger.info("Reference image loaded successfully")
        
        logger.info(f"Returning result for session: {session_id}")
        
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
        logger.error(f"Error loading result: {str(e)}")
        return jsonify({'error': f'Failed to load result: {str(e)}'}), 500

@app.route('/delete_result/<session_id>', methods=['DELETE'])
def delete_result(session_id):
    """Delete a saved result and its files"""
    try:
        logger.info(f"Request to delete result for session: {session_id}")
        
        # Find and load metadata
        saved_results_dir = app.config['SAVED_RESULTS_FOLDER']
        metadata_files = [f for f in os.listdir(saved_results_dir) 
                         if f.startswith('metadata_') and session_id in f]
        
        if not metadata_files:
            logger.warning(f"No metadata found for session: {session_id}")
            return jsonify({'error': 'Result not found'}), 404
        
        metadata_path = os.path.join(saved_results_dir, metadata_files[0])
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Delete files (don't delete reference as it's shared)
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
                    logger.info(f"Deleted file: {file_path}")
            except Exception as e:
                logger.warning(f"Cannot delete file {file_path}: {e}")
        
        logger.info(f"Successfully deleted {len(deleted_files)} files for session: {session_id}")
        
        return jsonify({
            'success': True,
            'message': f'Deleted {len(deleted_files)} files',
            'deleted_files': deleted_files
        })
        
    except Exception as e:
        logger.error(f"Error deleting result: {str(e)}")
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
    logger.warning("File upload too large")
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    logger.warning(f"Endpoint not found: {request.url}")
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/embeddings/cache', methods=['GET'])
def get_embeddings_cache_info():
    """Get information about embeddings cache"""
    try:
        cache_path = get_embeddings_cache_path()
        reference_faces_dir = app.config['REFERENCE_FACES_FOLDER']
        
        # Count reference faces
        face_count = len([f for f in os.listdir(reference_faces_dir) if allowed_file(f)])
        
        cache_info = {
            'cache_exists': os.path.exists(cache_path),
            'reference_faces_count': face_count,
            'cache_path': cache_path
        }
        
        if cache_info['cache_exists']:
            cache_stat = os.stat(cache_path)
            cache_info.update({
                'cache_size_mb': round(cache_stat.st_size / 1024 / 1024, 2),
                'cache_modified': datetime.fromtimestamp(cache_stat.st_mtime).isoformat(),
            })
        
        return jsonify(cache_info)
    except Exception as e:
        logger.error(f"Error getting cache info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/embeddings/rebuild', methods=['POST'])
def rebuild_embeddings_cache():
    """Force rebuild embeddings cache"""
    try:
        logger.info("Force rebuilding embeddings cache...")
        
        # Delete existing cache
        cache_path = get_embeddings_cache_path()
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info("Deleted existing cache")
        
        # Create new cache
        embeddings_cache = load_or_create_embeddings_cache()
        
        return jsonify({
            'success': True,
            'message': f'Rebuilt embeddings cache with {len(embeddings_cache)} faces',
            'faces_count': len(embeddings_cache)
        })
    except Exception as e:
        logger.error(f"Error rebuilding cache: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting BeautyStudio Web Application with AI Face Matching...")
    logger.info("Starting BeautyStudio web application with DeepFace")
    
    # Pre-compute embeddings on startup for better performance
    logger.info("Pre-computing face embeddings for faster matching...")
    try:
        embeddings_cache = load_or_create_embeddings_cache()
        if embeddings_cache:
            logger.info(f"✅ Ready with {len(embeddings_cache)} face embeddings!")
        else:
            logger.warning("⚠️ No reference faces found for embedding")
    except Exception as e:
        logger.error(f"❌ Error pre-computing embeddings: {str(e)}")
    
    # Initialize models
    if initialize_models():
        logger.info("🌟 BeautyStudio Web App with AI is ready!")
        print("🌟 BeautyStudio Web App with AI is ready!")
        print("📝 Access the application at: http://localhost:5000")
        print("📄 Log saved to file: beauty_transform_app.log")
        print("🎭 Application will automatically select the best matching reference face!")
        print("⚡ Fast face matching with pre-computed embeddings!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("❌ Failed to initialize models")
