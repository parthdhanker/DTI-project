import os
import uuid
import io
from flask import Flask, render_template, request, redirect, send_from_directory, jsonify, send_file, url_for
from PIL import Image
import numpy as np
import onnxruntime as ort
import cv2
import requests

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
SEGMENT_FOLDER = 'static/segments'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
GOOGLE_API_KEY = "Parth API key"
MAPS_ZOOM_LEVEL = 16

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, SEGMENT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Load ONNX models
color_model_path = r"C:\Users\lenovo\Desktop\Dit project\best_model.onnx"
seg_model_path = r"C:\Users\lenovo\Desktop\Dit project\segmentation model.onnx"

color_session = ort.InferenceSession(color_model_path, providers=['CPUExecutionProvider'])
seg_session = ort.InferenceSession(seg_model_path, providers=['CPUExecutionProvider'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def colorize_image(image_path, output_path):
    img = Image.open(image_path).convert('L').resize((256, 256))
    img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, 256, 256, 1)

    input_name = color_session.get_inputs()[0].name
    output_name = color_session.get_outputs()[0].name
    prediction = color_session.run([output_name], {input_name: img_array})[0]

    output_image = ((prediction[0] + 1) * 127.5).astype(np.uint8)
    Image.fromarray(output_image).save(output_path)

def segment_image(colorized_path, segmented_path):
    img = cv2.imread(colorized_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img_input = img.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)  # (1, 256, 256, 3)

    input_name = seg_session.get_inputs()[0].name
    output = seg_session.run(None, {input_name: img_input})[0]  # (1, 256, 256, 7)
    pred = np.argmax(output.squeeze(), axis=-1).astype(np.uint8)

    color_map = {
        0: (255, 0, 0),     # Urban Land
        1: (0, 255, 0),     # Agriculture Land
        2: (160, 82, 45),   # Rangeland
        3: (0, 100, 0),     # Forest Land
        4: (0, 255, 255),   # Water
        5: (255, 255, 0),   # Barren Land
        6: (128, 128, 128)  # Unknown
    }

    label_names = {
        0: "Urban Land",
        1: "Agriculture Land",
        2: "Rangeland",
        3: "Forest Land",
        4: "Water",
        5: "Barren Land",
        6: "Unknown"
    }

    segmented_img = np.zeros((256, 256, 3), dtype=np.uint8)
    for label, color in color_map.items():
        segmented_img[pred == label] = color

    Image.fromarray(segmented_img).save(segmented_path)

    total = pred.size
    breakdown = {
        label_names[label]: np.sum(pred == label) / total * 100
        for label in range(7)
    }

    return [f"{label}: {percent:.2f}%" for label, percent in breakdown.items()]

def get_google_maps_image(lat, lon, zoom=17):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size=640x640&maptype=satellite&key={GOOGLE_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = f"{uuid.uuid4().hex}.png"
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    # Colorize the image
    colorized_filename = f"colorized_{filename}"
    colorized_path = os.path.join(OUTPUT_FOLDER, colorized_filename)
    colorize_image(upload_path, colorized_path)

    # Segment the colorized image
    segmented_filename = f"segmented_{filename}"
    segmented_path = os.path.join(SEGMENT_FOLDER, segmented_filename)
    percentages = segment_image(colorized_path, segmented_path)

    return render_template('segmentation_result.html',
                         colorized_filename=colorized_filename,
                         segmented_filename=segmented_filename,
                         percentages=percentages)

@app.route('/location_segment', methods=['POST'])
def location_segment():
    try:
        # Get data from form
        lat = request.form.get('latitude')
        lng = request.form.get('longitude')
        
        if not lat or not lng:
            return jsonify({'success': False, 'error': 'Missing latitude or longitude'}), 400
        
        try:
            lat = float(lat)
            lng = float(lng)
        except ValueError:
            return jsonify({'success': False, 'error': 'Coordinates must be numbers'}), 400
        
        # Validate coordinates
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            return jsonify({'success': False, 'error': 'Invalid coordinates'}), 400

        # Get image from Google Maps
        try:
            img = get_google_maps_image(lat, lng)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to get map image: {str(e)}'}), 500
        
        # Generate filenames
        file_id = uuid.uuid4().hex
        colorized_filename = f"colorized_{file_id}.png"
        segmented_filename = f"segmented_{file_id}.png"
        
        # Save images
        colorized_path = os.path.join(OUTPUT_FOLDER, colorized_filename)
        segmented_path = os.path.join(SEGMENT_FOLDER, segmented_filename)
        
        try:
            img.save(colorized_path)
            # Perform segmentation
            percentages = segment_image(colorized_path, segmented_path)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Processing failed: {str(e)}'}), 500
        
        # Convert coordinates to floats before passing to template
        return jsonify({
            'success': True,
            'redirect': url_for('segmentation_result', 
                              colorized_filename=colorized_filename,
                              segmented_filename=segmented_filename,
                              lat=float(lat),
                              lng=float(lng))
        })
        
    except Exception as e:
        app.logger.error(f"Error in location_segment: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route('/segmentation_result')
def segmentation_result():
    try:
        colorized_filename = request.args.get('colorized_filename')
        segmented_filename = request.args.get('segmented_filename')
        lat = request.args.get('lat')
        lng = request.args.get('lng')
        
        if not all([colorized_filename, segmented_filename]):
            raise ValueError("Missing required parameters")
            
        # Get percentages from the segmented image
        colorized_path = os.path.join(OUTPUT_FOLDER, colorized_filename)
        segmented_path = os.path.join(SEGMENT_FOLDER, segmented_filename)
        percentages = segment_image(colorized_path, segmented_path)
            
        coordinates = {'lat': lat, 'lng': lng} if lat and lng else None
        
        return render_template('segmentation_result.html',
                           colorized_filename=colorized_filename,
                           segmented_filename=segmented_filename,
                           percentages=percentages,
                           coordinates=coordinates)
    except Exception as e:
        app.logger.error(f"Error in segmentation_result: {str(e)}")
        return render_template('index.html', error=str(e))

@app.route('/static/map_preview')
def map_preview():
    try:
        lat = request.args.get('lat')
        lng = request.args.get('lng')
        
        if not lat or not lng:
            return "Missing coordinates", 400
        
        img = get_google_maps_image(float(lat), float(lng), zoom=15)
        img_io = io.BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in map_preview: {str(e)}")
        return str(e), 500

@app.route('/download/<folder>/<filename>')
def download_file(folder, filename):
    folder_map = {
        'outputs': OUTPUT_FOLDER,
        'segments': SEGMENT_FOLDER,
        'uploads': UPLOAD_FOLDER
    }
    if folder in folder_map:
        return send_from_directory(folder_map[folder], filename, as_attachment=True)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)