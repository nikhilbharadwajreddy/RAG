from flask import Flask, render_template, request
import os
import subprocess
from process_file import process_document
import json 
import io
import contextlib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/run-script', methods=['POST'])
def run_script():
    uploaded_file = request.files.get('file')
    if uploaded_file:
        filename = uploaded_file.filename
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(temp_path)

        # 🪄 Capture all printed logs
        log_output = io.StringIO()
        with contextlib.redirect_stdout(log_output):
            result = process_document(temp_path)

        logs = log_output.getvalue()
        return render_template('upload.html', upload_message=f"File processed: {filename}\n\n{logs}")

    return render_template('upload.html', upload_message="No file selected.")


@app.route('/run-secondary-script', methods=['POST'])
def run_secondary():
    try:
        result = subprocess.run(
            ['python3', 'delete_pinecone_data.py'],
            capture_output=True, text=True, check=True
        )
        output = result.stdout
        return render_template('upload.html', delete_message=f" Deleted data:\n\n{output}")
    except subprocess.CalledProcessError as e:
        return render_template('upload.html', delete_message=f"Error deleting data:\n\n{e.stderr}")


if __name__ == '__main__':
    app.run(port=5001, debug=True)
