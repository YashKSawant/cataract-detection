import os
from flask import *
from werkzeug.utils import secure_filename
from utils import allowed_file, addFileInJson
from model import *

app = Flask(__name__)

UPLOAD_FOLDER = './ImageUploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/')
def main():
    return render_template('home.html', title="Cataract Detection")


@app.route('/index')
def index():
    return render_template('index.html', title='Cataract Detection')


@app.route('/settings')
def settings():
    return render_template('settings.html', title='Cataract Detection')


@app.route('/single')
def single():
    return render_template('singleDetection.html', title='Cataract Detection')


@app.route('/uploadImage', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp

    files = request.files.getlist('files[]')

    errors = {}
    success = False

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
            addFileInJson(filename)
        else:
            errors[file.filename] = 'File type is not allowed'

    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 206
        return resp
    if success:
        resp = jsonify({'message': 'Files successfully uploaded'})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp


if __name__ == "__main__":
    app.run(debug=True)
