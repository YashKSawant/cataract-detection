import json

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def addFileInJson(filename):
    data = json.loads(open("data.json", "r").read())
    data.append(filename)
    open("data.json", "w").write(json.dumps(data, indent=4))
