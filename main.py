from flask import *
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('home.html', title="Cataract Detection")

if __name__=="__main__":
    app.run()
