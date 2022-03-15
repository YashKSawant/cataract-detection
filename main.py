from flask import *
app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html', title="Cataract Detection")

@app.route('/index')
def index():
    return render_template('index.html',title='Cataract Detection')

@app.route('/settings')
def settings():
    return render_template('settings.html',title='Cataract Detection')

if __name__=="__main__":
    app.run(debug=True)
