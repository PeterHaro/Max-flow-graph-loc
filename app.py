import os
import socket
import json

from flask import Flask, render_template, request

from matchmaking import Matchmaking, MatchmakingMethodology
from whitenoise import WhiteNoise

app = Flask(__name__)
app.secret_key = 'suppasecret'
static_dir = os.path.join(os.path.dirname(__file__), 'static')
app.wsgi_app = WhiteNoise(app.wsgi_app, root=static_dir, prefix='static/')

matchmaker = Matchmaking()


@app.route("/")
def index():
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        return render_template('index.html', hostname=host_name, ip=host_ip)
    except:
        return render_template('error.html')


@app.route("/mm", methods=["POST"])
def matchmake():
    content = request.get_json()
    offers = content["offers"]
    buyerId = content["buyer"]
    result = json.dumps(matchmaker.perform_matchmaking(offers, int(buyerId), MatchmakingMethodology.MAX_FLOW))
    return result


if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run()
