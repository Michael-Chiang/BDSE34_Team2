from flask import Flask
from .route import init_route


def create_app():
    app = Flask(__name__)

    init_route(app)

    return app