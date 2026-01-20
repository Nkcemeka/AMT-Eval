from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import dash
from dash_app.database import SessionLocal, User, load_questions, load_responses
from flask import Flask
from flask_login import LoginManager, logout_user, current_user
import os
from dash.dependencies import Input, Output, State
from dotenv import load_dotenv
import random
import json
from pathlib import Path

FILE_PATH = Path(__file__).resolve().parent

# Exposing the Flask Server to enable configuration for logging in
load_dotenv(dotenv_path=FILE_PATH / ".env") # helps us load the secret key
server = Flask(__name__)

# Instantiate Dash app.
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP], \
        use_pages=True, suppress_callback_exceptions=True)

# Updating the Flask Server configuration with Secret Key to encrypt the user session cookie
server.config.update(SECRET_KEY=os.getenv("SECRET_KEY"))

# Login manager object will be used to login / logout users
login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = "/login"

# Loads the user from the database
@login_manager.user_loader
def load_user(user_id):
    session = SessionLocal()  # connect to your DB
    try:
        return session.query(User).filter_by(id=user_id).first()
    finally:
        session.close()


# Define main app layout
app.layout = html.Div(
    [
        dcc.Store(id="user-selections", data={}, storage_type="session"),
        dcc.Store(id="selection-changes", data={}, storage_type="session"),
        dcc.Store(id="finished", data=False, storage_type="session"),
        dcc.Store(id="init-login", data=False, storage_type="session"),
        #dcc.Store(id='lq', data=-1, storage_type="session"), # tracks the last filled question
        dcc.Store(id='questions', data=[], storage_type="session"),
        dcc.Store(id='order', data='', storage_type="session"),
        dcc.Location(id="url", refresh=True),
        dbc.NavbarSimple(id="navbar",
            children=[],
            brand="",
            brand_href="/",
            color="white",
            dark=False,
        ),
        dash.page_container
    ]
)

# Load questions for the current user
@app.callback(
    Output("questions", "data"),
    Input("questions", "data")
)
def get_questions(_: list):
    return load_questions()

# Callback to handle selections
@app.callback(
    Output("user-selections", "data"),
    Input("selection-changes", "data"),
    Input("init-login", "data"),
    State("user-selections", "data"),
)
def selection_handler(changes, _, selections):
    if not current_user.is_authenticated:
        return {}
    
    if not selections:
        # if selections are empty, check if the database
        # has some selections already
        selections =  load_responses(current_user.get_id())
    
    # Check if there are changes to the selections from the user
    # if yes, add them
    if changes:
        for key, val in changes.items():
            selections[key] = val

    return selections

# Once we login, we create a random order
# map for the questions. This map will be 
# useful for reloading state if the transcriber
# wants to do so.
@app.callback(
    Output("order", "data"),
    Input("questions", "data"),
)
def gen_order(questions):
    if not current_user.is_authenticated:
        return ''
    num_questions = len(questions)

    # Check if order is empty. If it is, it means
    # we are in a new fresh session and order state has not
    # been saved to the dB
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(id=current_user.get_id()).first()
        order = user.order 
        if len(order) > 0:
            return order
        
        # Here we generate the order if it doesn't exist in the database
        # and store it there as well
        order = []
        for i in range(num_questions):
            # 0th index is for transcription A and 1th index is
            # for transcription B
            item = [0, 1]
            random.shuffle(item)
            order.append((item[0], item[1]))
        
        # jsonify order and return it
        order = json.dumps(order)
        user.order = order
        session.commit()
        return order
    finally:
        session.close()


# Router callback to deal with
# navigation issues
@app.callback(
    Output("url", "pathname"),
    Output("navbar", "children"),
    Output("navbar", "brand"),
    Input("url", "pathname"),
    Input("init-login", "data"),
    Input("finished", "data")
)
def router(page, _, finished: bool):
    # if user is not authenticated and we are not on the signup
    # page, we go to Login
    if not current_user.is_authenticated and page != "/signup":
        return "/login", [], ""

    # define navigation bar after login
    fullnav = [
                dbc.NavItem(dbc.NavLink("Home", href="/")),
                dbc.NavItem(dbc.NavLink("Instructions", href="/instructions")),
                dbc.NavItem(dbc.NavLink("Questions", href="/questions")),
                dbc.NavItem(dbc.NavLink("Logout", href="/logout")),
            ]
    
    thankyou_nav = [
        dbc.NavItem(dbc.NavLink("Logout", href="/logout")),
    ]
    
    if page == "/login" and current_user.is_authenticated:
        return "/", fullnav, "User Study"
    
    if page == "/logout" and current_user.is_authenticated:
        logout_user()
        return "/login", fullnav, ""
    
    # If the user is authenticated
    # Check if he is done with the questions
    if finished:
        return "/thanks", thankyou_nav, "User Study"
    
    if page == "/signup":
        return dash.no_update, [dbc.NavItem(html.A("Login", href="/login",\
                        className="nav-hover", id="nav-login")),], ""
    return dash.no_update, fullnav, "User Study"

if __name__ == '__main__':
    app.run(debug=True)
