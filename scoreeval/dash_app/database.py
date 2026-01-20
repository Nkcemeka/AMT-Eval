"""
    File: database.py
    Description: Contains code functionality for handling the
                 database for the dash app.
"""

# Import necesaary libraries
import json
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from pathlib import Path
import bcrypt
from flask_login import UserMixin
from typing import Optional

# Get the path of this file
FILE_PATH = Path(__file__).resolve().parent

with open(FILE_PATH / "assets/questions.json", "r") as f:
    questions = json.load(f)["questions"]

# Instantiate base
DB_PATH = FILE_PATH / "study.db"
ENGINE = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=ENGINE)
Base = declarative_base()

# We will have various classes for our database
# UserMixin gives our class is_authenticated, get_id etc
class User(Base, UserMixin):
    __tablename__ = 'users'
    id = Column(String, primary_key=True, unique=True)
    password = Column(String, nullable=False)
    order = Column(String, nullable=False)

    # lq stands for last question when user logs out without finishing
    #lq = Column(int, nullable=False, default=-1) 
    responses = relationship("Response", back_populates="user")

class Question(Base):
    __tablename__ = 'questions'
    id = Column(String, primary_key=True)
    type = Column(String)
    refAddress = Column(String) # address to main ref.audio
    aTrans = Column(String) # address to transcription A image
    bTrans = Column(String) # address to transcription B image
    aTrans_audio = Column(String) # address to audio for transcription A image
    bTrans_audio = Column (String) # address to audio for transcription B image
    responses=relationship("Response", back_populates="question")

class Response(Base):
    __tablename__ = 'responses'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    type = Column(String, nullable=False) # can be a MIDI/Score response
    question_id = Column(String, ForeignKey("questions.id"), nullable=False)
    choice = Column(String, nullable=False) # string is a json-dumped dict (key is dimension: value is choice)
    user = relationship("User", back_populates="responses")
    question = relationship("Question", back_populates="responses")

    # We don;t multiple responses for the same question
    __table_args__ = (
        UniqueConstraint("user_id", "question_id"),
    )

def question_to_dict(q):
    # Make database object JSON serializable by
    # returning a dict
    return {
        'id': q.id,
        'type': q.type,
        'refAddress': q.refAddress,
        'aTrans': q.aTrans,
        'bTrans': q.bTrans,
        'aTrans_audio': q.aTrans_audio,
        'bTrans_audio': q.bTrans_audio
    }

def load_questions():
    session = SessionLocal()
    try:
        q_list = []

        # Extract questions and responses for user_id in question
        q_db = session.query(Question).order_by(Question.id).all()

        for i, each in enumerate(q_db):
            q_dict = question_to_dict(each)
            q_list.append(q_dict)
        return q_list
    finally:
        session.close()

def load_responses(user_id: str):
    session = SessionLocal()
    try:
        sel_dict = {}

        # Extract questions and responses for user_id in question
        q_db = session.query(Question).order_by(Question.id).all()
        response_db = session.query(Response).filter_by(user_id=user_id).all()
        r_map = {
            r.question_id: r.choice for r in response_db
        }

        for i, each in enumerate(q_db):
            selection = r_map.get(each.id, None)
            if selection is not None:
                sel_dict[str(i)] = json.loads(selection)
        return sel_dict
    finally:
        session.close()

def auth_func(username: str, passwd: str) -> bool:
    # Create a session
    try:
        session = SessionLocal()
        user = session.query(User).filter_by(id=username).first()
        if user is None:
            return False

        return bcrypt.checkpw(passwd.encode(), user.password)
    finally:
        session.close()

def get_user_details():
    # Connect to the database
    # Create a session
    session = SessionLocal()
    result = session.query(User.id, User.password).all()
    session.close()
    return result

def add_response(response_dict: dict, user_id: str):
    # Connect to the database
    # Create a session
    session = SessionLocal()
    q_db = session.query(Question).order_by(Question.id).all()
    q_list = []
    for i, each in enumerate(q_db):
            q_dict = question_to_dict(each)
            q_list.append(q_dict)

    for key in response_dict.keys():
        key_int = int(key)
        quest_id = str(Path(q_list[key_int]["refAddress"]).stem)
        resp_type = q_list[key_int]["type"]
        choice = json.dumps(response_dict[key])

        existing = session.query(Response).filter_by(
            user_id=user_id,
            question_id=quest_id,
            type=resp_type
        ).first()

        if existing:
            existing.choice = choice
        else:
            session.add(Response(
                user_id=user_id,
                question_id=quest_id,
                choice=choice,
                type=resp_type
            ))
    
    session.commit()
    session.close()

def hash_passwd(passwd: str):
    return bcrypt.hashpw(passwd.encode(), bcrypt.gensalt())

def create_account(username, password) -> Optional[bool]:
    if username is None or password is None:
        return None
    session = SessionLocal()
    hash = hash_passwd(password)
    existing = session.query(User).filter_by(
            id=username
        ).first()
    
    try:
        if not existing:
            new_user = User(id=username, password=hash, order="")
            session.add(new_user)
            session.commit()
            return True
        else:
            return False
    finally:
        session.close()

def init_database():
    # Setup the database
    Base.metadata.create_all(ENGINE)

    # Create a session
    session = SessionLocal()

    for each in questions:
        # add each question in the question bank
        id = str(Path(each["audio"]).stem)
        question = Question(id=id, aTrans=each["images"][0], refAddress=each["audio"],
                    bTrans=each["images"][1], aTrans_audio=each["audio_trans"][0],\
                    bTrans_audio=each["audio_trans"][1], type=each["type"])
        session.add(question)

    session.commit()
    session.close()


if __name__ == "__main__":
    init_database()
