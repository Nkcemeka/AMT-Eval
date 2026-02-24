"""
    File: database.py
    Description: Contains code functionality for handling the
                 database for the dash app.
"""

# Import necesaary libraries
import json
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, UniqueConstraint, text, func
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import IntegrityError
from pathlib import Path
import bcrypt
from flask_login import UserMixin
from typing import Optional

# Get the path of this file
FILE_PATH = Path(__file__).resolve().parent

USERS_PER_QUESTION = 10
NUM_QUESTIONS = 12

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
    lq = Column(Integer, nullable=False, default=-1) 
    responses = relationship("Response", back_populates="user")
    assigned_questions = relationship("Question", secondary="question_assignments", \
                                      back_populates="assigned_users", 
                                  order_by="QuestionAssignment.order_index")

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
    assigned_users = relationship("User", secondary="question_assignments", \
                                  back_populates="assigned_questions")

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

class QuestionAssignment(Base):
    __tablename__ = "question_assignments"
    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    question_id = Column(String, ForeignKey("questions.id"), primary_key=True)
    order_index = Column(Integer, nullable=False) # added this to prevent error when logging out and back in


def trigger(engine):
    with engine.connect() as conn:
        conn.execute(text(f"""
            CREATE TRIGGER IF NOT EXISTS limit_users_per_question
            BEFORE INSERT ON question_assignments
            BEGIN
                SELECT
                    CASE
                        WHEN ( 
                            SELECT COUNT(*)
                            FROM question_assignments
                            WHERE question_id = NEW.question_id
                        ) >= {USERS_PER_QUESTION}
                        THEN RAISE(ABORT, 'A question cannot have more than {USERS_PER_QUESTION} users')
                    END;
            END;
        """))
        conn.commit()

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

def load_questions2(user_id: str, NUM_QUESTIONS: int=NUM_QUESTIONS):
    """ 
        This loads the questions for a given user_id.
        If a user has not been assigned a set of questions, 
        it assigns the user and then returns the assigned
        questions.

        Args:
            user_id (str): User id of the user
            NUM_QUESTIONS (int): Number of questions to be assigned to user

        Returns:
            final_qdb (list): A list of final set of questions for user
    """
    session = SessionLocal()
    final_qdb = []
    try:
        user = session.get(User, user_id)

        # If questions are already assigned for this user, just load and return
        if user.assigned_questions:
            return [
                question_to_dict(each) for each in user.assigned_questions
            ]
        
        # Otherwise, assign questions to user and load them
        q_db = session.query(Question).outerjoin(\
            QuestionAssignment).group_by(Question.id).having(func.count(\
            QuestionAssignment.user_id) < USERS_PER_QUESTION).order_by(func.random(), Question.type).all()
        
        for i, each in enumerate(q_db):
            # we shouldn't get into this if condition ideally,
            # so take it off
            if user in each.assigned_users:
                continue

            if len(final_qdb) >= NUM_QUESTIONS:
                break

            with session.begin_nested(): # create a savepoint
                #each.assigned_users.append(user)
                try:
                    # Trigger fires here (important when two users are racing to write to dB)
                    # if this fails, we rollback
                    session.execute(
                        QuestionAssignment.__table__.insert().values(
                            user_id=user.id,
                            question_id=each.id,
                            order_index=len(final_qdb)
                        )
                    )

                    session.flush()
                    q_dict = question_to_dict(each)
                    final_qdb.append(q_dict)
                except IntegrityError:
                    # rollback to savepoint
                    session.rollback()
                    continue

        session.commit()
        return final_qdb
    finally:
        session.close()

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

def load_lq(user_id: str):
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(id=user_id).first()
        lq = user.lq
        return lq
    finally:
        session.close()

def save_lq(user_id: str, lq):
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(id=user_id).first()
        user.lq = lq
        session.commit()
    finally:
        session.close()

def load_responses(user_id: str):
    session = SessionLocal()
    try:
        sel_dict = {}

        # Extract questions and responses for user_id in question
        # q_db = session.query(Question).order_by(Question.id).all()
        user = session.get(User, user_id)

        # If questions are already assigned for this user, just load and return
        if user.assigned_questions:
            q_list =  [
                question_to_dict(each) for each in user.assigned_questions
            ]
        else:
            # if no assigned questions, raise error; user should
            #raise RuntimeError(f"User {user_id} should have assigned questions!")
            return {}
        
        response_db = session.query(Response).filter_by(user_id=user_id).all()
        r_map = {
            r.question_id: r.choice for r in response_db
        }

        for i, each in enumerate(q_list):
            selection = r_map.get(each["id"], None)
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
    user = session.get(User, user_id)

    # If questions are already assigned for this user, just load and return
    if user.assigned_questions:
        q_list =  [
            question_to_dict(each) for each in user.assigned_questions
        ]
    else:
        raise RuntimeError("User not assigned questions yet!")

    for key in response_dict.keys():
        key_int = int(key)
        #quest_id = str(Path(q_list[key_int]["refAddress"]).stem)
        quest_id = str(Path(q_list[key_int]["id"]).stem)
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

        if each["type"] == "MIDI":
            question = Question(id=id, aTrans=each["images"][0], refAddress=each["audio"],
                        bTrans=each["images"][1], aTrans_audio=each["audio_trans"][0],\
                        bTrans_audio=each["audio_trans"][1], type=each["type"])
        elif each["type"] == "SCORE":
            suffix_name1 = Path(each["audio_trans"][0]).stem.split("_")[-1]
            suffix_name2 = Path(each["audio_trans"][1]).stem.split("_")[-1]
            id = Path(each["audio"]).stem + "_" + suffix_name1 + "_" + suffix_name2
            question = Question(id=id, aTrans=each["images"][0], refAddress=each["audio"],
                        bTrans=each["images"][1], aTrans_audio=each["audio_trans"][0],\
                        bTrans_audio=each["audio_trans"][1], type=each["type"])
        else:
            raise ValueError(f"Unknown data type: {each[type]}")
        session.add(question)

    session.commit()
    session.close()


if __name__ == "__main__":
    init_database()
