from untimedTest import *

from flask import Flask, render_template, redirect, url_for, request, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from io import BytesIO
from flask import send_file
from flask_caching import Cache
from flask import send_from_directory, jsonify, session
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///school.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # 5 minutes
cache = Cache(app)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Database Models
class Teacher(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password_hash = db.Column(db.String(200))
    students = db.relationship('Student', backref='teacher', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_id(self):
        return f"teacher_{self.id}"

class Student(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password_hash = db.Column(db.String(200))
    total_points = db.Column(db.Integer, default=0)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teacher.id'))
    games = db.relationship('GameTracker', backref='student', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_id(self):
        return f"student_{self.id}"
    
class GameTracker(db.Model):
    __tablename__ = 'game_tracker'
    GameID = db.Column(db.Integer, primary_key=True)
    studentID = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    Points = db.Column(db.Integer, nullable=False)
    GameType = db.Column(db.Integer, nullable=False)
    
    __table_args__ = (
        db.CheckConstraint('GameType IN (1, 2, 3)', name='valid_game_type'),
    )

    def __repr__(self):
        return f'<Game {self.GameType} - Points: {self.Points}>'
    
@login_manager.user_loader
def load_user(user_id):
    try:
        if user_id.startswith('teacher_'):
            _, tid = user_id.split('_')
            return Teacher.query.get(int(tid))
        elif user_id.startswith('student_'):
            _, sid = user_id.split('_')
            return Student.query.get(int(sid))
    except:
        return None

# Routes
@app.route('/')
def home():
    return redirect(url_for('login'))

# Routes (keep other routes the same as previous version)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check teachers first
        user = Teacher.query.filter_by(username=username).first()
        if not user:
            # Then check students
            user = Student.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            if isinstance(user, Teacher):
                return redirect(url_for('teacher_dashboard'))
            else:
                return redirect(url_for('student_dashboard'))
        return '<script>alert("Invalid credentials");window.location.href="/login";</script>'
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        teacher_id = request.form.get('teacher_id')

        # Check if username exists in either table
        if Teacher.query.filter_by(username=username).first() or Student.query.filter_by(username=username).first():
            return '<script>alert("Username already exists");window.location.href="/register";</script>'

        if role == 'teacher':
            new_teacher = Teacher(username=username)
            new_teacher.set_password(password)
            db.session.add(new_teacher)
        elif role == 'student':
            if not teacher_id:
                return '<script>alert("Please select a teacher");window.location.href="/register";</script>'
            new_student = Student(
                username=username,
                teacher_id=teacher_id,
                total_points=0
            )
            new_student.set_password(password)
            db.session.add(new_student)
        
        db.session.commit()
        return redirect(url_for('login'))
    
    teachers = Teacher.query.all()
    return render_template('register.html', teachers=teachers)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/teacher')
@login_required
def teacher_dashboard():
    if not isinstance(current_user, Teacher):
        return redirect(url_for('login'))
    
    students = Student.query.filter_by(teacher_id=current_user.id)\
                           .order_by(Student.total_points.desc())\
                           .all()
    
    return render_template('teacher_dashboard.html', students=students)

@app.route('/student')
@login_required
def student_dashboard():
    if not isinstance(current_user, Student):
        return redirect(url_for('login'))
    return render_template('student_dashboard.html')


@app.route('/untimed-test', methods=['GET', 'POST'])
@login_required
def untimedTest():
    if request.method == "POST":
        n = int(request.form["number-input"])
        images, answers = randomImAns(n)
        return render_template("play_untimed_test.html", 
                               ims=images, 
                               ans=answers, 
                               n=n, 
                               i=0,
                               points = 0,
                               score = 0)
    # GET request
    return render_template('untimed_test.html')


@app.route('/match-game', methods=['GET', 'POST'])
@login_required
def matchGame():
    if request.method == 'POST':
        data = request.get_json()
        points = data.get("points", 0)
        i = data.get("i", 5)
    else:
        points = 0
        i = 5

    pairs = {}
    n = 0
    images, answers = randomImAns(6)
    n = answers.count('0')
    corr = []
    for j in range(len(images)):
        pairs[images[j]] = answers[j]
        if answers[j] == '0':
            corr.append(images[j])
            print(corr)

    return render_template("match-game.html", 
                           correct_images = corr, 
                           n=n, 
                           i=i, 
                           points=points,
                           images=images,
                           data=pairs)


@app.route('/leaderboard')
@login_required
def leaderboard():
    if isinstance(current_user, Student):
        classmates = Student.query.filter_by(teacher_id=current_user.teacher_id)\
                                  .order_by(Student.total_points.desc())\
                                  .all()
        return render_template('leaderboard.html', students=classmates, is_student=True)
    
    elif isinstance(current_user, Teacher):
        return redirect(url_for('teacher_dashboard'))
    
    return redirect(url_for('login'))

@app.route('/game_plot/<int:game_type>')
@login_required
def game_plot(game_type):
    if not isinstance(current_user, Student):
        return redirect(url_for('login'))
    
    attempts = GameTracker.query.filter_by(
        studentID=current_user.id,
        GameType=game_type
    ).order_by(GameTracker.GameID.desc()).limit(5).all()[::-1]
    
    plt.switch_backend('Agg')
    fig, ax = plt.subplots()
    
    if attempts:
        x = list(range(1, len(attempts)+1))
        points = [attempt.Points for attempt in attempts]
        ax.plot(x, points, 'b-o')
        ax.set_xlabel('Attempts')
        ax.set_ylabel('Points')
        ax.set_title(f'Game {game_type} History')
    else:
        ax.text(0.5, 0.5, 'No Data Available', 
               ha='center', va='center', fontsize=12)
        ax.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/analytics')
@login_required
def analytics():
    if not isinstance(current_user, Student):
        return redirect(url_for('login'))
    return render_template('analytics.html', 
                         total_points=current_user.total_points)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # Only create sample data if database is empty
        if not Teacher.query.first():
            # Create sample teacher
            teacher = Teacher(username="math_teacher")
            teacher.set_password("teacher123")
            db.session.add(teacher)
            db.session.commit()

            # Create main sample student
            student = Student(username="star_student", teacher_id=teacher.id)
            student.set_password("student123")
            db.session.add(student)
            db.session.commit()

            # Generate 5 attempts for each game type
            import random
            game_data = []
            
            # Game 1 attempts (50-150 points)
            for _ in range(5):
                game_data.append((
                    student.id,
                    1,
                    random.randint(50, 150)
                ))

            # Game 2 attempts (80-200 points)
            for _ in range(5):
                game_data.append((
                    student.id,
                    2,
                    random.randint(80, 200)
                ))

            # Game 3 attempts (100-300 points)
            for _ in range(5):
                game_data.append((
                    student.id,
                    3,
                    random.randint(100, 300)
                ))

            # Add all game attempts and calculate total points
            total_points = 0
            for student_id, game_type, points in game_data:
                game = GameTracker(
                    studentID=student_id,
                    GameType=game_type,
                    Points=points
                )
                db.session.add(game)
                total_points += points

            # Update student's total points
            student.total_points = total_points
            db.session.commit()

            print(f"Created 15 game attempts for {student.username}")
            print(f"Total points: {total_points}")
            
    app.run(debug=True)
