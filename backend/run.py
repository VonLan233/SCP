from app import create_app, db
from app.models.student import Student
from app.models.class_model import Class  
from app.models.grade import Grade
from app.models.exam import Exam
from app.models.score import Score

app = create_app()

@app.shell_context_processor
def make_shell_context():
    return {
        'db': db,
        'Student': Student,
        'Class': Class, 
        'Grade': Grade,
        'Exam': Exam,
        'Score': Score
    }

if __name__ == '__main__':
    app.run(debug=True)