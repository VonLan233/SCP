# 创建 migrations/add_class_count.py

from app import create_app, db
from sqlalchemy import text

def add_class_count_column():
    app = create_app()
    with app.app_context():
        print("开始添加 class_count 列到 grades 表...")
        
        # 检查列是否已存在
        inspector = db.inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('grades')]
        
        if 'class_count' not in columns:
            print("添加 class_count 列...")
            db.engine.execute(text('ALTER TABLE grades ADD COLUMN class_count INTEGER DEFAULT 0;'))
            
            # 更新现有记录的班级数量
            print("更新现有年级的班级数量...")
            result = db.engine.execute(text('''
                UPDATE grades
                SET class_count = (
                    SELECT COUNT(*)
                    FROM classes
                    WHERE classes.grade_id = grades.id
                )
                WHERE EXISTS (
                    SELECT 1
                    FROM classes
                    WHERE classes.grade_id = grades.id
                );
            '''))
            
            print(f"成功更新 {result.rowcount} 个年级的班级数量")
            print("class_count 列已添加并初始化")
        else:
            print("class_count 列已存在，无需添加")

if __name__ == "__main__":
    add_class_count_column()