// src/components/FilterPanel/FilterPanel.tsx
import React from 'react';
import { Grade, Class } from '../../types';
import './FilterPanel.css';

interface FilterPanelProps {
  grades: Grade[];
  classes: Class[];
  selectedGrade: number | null;
  selectedClass: number | null;
  viewMode: 'student' | 'class' | 'grade';
  onGradeChange: (gradeId: number | null) => void; // 修改类型以接受null
  onClassChange: (classId: number | null) => void; // 修改类型以接受null
  onViewModeChange: (mode: 'student' | 'class' | 'grade') => void;
}

const FilterPanel: React.FC<FilterPanelProps> = ({
  grades,
  classes,
  selectedGrade,
  selectedClass,
  viewMode,
  onGradeChange,
  onClassChange,
  onViewModeChange
}) => {
  // 添加调试日志
  console.log('FilterPanel 渲染:', { 
    gradesCount: grades.length, 
    classesCount: classes.length,
    selectedGrade, 
    selectedClass, 
    viewMode 
  });

  return (
    <div className="filter-panel">
      <h3>筛选条件</h3>
      
      <div className="filter-group">
        <label htmlFor="view-mode">视图模式</label>
        <select
          id="view-mode"
          value={viewMode}
          onChange={(e) => {
            console.log('视图模式变更:', e.target.value);
            onViewModeChange(e.target.value as 'student' | 'class' | 'grade');
          }}
        >
          <option value="student">学生</option>
          <option value="class">班级</option>
          <option value="grade">年级</option>
        </select>
      </div>
      
      <div className="filter-group">
        <label htmlFor="grade-select">年级</label>
        <select
          id="grade-select"
          value={selectedGrade || ''}
          onChange={(e) => {
            const value = e.target.value ? Number(e.target.value) : null;
            console.log('年级选择变更:', value);
            onGradeChange(value);
          }}
        >
          <option value="">选择年级</option>
          {grades.map(grade => (
            <option key={grade.id} value={grade.id}>{grade.name}</option>
          ))}
        </select>
      </div>
      
      {(viewMode === 'student' || viewMode === 'class') && (
        <div className="filter-group">
          <label htmlFor="class-select">班级</label>
          <select
            id="class-select"
            value={selectedClass || ''}
            onChange={(e) => {
              const value = e.target.value ? Number(e.target.value) : null;
              console.log('班级选择变更:', value);
              onClassChange(value);
            }}
            disabled={!selectedGrade}
          >
            <option value="">选择班级</option>
            {classes.map(cls => (
              <option key={cls.id} value={cls.id}>{cls.name}</option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
};

export default FilterPanel;