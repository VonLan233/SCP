import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import './ClassAnalysis.css';

interface ClassAnalysisProps {
  classId: number;
  className: string;
  averageData: any[];
}

const ClassAnalysis: React.FC<ClassAnalysisProps> = ({
  classId,
  className,
  averageData
}) => {
  // 模拟数据 - 在实际应用中应从API获取
  const scoreDistribution = [
    { name: '优秀 (90-100)', value: 8 },
    { name: '良好 (80-89)', value: 12 },
    { name: '中等 (70-79)', value: 6 },
    { name: '及格 (60-69)', value: 3 },
    { name: '不及格 (<60)', value: 1 }
  ];
  
  const subjectPerformance = [
    { subject: '语文', average: 85 },
    { subject: '数学', average: 82 },
    { subject: '英语', average: 88 },
    { subject: '物理', average: 79 },
    { subject: '化学', average: 81 },
    { subject: '生物', average: 84 }
  ];
  
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#FF0000'];

  return (
    <div className="class-analysis">
      <div className="class-header">
        <h2>{className}班级分析</h2>
      </div>
      
      <div className="analysis-grid">
        <div className="analysis-card">
          <h3>成绩分布</h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={scoreDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={true}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                >
                  {scoreDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="analysis-card">
          <h3>学科表现</h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={subjectPerformance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="subject" />
                <YAxis domain={[60, 100]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="average" name="平均分" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="analysis-card full-width">
          <h3>班级成绩趋势</h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={averageData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis domain={[60, 100]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="average" name="班级平均分" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      
      <div className="insights-container">
        <div className="insight-card">
          <h3>班级洞察</h3>
          <ul className="insights-list">
            <li className="insight-item">
              <div className="insight-highlight">优势科目: 英语</div>
              <p>班级在英语科目上表现最佳，平均分达到88分，可以考虑分享英语学习方法。</p>
            </li>
            <li className="insight-item">
              <div className="insight-highlight">需改进科目: 物理</div>
              <p>物理科目平均分为79分，是所有科目中最低的，建议加强此科目的教学。</p>
            </li>
            <li className="insight-item">
              <div className="insight-highlight">成绩稳定性</div>
              <p>班级整体成绩趋势稳步上升，表明教学方法有效。</p>
            </li>
          </ul>
        </div>
        
        <div className="insight-card">
          <h3>教学建议</h3>
          <ul className="insights-list">
            <li className="insight-item">
              <div className="insight-highlight">1. 分组辅导</div>
              <p>针对不同成绩层次的学生采用分组辅导策略，特别关注后10%的学生。</p>
            </li>
            <li className="insight-item">
              <div className="insight-highlight">2. 增强物理教学</div>
              <p>在物理科目上增加实验教学和实际应用案例，提高学生兴趣和理解。</p>
            </li>
            <li className="insight-item">
              <div className="insight-highlight">3. 经验分享</div>
              <p>组织成绩优秀的学生分享学习方法和经验，促进同伴学习。</p>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ClassAnalysis;