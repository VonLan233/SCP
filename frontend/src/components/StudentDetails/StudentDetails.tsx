// src/components/StudentDetails/StudentDetails.tsx
import React, { useState, useEffect } from 'react';
import { ExamScore, ClassAverage } from '../../types';
import { TrendingUp, TrendingDown, AlertTriangle, Award } from 'lucide-react';
import './StudentDetails.css';

interface StudentDetailsProps {
  studentId: number;
  studentName: string;
  scoreData: ExamScore[];
  classAverageData: ClassAverage[];
}

const StudentDetails: React.FC<StudentDetailsProps> = ({
  studentId,
  studentName,
  scoreData,
  classAverageData
}) => {
  // 计算统计数据
  const [stats, setStats] = useState({
    average: 0,
    highest: 0,
    highestDate: '',
    lowest: 0,
    lowestDate: '',
    improvement: 0,
    comparedToClass: 0,
    rank: '0/0',
    rankPercentile: '0%'
  });

  useEffect(() => {
    if (scoreData.length > 0) {
      // 过滤有实际分数的考试
      const actualScores = scoreData.filter(score => score.actual !== null);
      
      if (actualScores.length > 0) {
        // 计算平均分
        const sum = actualScores.reduce((acc, score) => acc + (score.actual || 0), 0);
        const average = sum / actualScores.length;
        
        // 找出最高分和最低分
        const highest = Math.max(...actualScores.map(score => score.actual || 0));
        const highestScore = actualScores.find(score => score.actual === highest);
        
        const lowest = Math.min(...actualScores.map(score => score.actual || 0));
        const lowestScore = actualScores.find(score => score.actual === lowest);
        
        // 计算最近的进步
        const recentScores = actualScores.slice(-3);
        const firstRecent = recentScores[0]?.actual || 0;
        const lastRecent = recentScores[recentScores.length - 1]?.actual || 0;
        const improvement = lastRecent - firstRecent;
        
        // 与班级平均分比较
        const lastScoreIndex = actualScores.length - 1;
        const lastScore = actualScores[lastScoreIndex]?.actual || 0;
        const correspondingClassAvg = classAverageData[lastScoreIndex]?.average || 0;
        const comparedToClass = lastScore - correspondingClassAvg;
        
        setStats({
          average: parseFloat(average.toFixed(1)),
          highest,
          highestDate: highestScore?.date || '',
          lowest,
          lowestDate: lowestScore?.date || '',
          improvement,
          comparedToClass,
          rank: '3/28', // 模拟数据，实际应从API获取
          rankPercentile: '前10%' // 模拟数据，实际应从API获取
        });
      }
    }
  }, [scoreData, classAverageData]);

  return (
    <div className="student-details">
      <div className="student-header">
        <h2>{studentName}的学习表现详情</h2>
      </div>
      
      <div className="stats-overview">
        <div className="stat-box">
          <h3>总体表现</h3>
          <div className="stat-grid">
            <div className="stat-item">
              <span className="stat-label">平均分</span>
              <span className="stat-value">{stats.average}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">最高分</span>
              <span className="stat-value">{stats.highest}</span>
              <span className="stat-date">{stats.highestDate}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">最低分</span>
              <span className="stat-value">{stats.lowest}</span>
              <span className="stat-date">{stats.lowestDate}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">近期进步</span>
              <span className={`stat-value ${stats.improvement > 0 ? 'positive' : stats.improvement < 0 ? 'negative' : ''}`}>
                {stats.improvement > 0 ? '+' : ''}{stats.improvement}
              </span>
            </div>
          </div>
        </div>
        
        <div className="stat-box">
          <h3>班级对比</h3>
          <div className="stat-grid">
            <div className="stat-item">
              <span className="stat-label">对比班级平均</span>
              <span className={`stat-value ${stats.comparedToClass > 0 ? 'positive' : stats.comparedToClass < 0 ? 'negative' : ''}`}>
                {stats.comparedToClass > 0 ? '+' : ''}{stats.comparedToClass}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">班级排名</span>
              <span className="stat-value">{stats.rank}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">排名百分比</span>
              <span className="stat-value">{stats.rankPercentile}</span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="performance-analysis">
        <h3>表现分析</h3>
        <div className="analysis-list">
          {stats.improvement > 0 && (
            <div className="analysis-item positive">
              <TrendingUp className="analysis-icon" />
              <div className="analysis-content">
                <h4>稳定上升趋势</h4>
                <p>学生在最近几次考试中表现出稳定的进步，成绩持续提高。</p>
              </div>
            </div>
          )}
          
          {stats.improvement < 0 && (
            <div className="analysis-item negative">
              <TrendingDown className="analysis-icon" />
              <div className="analysis-content">
                <h4>成绩有所下滑</h4>
                <p>学生在最近几次考试中表现出下降趋势，可能需要额外关注。</p>
              </div>
            </div>
          )}
          
          {stats.comparedToClass > 5 && (
            <div className="analysis-item positive">
              <Award className="analysis-icon" />
              <div className="analysis-content">
                <h4>优秀表现</h4>
                <p>学生成绩显著高于班级平均水平，表现优异。</p>
              </div>
            </div>
          )}
          
          {stats.comparedToClass < -5 && (
            <div className="analysis-item warning">
              <AlertTriangle className="analysis-icon" />
              <div className="analysis-content">
                <h4>需要提升</h4>
                <p>学生成绩低于班级平均水平，可能需要额外辅导或关注。</p>
              </div>
            </div>
          )}
          
          {/* 添加更多的分析条目 */}
        </div>
      </div>
      
      <div className="recommendations">
        <h3>学习建议</h3>
        <div className="recommendation-list">
          <div className="recommendation-item">
            <div className="recommendation-number">1</div>
            <div className="recommendation-content">
              <h4>强化弱点科目</h4>
              <p>根据分析，建议关注数学中的几何部分，加强练习和理解。</p>
            </div>
          </div>
          
          <div className="recommendation-item">
            <div className="recommendation-number">2</div>
            <div className="recommendation-content">
              <h4>制定学习计划</h4>
              <p>建议制定更有针对性的学习计划，每周定期复习重点内容。</p>
            </div>
          </div>
          
          <div className="recommendation-item">
            <div className="recommendation-number">3</div>
            <div className="recommendation-content">
              <h4>参与小组讨论</h4>
              <p>通过参与小组讨论，加深对复杂概念的理解和应用能力。</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StudentDetails;