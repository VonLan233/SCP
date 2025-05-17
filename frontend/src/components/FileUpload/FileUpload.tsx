// src/components/FileUpload/FileUpload.tsx
import React, { useState } from 'react';
import { X } from 'lucide-react';
import { uploadScoreSheet, uploadClassData, uploadStudentData } from '../../api';
import './FileUpload.css';

// 定义上传文件类型
type UploadType = 'scores' | 'classes' | 'students';

interface FileUploadProps {
  onClose: () => void;
  onUploadSuccess?: () => void; // 添加上传成功回调
  initialTab?: UploadType; // 初始选中的标签页
}

const FileUpload: React.FC<FileUploadProps> = ({ onClose, onUploadSuccess, initialTab = 'scores' }) => {
  const [activeTab, setActiveTab] = useState<UploadType>(initialTab);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      
      // 验证文件类型 - 更宽松的检查，因为某些浏览器可能返回不同的MIME类型
      const validExtensions = ['.csv', '.xls', '.xlsx'];
      const fileExtension = selectedFile.name.substring(selectedFile.name.lastIndexOf('.')).toLowerCase();
      
      if (!validExtensions.includes(fileExtension)) {
        setError('请上传有效的Excel或CSV文件');
        setFile(null);
        return;
      }
      
      setFile(selectedFile);
      setError('');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file) {
      setError('请先选择文件');
      return;
    }
    
    setUploading(true);
    setError('');
    setMessage('');
    
    try {
      console.log(`开始上传${getTabName()}文件:`, file.name);
      
      let response;
      
      // 根据当前选择的标签页决定上传端点
      switch (activeTab) {
        case 'scores':
          response = await uploadScoreSheet(file);
          break;
        case 'classes':
          response = await uploadClassData(file);
          break;
        case 'students':
          response = await uploadStudentData(file);
          break;
      }
      
      console.log('上传响应:', response);
      
      if (response.success) {
        // 处理不同类型上传的响应
        if (activeTab === 'scores') {
          const processedRecords = response.data?.processed_records || 0;
          const autoCreated = response.data?.auto_created || {};
          const errors = response.data?.errors || [];
          
          let successMessage = `${getTabName()}文件上传成功，处理了 ${processedRecords} 条记录。`;
          
          // 添加自动创建的信息
          if (autoCreated.students > 0 || autoCreated.classes > 0 || autoCreated.exams > 0) {
            successMessage += ` 系统自动创建了 ${autoCreated.students || 0} 名学生、${autoCreated.classes || 0} 个班级和 ${autoCreated.exams || 0} 次考试。`;
          }
          
          if (errors.length > 0) {
            successMessage += ` 有 ${errors.length} 条记录处理失败。`;
            console.log('处理错误:', errors);
          }
          
          setMessage(successMessage);
        } else if (activeTab === 'classes') {
          const processedRecords = response.data?.processed_records || 0;
          const created = response.data?.created || {};
          const updated = response.data?.updated || {};
          
          let successMessage = `${getTabName()}文件上传成功，处理了 ${processedRecords} 条记录。`;
          successMessage += ` 创建了 ${created.classes || 0} 个班级、${created.grades || 0} 个年级、${created.teachers || 0} 名教师。`;
          successMessage += ` 更新了 ${updated.classes || 0} 个班级。`;
          
          setMessage(successMessage);
        } else if (activeTab === 'students') {
          const processedRecords = response.data?.processed_records || 0;
          const created = response.data?.created || {};
          const updated = response.data?.updated || {};
          
          let successMessage = `${getTabName()}文件上传成功，处理了 ${processedRecords} 条记录。`;
          successMessage += ` 创建了 ${created.students || 0} 名学生、${created.classes || 0} 个班级、${created.grades || 0} 个年级。`;
          successMessage += ` 更新了 ${updated.students || 0} 名学生。`;
          
          setMessage(successMessage);
        }
        
        // 调用成功回调
        if (onUploadSuccess) {
          setTimeout(() => {
            onUploadSuccess();
          }, 500); // 短暂延迟，确保后端处理完成
        }
        
        // 添加延迟关闭
        // setTimeout(() => {
        //   onClose();
        // }, 3000);
      } else {
        setError(response.message || `${getTabName()}文件上传失败`);
      }
    } catch (err) {
      console.error('上传过程中发生错误:', err);
      setError('上传过程中发生错误，请检查文件格式或网络连接');
    } finally {
      setUploading(false);
    }
  };

  // 获取当前标签页的显示名称
  const getTabName = () => {
    switch (activeTab) {
      case 'scores':
        return '成绩';
      // case 'classes':
      //   return '班级';
      case 'students':
        return '学生';
      default:
        return '';
    }
  };

  // 获取当前标签页的文件格式要求
  const getFormatRequirements = () => {
    switch (activeTab) {
      case 'scores':
        return (
          <ul className="requirements-list">
            <li>文件应包含以下列: student_id(学生ID), exam_date(考试日期), score(分数), subject(科目)</li>
            <li>可选列: class_name(班级名称), exam_name(考试名称), student_name(学生姓名)</li>
            <li>确保第一行是列标题</li>
            <li>日期格式应为YYYY-MM-DD</li>
            <li>分数应为0-100之间的数字</li>
          </ul>
        );
      // case 'classes':
        // return (
        //   <ul className="requirements-list">
        //     <li>必需列: class_name(班级名称)</li>
        //     <li>可选列: class_id(班级ID), grade_name(年级名称), teacher_name(班主任姓名), student_count(学生数量)</li>
        //     <li>确保第一行是列标题</li>
        //     <li>如果班级名称中包含年级信息（如"高一(1)班"），系统会自动识别年级</li>
        //   </ul>
        // );
      case 'students':
        return (
          <ul className="requirements-list">
            <li>必需列: student_id(学生学号), student_name(学生姓名)</li>
            {/* <li>至少包含以下列之一: </li> */}
            <li>可选列: choose_subject(选课), contact(联系方式), other_info(其他信息)</li>
            <li>确保第一行是列标题</li>
            <li>如未提供班级信息，将使用默认班级</li>
          </ul>
        );
      default:
        return null;
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal">
        <div className="modal-header">
          <h2>上传{getTabName()}数据</h2>
          <button className="close-button" onClick={onClose}>
            <X className="icon" />
          </button>
        </div>
        
        <div className="modal-body">
          {/* 标签页切换 */}
          <div className="upload-tabs">
            <button 
              className={`upload-tab ${activeTab === 'scores' ? 'active' : ''}`}
              onClick={() => {
                setActiveTab('scores');
                setFile(null);
                setError('');
                setMessage('');
              }}
            >
              成绩数据
            </button>
            {/* <button 
              className={`upload-tab ${activeTab === 'classes' ? 'active' : ''}`}
              onClick={() => {
                setActiveTab('classes');
                setFile(null);
                setError('');
                setMessage('');
              }}
            >
              班级数据
            </button> */}
            <button 
              className={`upload-tab ${activeTab === 'students' ? 'active' : ''}`}
              onClick={() => {
                setActiveTab('students');
                setFile(null);
                setError('');
                setMessage('');
              }}
            >
              学生数据
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="file-upload">选择Excel或CSV文件</label>
              <input
                type="file"
                id="file-upload"
                accept=".xlsx,.xls,.csv"
                onChange={handleFileChange}
                disabled={uploading}
              />
              <p className="help-text">支持的文件格式: Excel (.xlsx, .xls) 或 CSV (.csv)</p>
              {error && <p className="error-text">{error}</p>}
              {message && <p className="success-text">{message}</p>}
            </div>
            
            <div className="form-group">
              <h3>文件格式要求:</h3>
              {getFormatRequirements()}
              
              <div className="template-download">
                <p>需要模板？<a href={`/templates/${activeTab}_template.csv`} download>下载{getTabName()}数据模板</a></p>
              </div>
            </div>
            
            <div className="form-actions">
              <button type="button" className="btn btn-secondary" onClick={onClose} disabled={uploading}>
                取消
              </button>
              <button type="submit" className="btn btn-primary" disabled={!file || uploading}>
                {uploading ? '上传中...' : '上传'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default FileUpload;