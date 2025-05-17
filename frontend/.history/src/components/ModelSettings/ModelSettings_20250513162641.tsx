import React, { useState } from 'react';
import './ModelSettings.css';

interface ModelSettingsProps {
  onClose: () => void;
  onSave: (settings: ModelSettingsType) => void;
  currentSettings: ModelSettingsType;
  selectedSubject: string;
  availableSubjects: string[];
  onSubjectChange: (subject: string) => void;
}

export interface ModelSettingsType {
  timeSteps: number;
  predictionSteps: number;
  epochs: number;
  batchSize: number;
  confidenceInterval: number;
  force_retrain: boolean;
}

const ModelSettings: React.FC<ModelSettingsProps> = ({
  onClose,
  onSave,
  currentSettings,
  selectedSubject,
  availableSubjects,
  onSubjectChange
}) => {
  const [settings, setSettings] = useState<ModelSettingsType>(currentSettings);
  const [localSubject, setLocalSubject] = useState<string>(selectedSubject);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setSettings({
      ...settings,
      [name]: type === 'checkbox' ? checked : parseInt(value, 10)
    });
  };

  const handleSubjectChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setLocalSubject(e.target.value);
    onSubjectChange(e.target.value);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(settings);
    onClose();
  };

  return (
    <div className="modal-overlay">
      <div className="modal model-settings-modal">
        <div className="modal-header">
          <h2>预测模型设置</h2>
          <button className="close-button" onClick={onClose}>
            ×
          </button>
        </div>
        
        <div className="modal-body">
          <form onSubmit={handleSubmit}>
            {/* 科目选择 */}
            <div className="form-group">
              <label htmlFor="modelSubject">预测科目</label>
              <select 
                id="modelSubject"
                value={localSubject} 
                onChange={handleSubjectChange}
                className="form-control"
              >
                <option value="all">全部科目</option>
                {availableSubjects.map(subject => (
                  <option key={subject} value={subject}>{subject}</option>
                ))}
              </select>
              <p className="help-text">
                选择要预测的科目。每个科目将使用单独的预测模型。
              </p>
            </div>

            <div className="form-group">
              <label htmlFor="timeSteps">历史时间步长</label>
              <input
                type="number"
                id="timeSteps"
                name="timeSteps"
                min="1"
                max="20"
                value={settings.timeSteps}
                onChange={handleChange}
              />
              <p className="help-text">模型将使用多少历史考试数据点来预测未来。较大的值可能提高精确度，但需要更多数据。</p>
            </div>
            
            <div className="form-group">
              <label htmlFor="predictionSteps">预测步长</label>
              <input
                type="number"
                id="predictionSteps"
                name="predictionSteps"
                min="1"
                max="10"
                value={settings.predictionSteps}
                onChange={handleChange}
              />
              <p className="help-text">模型将预测未来多少次考试的成绩。预测更远的未来可能降低精确度。</p>
            </div>
            
            {/* <div className="form-group">
              <label htmlFor="epochs">训练轮数</label>
              <input
                type="number"
                id="epochs"
                name="epochs"
                min="10"
                max="500"
                value={settings.epochs}
                onChange={handleChange}
              />
              <p className="help-text">模型训练的迭代次数。较大的值可能提高精确度，但会增加训练时间。</p>
            </div> */}
            
            {/* <div className="form-group">
              <label htmlFor="batchSize">批次大小</label>
              <input
                type="number"
                id="batchSize"
                name="batchSize"
                min="8"
                max="128"
                step="8"
                value={settings.batchSize}
                onChange={handleChange}
              />
              <p className="help-text">每次更新模型参数时使用的样本数量。</p>
            </div> */}
            
            <div className="form-group">
              <label htmlFor="confidenceInterval">置信区间 (%)</label>
              <input
                type="number"
                id="confidenceInterval"
                name="confidenceInterval"
                min="70"
                max="99"
                value={settings.confidenceInterval}
                onChange={handleChange}
              />
              <p className="help-text">预测结果的置信区间宽度。较高的值会产生更宽的预测范围。</p>
            </div>

            {/* 强制重新训练选项 */}
            <div className="form-group checkbox-group">
              <div className="checkbox-container">
                <input
                  type="checkbox"
                  id="force_retrain"
                  name="force_retrain"
                  checked={settings.force_retrain || false}
                  onChange={handleChange}
                />
                <label htmlFor="force_retrain">强制重新训练模型</label>
              </div>
              <p className="help-text">勾选此选项将忽略已有的模型，强制进行重新训练。</p>
            </div>
            
            <div className="form-actions">
              <button type="button" className="btn btn-secondary" onClick={onClose}>
                取消
              </button>
              <button type="submit" className="btn btn-primary">
                保存设置
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ModelSettings;