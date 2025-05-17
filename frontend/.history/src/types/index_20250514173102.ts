export interface Student {
  id: string;
  student_id: string;
  name: string;
  average: number;
  trend: 'up' | 'down' | 'stable';
  alerts: number;
  classId: number;
  gradeId: number;
}

export interface ExamScore {
  examId: number;
  subject: string;
  date: string;
  actual: number | null;
  predicted: number | null;
  lower: number | null;
  upper: number | null;
}

export interface ClassAverage {
  examId: number;
  date: string;
  subject: string;  
  average: number;
}

export interface Class {
  id: number;
  name: string;
  gradeId: number;
}

export interface Grade {
  id: number;
  name: string;
}

// API类型定义
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// 文件上传响应
export interface FileUploadResponse {
  success: boolean;
  message?: string;
  data?: {
    processed_records?: number;
    errors?: string[];
  };
}
