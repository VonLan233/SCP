// src/api/index.ts
import {
  Student,
  ExamScore,
  ClassAverage,
  Class,
  Grade,
  ApiResponse,
  FileUploadResponse,
} from "../types";

const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || "http://localhost:8000/api";

// 获取学生列表
export async function getStudents(
  gradeId?: number,
  classId?: number
): Promise<ApiResponse<Student[]>> {
  try {
    let url = `${API_BASE_URL}/students`;
    const params = new URLSearchParams();

    if (gradeId) params.append("gradeId", gradeId.toString());
    if (classId) params.append("classId", classId.toString());

    if (params.toString()) url += `?${params.toString()}`;

    const response = await fetch(url);
    return await response.json();
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "获取学生列表失败",
    };
  }
}

// 获取学生详情
export async function getStudentDetail(
  studentId: string
): Promise<ApiResponse<Student>> {
  try {
    const response = await fetch(`${API_BASE_URL}/students/${studentId}`);
    return await response.json();
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "获取学生详情失败",
    };
  }
}

// 获取学生成绩历史
export async function getStudentScores(
  studentId: string
): Promise<ApiResponse<ExamScore[]>> {
  try {
    const response = await fetch(
      `${API_BASE_URL}/students/${studentId}/scores`
    );
    return await response.json();
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "获取学生成绩历史失败",
    };
  }
}

// 获取班级平均分
export async function getClassAverage(
  classId: number
): Promise<ApiResponse<ClassAverage[]>> {
  try {
    const response = await fetch(`${API_BASE_URL}/classes/${classId}/average`);
    return await response.json();
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "获取班级平均分失败",
    };
  }
}

// 获取年级平均分
export async function getGradeAverage(
  gradeId: number
): Promise<ApiResponse<ClassAverage[]>> {
  try {
    const response = await fetch(`${API_BASE_URL}/grades/${gradeId}/average`);
    return await response.json();
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "获取年级平均分失败",
    };
  }
}

// 获取班级列表
export async function getClasses(
  gradeId?: number
): Promise<ApiResponse<Class[]>> {
  try {
    let url = `${API_BASE_URL}/classes`;
    if (gradeId) url += `?gradeId=${gradeId}`;

    const response = await fetch(url);
    return await response.json();
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "获取班级列表失败",
    };
  }
}

// 获取年级列表
export async function getGrades(): Promise<ApiResponse<Grade[]>> {
  try {
    const response = await fetch(`${API_BASE_URL}/grades`);
    return await response.json();
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "获取年级列表失败",
    };
  }
}

// 在 src/api/index.ts 文件中

// 预测学生未来成绩
export async function predictStudentScore(
  studentId: string
): Promise<ApiResponse<ExamScore[]>> {
  console.log(`调用API预测学生 ${studentId} 的成绩`);

  try {
    // 确保使用正确的URL
    const url = `${API_BASE_URL}/students/${studentId}/predict`;
    console.log(`请求URL: ${url}`);

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        steps: 3,
      }),
    });

    // 输出响应状态
    console.log(`预测API响应状态: ${response.status} ${response.statusText}`);

    // 先获取响应文本
    const responseText = await response.text();
    console.log(`预测API响应内容: ${responseText.substring(0, 200)}...`);

    try {
      // 尝试解析JSON
      const data = JSON.parse(responseText);
      return data;
    } catch (parseError) {
      console.error("解析预测响应失败:", parseError);

      // 检查是否为HTML响应
      if (
        responseText.toLowerCase().includes("<!doctype") ||
        responseText.toLowerCase().includes("<html")
      ) {
        console.error("服务器返回了HTML页面而非JSON响应");
        return {
          success: false,
          error: "服务器错误：返回了网页而不是API响应",
        };
      }

      return {
        success: false,
        error: "服务器返回了无效的响应格式",
      };
    }
  } catch (error) {
    console.error("预测API调用失败:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "预测学生成绩失败",
    };
  }
}

// 预测班级平均分
export async function predictClassAverage(
  classId: number
): Promise<ApiResponse<ClassAverage[]>> {
  try {
    const response = await fetch(`${API_BASE_URL}/classes/${classId}/predict`, {
      method: "POST",
    });
    return await response.json();
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "预测班级平均分失败",
    };
  }
}

// 预测年级平均分
export async function predictGradeAverage(
  gradeId: number
): Promise<ApiResponse<ClassAverage[]>> {
  try {
    const response = await fetch(`${API_BASE_URL}/grades/${gradeId}/predict`, {
      method: "POST",
    });
    return await response.json();
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "预测年级平均分失败",
    };
  }
}

// 上传学生成绩表格
// 上传学生成绩表格
// 在 src/api/index.ts 文件中
// src/api.ts 中添加新的上传函数

/**
 * 上传班级数据
 * @param file 班级数据文件
 */
export const uploadClassData = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/api/upload/classes", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("上传班级数据错误:", error);
    return {
      success: false,
      message: "上传过程中发生错误",
    };
  }
};

/**
 * 上传学生数据
 * @param file 学生数据文件
 */
export const uploadStudentData = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/api/upload/students", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("上传学生数据错误:", error);
    return {
      success: false,
      message: "上传过程中发生错误",
    };
  }
};

// 上传学生成绩表格
export async function uploadScoreSheet(file: File): Promise<any> {
  try {
    const formData = new FormData();
    formData.append("file", file);

    console.log(
      `正在上传文件: ${file.name}, 大小: ${file.size} 字节, 类型: ${file.type}`
    );

    const response = await fetch(`${API_BASE_URL}/upload/scores`, {
      method: "POST",
      body: formData,
    });

    // 先获取响应文本
    const responseText = await response.text();
    console.log(`服务器响应: ${responseText.substring(0, 200)}...`);

    try {
      // 尝试解析JSON
      const data = JSON.parse(responseText);
      return data;
    } catch (parseError) {
      console.error("解析服务器响应失败:", parseError);

      return {
        success: false,
        message: "服务器返回了无效的响应格式",
      };
    }
  } catch (error) {
    console.error("上传过程中发生网络错误:", error);
    return {
      success: false,
      message: error instanceof Error ? error.message : "上传成绩表格失败",
    };
  }
}
// 添加到src/api/index.ts文件中

// 按学号搜索学生
export async function searchStudentById(
  studentId: string
): Promise<ApiResponse<Student[]>> {
  try {
    const url = `${API_BASE_URL}/search/students?student_id=${studentId}`;
    console.log(`搜索学生，URL: ${url}`);

    const response = await fetch(url);
    return await response.json();
  } catch (error) {
    console.error("按学号搜索学生失败:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "按学号搜索学生失败",
    };
  }
}

// 也可以添加一个更通用的搜索函数，支持多种搜索条件
export async function searchStudents(params: {
  student_id?: string;
  name?: string;
}): Promise<ApiResponse<Student[]>> {
  try {
    const searchParams = new URLSearchParams();
    if (params.student_id) searchParams.append("student_id", params.student_id);
    if (params.name) searchParams.append("name", params.name);

    const url = `${API_BASE_URL}/search/students?${searchParams.toString()}`;
    console.log(`搜索学生，URL: ${url}`);

    const response = await fetch(url);
    return await response.json();
  } catch (error) {
    console.error("搜索学生失败:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "搜索学生失败",
    };
  }
}
// 在现有类型基础上补充以下类型

// 学生成绩分析结果
export interface StudentAnalysis {
  student_id: string;
  statistics: {
    average: number;
    std_dev: number;
    max: number;
    min: number;
    range: number;
  };
  trend: {
    value: number;
    type: "显著提升" | "稳步提升" | "显著下降" | "轻微下降" | "基本稳定";
    recent_average: number;
    historical_average: number;
  };
  volatility: {
    value: number;
    level: "高波动" | "中等波动" | "低波动";
  };
  outliers: Array<{
    index: number;
    score: number;
    z_score: number;
  }>;
  recommendations: string[];
  next_exam_prediction?: {
    score: number;
    lower_bound: number;
    upper_bound: number;
  };
}

// 成绩异常检测结果
export interface PerformanceAnomaly {
  student_id: string;
  student_name: string;
  type: "outperformer" | "underperformer" | "high_volatility" | "sudden_change";
  description: string;
  scores: number[];
  avg_score: number;
}

// 班级分析结果
export interface ClassAnalysisResult {
  class_id: number;
  predictions: ClassAverage[];
  anomalies: PerformanceAnomaly[];
  improvement_opportunities: string[];
  strengths: string[];
  recommendations: string[];
  predictions_trend?: {
    direction: "upward" | "downward" | "stable";
    magnitude: number;
    description: string;
  };
}

// 模型配置参数
export interface ModelSettings {
  timeSteps: number;
  predictionSteps: number;
  epochs: number;
  batchSize: number;
  confidenceInterval: number;
  modelType?: string;
  force_retrain?: boolean;
}

// 模型状态信息
export interface ModelStatus {
  trained: boolean;
  metrics: {
    MSE: number;
    MAE: number;
    RMSE: number;
    R2: number;
  } | null;
  last_trained: string | null;
  model_type: string;
  available_data: number;
}

// 服务状态信息
export interface ServiceStatus {
  service: {
    name: string;
    version: string;
    default_model_type: string;
    model_path: string;
    cached_models: number;
  };
  models: {
    trained_count: number;
    model_types: Record<string, number>;
    default_params: Record<string, any>;
    training_config: Record<string, any>;
  };
  data: {
    total_students: number;
    total_scores: number;
    total_classes: number;
    avg_scores_per_student: number;
  };
}
// 在现有 src/api/index.ts 文件末尾添加以下内容
// 以下是高级预测API封装，直接集成到现有API模块中

/**
 * 高级预测API - 使用新的预测服务端点
 */

// 高级学生成绩预测API基础URL
const PREDICTION_API = `${API_BASE_URL}/prediction`;

// 高级预测学生成绩
// 替换 src/api/index.ts 中的 predictStudentScoresAdvanced 函数

export async function predictStudentScoresAdvanced(
  studentId: string,
  steps: number = 1,
  modelParams?: any
): Promise<ApiResponse<ExamScore[]>> {
  try {
    console.log(`调用高级API预测学生 ${studentId} 的成绩，参数:`, {
      steps,
      modelParams,
    });

    const response = await fetch(`${PREDICTION_API}/student/${studentId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        steps,
        modelParams,
        modelType: "student_predictor", // 默认使用StudentScorePredictor
        subject: modelParams?.subject || "总分", // 添加科目参数
      }),
    });

    console.log(
      `高级预测API响应状态: ${response.status} ${response.statusText}`
    );

    // 解析响应
    const data = await response.json();
    // 检查数据结构
    if (data.success && data.data) {
      // 检查data是否包含必要的字段
      const firstItem = data.data[0];
      console.log("数据结构示例:", firstItem);

      // 确保数值类型正确
      data.data = data.data.map((item: { predicted: number | null; lower: number | null; upper: number | null }) => ({
        ...item,
        predicted: item.predicted !== null ? Number(item.predicted) : null,
        lower: item.lower !== null ? Number(item.lower) : null,
        upper: item.upper !== null ? Number(item.upper) : null,
      }));
    }
    return data;
  } catch (error) {
    console.error("高级预测API调用失败:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "预测学生成绩失败",
    };
  }
}

// 修改 getModelStatus 函数以支持科目参数
export async function getModelStatus(
  studentId: string,
  subject?: string,
  modelType?: string
): Promise<ApiResponse<any>> {
  try {
    let url = `${PREDICTION_API}/model/status/${studentId}`;
    const params = new URLSearchParams();

    if (modelType) {
      params.append("modelType", modelType);
    }

    if (subject) {
      params.append("subject", subject);
    }

    if (params.toString()) {
      url += `?${params.toString()}`;
    }

    const response = await fetch(url);
    return await response.json();
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "获取模型状态失败",
    };
  }
}

// 预测班级平均分(高级版)
export async function predictClassAverageAdvanced(
  classId: number
): Promise<ApiResponse<ClassAverage[]>> {
  try {
    const response = await fetch(`${PREDICTION_API}/class/${classId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        steps: 3,
      }),
    });
    return await response.json();
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "预测班级平均分失败",
    };
  }
}

// 获取班级异常表现
export async function getClassAnomalies(
  classId: number
): Promise<ApiResponse<any>> {
  try {
    const response = await fetch(
      `${PREDICTION_API}/analysis/class/${classId}/anomalies`
    );
    return await response.json();
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "获取班级异常表现失败",
    };
  }
}

// 获取班级预测洞察
export async function getClassInsights(
  classId: number
): Promise<ApiResponse<any>> {
  try {
    const response = await fetch(`${PREDICTION_API}/insights/class/${classId}`);
    return await response.json();
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "获取班级洞察失败",
    };
  }
}
