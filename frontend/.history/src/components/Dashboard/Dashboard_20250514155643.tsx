// src/components/Dashboard/Dashboard.tsx
import React, { useState, useEffect, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
} from "recharts";
import {
  Plus,
  User,
  Users,
  Award,
  AlertTriangle,
  Book,
  TrendingUp,
  Calendar,
  Settings,
} from "lucide-react";
import { Student, ExamScore, ClassAverage, Class, Grade } from "../../types";
import {
  getStudents,
  getStudentDetail,
  getStudentScores,
  getClassAverage,
  getGradeAverage,
  getClasses,
  getGrades,
  predictStudentScore,
  predictClassAverage,
  predictGradeAverage,
  searchStudentById,
  searchStudents,
  predictStudentScoresAdvanced,
  predictClassAverageAdvanced,
  getModelStatus,
  getClassAnomalies,
  getClassInsights,
} from "../../api";
import "./Dashboard.css";
import FileUpload from "../FileUpload/FileUpload";
import FilterPanel from "../FilterPanel/FilterPanel";
import SearchBar from "../SearchBar/SearchBar";
import ModelSettings, {
  ModelSettingsType,
} from "../ModelSettings/ModelSettings";

// 这个仪表盘组件用于展示学生、班级或年级的数据
const Dashboard: React.FC = () => {
  // 状态管理
  const [students, setStudents] = useState<Student[]>([]);
  const [selectedStudent, setSelectedStudent] = useState<string | null>(null);
  const [studentData, setStudentData] = useState<Student | null>(null);
  const [scoreData, setScoreData] = useState<ExamScore[]>([]);
  const [classAverageData, setClassAverageData] = useState<ClassAverage[]>([]);
  const [activeTab, setActiveTab] = useState("prediction");
  const [classes, setClasses] = useState<Class[]>([]);
  const [grades, setGrades] = useState<Grade[]>([]);
  const [searchResults, setSearchResults] = useState<Student[]>([]);
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [selectedSubject, setSelectedSubject] = useState<string>("all");
  const [availableSubjects, setAvailableSubjects] = useState<string[]>([]);

  // 在Dashboard.tsx的状态部分添加以下状态变量
  const [useAdvancedModel, setUseAdvancedModel] = useState(true); // 是否使用高级模型
  const [modelStatus, setModelStatus] = useState<any>(null); // 模型状态
  const [showModelSettings, setShowModelSettings] = useState(false); // 是否显示模型设置
  // 更新 Dashboard.tsx 中的 modelSettings 初始状态
  const [modelSettings, setModelSettings] = useState({
    timeSteps: 5,
    predictionSteps: 3,
    epochs: 100,
    batchSize: 32,
    confidenceInterval: 95,
    force_retrain: false,
  });
  // 添加需要的状态
  // 在 Dashboard.tsx 中添加状态变量
  const [showDropdown, setShowDropdown] = useState(false);
  const [uploadTab, setUploadTab] = useState<"scores" | "classes" | "students">(
    "scores"
  );
  const [selectedClass, setSelectedClass] = useState<number | null>(null);
  const [selectedGrade, setSelectedGrade] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<"student" | "class" | "grade">(
    "student"
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showUploadModal, setShowUploadModal] = useState(false);
  // 添加一个刷新数据的函数
  const refreshData = async () => {
    console.log("刷新数据...");
    setLoading(true);

    try {
      // 重新获取年级数据
      const gradesResponse = await getGrades();
      if (gradesResponse.success && gradesResponse.data) {
        setGrades(gradesResponse.data);

        // 如果有年级数据，默认选择第一个年级
        if (gradesResponse.data.length > 0 && !selectedGrade) {
          setSelectedGrade(gradesResponse.data[0].id);
        }
      }

      // 如果已选择了年级，重新获取班级数据
      if (selectedGrade) {
        const classesResponse = await getClasses(selectedGrade);
        if (classesResponse.success && classesResponse.data) {
          setClasses(classesResponse.data);

          // 如果有班级数据，默认选择第一个班级
          if (classesResponse.data.length > 0 && !selectedClass) {
            setSelectedClass(classesResponse.data[0].id);
          }
        }
      }

      // 如果已选择了班级，重新获取学生列表
      // 在refreshData函数中
      if (selectedClass) {
        // 类型转换，确保传递的是number | undefined而不是number | null
        const gradeId = selectedGrade || undefined;
        const classId = selectedClass || undefined;
        const studentsResponse = await getStudents(gradeId, classId);
        if (studentsResponse.success && studentsResponse.data) {
          setStudents(studentsResponse.data);

          // 如果有学生数据，默认选择第一个学生
          if (studentsResponse.data.length > 0 && !selectedStudent) {
            setSelectedStudent(studentsResponse.data[0].student_id);
          }
        }
      }
    } catch (err) {
      console.error("刷新数据失败:", err);
      setError("刷新数据失败");
    } finally {
      setLoading(false);
    }
  };

  // 初始数据加载
  useEffect(() => {
    const fetchInitialData = async () => {
      setLoading(true);
      setError("");

      try {
        console.log("加载初始数据...");

        // 获取年级列表
        const gradesResponse = await getGrades();
        console.log("年级响应:", gradesResponse);

        if (gradesResponse.success && gradesResponse.data) {
          setGrades(gradesResponse.data);

          // 如果有年级数据，默认选择第一个年级
          if (gradesResponse.data.length > 0) {
            const firstGradeId = gradesResponse.data[0].id;
            console.log("选择默认年级:", firstGradeId);
            setSelectedGrade(firstGradeId);

            // 立即加载该年级的班级
            const classesResponse = await getClasses(firstGradeId);
            console.log("班级响应:", classesResponse);

            if (classesResponse.success && classesResponse.data) {
              setClasses(classesResponse.data);

              // 如果有班级数据，默认选择第一个班级
              if (classesResponse.data.length > 0) {
                const firstClassId = classesResponse.data[0].id;
                console.log("选择默认班级:", firstClassId);
                setSelectedClass(firstClassId);

                // 立即加载该班级的学生
                const studentsResponse = await getStudents(
                  firstGradeId,
                  firstClassId
                );
                console.log("学生响应:", studentsResponse);

                if (studentsResponse.success && studentsResponse.data) {
                  setStudents(studentsResponse.data);

                  // 如果有学生数据，默认选择第一个学生
                  if (studentsResponse.data.length > 0) {
                    const firstStudentId = studentsResponse.data[0].student_id;
                    console.log("选择默认学生:", firstStudentId);
                    setSelectedStudent(firstStudentId);
                  }
                }
              }
            }
          }
        }
      } catch (err) {
        console.error("加载初始数据失败:", err);
        setError("加载初始数据失败");
      } finally {
        setLoading(false);
      }
    };

    fetchInitialData();
  }, []);

  // 检查数据是否正确加载
  useEffect(() => {
    console.log("viewMode:", viewMode);
    console.log("studentData:", studentData);
    console.log("selectedClass:", selectedClass);
    console.log("selectedGrade:", selectedGrade);
    console.log("classes:", classes);
    console.log("grades:", grades);
  }, [viewMode, studentData, selectedClass, selectedGrade, classes, grades]);

  // 当选择的年级变化时，加载对应的班级列表
  useEffect(() => {
    if (selectedGrade) {
      const fetchClasses = async () => {
        setLoading(true);
        try {
          const classesResponse = await getClasses(selectedGrade);
          if (classesResponse.success && classesResponse.data) {
            setClasses(classesResponse.data);

            // 如果有班级数据，默认选择第一个班级
            if (classesResponse.data.length > 0) {
              setSelectedClass(classesResponse.data[0].id);
            } else {
              setSelectedClass(null);
            }
          }
        } catch (err) {
          setError("加载班级列表失败");
          console.error(err);
        } finally {
          setLoading(false);
        }
      };

      fetchClasses();
    }
  }, [selectedGrade]);

  // 当选择的班级变化时，加载学生列表
  useEffect(() => {
    if (selectedClass) {
      const fetchStudents = async () => {
        setLoading(true);
        try {
          // 类型转换，确保传递的是number | undefined而不是number | null
          const gradeId = selectedGrade || undefined;
          const classId = selectedClass || undefined;

          const studentsResponse = await getStudents(gradeId, classId);
          if (studentsResponse.success && studentsResponse.data) {
            setStudents(studentsResponse.data);

            // 如果有学生数据，默认选择第一个学生
            if (studentsResponse.data.length > 0) {
              setSelectedStudent(studentsResponse.data[0].student_id);
            } else {
              setSelectedStudent(null);
              setStudentData(null);
            }
          }
        } catch (err) {
          setError("加载学生列表失败");
          console.error(err);
        } finally {
          setLoading(false);
        }
      };

      fetchStudents();
    }
  }, [selectedClass, selectedGrade]);

  // 当选择的学生变化时，加载学生详情和成绩数据
  useEffect(() => {
    if (selectedStudent && viewMode === "student") {
      const fetchStudentData = async () => {
        setLoading(true);
        try {
          // 获取学生详情
          const studentDetailResponse = await getStudentDetail(selectedStudent);
          if (studentDetailResponse.success && studentDetailResponse.data) {
            setStudentData(studentDetailResponse.data);
          }

          // 获取学生成绩历史和预测
          const scoreResponse = await getStudentScores(selectedStudent);
          if (scoreResponse.success && scoreResponse.data) {
            setScoreData(scoreResponse.data);
          }

          // 获取班级平均分，用于比较
          if (selectedClass) {
            const classAverageResponse = await getClassAverage(selectedClass);
            if (classAverageResponse.success && classAverageResponse.data) {
              setClassAverageData(classAverageResponse.data);
            }
          }
        } catch (err) {
          setError("加载学生数据失败");
          console.error(err);
        } finally {
          setLoading(false);
        }
      };

      fetchStudentData();
    } else if (selectedClass && viewMode === "class") {
      // 加载班级数据
      const fetchClassData = async () => {
        setLoading(true);
        try {
          const classAverageResponse = await getClassAverage(selectedClass);
          if (classAverageResponse.success && classAverageResponse.data) {
            setClassAverageData(classAverageResponse.data);
          }
        } catch (err) {
          setError("加载班级数据失败");
          console.error(err);
        } finally {
          setLoading(false);
        }
      };

      fetchClassData();
    } else if (selectedGrade && viewMode === "grade") {
      // 加载年级数据
      const fetchGradeData = async () => {
        setLoading(true);
        try {
          const gradeAverageResponse = await getGradeAverage(selectedGrade);
          if (gradeAverageResponse.success && gradeAverageResponse.data) {
            setClassAverageData(gradeAverageResponse.data);
          }
        } catch (err) {
          setError("加载年级数据失败");
          console.error(err);
        } finally {
          setLoading(false);
        }
      };

      fetchGradeData();
    }
  }, [selectedStudent, selectedClass, selectedGrade, viewMode]);

  useEffect(() => {
    // 当成绩数据加载后，提取所有可用科目
    if (scoreData.length > 0) {
      const subjects = new Set<string>();
      scoreData.forEach((score) => {
        if (score.subject) {
          subjects.add(score.subject);
        }
      });
      setAvailableSubjects(Array.from(subjects));
    }
  }, [scoreData]);
  // 添加调试输出，确认数据格式

  useEffect(() => {
    // 检查数据中是否包含预测值和置信区间
    if (scoreData && scoreData.length > 0) {
      console.log("检查预测数据:");
      console.log("有预测值的项目:", scoreData.filter(d => d.predicted !== null).length);
      console.log("有上界的项目:", scoreData.filter(d => d.upper !== null).length);
      console.log("有下界的项目:", scoreData.filter(d => d.lower !== null).length);
      console.log("示例数据项:", scoreData.find(d => d.predicted !== null));
    }
  }, [scoreData]);

  const filteredScoreData = useMemo(() => {
    let data = [];
    if (selectedSubject === "all") {
      data = [...scoreData];
    } else {
      data = scoreData.filter((score) => score.subject === selectedSubject);
    }

    // 按照日期排序（从早到晚）
    return data.sort((a, b) => {
      const dateA = new Date(a.date);
      const dateB = new Date(b.date);
      return dateA.getTime() - dateB.getTime();
    });
  }, [scoreData, selectedSubject]);

  const filteredClassAverageData = useMemo(() => {
    let data = [];
    if (selectedSubject === "all") {
      data = [...classAverageData];
    } else {
      data = classAverageData.filter(
        (score) => score.subject === selectedSubject
      );
    }

    // 按照日期排序（从早到晚）
    return data.sort((a, b) => {
      const dateA = new Date(a.date);
      const dateB = new Date(b.date);
      return dateA.getTime() - dateB.getTime();
    });
  }, [classAverageData, selectedSubject]);

  // 处理视图模式切换
  const handleViewModeChange = (mode: "student" | "class" | "grade") => {
    setViewMode(mode);
    setActiveTab("prediction");
  };

  // 修改 src/components/Dashboard/Dashboard.tsx
  const SubjectSelector = () => (
    <div className="subject-selector">
      <select
        value={selectedSubject}
        onChange={(e) => setSelectedSubject(e.target.value)}
        className="subject-select"
      >
        <option value="all">全部科目</option>
        {availableSubjects.map((subject) => (
          <option key={subject} value={subject}>
            {subject}
          </option>
        ))}
      </select>
    </div>
  );

  const handleSearch = async (studentId: string) => {
    setLoading(true);
    setError("");
    setShowSearchResults(true);

    try {
      console.log(`正在按学号搜索: ${studentId}`);
      const response = await searchStudentById(studentId);

      if (response.success && response.data) {
        setSearchResults(response.data);

        // 如果只有一个结果，直接选择该学生
        if (response.data.length === 1) {
          const student = response.data[0];
          setSelectedStudent(student.student_id);

          // 如果需要，更新班级和年级的选择
          // setSelectedClass(student.class_id);
          // setSelectedGrade(student.grade_id);

          // 切换到学生视图模式
          setViewMode("student");

          // 清除搜索结果显示
          setShowSearchResults(false);
        }
      } else {
        setSearchResults([]);
        setError("未找到匹配的学生");
      }
    } catch (err) {
      console.error("搜索失败:", err);
      setError(
        "搜索失败: " + (err instanceof Error ? err.message : String(err))
      );
    } finally {
      setLoading(false);
    }
  };

  // 选择搜索结果中的学生
  const handleSelectSearchResult = (student: Student) => {
    setSelectedStudent(student.student_id);
    // setSelectedClass(student.class_id);
    // setSelectedGrade(student.grade_id);
    setViewMode("student");
    setShowSearchResults(false);
  };

  // 处理预测请求
  // 处理预测请求
  // 修改 handlePredict 函数
  const handlePredict = async () => {
    console.log("预测按钮被点击，模式:", viewMode);
    console.log("选择的学生ID:", selectedStudent);
    console.log("选择的班级ID:", selectedClass);
    console.log("选择的年级ID:", selectedGrade);
    console.log("选择的科目:", selectedSubject);

    setLoading(true);
    setError("");

    try {
      if (viewMode === "student" && selectedStudent) {
        console.log(
          `开始为学生 ${selectedStudent} 的 ${selectedSubject} 科目请求预测，使用${
            useAdvancedModel ? "高级" : "标准"
          }模型`
        );

        let predictionResponse;

        if (useAdvancedModel) {
          // 使用高级预测API，传递科目参数
          predictionResponse = await predictStudentScoresAdvanced(
            selectedStudent.toString(),
            modelSettings.predictionSteps,
            {
              ...modelSettings,
              subject: selectedSubject !== "all" ? selectedSubject : "总分", // 如果选择"all"，则使用"总分"作为默认科目
            }
          );
        } else {
          // 使用原有的预测API
          // 如果需要支持原始API按科目预测，也需要修改此部分
          predictionResponse = await predictStudentScore(selectedStudent);
        }

        console.log("预测响应:", predictionResponse);

        if (predictionResponse.success && predictionResponse.data) {
          console.log("原始预测数据:", predictionResponse.data);
          // 格式化数值，保留两位小数
          const formattedData = predictionResponse.data.map((item) => ({
            ...item,
            predicted:
              item.predicted !== null
                ? Number(item.predicted.toFixed(2))
                : null,
            lower: item.lower !== null ? Number(item.lower.toFixed(2)) : null,
            upper: item.upper !== null ? Number(item.upper.toFixed(2)) : null,
          }));
          setScoreData(formattedData);
          // 如果使用高级模型，可以尝试获取模型状态
          if (useAdvancedModel) {
            try {
              // 获取模型状态时也传递科目信息
              const statusResponse = await getModelStatus(
                selectedStudent.toString(),
                selectedSubject !== "all" ? selectedSubject : "总分"
              );
              if (statusResponse.success && statusResponse.data) {
                setModelStatus(statusResponse.data);
              }
            } catch (err) {
              console.error("获取模型状态失败:", err);
            }
          }
          // 确保在预测成功后总是获取模型状态
          try {
            console.log(
              `获取模型状态: 学生ID ${selectedStudent}, 科目 ${selectedSubject}`
            );
            const statusResponse = await getModelStatus(
              selectedStudent.toString(),
              selectedSubject !== "all" ? selectedSubject : "总分"
            );

            console.log("模型状态响应:", statusResponse);

            if (statusResponse.success && statusResponse.data) {
              setModelStatus(statusResponse.data);
              console.log("模型状态已更新:", statusResponse.data);
            } else {
              console.warn("获取模型状态失败:", statusResponse.error);
            }
          } catch (statusErr) {
            console.error("获取模型状态时发生错误:", statusErr);
          }
          const hasPredictions = predictionResponse.data.some(
            (item) => item.predicted !== null && item.predicted !== undefined
          );
          console.log("数据包含预测结果:", hasPredictions);
          setScoreData(predictionResponse.data);
          // 检查数据更新后是否触发了useMemo
          console.log("更新后的filteredScoreData:", filteredScoreData);

          alert("预测已完成!");
        } else {
          throw new Error(predictionResponse.error || "预测失败");
        }
      } else if (viewMode === "class" && selectedClass) {
        console.log(
          `开始为班级 ${selectedClass} 请求预测，使用${
            useAdvancedModel ? "高级" : "标准"
          }模型`
        );

        let predictionResponse;

        if (useAdvancedModel) {
          // 使用高级预测API
          predictionResponse = await predictClassAverageAdvanced(selectedClass);

          // 尝试获取班级洞察
          try {
            const insightsResponse = await getClassInsights(selectedClass);
            if (insightsResponse.success && insightsResponse.data) {
              console.log("班级洞察:", insightsResponse.data);
              // 这里可以处理班级洞察数据，例如更新UI显示
            }
          } catch (err) {
            console.error("获取班级洞察失败:", err);
          }
        } else {
          // 使用原有的预测API
          predictionResponse = await predictClassAverage(selectedClass);
        }

        console.log("班级预测响应:", predictionResponse);

        if (predictionResponse.success && predictionResponse.data) {
          setClassAverageData(predictionResponse.data);
          alert("班级预测已完成!");
        } else {
          throw new Error(predictionResponse.error || "预测失败");
        }
      } else if (viewMode === "grade" && selectedGrade) {
        console.log(`开始为年级 ${selectedGrade} 请求预测`);
        const predictionResponse = await predictGradeAverage(selectedGrade);
        console.log("年级预测响应:", predictionResponse);

        if (predictionResponse.success && predictionResponse.data) {
          setClassAverageData(predictionResponse.data);
          alert("年级预测已完成!");
        } else {
          throw new Error(predictionResponse.error || "预测失败");
        }
      } else {
        console.error("无法预测: 未选择学生/班级/年级或视图模式不正确");
        setError("请先选择要预测的学生/班级/年级");
      }
    } catch (err) {
      console.error("预测过程中发生错误:", err);
      setError(
        "预测失败: " + (err instanceof Error ? err.message : String(err))
      );
      alert("预测失败: " + (err instanceof Error ? err.message : String(err)));
    } finally {
      setLoading(false);
    }
  };
  // 在Dashboard.tsx中，找到"统计卡片"部分，计算并动态更新"近期提升"和"预测准确度"的值
  // 可以在useEffect或useMemo中计算这些值

  // 添加以下useMemo来计算卡片数据
  const cardData = useMemo(() => {
    // 默认值
    const defaultData = {
      nextExamPrediction: { predicted: "N/A", lower: "N/A", upper: "N/A" },
      recentImprovement: { value: "+0.0", period: "最近3次考试" },
      predictAccuracy: { value: "0.0", period: "最近5次预测" },
      average: { value: "0.0", period: "全部考试" },
      highest: { value: "0.0", date: "" },
      lowest: { value: "0.0", date: "" },
      // 班级视图的默认值
      classAverage: { value: "0.0", period: "全部考试" },
      classHighest: { value: "0.0", date: "" },
      classLowest: { value: "0.0", date: "" },
      classTrend: { value: "稳定", change: "0.0" },
      passRate: { value: "0.0", threshold: "60分" },
      excellentRate: { value: "0.0", threshold: "85分" },
      classRank: { value: "N/A", total: "N/A", percentile: "N/A" }
    };

    // 如果没有数据，返回默认值
    if (!filteredScoreData || filteredScoreData.length === 0) {
      return defaultData;
    }

    try {
      // 1. 下次考试预测
      const predictionItem = filteredScoreData.find(
        (d) => d.predicted !== null
      );
      const nextExamPrediction = {
        predicted:
          predictionItem?.predicted !== null
            ? predictionItem?.predicted?.toFixed(1) ?? "N/A"
            : "N/A",
        lower:
          predictionItem?.lower !== null
            ? predictionItem?.lower?.toFixed(1) ?? "N/A"
            : "N/A",
        upper:
          predictionItem?.upper !== null
            ? predictionItem?.upper?.toFixed(1) ?? "N/A"
            : "N/A",
      };

      // 2. 仅考虑实际分数
      const actualScores = filteredScoreData
        .filter((d) => d.actual !== null)
        .sort(
          (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
        );

      // 3. 近期提升
      let recentImprovement = { value: "+0.0", period: "最近3次考试" };
      if (actualScores.length >= 3) {
        const recentScores = actualScores.slice(-3);
        const improvement =
          (recentScores[recentScores.length - 1].actual ?? 0) -
          (recentScores[0].actual ?? 0);
        recentImprovement = {
          value: (improvement >= 0 ? "+" : "") + improvement.toFixed(1),
          period: "最近3次考试",
        };
      }

      // 4. 预测准确度
      let predictAccuracy = { value: "0.0", period: "最近5次预测" };
      const scorePairs = filteredScoreData.filter(
        (d) => d.actual !== null && d.predicted !== null
      );

      if (scorePairs.length > 0) {
        // 计算平均绝对误差百分比
        const errors = scorePairs.map((d) =>
          d.actual !== null && d.predicted !== null
            ? Math.abs((d.actual - d.predicted) / d.actual) * 100
            : 0
        );
        const avgError =
          errors.reduce((sum, err) => sum + err, 0) / errors.length;
        const accuracy = Math.max(0, 100 - avgError);

        predictAccuracy = {
          value: accuracy.toFixed(1),
          period: `最近${scorePairs.length}次预测`,
        };
      }

      // 5. 平均分、最高分、最低分
      let average = { value: "0.0", period: "全部考试" };
      let highest = { value: "0.0", date: "" };
      let lowest = { value: "0.0", date: "" };

      if (actualScores.length > 0) {
        const sum = actualScores.reduce(
          (total, score) => total + (score.actual ?? 0),
          0
        );
        const avg = sum / actualScores.length;

        const highestScore = actualScores.reduce(
          (max, score) =>
            score.actual !== null && score.actual > max.score
              ? { score: score.actual, date: score.date }
              : max,
          { score: -Infinity, date: "" }
        );

        const lowestScore = actualScores.reduce(
          (min, score) =>
            score.actual !== null && score.actual < min.score
              ? { score: score.actual, date: score.date }
              : min,
          { score: Infinity, date: "" }
        );

        average = {
          value: avg.toFixed(1),
          period: "全部考试",
        };

        highest = {
          value: highestScore.score.toFixed(1),
          date: new Date(highestScore.date).toLocaleDateString(),
        };

        lowest = {
          value: lowestScore.score.toFixed(1),
          date: new Date(lowestScore.date).toLocaleDateString(),
        };
      }

      return {
        nextExamPrediction,
        recentImprovement,
        predictAccuracy,
        average,
        highest,
        lowest,
      };
    } catch (error) {
      console.error("计算卡片数据时出错:", error);
      return defaultData;
    }
  }, [filteredScoreData]);

  // 然后在卡片中使用这些计算值

  const ModelSwitch = () => (
    <div className="model-switch">
      <label className="switch">
        <input
          type="checkbox"
          checked={useAdvancedModel}
          onChange={() => setUseAdvancedModel(!useAdvancedModel)}
        />
        <span className="slider"></span>
      </label>
      <span className="switch-label">
        {useAdvancedModel ? "高级预测模型" : "标准预测模型"}
      </span>
      {useAdvancedModel && (
        <button
          className="btn-icon"
          onClick={() => setShowModelSettings(true)}
          title="模型设置"
        >
          <Settings className="icon" size={18} />
        </button>
      )}
    </div>
  );

  // 渲染页面
  return (
    <div className="dashboard">
      {/* 头部 */}
      <header className="dashboard-header">
        <div className="container">
          <h1>学生成绩预测系统</h1>
          <div className="header-actions">
            <div className="search-container">
              <SearchBar
                onSearch={handleSearch}
                placeholder="输入学号搜索..."
              />
            </div>
            <div className="dropdown">
              <button
                className="btn btn-secondary"
                onClick={() => setShowDropdown(!showDropdown)}
              >
                上传数据
                <span className="dropdown-icon">
                  {showDropdown ? "▲" : "▼"}
                </span>
              </button>
              {showDropdown && (
                <div className="dropdown-menu">
                  <button
                    className="dropdown-item"
                    onClick={() => {
                      setShowUploadModal(true);
                      setUploadTab("scores");
                      setShowDropdown(false);
                    }}
                  >
                    上传成绩表格
                  </button>
                  {/* <button
                    className="dropdown-item"
                    onClick={() => {
                      setShowUploadModal(true);
                      setUploadTab("classes");
                      setShowDropdown(false);
                    }}
                  >
                    上传班级数据
                  </button> */}
                  <button
                    className="dropdown-item"
                    onClick={() => {
                      setShowUploadModal(true);
                      setUploadTab("students");
                      setShowDropdown(false);
                    }}
                  >
                    上传学生数据
                  </button>
                </div>
              )}
            </div>
            <button className="btn btn-primary">
              <Calendar className="icon" />
              2024-2025学年
            </button>
            <button className="btn btn-icon">
              <User className="icon" />
            </button>
          </div>
        </div>
      </header>

      {showSearchResults && searchResults.length > 0 && (
        <div className="search-results">
          <div className="search-results-header">
            <h3>搜索结果</h3>
            <button
              className="btn-close"
              onClick={() => setShowSearchResults(false)}
            >
              &times;
            </button>
          </div>
          <div className="search-results-list">
            {searchResults.map((student) => (
              <div
                key={student.id}
                className="search-result-item"
                onClick={() => handleSelectSearchResult(student)}
              >
                <div className="student-info">
                  <User className="icon" />
                  <div className="student-details">
                    <div className="student-name">{student.name}</div>
                    <div className="student-class">学号: {student.id}</div>
                  </div>
                </div>
                <div className={`trend trend-${student.trend}`}>
                  {student.trend === "up"
                    ? "↑"
                    : student.trend === "down"
                    ? "↓"
                    : "→"}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 主内容区域 */}
      <div className="dashboard-content">
        {/* 侧边栏 */}
        <aside className="sidebar">
          {/* 筛选面板 */}
          {/* // 在Dashboard.tsx中修改这部分代码 */}
          <FilterPanel
            grades={grades}
            classes={classes}
            selectedGrade={selectedGrade}
            selectedClass={selectedClass}
            viewMode={viewMode}
            onGradeChange={(id) => {
              console.log("Dashboard收到年级变更:", id);
              setSelectedGrade(id);
              if (id === null) {
                setSelectedClass(null);
                setSelectedStudent(null);
              }
            }}
            onClassChange={(id) => {
              console.log("Dashboard收到班级变更:", id);
              setSelectedClass(id);
              if (id === null) {
                setSelectedStudent(null);
              }
            }}
            onViewModeChange={handleViewModeChange}
          />

          {/* 学生列表 - 仅在学生视图模式下显示 */}
          {viewMode === "student" && (
            <div className="student-list-container">
              <div className="section-header">
                <h2>学生列表</h2>
                {/* <button className="btn-icon">
                  <Plus className="icon" />
                </button> */}
              </div>

              <div className="student-list">
                {students.map((student) => (
                  <button
                    key={student.id}
                    className={`student-item ${
                      selectedStudent === student.student_id ? "active" : ""
                    }`}
                    onClick={() => setSelectedStudent(student.student_id)}
                  >
                    <div className="student-info">
                      <User className="icon" />
                      <span>{student.name}</span>
                    </div>
                    {student.alerts > 0 && (
                      <span className="alert-badge">{student.alerts}</span>
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* 导航菜单 */}
          <nav className="navigation">
            <h3>仪表盘</h3>
            <div className="nav-items">
              <button className="nav-item">
                <Users className="icon" />
                <span>班级总览</span>
              </button>
              <button className="nav-item">
                <AlertTriangle className="icon" />
                <span>风险学生</span>
              </button>
              <button className="nav-item">
                <Award className="icon" />
                <span>绩效报告</span>
              </button>
              <button className="nav-item">
                <Book className="icon" />
                <span>课程分析</span>
              </button>
            </div>
          </nav>
        </aside>

        {/* 主内容区域 */}
        <main className="main-content">
          {loading ? (
            <div className="loading">加载中...</div>
          ) : error ? (
            <div className="error">{error}</div>
          ) : (
            <>
              {/* 标题区域 */}
              <div className="content-header">
                <div>
                  <h2>
                    {viewMode === "student"
                      ? studentData?.name
                      : viewMode === "class"
                      ? classes.find((c) => c.id === selectedClass)?.name
                      : grades.find((g) => g.id === selectedGrade)?.name}
                  </h2>
                  <p>
                    {viewMode === "student"
                      ? `学生ID: ${studentData?.student_id}`
                      : viewMode === "class"
                      ? `班级ID: ${selectedClass}`
                      : `年级ID: ${selectedGrade}`}
                  </p>
                </div>
                {viewMode === "student" && studentData && (
                  <div className="stats">
                    <div className="stat-item">
                      <p>平均分</p>
                      <p className="stat-value">{studentData.average}</p>
                    </div>
                    <div className="stat-item">
                      <p>趋势</p>
                      <p className={`stat-value trend-${studentData.trend}`}>
                        {studentData.trend === "up"
                          ? "↑"
                          : studentData.trend === "down"
                          ? "↓"
                          : "→"}
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* 标签页 */}
              <div className="tabs-container">
                <div className="tabs">
                  <button
                    className={`tab ${
                      activeTab === "prediction" ? "active" : ""
                    }`}
                    onClick={() => setActiveTab("prediction")}
                  >
                    成绩预测
                  </button>
                  <button
                    className={`tab ${activeTab === "history" ? "active" : ""}`}
                    onClick={() => setActiveTab("history")}
                  >
                    成绩历史
                  </button>
                  {viewMode === "student" && (
                    <button
                      className={`tab ${
                        activeTab === "comparison" ? "active" : ""
                      }`}
                      onClick={() => setActiveTab("comparison")}
                    >
                      班级对比
                    </button>
                  )}
                </div>

                {/* 标签页内容 */}
                <div className="tab-content">
                  {activeTab === "prediction" && (
                    <div className="prediction-tab">
                      <div className="tab-header">
                        <h3>成绩预测与波动范围</h3>
                        <div className="tab-actions">
                          <SubjectSelector />
                          <ModelSwitch />
                          <button
                            className="btn btn-primary"
                            onClick={handlePredict}
                            disabled={loading}
                          >
                            {loading ? "预测中..." : "更新预测"}
                          </button>
                        </div>
                      </div>

                      {/* 预测图表 */}
                      <div className="chart-container">
                        <ResponsiveContainer width="100%" height={300}>
                          <LineChart
                            data={
                              viewMode === "student"
                                ? filteredScoreData
                                : filteredClassAverageData
                            }
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="date" />
                            <YAxis domain={[60, 100]} />
                            <Tooltip />
                            <Legend />
                            <Line
                              type="monotone"
                              dataKey={
                                viewMode === "student" ? "actual" : "average"
                              }
                              stroke="#4f46e5"
                              name={
                                viewMode === "student"
                                  ? "实际分数"
                                  : "实际平均分"
                              }
                              strokeWidth={2}
                              dot={{ r: 4 }}
                              isAnimationActive={false}
                            />

                            {/* 确保预测线条始终可见 */}
                            <Line
                              type="monotone"
                              dataKey="predicted"
                              stroke="#22c55e"
                              name="预测分数"
                              strokeWidth={2}
                              strokeDasharray="5 5"
                              dot={{ r: 4 }}
                              connectNulls={true}
                              isAnimationActive={false}
                            />

                            {/* 确保置信区间显示 */}
                            <Area
                              type="monotone"
                              dataKey="lower"
                              stroke="none"
                              fill="#22c55e"
                              fillOpacity={0.2}
                              name="预测下限"
                              isAnimationActive={false}
                            />
                            <Area
                              type="monotone"
                              dataKey="upper"
                              stroke="none"
                              fill="#22c55e"
                              fillOpacity={0.2}
                              name="预测上限"
                              isAnimationActive={false}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>

                      {/* 模型状态信息 - 新增 */}
                      {useAdvancedModel && modelStatus && (
                        <div className="model-status-info">
                          <div className="status-badge">
                            <span className="status-icon">
                              {modelStatus.trained ? "✓" : "!"}
                            </span>
                            <span className="status-text">
                              {modelStatus.trained
                                ? `模型已训练 (${new Date(
                                    modelStatus.last_trained
                                  ).toLocaleDateString()})`
                                : "模型未训练"}
                            </span>
                          </div>
                          {modelStatus.trained && modelStatus.metrics && (
                            <div className="model-metrics">
                              <span className="metric">
                                准确度:{" "}
                                {(modelStatus.metrics.R2 * 100).toFixed(1)}%
                              </span>
                              <span className="metric">
                                预测误差: ±{modelStatus.metrics.MAE.toFixed(1)}
                                分
                              </span>
                            </div>
                          )}
                        </div>
                      )}

                      {/* 统计卡片 */}
                      <div className="stats-cards">
                        {/* // 然后在卡片中使用这些数据 // 在"下次考试预测"卡片中: */}
                        <div className="stat-card primary">
                          <p className="card-label">下次考试预测</p>
                          <p className="card-value">
                            {cardData.nextExamPrediction.predicted}
                          </p>
                          <p className="card-subtitle">
                            范围: {cardData.nextExamPrediction.lower}-
                            {cardData.nextExamPrediction.upper}
                          </p>
                        </div>
                        {/* // 在"近期提升"卡片中: */}
                        <div className="stat-card success">
                          <p className="card-label">近期提升</p>
                          <p className="card-value">
                            {cardData.recentImprovement.value} 分
                          </p>
                          <p className="card-subtitle">
                            {cardData.recentImprovement.period}
                          </p>
                        </div>
                        {/* // 在"预测准确度"卡片中: */}
                        <div className="stat-card info">
                          <p className="card-label">预测准确度</p>
                          <p className="card-value">
                            {cardData.predictAccuracy.value}%
                          </p>
                          <p className="card-subtitle">
                            {cardData.predictAccuracy.period}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === "history" && (
                    <div className="history-tab">
                      <h3>成绩历史与分析</h3>

                      {/* 历史图表 */}
                      <div className="chart-container">
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart
                            data={
                              viewMode === "student"
                                ? scoreData.filter((d) => d.actual !== null)
                                : classAverageData
                            }
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="date" />
                            <YAxis domain={[60, 100]} />
                            <Tooltip />
                            <Legend />
                            <Bar
                              dataKey={
                                viewMode === "student" ? "actual" : "average"
                              }
                              name={viewMode === "student" ? "分数" : "平均分"}
                              fill="#4f46e5"
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      {/* 统计卡片 */}
                      <div className="stat-card">
                        <p className="card-label">平均分</p>
                        <p className="card-value">{cardData.average.value}</p>
                        <p className="card-subtitle">
                          {cardData.average.period}
                        </p>
                      </div>
                      <div className="stat-card">
                        <p className="card-label">最高分</p>
                        <p className="card-value">{cardData.highest.value}</p>
                        <p className="card-subtitle">{cardData.highest.date}</p>
                      </div>
                      <div className="stat-card">
                        <p className="card-label">最低分</p>
                        <p className="card-value">{cardData.lowest.value}</p>
                        <p className="card-subtitle">{cardData.lowest.date}</p>
                      </div>
                    </div>
                  )}

                  {activeTab === "comparison" && viewMode === "student" && (
                    <div className="comparison-tab">
                      <h3>与班级平均分对比</h3>

                      {/* 对比图表 */}
                      <div className="chart-container">
                        <ResponsiveContainer width="100%" height={300}>
                          <LineChart
                            data={scoreData
                              .filter((d) => d.actual !== null)
                              .map((item, index) => ({
                                ...item,
                                classAverage:
                                  classAverageData[index]?.average || null,
                              }))}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="date" />
                            <YAxis domain={[60, 100]} />
                            <Tooltip />
                            <Legend />
                            <Line
                              type="monotone"
                              dataKey="actual"
                              stroke="#4f46e5"
                              name="学生分数"
                              strokeWidth={2}
                            />
                            <Line
                              type="monotone"
                              dataKey="classAverage"
                              stroke="#f97316"
                              name="班级平均分"
                              strokeWidth={2}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>

                      {/* 统计卡片 */}
                      <div className="stats-cards">
                        <div className="stat-card secondary">
                          <p className="card-label">高于平均分</p>
                          <p className="card-value">+5.6 分</p>
                          <p className="card-subtitle">平均差异</p>
                        </div>
                        <div className="stat-card warning">
                          <p className="card-label">班级排名</p>
                          <p className="card-value">3/28</p>
                          <p className="card-subtitle">前10%</p>
                        </div>
                        <div className="stat-card info">
                          <p className="card-label">相对成长速度</p>
                          <p className="card-value">2倍</p>
                          <p className="card-subtitle">最近5次考试</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* 额外分析区域 */}
              <div className="insights-container">
                <div className="insight-card">
                  <h3>表现分析</h3>
                  <ul className="insight-list">
                    <li className="insight-item">
                      <TrendingUp className="icon success" />
                      <p>
                        <span>强劲的提升趋势</span>
                        在最近5次考试中，表明学习习惯有效。
                      </p>
                    </li>
                    <li className="insight-item">
                      <AlertTriangle className="icon warning" />
                      <p>
                        <span>轻微的不一致性</span>
                        在二月份的预测与实际分数之间，可能是由于外部因素导致。
                      </p>
                    </li>
                    <li className="insight-item">
                      <Award className="icon primary" />
                      <p>
                        <span>持续高于班级平均</span>
                        且在最近的考试中差距逐渐扩大。
                      </p>
                    </li>
                  </ul>
                </div>

                <div className="insight-card">
                  <h3>建议</h3>
                  <ul className="insight-list">
                    <li className="insight-item">
                      <div className="numbered-icon">1</div>
                      <p>
                        <span>考虑增加额外挑战</span>以保持学习积极性和动力。
                      </p>
                    </li>
                    <li className="insight-item">
                      <div className="numbered-icon">2</div>
                      <p>
                        <span>分析二月份的表现因素</span>
                        以了解临时性下降的原因。
                      </p>
                    </li>
                    <li className="insight-item">
                      <div className="numbered-icon">3</div>
                      <p>
                        <span>考虑同伴辅导机会</span>帮助提高班级整体表现。
                      </p>
                    </li>
                  </ul>
                </div>
              </div>
            </>
          )}
        </main>
      </div>

      {/* 文件上传模态窗口 */}
      {/* // 修改文件上传模态窗口部分 */}
      {/* 文件上传模态窗口 */}
      {showUploadModal && (
        <FileUpload
          onClose={() => setShowUploadModal(false)}
          onUploadSuccess={refreshData}
          initialTab={uploadTab}
        />
      )}
      {showModelSettings && (
        <ModelSettings
          onClose={() => setShowModelSettings(false)}
          onSave={(newSettings) => {
            setModelSettings(newSettings);
            // 可以选择立即应用新设置
            handlePredict();
          }}
          currentSettings={modelSettings}
          selectedSubject={selectedSubject}
          availableSubjects={availableSubjects}
          onSubjectChange={setSelectedSubject}
        />
      )}
    </div>
  );
};

export default Dashboard;
