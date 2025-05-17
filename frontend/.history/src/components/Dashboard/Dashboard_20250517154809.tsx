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
  TrendingDown,
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
  const [modelSettings, setModelSettings] = useState({
    timeSteps: 5,
    predictionSteps: 3,
    epochs: 50, // 减少 epochs，线性回归不需要那么多
    batchSize: 32,
    confidenceInterval: 95,
    force_retrain: true,
    modelType: "linear_regression", // 添加默认模型类型
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
      loadStudentData(selectedStudent).catch((err) => {
        setError(
          "加载学生数据失败: " +
            (err instanceof Error ? err.message : String(err))
        );
      });
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
  // 当成绩数据首次加载后，提取所有可用科目，但不要在每次预测后都更新
  if (scoreData.length > 0 && availableSubjects.length === 0) {
    const subjects = new Set<string>();
    scoreData.forEach((score) => {
      if (score.subject) {
        subjects.add(score.subject);
      }
    });
    setAvailableSubjects(Array.from(subjects));
  }
}, [scoreData, availableSubjects.length]); // 添加 availableSubjects.length 作为依赖项
  // 添加调试输出，确认数据格式

  useEffect(() => {
    // 检查数据中是否包含预测值和置信区间
    if (scoreData && scoreData.length > 0) {
      console.log("检查预测数据:");
      console.log(
        "有预测值的项目:",
        scoreData.filter((d) => d.predicted !== null).length
      );
      console.log(
        "有上界的项目:",
        scoreData.filter((d) => d.upper !== null).length
      );
      console.log(
        "有下界的项目:",
        scoreData.filter((d) => d.lower !== null).length
      );
      console.log(
        "示例数据项:",
        scoreData.find((d) => d.predicted !== null)
      );
    }
  }, [scoreData]);

  const filteredScoreData = useMemo(() => {
    console.log("过滤数据，当前选择科目:", selectedSubject);
    console.log("可用科目:", availableSubjects);
    console.log("原始数据长度:", scoreData.length);

    let data = [];
    if (selectedSubject === "all") {
      data = [...scoreData];
    } else {
      data = scoreData.filter((score) => score.subject === selectedSubject);
    }

    console.log("过滤后数据长度:", data.length);

    // 按照日期排序（从早到晚）
    return data.sort((a, b) => {
      const dateA = new Date(a.date);
      const dateB = new Date(b.date);
      return dateA.getTime() - dateB.getTime();
    });
  }, [scoreData, selectedSubject, availableSubjects]); // 确保所有依赖都包含

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

  // 添加特定的事件处理逻辑，确保科目选择器可用
  // 添加特定的事件处理逻辑，确保科目选择器可用
  useEffect(() => {
    // 为科目选择器添加特殊的点击处理
    const handleSubjectSelectorClick = (e: Event) => {
      // 阻止其他事件处理程序干扰
      e.stopPropagation();
    };

    // 获取科目选择器元素
    const selector = document.querySelector(".subject-select");
    if (selector) {
      selector.addEventListener("click", handleSubjectSelectorClick);
      selector.addEventListener("change", handleSubjectSelectorClick);
    }

    // 清理函数
    return () => {
      if (selector) {
        selector.removeEventListener("click", handleSubjectSelectorClick);
        selector.removeEventListener("change", handleSubjectSelectorClick);
      }
    };
  }, [scoreData, filteredScoreData]); // 在数据更新后重新绑定事件

  // 修改 filteredScoreData 的 useMemo 依赖项

  // 处理视图模式切换
  const handleViewModeChange = (mode: "student" | "class" | "grade") => {
    setViewMode(mode);
    setActiveTab("prediction");
  };

 const SubjectSelector = () => (
  <div className="subject-selector">
    <select
      key="subject-selector" // 添加固定的键，防止重渲染导致状态丢失
      value={selectedSubject}
      onChange={(e) => {
        e.stopPropagation();
        setSelectedSubject(e.target.value);
        console.log("科目改变为:", e.target.value);
      }}
      className="subject-select"
      disabled={loading}
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

  // 修改 Dashboard.tsx 中的相关函数

  // 1. 修改 handleSearch 函数，确保在选择学生后加载学生详情
  const handleSearch = async (studentId: string) => {
    setLoading(true);
    setError("");
    setShowSearchResults(true);

    try {
      console.log(`正在按学号搜索: ${studentId}`);
      const response = await searchStudentById(studentId);

      if (response.success && response.data && response.data.length > 0) {
        setSearchResults(response.data);

        // 如果只有一个结果，直接选择该学生
        if (response.data.length === 1) {
          const student = response.data[0];
          console.log("自动选择搜索结果:", student);

          // 设置学生ID
          setSelectedStudent(student.student_id);

          // 立即加载该学生的详细数据
          await loadStudentData(student.id);

          // 切换到学生视图模式
          setViewMode("student");

          // 清除搜索结果显示
          setShowSearchResults(false);
        }
      } else {
        setSearchResults([]);
        // setError("未找到匹配的学生");
      }
    } catch (err) {
      console.error("搜索失败:", err);
      // setError(
      //   "搜索失败: " + (err instanceof Error ? err.message : String(err))
      // );
    } finally {
      setLoading(false);
    }
  };

  // 2. 修改 handleSelectSearchResult 函数，确保在选择学生后加载学生详情
  const handleSelectSearchResult = async (student: Student) => {
    console.log("手动选择搜索结果:", student);

    // 设置loading状态
    setLoading(true);

    try {
      // 设置学生ID
      console.log("选择的学生对象:", student);
      setSelectedStudent(student.student_id);

      // 立即加载该学生的详细数据
      await loadStudentData(student.id);

      // 切换到学生视图模式
      setViewMode("student");
      setShowSearchResults(false);
      setError("");
    } catch (err) {
      console.error("加载学生详情失败:", err);
      setError(
        "加载学生详情失败: " +
          (err instanceof Error ? err.message : String(err))
      );
    } finally {
      setLoading(false);
    }
  };

  // 3. 添加一个专门用于加载学生详情数据的函数
  const loadStudentData = async (studentId: string) => {
    try {
      // 获取学生详情
      const studentDetailResponse = await getStudentDetail(studentId);
      if (studentDetailResponse.success && studentDetailResponse.data) {
        setStudentData(studentDetailResponse.data);
      } else {
        throw new Error("获取学生详情失败");
      }

      // 获取学生成绩历史
      const scoreResponse = await getStudentScores(studentId);
      if (scoreResponse.success && scoreResponse.data) {
        setScoreData(scoreResponse.data);

        // 提取可用科目
        const subjects = new Set<string>();
        scoreResponse.data.forEach((score) => {
          if (score.subject) {
            subjects.add(score.subject);
          }
        });
        setAvailableSubjects(Array.from(subjects));
      }

      // 获取班级平均分（如果有选择班级）
      if (selectedClass) {
        const classAverageResponse = await getClassAverage(selectedClass);
        if (classAverageResponse.success && classAverageResponse.data) {
          setClassAverageData(classAverageResponse.data);
        }
      }

      return true;
    } catch (err) {
      console.error("加载学生数据失败:", err);
      throw err;
    }
  };
  // 处理预测请求
  // 修改 handlePredict 函数
  const handlePredict = async () => {
    console.log("预测按钮被点击，模式:", viewMode);
    console.log("选择的学生ID:", selectedStudent);
    console.log("选择的班级ID:", selectedClass);
    console.log("选择的年级ID:", selectedGrade);
    console.log("选择的科目:", selectedSubject);
    console.log("是否重新训练", modelSettings.force_retrain);
    const currentSubject = selectedSubject;
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
          // // 确保在获取新的可用科目
          // if (formattedData.length > 0) {
          //   const newSubjects = new Set<string>();
          //   formattedData.forEach((score) => {
          //     if (score.subject) {
          //       newSubjects.add(score.subject);
          //     }
          //   });
          //   // 更新可用科目列表
          //   setAvailableSubjects(Array.from(newSubjects));
          // }

          // 如果使用高级模型，可以尝试获取模型状态
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
  predictionResponse = await predictStudentScore(selectedStudent);
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
          // 在 src/components/Dashboard/Dashboard.tsx 中
          // 修改 handlePredict 函数结尾部分

          // 修改 handlePredict 函数中预测成功后的部分
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

            // 确保在获取新的可用科目
            // if (formattedData.length > 0) {
            //   const newSubjects = new Set<string>();
            //   formattedData.forEach((score) => {
            //     if (score.subject) {
            //       newSubjects.add(score.subject);
            //     }
            //   });
            //   // 更新可用科目列表
            //   setAvailableSubjects(Array.from(newSubjects));
            // }

            // 其他代码...

            alert("预测已完成!");
            // 在预测完成后，确保恢复原来选择的科目
            setSelectedSubject(currentSubject);

            // 添加一个小延迟，确保UI重新渲染
            // setTimeout(() => {
            //   // 强制刷新科目选择器状态
            //   setSelectedSubject(selectedSubject);
            // }, 100);
          }
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
      classRank: { value: "N/A", total: "N/A", percentile: "N/A" },
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

  // 添加在其他 useMemo 后面，cardData 之后
  const analysisData = useMemo(() => {
    // 默认值
    const defaultAnalysis = {
      performances: [
        {
          type: "success",
          title: "强劲的提升趋势",
          description: "在最近5次考试中，表明学习习惯有效。",
        },
        {
          type: "warning",
          title: "轻微的不一致性",
          description: "在二月份的预测与实际分数之间，可能是由于外部因素导致。",
        },
        {
          type: "primary",
          title: "持续高于班级平均",
          description: "且在最近的考试中差距逐渐扩大。",
        },
      ],
      suggestions: [
        {
          title: "考虑增加额外挑战",
          description: "以保持学习积极性和动力。",
        },
        {
          title: "分析二月份的表现因素",
          description: "以了解临时性下降的原因。",
        },
        {
          title: "考虑同伴辅导机会",
          description: "帮助提高班级整体表现。",
        },
      ],
    };

    // 如果没有数据，返回默认值
    if (!filteredScoreData || filteredScoreData.length === 0) {
      return defaultAnalysis;
    }

    try {
      // 获取实际分数数据
      const actualScores = filteredScoreData
        .filter((d) => d.actual !== null)
        .sort(
          (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
        );

      // 分析结果和建议
      const performances = [];
      const suggestions = [];

      if (actualScores.length >= 3) {
        const recentScores = actualScores.slice(-5); // 考虑最近5次考试，提供更多数据点

        if (recentScores.length >= 3) {
          // 计算所有相邻考试的分数差
          const scoreDiffs = [];
          for (let i = 1; i < recentScores.length; i++) {
            scoreDiffs.push(
              (recentScores[i].actual || 0) - (recentScores[i - 1].actual || 0)
            );
          }

          // 计算平均变化和标准差来评估趋势稳定性
          const avgDiff =
            scoreDiffs.reduce((sum, diff) => sum + diff, 0) / scoreDiffs.length;
          const variance =
            scoreDiffs.reduce(
              (sum, diff) => sum + Math.pow(diff - avgDiff, 2),
              0
            ) / scoreDiffs.length;
          const stdDev = Math.sqrt(variance);

          // 计算首尾差值
          const firstScore = recentScores[0].actual || 0;
          const lastScore = recentScores[recentScores.length - 1].actual || 0;
          const totalImprovement = lastScore - firstScore;

          // 计算线性趋势强度: R² 值 (线性拟合度)
          let rSquared = 0;
          if (recentScores.length >= 3) {
            // 计算线性回归
            let sumX = 0,
              sumY = 0,
              sumXY = 0,
              sumX2 = 0,
              sumY2 = 0;
            for (let i = 0; i < recentScores.length; i++) {
              const x = i;
              const y = recentScores[i].actual || 0;
              sumX += x;
              sumY += y;
              sumXY += x * y;
              sumX2 += x * x;
              sumY2 += y * y;
            }
            const n = recentScores.length;
            const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            const intercept = (sumY - slope * sumX) / n;

            // 计算拟合优度 R²
            let yMean = sumY / n;
            let totalSS = 0,
              residualSS = 0;
            for (let i = 0; i < recentScores.length; i++) {
              const y = recentScores[i].actual || 0;
              const predicted = slope * i + intercept;
              totalSS += Math.pow(y - yMean, 2);
              residualSS += Math.pow(y - predicted, 2);
            }

            rSquared = 1 - residualSS / totalSS;
          }

          // 标准差与平均分的比值（变异系数）评估波动性
          const variationCoef =
            stdDev /
            (recentScores.reduce((sum, score) => sum + (score.actual || 0), 0) /
              recentScores.length);

          // 根据综合指标判断趋势
          const trendStrength = Math.abs(avgDiff) * Math.sqrt(rSquared); // 趋势强度
          const consistencyThreshold = 0.15; // 波动一致性阈值

          // 判断趋势类型
          if (
            Math.abs(totalImprovement) >= 5 &&
            rSquared >= 0.6 &&
            variationCoef < consistencyThreshold
          ) {
            // 强趋势 - 总体变化大，相关性高，波动小
            if (avgDiff > 0) {
              performances.push({
                type: "success",
                title: "明确的上升趋势",
                description: `在最近${
                  recentScores.length
                }次考试中呈现持续上升趋势，总提升${Math.abs(
                  totalImprovement
                ).toFixed(1)}分，趋势一致性强。`,
              });
              suggestions.push({
                title: "继续保持当前学习方法",
                description: "已建立了稳定有效的学习模式。",
              });
            } else {
              performances.push({
                type: "warning",
                title: "明确的下降趋势",
                description: `在最近${
                  recentScores.length
                }次考试中呈现持续下降趋势，总下降${Math.abs(
                  totalImprovement
                ).toFixed(1)}分，需要重点关注。`,
              });
              suggestions.push({
                title: "全面评估学习状况",
                description: "找出成绩持续下滑的根本原因。",
              });
            }
          } else if (Math.abs(totalImprovement) >= 3 && rSquared >= 0.4) {
            // 中等趋势 - 有一定变化，相关性中等
            if (avgDiff > 0) {
              performances.push({
                type: "success",
                title: "逐步提升趋势",
                description: `在最近${recentScores.length}次考试中有${Math.abs(
                  totalImprovement
                ).toFixed(1)}分的整体提升，进步稳健。`,
              });
              suggestions.push({
                title: "适当增加学习挑战",
                description: "巩固已有进步并提高学习效率。",
              });
            } else {
              performances.push({
                type: "warning",
                title: "轻微下降趋势",
                description: `在最近${recentScores.length}次考试中有${Math.abs(
                  totalImprovement
                ).toFixed(1)}分的整体下滑，需要关注。`,
              });
              suggestions.push({
                title: "针对性调整学习方法",
                description: "找出可能的知识盲点和学习障碍。",
              });
            }
          } else if (variationCoef >= 0.1) {
            // 高波动性 - 无明显趋势但波动大
            performances.push({
              type: "warning",
              title: "成绩波动明显",
              description: `在最近${
                recentScores.length
              }次考试中成绩起伏较大，标准差达${stdDev.toFixed(
                1
              )}分，缺乏稳定性。`,
            });
            suggestions.push({
              title: "注重学习的稳定性和连贯性",
              description: "减少考试发挥不稳定的因素，建立系统学习习惯。",
            });
          } else {
            // 稳定表现 - 波动小且无明显趋势
            performances.push({
              type: "primary",
              title: "稳定的表现",
              description: `在最近${recentScores.length}次考试中表现保持稳定，波动幅度小。`,
            });
            suggestions.push({
              title: "寻找突破瓶颈的方法",
              description: "在保持稳定的基础上，尝试新的学习策略提升成绩。",
            });
          }

          // 添加关于最近一次考试的评价
          if (recentScores.length >= 2) {
            const latestDiff =
              (recentScores[recentScores.length - 1].actual || 0) -
              (recentScores[recentScores.length - 2].actual || 0);

            if (Math.abs(latestDiff) >= 8) {
              if (latestDiff > 0) {
                performances.push({
                  type: "success",
                  title: "最近考试大幅进步",
                  description: `最近一次考试比前一次提高了${latestDiff.toFixed(
                    1
                  )}分，表现优秀。`,
                });
              } else {
                performances.push({
                  type: "warning",
                  title: "最近考试明显退步",
                  description: `最近一次考试比前一次下降了${Math.abs(
                    latestDiff
                  ).toFixed(1)}分，需要分析原因。`,
                });
                suggestions.push({
                  title: "分析最近考试退步原因",
                  description: "检查是否是特定知识点问题或考试状态影响。",
                });
              }
            }
          }
        }
      }

      // 分析预测准确度
      const scorePairs = filteredScoreData.filter(
        (d) => d.actual !== null && d.predicted !== null
      );
      if (scorePairs.length > 0) {
        // 计算最大误差
        const maxError = Math.max(
          ...scorePairs.map((d) =>
            d.actual !== null && d.predicted !== null
              ? Math.abs(d.actual - d.predicted)
              : 0
          )
        );

        if (maxError > 10) {
          performances.push({
            type: "warning",
            title: "预测与实际分数存在较大差异",
            description: `最大误差达${maxError.toFixed(
              1
            )}分，可能受外部因素影响。`,
          });
          suggestions.push({
            title: "关注影响成绩波动的因素",
            description: "以提高未来预测的准确性。",
          });
        } else if (maxError < 5) {
          performances.push({
            type: "success",
            title: "预测模型表现优秀",
            description: `预测与实际分数接近，最大误差仅${maxError.toFixed(
              1
            )}分。`,
          });
        }
      }

      // 与班级平均的对比
      if (
        classAverageData &&
        classAverageData.length > 0 &&
        actualScores.length > 0
      ) {
        // 找到可比较的数据点
        const comparablePoints = actualScores.filter((score) =>
          classAverageData.some(
            (classScore) =>
              classScore.date === score.date && classScore.average !== null
          )
        );

        if (comparablePoints.length > 0) {
          const differences = comparablePoints.map((score) => {
            const classScore = classAverageData.find(
              (c) => c.date === score.date
            );
            return classScore
              ? (score.actual || 0) - (classScore.average || 0)
              : 0;
          });

          const avgDifference =
            differences.reduce((sum, diff) => sum + diff, 0) /
            differences.length;

          if (avgDifference > 5) {
            performances.push({
              type: "primary",
              title: "持续高于班级平均",
              description: `平均超出${avgDifference.toFixed(1)}分，表现突出。`,
            });
            suggestions.push({
              title: "考虑参与更高水平的竞赛",
              description: "进一步挑战自我。",
            });
          } else if (avgDifference > 0) {
            performances.push({
              type: "primary",
              title: "略高于班级平均",
              description: `平均高出${avgDifference.toFixed(1)}分。`,
            });
          } else if (avgDifference < -5) {
            performances.push({
              type: "warning",
              title: "低于班级平均",
              description: `平均落后${Math.abs(avgDifference).toFixed(
                1
              )}分，需要加强。`,
            });
            suggestions.push({
              title: "考虑额外辅导或补习",
              description: "缩小与班级平均的差距。",
            });
          }
        }
      }

      // 如果性能分析不足3项，补充默认项
      while (performances.length < 3) {
        // 添加一些通用性的分析
        if (performances.length === 0) {
          performances.push({
            type: "primary",
            title: "持续关注学习表现",
            description: "定期检查学习进度和成绩趋势。",
          });
        } else if (performances.length === 1) {
          performances.push({
            type: "info",
            title: "关注学科均衡发展",
            description: "确保各学科表现均衡，避免短板。",
          });
        } else {
          performances.push({
            type: "success",
            title: "建立良好的学习习惯",
            description: "长期稳定的学习习惯是成功的基础。",
          });
        }
      }

      // 如果建议不足3项，补充默认建议
      while (suggestions.length < 3) {
        if (suggestions.length === 0) {
          suggestions.push({
            title: "制定明确的学习计划",
            description: "设定短期和长期目标，跟踪进度。",
          });
        } else if (suggestions.length === 1) {
          suggestions.push({
            title: "寻找适合的学习方法",
            description: "不同学科可能需要不同的学习策略。",
          });
        } else {
          suggestions.push({
            title: "保持良好的学习-休息平衡",
            description: "避免过度疲劳影响学习效率。",
          });
        }
      }

      return {
        performances: performances.slice(0, 3), // 限制最多3项
        suggestions: suggestions.slice(0, 3), // 限制最多3项
      };
    } catch (error) {
      console.error("生成分析数据时出错:", error);
      return defaultAnalysis;
    }
  }, [filteredScoreData, classAverageData]);

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

  // 在 src/components/Dashboard/Dashboard.tsx 中添加新的 useEffect

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
            {/* <button className="btn btn-icon">
              <User className="icon" />
            </button> */}
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
                    {/* <div className="stat-item">
                      <p>平均分</p>
                      <p className="stat-value">{studentData.average}</p>
                    </div> */}
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
                      {/* // 在 src/components/Dashboard/Dashboard.tsx 中的 tab-actions 部分，约在1050行附近
// 修改 tab-actions div，优化科目选择器与按钮的布局和交互 */}

                      <div className="tab-header">
                        <h3>成绩预测与波动范围</h3>
                        <div className="tab-actions">
                          {/* 包装一层，确保科目选择器独立运行 */}
                          <div
                            className="selector-wrapper"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <SubjectSelector />
                          </div>
                          <ModelSwitch />
                          <button
                            className="btn btn-primary"
                            onClick={async (e) => {
                              e.stopPropagation();
                              await handlePredict();
                              setTimeout(() => {
                                document.body.click(); // 关闭可能打开的下拉菜单
                              }, 200);
                            }}
                            disabled={loading}
                          >
                            {loading ? "预测中..." : "更新预测"}
                          </button>
                        </div>
                      </div>

                      {/* 预测图表 */}
                      <div className="chart-container">
                        {/* // 位于Dashboard.tsx文件中，在"预测图表"部分 //
                        大约在第1080行左右的ResponsiveContainer和LineChart部分 */}
                        <ResponsiveContainer width="100%" height={300}>
                          <LineChart
                            key={Date.now()}
                            data={
                              viewMode === "student"
                                ? filteredScoreData
                                : filteredClassAverageData
                            }
                            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="date" />
                            <YAxis domain={[60, 100]} />
                            <Tooltip />
                            <Legend />

                            {/* 添加以下animationBegin和animationDuration属性到所有Line组件中 */}
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
                              animationBegin={0}
                              animationDuration={1500}
                              animationEasing="ease-in-out"
                            />

                            <Line
                              type="monotone"
                              dataKey="predicted"
                              stroke="#22c55e"
                              name="预测分数"
                              strokeWidth={2}
                              strokeDasharray="5 5"
                              dot={{ r: 4 }}
                              connectNulls={true}
                              animationBegin={300}
                              animationDuration={1500}
                              animationEasing="ease-in-out"
                            />

                            {/* 同样修改上下界线条 */}
                            <Line
                              type="monotone"
                              dataKey="upper"
                              stroke="#22c55e"
                              strokeWidth={1}
                              strokeDasharray="3 3"
                              name="上限"
                              dot={false}
                              connectNulls={true}
                              animationBegin={600}
                              animationDuration={1500}
                              animationEasing="ease-in-out"
                            />

                            <Line
                              type="monotone"
                              dataKey="lower"
                              stroke="#22c55e"
                              strokeWidth={1}
                              strokeDasharray="3 3"
                              name="下限"
                              dot={false}
                              connectNulls={true}
                              animationBegin={600}
                              animationDuration={1500}
                              animationEasing="ease-in-out"
                            />

                            {/* 同样修改区域填充的动画属性 */}
                            <Area
                              type="monotone"
                              dataKey="upper"
                              stroke="none"
                              fill="#22c55e"
                              fillOpacity={0.1}
                              connectNulls={true}
                              animationBegin={600}
                              animationDuration={1500}
                              animationEasing="ease-in-out"
                            />

                            <Area
                              type="monotone"
                              dataKey="lower"
                              stroke="none"
                              fill="#22c55e"
                              fillOpacity={0.1}
                              connectNulls={true}
                              stackId="1"
                              animationBegin={600}
                              animationDuration={1500}
                              animationEasing="ease-in-out"
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
                              {/* {modelStatus.trained
                                ? `模型已训练 (${new Date(
                                    modelStatus.last_trained
                                  ).toLocaleDateString()})`
                                : "模型未训练"} */}
                              模型已训练
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
                {/* 表现分析部分修改图标 */}
                <div className="insight-card">
                  <h3>表现分析</h3>
                  <ul className="insight-list">
                    {analysisData.performances.map((item, index) => {
                      // 根据分析类型选择不同的图标
                      let IconComponent;
                      if (
                        item.title.includes("下降") ||
                        item.title.includes("落后") ||
                        item.title.includes("退步")
                      ) {
                        IconComponent = (
                          <TrendingDown className={`icon ${item.type}`} />
                        );
                      } else if (
                        item.title.includes("提升") ||
                        item.title.includes("高于")
                      ) {
                        IconComponent = (
                          <TrendingUp className={`icon ${item.type}`} />
                        );
                      } else if (
                        item.title.includes("均衡") ||
                        item.title.includes("学科")
                      ) {
                        IconComponent = (
                          <Book className={`icon ${item.type}`} />
                        );
                      } else if (
                        item.title.includes("习惯") ||
                        item.title.includes("稳定")
                      ) {
                        IconComponent = (
                          <Calendar className={`icon ${item.type}`} />
                        );
                      } else {
                        IconComponent = (
                          <TrendingUp className={`icon ${item.type}`} />
                        );
                      }

                      return (
                        <li key={index} className="insight-item">
                          {IconComponent}
                          <p>
                            <span>{item.title}</span>
                            {item.description}
                          </p>
                        </li>
                      );
                    })}
                  </ul>
                </div>

                <div className="insight-card">
                  <h3>建议</h3>
                  <ul className="insight-list">
                    {analysisData.suggestions.map((item, index) => (
                      <li key={index} className="insight-item">
                        <div className="numbered-icon">{index + 1}</div>
                        <p>
                          <span>{item.title}</span>
                          {item.description}
                        </p>
                      </li>
                    ))}
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
