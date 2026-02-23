'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Command } from '@tauri-apps/plugin-shell';
import { open } from '@tauri-apps/plugin-dialog';
import { resolveResource } from '@tauri-apps/api/path';
import { convertFileSrc, invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { FolderOpen, Play, Square, Save, Activity, Terminal, CheckCircle, AlertCircle, BarChart2, Layers, Download, Cpu, Sun, Moon, Database, Zap, Search, History } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { save } from '@tauri-apps/plugin-dialog';
import { writeTextFile } from '@tauri-apps/plugin-fs';
import QRCodeConnect from './components/QRCodeConnect';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface LogEntry {
  type: 'info' | 'error' | 'success';
  message: string;
  timestamp: string;
}

interface TrainingStatus {
  epoch: number;
  total_epochs: number;
  train_accuracy: string;
  train_loss: string;
  val_accuracy: string;
  val_loss: string;
  status: string;
  learning_rate?: string;
}

interface Preset {
  name: string;
  epochs: number;
  batchSize: number;
  model: string;
  learningRate: number;
}

const PRESETS: Preset[] = [
  { name: 'Quick Test', epochs: 2, batchSize: 16, model: 'resnet18', learningRate: 0.001 },
  { name: 'Standard', epochs: 10, batchSize: 32, model: 'resnet18', learningRate: 0.001 },
  { name: 'High Performance', epochs: 20, batchSize: 64, model: 'resnet50', learningRate: 0.0001 },
  { name: 'Deep Learning', epochs: 30, batchSize: 32, model: 'eva02', learningRate: 0.0001 },
];

interface EvalResult {
  status: string;
  report: Record<string, any>;
  confusion_matrix_path: string;
  total_epochs: number;
  test_size: number;
}
interface StoredExperiment {
  id: string;
  date: string;
  accuracy: string;
  model: string;
  training_metrics: {
    epoch: number;
    accuracy: number;
    loss: number;
  }[];
}


type TabularResult = {
  status: 'success' | 'error';
  message?: string;
  columns?: string[];
  data?: (string | number | null)[][];
  shape?: [number, number];
  dtypes?: Record<string, string>;
  missing?: Record<string, number>;
  loaded_path?: string;
};
type TabAction = 'load' | 'drop_missing' | 'fill_missing' | 'label_encode' | 'one_hot_encode';
type FillMethod = 'mean' | 'median' | 'mode' | 'zero';

export default function Home() {
  const [datasetPath, setDatasetPath] = useState('');
  const [savePath, setSavePath] = useState('');
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(32);
  const [numWorkers, setNumWorkers] = useState(-1);
  const [model, setModel] = useState('resnet18');
  const [learningRate, setLearningRate] = useState(0.001);
  const [augmentationConfig, setAugmentationConfig] = useState({horizontalFlip: false,verticalFlip: false,rotation: { enabled: false, degrees: 15 },colorJitter: { enabled: false, brightness: 0.2, contrast: 0.2, saturation: 0.2 },randomResizedCrop: { enabled: false, scaleMin: 0.8, scaleMax: 1.0 }});
  const [zipDataset, setZipDataset] = useState(false);
  const [onlyZip, setOnlyZip] = useState(false);
  const [patience, setPatience] = useState(5);
  const [resumePath, setResumePath] = useState('');
  const [evaluateOnly, setEvaluateOnly] = useState(false);
  const [gpuAvailable, setGpuAvailable] = useState<boolean | null>(null);
  const [condaEnvs, setCondaEnvs] = useState<{name: string, isGpu: boolean | null}[]>([]);
  const [selectedEnv, setSelectedEnv] = useState<string>('system');
  const [scanningEnvs, setScanningEnvs] = useState(false);
  const [showPresets, setShowPresets] = useState(false);
  const [showExperiments, setShowExperiments] = useState(false);
  const [isLightMode, setIsLightMode] = useState(true);
  const [recentExperiments, setRecentExperiments] = useState<StoredExperiment[]>([]);
  const finalAccuracyRef = useRef<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [pid, setPid] = useState<number | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [chartData, setChartData] = useState<any[]>([]);
  const [progress, setProgress] = useState(0);
  const [currentStatus, setCurrentStatus] = useState<TrainingStatus | null>(null);
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null);
  const [matrixImageUrl, setMatrixImageUrl] = useState<string | null>(null);
  const [imageLoadError, setImageLoadError] = useState(false);
  const [activeTab, setActiveTab] = useState<'logs' | 'charts' | 'results' | 'data' | 'system' | 'compare' | 'insights' | 'runs'>('logs');
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  // Tabular / GPU state
  const [tabFile, setTabFile] = useState('');
  const [tabAction, setTabAction] = useState<TabAction>('load');
  const [fillMethod, setFillMethod] = useState<FillMethod>('mean');
  const [encodeColumns, setEncodeColumns] = useState('');
  const [tabOutPath, setTabOutPath] = useState('');
  const [tabLoading, setTabLoading] = useState(false);
  const [tabResult, setTabResult] = useState<TabularResult | null>(null);
  const [systemInfo, setSystemInfo] = useState<any | null>(null);
  const [systemLoading, setSystemLoading] = useState(false);
  const [systemError, setSystemError] = useState<string | null>(null);
  // AutoML State
  const [isAutoMLRunning, setIsAutoMLRunning] = useState(false);
  const [autoMLTrials, setAutoMLTrials] = useState<{trial: number; n_trials: number; params: {learning_rate: number; batch_size: number; optimizer: string}; val_accuracy: number}[]>([]);
  const [autoMLBestParams, setAutoMLBestParams] = useState<{learning_rate: number; batch_size: number; optimizer: string} | null>(null);
  const [autoMLProgress, setAutoMLProgress] = useState(0);
  const [autoMLTrialCount, setAutoMLTrialCount] = useState(10);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const commandRef = useRef<Command<string> | null>(null);
  const childRef = useRef<any>(null);
  const autoMLChildRef = useRef<any>(null);
  const metricsRef = useRef<any[]>([]);
  // Ref that always points to the latest startTraining — avoids stale closure in listeners
  const startTrainingRef = useRef<() => void>(() => {});
  const [runs, setRuns] = useState<any[]>([]);

  // Dataset Analysis State
  const [datasetStats, setDatasetStats] = useState<any | null>(null);
  const [analyzingDataset, setAnalyzingDataset] = useState(false);
  // Check GPU availability
  useEffect(() => {
    async function checkGpu() {
      setGpuAvailable(null);
      try {
        let cmdName: string;
        let args: string[];

        if (selectedEnv.startsWith('conda:')) {
          const envName = selectedEnv.replace('conda:', '');
          cmdName = 'conda';
          args = ['run', '-n', envName, 'python', '-c', 'import torch; print(torch.cuda.is_available())'];
        } else {
          cmdName = await resolvePythonInterpreter();
          args = ['-c', 'import torch; print(torch.cuda.is_available())'];
        }
        
        const cmd = Command.create(cmdName, args);
        const res = await cmd.execute();
        setGpuAvailable(res.code === 0 && res.stdout.trim() === 'True');
      } catch {
        setGpuAvailable(false);
      }
    }
    checkGpu();
  }, [selectedEnv]);

  const pickTabFile = useCallback(async () => {
    const selected = await open({ multiple: false, filters: [{ name: 'Data Files', extensions: ['csv', 'xlsx', 'xls'] }] });
    if (typeof selected === 'string') setTabFile(selected);
  }, []);

  const runTabular = useCallback(async () => {
    if (!tabFile) return;
    setTabLoading(true);
    setTabResult(null);
    try {
      const isProcess = tabAction !== 'load';
      let paramsJson: string | undefined;
      if (isProcess) {
        const p: Record<string, unknown> = { operation: tabAction };
        if (tabAction === 'fill_missing') p.method = fillMethod;
        if (tabAction === 'label_encode' || tabAction === 'one_hot_encode')
          p.columns = encodeColumns.split(',').map(s => s.trim()).filter(Boolean);
        paramsJson = JSON.stringify(p);
      }
      const raw: string = await invoke('run_tabular_processor', {
        file: tabFile, action: isProcess ? 'process' : 'load',
        params: paramsJson, out: tabOutPath || undefined,
      });
      setTabResult(JSON.parse(raw));
    } catch (err) {
      setTabResult({ status: 'error', message: String(err) });
    } finally {
      setTabLoading(false);
    }
  }, [tabFile, tabAction, fillMethod, encodeColumns, tabOutPath]);

  const scanCondaEnvs = async () => {
    setScanningEnvs(true);
    addLog('Scanning Conda environments...', 'info');
    try {
      const cmd = Command.create('conda', ['env', 'list', '--json']);
      const output = await cmd.execute();
      if (output.code === 0) {
        const data = JSON.parse(output.stdout);
        const envs = data.envs as string[];
        const detected = [];
        for (const envPath of envs) {
           const name = envPath === envs[0] ? 'base' : envPath.split(/[\\/]/).pop() || 'unknown';
           detected.push({ name, isGpu: null });
        }
        setCondaEnvs(detected);
        addLog(`Found ${detected.length} Conda environments. Testing for PyTorch GPU...`, 'info');
        
        const tested = [];
        for (const env of detected) {
           try {
             const testCmd = Command.create('conda', ['run', '-n', env.name, 'python', '-c', 'import torch; print(torch.cuda.is_available())']);
             const res = await testCmd.execute();
             const isGpu = res.stdout.trim() === 'True';
             tested.push({ ...env, isGpu });
             addLog(`Env '${env.name}' - PyTorch GPU: ${isGpu}`, isGpu ? 'success' : 'info');
           } catch {
             tested.push({ ...env, isGpu: false });
           }
        }
        setCondaEnvs(tested);
        
        const gpuEnv = tested.find(e => e.isGpu);
        if (gpuEnv && !gpuAvailable) {
          setSelectedEnv(`conda:${gpuEnv.name}`);
          addLog(`Auto-selected GPU environment: ${gpuEnv.name}`, 'success');
        }
      } else {
         addLog('Conda not found or failed to list envs.', 'error');
      }
    } catch (e) {
       addLog(`Failed to scan Conda envs: ${e}`, 'error');
    }
    setScanningEnvs(false);
  };
  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        if (e.key === 'Enter' && !isRunning && datasetPath) {
          startTraining();
        } else if (e.key === 'p') {
          e.preventDefault();
          setShowPresets(!showPresets);
          setShowExperiments(false);
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isRunning, datasetPath, showPresets, showExperiments]);

  // Keep ref in sync with the latest startTraining on every render
  // so the mobile_command listener always uses the current datasetPath
  useEffect(() => {
    startTrainingRef.current = startTraining;
  });

  // Listen for mobile commands
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    
    const setupListener = async () => {
      unlisten = await listen<string>('mobile_command', (event) => {
        const payload = event.payload;

        if (payload === 'stop_training') {
          addLog('⏹️ Stop command received from Mobile App!', 'error');
          if (childRef.current) childRef.current.kill();
          if (autoMLChildRef.current) autoMLChildRef.current.kill();
          setIsRunning(false);
          setIsAutoMLRunning(false);
          return;
        }

        // Try to parse as JSON (start_training / adjust_params)
        try {
          const cmd = JSON.parse(payload);
          if (cmd.action === 'start_training') {
            if (isRunning || isAutoMLRunning) {
              addLog('⚠️ Mobile requested start, but training is already in progress. Stop it first.', 'error');
              invoke('broadcast_log', {
                log: JSON.stringify({ type: 'training_busy', message: 'Training already in progress on desktop.' })
              }).catch(console.error);
            } else {
              addLog('▶️ Start training command received from Mobile App!', 'info');
              if (cmd.epochs !== undefined) setEpochs(Number(cmd.epochs));
              if (cmd.batch_size !== undefined) setBatchSize(Number(cmd.batch_size));
              if (cmd.learning_rate !== undefined) setLearningRate(Number(cmd.learning_rate));
              if (cmd.model !== undefined) setModel(String(cmd.model));
              // Use ref so we always call the latest startTraining (avoids stale closure over datasetPath)
              setTimeout(() => startTrainingRef.current(), 200);
            }
          } else if (cmd.action === 'adjust_params') {
            addLog('⚙️ Params adjusted from Mobile App', 'info');
            if (cmd.epochs !== undefined) { setEpochs(Number(cmd.epochs)); addLog(`  Epochs → ${cmd.epochs}`, 'info'); }
            if (cmd.batch_size !== undefined) { setBatchSize(Number(cmd.batch_size)); addLog(`  Batch Size → ${cmd.batch_size}`, 'info'); }
            if (cmd.learning_rate !== undefined) { setLearningRate(Number(cmd.learning_rate)); addLog(`  LR → ${cmd.learning_rate}`, 'info'); }
            if (cmd.model !== undefined) { setModel(String(cmd.model)); addLog(`  Model → ${cmd.model}`, 'info'); }
          } else if (cmd.action === 'select_dataset') {
            addLog('📂 Dataset selection requested from Mobile App', 'info');
            // Open the native folder picker on the desktop
            open({ directory: true, multiple: false, title: 'Select Dataset Folder' }).then((selected) => {
              if (selected && typeof selected === 'string') {
                setDatasetPath(selected);
                addLog(`📁 Dataset selected: ${selected}`, 'success');
                // Broadcast back to mobile so it can display the path
                invoke('broadcast_log', {
                  log: JSON.stringify({ type: 'dataset_update', path: selected })
                }).catch(console.error);
              } else {
                invoke('broadcast_log', {
                  log: JSON.stringify({ type: 'dataset_update', path: null })
                }).catch(console.error);
              }
            }).catch(() => {
              addLog('❌ Failed to open folder picker', 'error');
            });
          }
        } catch {
          // Unknown string command - ignore
        }
      });
    };
    
    setupListener();
    
    return () => {
      if (unlisten) unlisten();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const applyPreset = (preset: Preset) => {
    setEpochs(preset.epochs);
    setBatchSize(preset.batchSize);
    setModel(preset.model);
    setLearningRate(preset.learningRate);
    setShowPresets(false);
    addLog(`Applied preset: ${preset.name}`, 'info');
  }; // To store child handle for killing

  useEffect(() => {
    if (activeTab === 'logs') {
      // Use block: 'nearest' to prevent the entire browser window from scrolling
      logsEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [logs, activeTab]);
 
useEffect(() => {
  if (activeTab !== "system") return;

  // First load → show spinner
  fetchSystemInfo(true);

  const interval = setInterval(() => {
    fetchSystemInfo(false); // silent refresh
  }, 3000);

  return () => clearInterval(interval);

}, [activeTab]);

  useEffect(() => {
    async function loadMatrixImage() {
      if (evalResult?.confusion_matrix_path) {
        try {
          const { readFile } = await import('@tauri-apps/plugin-fs');
          const fileContents = await readFile(evalResult.confusion_matrix_path);
          const blob = new Blob([fileContents], { type: 'image/png' });
          const url = URL.createObjectURL(blob);
          
          setMatrixImageUrl(url);
          setImageLoadError(false);
          addLog(`Loaded confusion matrix via Blob URL: ${url}`, 'info');

          // Encode as base64 and broadcast to mobile app
          try {
            const base64 = btoa(
              Array.from(fileContents).map((b: number) => String.fromCharCode(b)).join('')
            );
            invoke('broadcast_log', {
              log: JSON.stringify({ type: 'confusion_matrix', data: `data:image/png;base64,${base64}` })
            }).catch(console.error);
            addLog('📱 Confusion matrix sent to mobile', 'success');
          } catch (encErr) {
            console.error('Failed to encode matrix for mobile:', encErr);
          }
        } catch (err) {
          addLog(`Failed to load matrix image file: ${err}`, 'error');
           try {
             // Fallback
             const normalizedPath = evalResult.confusion_matrix_path.replace(/\\/g, '/');
             const assetUrl = convertFileSrc(normalizedPath);
             setMatrixImageUrl(assetUrl);
             addLog(`Attempting fallback to asset URL: ${assetUrl}`, 'info');
           } catch (e) {
             console.error(e);
           }
        }
      }
    }
    loadMatrixImage();
  }, [evalResult]);
  useEffect(() => {
  let interval: NodeJS.Timeout;

  if (isRunning) {
    interval = setInterval(() => {
      setElapsedSeconds(prev => prev + 1);
    }, 1000);
  }

  return () => {
    if (interval) clearInterval(interval);
  };
}, [isRunning]);

  const addLog = (message: string, type: 'info' | 'error' | 'success' = 'info') => {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
    setLogs(prev => [...prev, { type, message, timestamp }]);
  };

  const handlePickDataset = async () => {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: 'Select Dataset Directory'
      });
      if (selected && typeof selected === 'string') {
        setDatasetPath(selected);
        if (!savePath) setSavePath(selected);
        // Notify connected mobile clients of the selected dataset
        invoke('broadcast_log', {
          log: JSON.stringify({ type: 'dataset_update', path: selected })
        }).catch(console.error);
        // Auto-analyze the dataset when selected
        analyzeDataset(selected);
      }
    } catch (err) {
      console.error(err);
      addLog(`Failed to open dialog: ${err}`, 'error');
    }
  };

  const analyzeDataset = async (path: string) => {
    if (!path) return;
    setAnalyzingDataset(true);
    setDatasetStats(null);
    try {
      const result = await invoke<string>('analyze_dataset', { path });
      const parsed = JSON.parse(result);
      setDatasetStats(parsed);
      if (parsed.status === 'success') {
        addLog(`Dataset analyzed: ${parsed.total_images} images, ${parsed.class_count} classes`, 'success');
      } else {
        addLog(`Dataset analysis failed: ${parsed.message}`, 'error');
      }
    } catch (err) {
      console.error(err);
      addLog(`Failed to analyze dataset: ${err}`, 'error');
    } finally {
      setAnalyzingDataset(false);
    }
  };

  const handlePickSavePath = async () => {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: 'Select Save Directory'
      });
      if (selected && typeof selected === 'string') {
        setSavePath(selected);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handlePickResumePath = async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [{ name: 'PyTorch Checkpoint', extensions: ['pth'] }],
        title: 'Select Checkpoint File to Resume From'
      });
      if (selected && typeof selected === 'string') {
        setResumePath(selected);
      }
    } catch (err) {
      console.error(err);
    }
  };
  async function resolvePythonInterpreter(): Promise<string> {
  const candidates = ['python', 'python3', 'py'];

  for (const cmd of candidates) {
    try {
      await Command.create(cmd, ['--version']).execute();
      return cmd;
    } catch {}
  }

  throw new Error('No Python interpreter found.');
}
  const startTraining = async () => {
    setElapsedSeconds(0);
  let experimentId = '';

  if (!datasetPath) {
    addLog('Please select a dataset path first.', 'error');
    return;
  }

  setIsRunning(true);
  setLogs([]);
  setChartData([]);
  metricsRef.current = [];
  setEvalResult(null);
  setProgress(0);
  setActiveTab('logs');
  setShowPresets(false);
  setShowExperiments(false);
  finalAccuracyRef.current = null;

  try {

    const scriptPath = await resolveResource('python_backend/script.py');
    if (!scriptPath) {
      throw new Error('Failed to resolve backend script path.');
    }

    addLog(`Resolved script path: ${scriptPath}`, 'info');

    const now = new Date();
    const timestamp = now.toISOString().replace(/[-:]/g, '').replace(/\..+/, '');
    experimentId = `exp_${timestamp}`;

    addLog(`Experiment ID: ${experimentId}`, 'info');

      const args = [
        scriptPath,
        '--path', datasetPath,
        '--epochs', epochs.toString(),
        '--batch_size', batchSize.toString(),
        '--model', model,
        '--learning_rate', learningRate.toString()
      ];
      args.push('--experiment_id', experimentId);
      if (savePath) args.push('--save_path', savePath);
      args.push('--num_workers', numWorkers.toString());
      if (zipDataset) args.push('--zip_dataset');
      if (onlyZip) args.push('--only_zip');
      args.push('--patience', patience.toString());
      if (resumePath) args.push('--resume', resumePath);
      if (evaluateOnly) args.push('--evaluate_only');
      args.push('--augmentation', JSON.stringify(augmentationConfig));

      let finalCmd: string;
      let finalArgs = args;

      if (selectedEnv.startsWith('conda:')) {
        const envName = selectedEnv.replace('conda:', '');
        finalCmd = 'conda';
        finalArgs = ['run', '-n', envName, '--no-capture-output', 'python', ...args];
      } else {
        finalCmd = await resolvePythonInterpreter();
      }

      addLog(`Starting command: ${finalCmd} ${finalArgs.join(' ')}`, 'info');

      const cmd = Command.create(finalCmd, finalArgs);
      commandRef.current = cmd;

      cmd.on('close', (data) => {
        addLog(`Process finished with code ${data.code}`, data.code === 0 ? 'success' : 'error');
        setIsRunning(false);
        setPid(null);
        if (data.code === 0) {
            addLog('Training/Task Complete successfully.', 'success');
            // Save to recent experiments
            // const expId = `exp_${Date.now()}`;
            setRecentExperiments(prev => [
          {
            id: experimentId,
            date: new Date().toLocaleDateString(),
            accuracy: finalAccuracyRef.current || 'N/A',
            model,
            training_metrics: metricsRef.current  
          },
          ...prev.slice(0, 9)
        ]);
        }
      });

      cmd.on('error', (error) => {
        addLog(`Command error: ${error}`, 'error');
        setIsRunning(false);
      });

      cmd.stdout.on('data', (line) => {
        // Broadcast to mobile app clients
        invoke('broadcast_log', { log: line }).catch(console.error);
        try {
          const data = JSON.parse(line);
          
          
          if (data.status === 'training') {
  setCurrentStatus(data);

  const newPoint = {
  epoch: data.epoch,
  accuracy: parseFloat(data.train_accuracy),
  loss: parseFloat(data.train_loss),
  val_accuracy: parseFloat(data.val_accuracy),
  val_loss: parseFloat(data.val_loss),
};

  metricsRef.current.push(newPoint);
  setChartData([...metricsRef.current]);
          finalAccuracyRef.current = data.accuracy;
            const prog = (data.epoch / data.total_epochs) * 100;
            setProgress(prog);
            if (activeTab === 'logs' && data.epoch === 1) setActiveTab('charts');
          } 
          else if (data.status === 'checkpoint') {
            addLog(`Checkpoint saved: ${data.message} at ${data.path}`, 'success');
          }
          else if (data.status === 'evaluation_complete') {
            setEvalResult(data);
            setActiveTab('results');
            addLog('Evaluation complete. Results available.', 'success');
          }
           else if (data.status === 'dataset_zip') {
             addLog(`Dataset zipped at: ${data.path}`, 'success');
           }
          else if (data.status === 'stopped_early') {
            addLog(`⏹ Early stopping triggered at epoch ${data.epoch}: ${data.message}`, 'success');
            setProgress(100);
          }
          else if (data.status === 'resumed') {
            addLog(`▶ Resumed from epoch ${data.message.replace('Resumed from epoch ', '')} (best acc so far: ${data.best_acc})`, 'info');
          }
          else if (data.status === 'error') {
            addLog(`Error from script: ${data.message}`, 'error');
          }
          else {
             addLog(line, 'info');
          }
        } catch (e) {
          addLog(line, 'info');
        }
      });

      cmd.stderr.on('data', (line) => {
        invoke('broadcast_log', { log: line }).catch(console.error);
        addLog(line, 'error');
      });

      const child = await cmd.spawn();
    childRef.current = child;
    setPid(child.pid);
    addLog(`Process started with PID: ${child.pid}`, 'info');

  } catch (err) {
    addLog(`Failed to spawn process: ${err}`, 'error');
    setIsRunning(false);
  }
};
const formatTime = (seconds: number) => {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  return [
    hrs > 0 ? String(hrs).padStart(2, '0') : null,
    String(mins).padStart(2, '0'),
    String(secs).padStart(2, '0')
  ].filter(Boolean).join(':');
};
const startAutoML = async () => {
    if (!datasetPath) {
      addLog('Please select a dataset path first.', 'error');
      return;
    }

    setIsAutoMLRunning(true);
    setAutoMLTrials([]);
    setAutoMLBestParams(null);
    setAutoMLProgress(0);
    setActiveTab('logs');

    try {
      const scriptPath = await resolveResource('python_backend/automl_sweep.py');
      if (!scriptPath) {
        throw new Error('Failed to resolve automl_sweep.py path.');
      }

      addLog(`[AutoML] Starting hyperparameter sweep with ${autoMLTrialCount} trials...`, 'info');

      const args = [
        scriptPath,
        '--path', datasetPath,
        '--model', model,
        '--n_trials', autoMLTrialCount.toString(),
        '--epochs_per_trial', '3',
        '--num_workers', numWorkers.toString()
      ];

      let finalCmd: string;
      let finalArgs = args;

      if (selectedEnv.startsWith('conda:')) {
        const envName = selectedEnv.replace('conda:', '');
        finalCmd = 'conda';
        finalArgs = ['run', '-n', envName, '--no-capture-output', 'python', ...args];
      } else {
        finalCmd = await resolvePythonInterpreter();
      }

      addLog(`[AutoML] Command: ${finalCmd} ${finalArgs.join(' ')}`, 'info');

      const cmd = Command.create(finalCmd, finalArgs);

      cmd.on('close', (data) => {
        addLog(`[AutoML] Sweep finished with code ${data.code}`, data.code === 0 ? 'success' : 'error');
        setIsAutoMLRunning(false);
      });

      cmd.on('error', (error) => {
        addLog(`[AutoML] Error: ${error}`, 'error');
        setIsAutoMLRunning(false);
      });

      cmd.stdout.on('data', (line) => {
        invoke('broadcast_log', { log: line }).catch(console.error);
        try {
          const data = JSON.parse(line);

          if (data.status === 'automl_started') {
            addLog(`[AutoML] Sweep started: ${data.n_trials} trials on ${data.device}`, 'info');
          } else if (data.status === 'automl_info') {
            addLog(`[AutoML] ${data.message}`, 'info');
          } else if (data.status === 'automl_trial') {
            setAutoMLTrials(prev => [...prev, data]);
            setAutoMLProgress(Math.round((data.trial / data.n_trials) * 100));
            addLog(`[AutoML] Trial ${data.trial}/${data.n_trials}: lr=${data.params.learning_rate.toExponential(2)}, bs=${data.params.batch_size}, opt=${data.params.optimizer} → acc=${(data.val_accuracy * 100).toFixed(2)}%`, 'info');
          } else if (data.status === 'automl_trial_error') {
            addLog(`[AutoML] Trial ${data.trial}/${data.n_trials} failed: ${data.message}`, 'error');
            setAutoMLProgress(Math.round((data.trial / data.n_trials) * 100));
          } else if (data.status === 'automl_complete') {
            setAutoMLBestParams(data.best_params);
            setAutoMLProgress(100);
            // Auto-apply best params
            setLearningRate(data.best_params.learning_rate);
            setBatchSize(data.best_params.batch_size);
            addLog(`[AutoML] Sweep complete! Best: lr=${data.best_params.learning_rate.toExponential(2)}, bs=${data.best_params.batch_size}, opt=${data.best_params.optimizer}, acc=${(data.best_accuracy * 100).toFixed(2)}%`, 'success');
            addLog(`[AutoML] Best learning rate and batch size have been auto-applied to your config.`, 'success');
          } else if (data.status === 'error') {
            addLog(`[AutoML] Error: ${data.message}`, 'error');
          } else {
            addLog(line, 'info');
          }
        } catch {
          addLog(line, 'info');
        }
      });

      cmd.stderr.on('data', (line) => {
        invoke('broadcast_log', { log: line }).catch(console.error);
        // Only log actual errors, skip Optuna/torch warnings
        if (line.trim() && !line.includes('[I ') && !line.includes('[W ')) {
          addLog(`[AutoML] ${line}`, 'error');
        }
      });

      const child = await cmd.spawn();
      autoMLChildRef.current = child;
    } catch (err) {
      addLog(`[AutoML] Failed to start sweep: ${err}`, 'error');
      setIsAutoMLRunning(false);
    }
  };

  const exportAsJson = async () => {
  try {
    console.log("Export triggered");

    if (!chartData.length) {
      addLog("No training data to export.", "error");
      return;
    }

    const payload = {
      metadata: {
        exported_at: new Date().toISOString(),
        dataset_path: datasetPath,
        model,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        num_workers: numWorkers,
        environment: selectedEnv,
        elapsed_seconds: elapsedSeconds
      },
      training_metrics: chartData,
      final_status: currentStatus,
      evaluation: evalResult,
      logs
    };

    const filePath = await save({
      filters: [{ name: "JSON", extensions: ["json"] }]
    });

    if (!filePath) {
      addLog("Export cancelled.", "info");
      return;
    }

    console.log("Saving to:", filePath);

    await writeTextFile(filePath, JSON.stringify(payload, null, 2));

    addLog("Experiment exported successfully.", "success");

  } catch (err) {
    console.error("EXPORT ERROR:", err);
    addLog(`Export failed: ${err}`, "error");
  }
};
const fetchSystemInfo = async (initial = false) => {
  try {
    if (initial) setSystemLoading(true);

    const raw = await invoke("get_system_info");
    const parsed = JSON.parse(raw as string);

    setSystemInfo(parsed);
    setSystemError(null);
  } catch (err) {
    setSystemError("Failed to load system information");
  } finally {
    if (initial) setSystemLoading(false);
  }
};
const getUsageColor = (value: number) => {
  if (value < 50) return "bg-emerald-500"
  if (value < 80) return "bg-yellow-500"
  return "bg-red-500"
}
const [compareA, setCompareA] = useState<StoredExperiment | null>(null);
const [compareB, setCompareB] = useState<StoredExperiment | null>(null);
const [compareData, setCompareData] = useState<any[]>([]);
useEffect(() => {
  console.log("Compare A:", compareA);
console.log("Compare B:", compareB);
  if (!compareA || !compareB) {
    setCompareData([]);
    return;
  }

  const maxEpochs = Math.max(
    compareA.training_metrics.length,
    compareB.training_metrics.length
  );

  const keyA = `${compareA.model}_${compareA.id}`;
const keyB = `${compareB.model}_${compareB.id}`;

const merged = Array.from({ length: maxEpochs }).map((_, i) => ({
  epoch: i + 1,
  [keyA]: compareA.training_metrics[i]?.accuracy ?? null,
  [keyB]: compareB.training_metrics[i]?.accuracy ?? null,
}));

  setCompareData(merged);
}, [compareA, compareB]);
const keyA = `${compareA?.model}_${compareA?.id}`;
const keyB = `${compareB?.model}_${compareB?.id}`;
// ===== INSIGHTS CALCULATIONS =====
const hasMetrics = chartData.length > 0;

const peakAccuracy = hasMetrics
  ? Math.max(...chartData.map(d => d.accuracy))
  : 0;

const lowestLoss = hasMetrics
  ? Math.min(...chartData.map(d => d.loss))
  : 0;

const bestEpoch = hasMetrics
  ? chartData.find(d => d.accuracy === peakAccuracy)?.epoch
  : null;

const accuracyGain = hasMetrics
  ? chartData[chartData.length - 1].accuracy - chartData[0].accuracy
  : 0;
  let overfitGap = 0;

if (hasMetrics && chartData.length > 0) {
  const last = chartData[chartData.length - 1];

  if (last.val_accuracy !== undefined) {
    overfitGap = last.accuracy - last.val_accuracy;
  }
}

// Simple trend detection
let trainingTrend = "Stable";


if (hasMetrics && chartData.length > 3) {
  const last = chartData[chartData.length - 1].accuracy;
  const prev = chartData[chartData.length - 3].accuracy;

  if (last > prev) trainingTrend = "Improving";
  else if (last < prev) trainingTrend = "Degrading";
}
// ===== ADVANCED INSIGHTS =====

let convergenceEpoch = null;
let stabilityScore = 0;
let efficiencyScore = 0;
let recommendation = "Training looks healthy.";

if (hasMetrics && chartData.length > 3) {

  // Convergence detection (when improvement becomes minimal)
  for (let i = 1; i < chartData.length; i++) {
    const diff = chartData[i].accuracy - chartData[i - 1].accuracy;
    if (Math.abs(diff) < 0.001) {
      convergenceEpoch = chartData[i].epoch;
      break;
    }
  }

  // Stability (variance of last 3 accuracies)
  const lastThree = chartData.slice(-3).map(d => d.accuracy);
  const mean = lastThree.reduce((a, b) => a + b, 0) / lastThree.length;
  const variance = lastThree.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / lastThree.length;
  stabilityScore = variance;

  // Efficiency score (gain per epoch)
  efficiencyScore = accuracyGain / chartData.length;

  // Recommendation logic
  if (trainingTrend === "Degrading") {
    recommendation = "Model accuracy is dropping. Consider reducing learning rate.";
  } else if (convergenceEpoch && convergenceEpoch < chartData.length - 2) {
    recommendation = "Model converged early. You may reduce total epochs.";
  } else if (accuracyGain < 0.01) {
    recommendation = "Minimal improvement. Try tuning hyperparameters.";
  }
}
const InsightCard = ({ title, children }: any) => (
  <div className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6">
    <div className="text-xs text-zinc-500 uppercase tracking-wider mb-2">
      {title}
    </div>
    <div className="text-2xl font-bold font-mono">
      {children}
    </div>
  </div>
);
const loadRuns = async () => {
  if (!savePath) return;

  try {
    const raw = await invoke('fetch_runs', {
      savePath: savePath
    });

    const parsed = JSON.parse(raw as string);
    setRuns(parsed);

  } catch (err) {
    console.error("Failed to load runs:", err);
  }
};
useEffect(() => {
  if (activeTab === "runs") {
    loadRuns();
  }
}, [activeTab]);

  return (
    <>
      <div data-theme={isLightMode ? 'light' : 'dark'} className={`min-h-screen font-sans bg-black text-zinc-100 theme-transition`}>
      {/* Header */}
      <header className="sticky top-0 z-50 flex items-center justify-between bg-black/80 backdrop-blur-md border-b border-zinc-800/50 px-8 py-4 mb-8">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 bg-zinc-900 rounded-full border border-zinc-800 overflow-hidden shrink-0">
             <img src="/epoq2.png" alt="EPOQ Logo" className="w-full h-full object-cover" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">
              EPOQ
            </h1>
            <p className="text-sm text-zinc-500 font-mono">v0.2.0-beta</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          {/* GPU Status Indicator */}
          <div className="flex items-center gap-2 px-4 py-2 bg-zinc-900/50 border border-zinc-800 rounded-full">
            <Cpu className={`w-4 h-4 ${gpuAvailable ? 'text-emerald-400' : 'text-zinc-500'}`} />
            <span className="text-xs font-medium">
              {gpuAvailable === null ? 'Checking...' : gpuAvailable ? 'GPU Available' : 'CPU Only'}
            </span>
          </div>
          <QRCodeConnect />
          {/* Recent Experiments Button */}
          <div className="relative">
            <button 
              onClick={() => {
                setShowExperiments(!showExperiments);
                setShowPresets(false);
              }}
              className="flex items-center gap-2 px-4 py-2 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 rounded-full text-sm text-zinc-300 transition-colors"
            >
              <Activity className="w-4 h-4" /> Recent
            </button>
            {showExperiments && (
              <div className="absolute right-0 mt-2 w-72 bg-zinc-900 border border-zinc-800 rounded-xl shadow-xl z-50 overflow-hidden">
                {recentExperiments.length > 0 ? (
                  <div className="max-h-64 overflow-y-auto">
                    {recentExperiments.map((exp) => (
                      <div key={exp.id} className="p-3 border-b last:border-b-0 border-zinc-800/50 hover:bg-zinc-800 transition-colors">
                        <div className="flex justify-between items-center">
                          <div>
                            <div className="text-sm text-zinc-300 font-medium">{exp.model}</div>
                            <div className="text-xs text-zinc-500">{exp.date}</div>
                          </div>
                          <div className="text-right">
                            <div className="text-sm font-mono text-emerald-400">{exp.accuracy !== 'N/A' ? `${(parseFloat(exp.accuracy)*100).toFixed(2)}%` : exp.accuracy}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-4 text-center text-sm text-zinc-500">No recent experiments</div>
                )}
              </div>
            )}
          </div>

          {/* Presets Button */}
          <div className="relative">
            <button 
              onClick={() => {
                setShowPresets(!showPresets);
                setShowExperiments(false);
              }}
              className="flex items-center gap-2 px-4 py-2 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 rounded-full text-sm text-zinc-300 transition-colors"
            >
              <Layers className="w-4 h-4" /> Presets
            </button>
            {showPresets && (
              <div className="absolute right-0 mt-2 w-48 bg-zinc-900 border border-zinc-800 rounded-xl shadow-xl z-50 overflow-hidden">
                {PRESETS.map((preset, i) => (
                  <button
                    key={i}
                    onClick={() => applyPreset(preset)}
                    className="w-full px-4 py-3 text-left text-sm text-zinc-300 hover:bg-zinc-800 transition-colors flex justify-between items-center"
                  >
                    <span>{preset.name}</span>
                    <span className="text-xs text-zinc-500 font-mono">{preset.epochs}ep/{preset.batchSize}bs</span>
                  </button>
                ))}
              </div>
            )}
          </div>

            {/* Theme Toggle */}
            <button
              onClick={() => setIsLightMode(!isLightMode)}
              className="flex items-center justify-center w-[38px] h-[38px] bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 rounded-full text-zinc-400 hover:text-white transition-colors ml-2"
              aria-label="Toggle theme"
            >
              {isLightMode ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
            </button>

          {/* Keyboard Shortcut Hint */}
          <span className="text-xs text-zinc-600 hidden lg:inline ml-2">Ctrl+Enter to start</span>
           
           {isRunning ? (
             <button 
               onClick={() => {
                   childRef.current?.kill();
                   addLog('Process killed by user.', 'error');
                   setIsRunning(false);
               }}
               className="flex items-center gap-2 px-6 py-2.5 bg-red-900/20 hover:bg-red-900/30 text-red-200 border border-red-900/30 rounded-full transition-all text-sm font-medium"
             >
               <Square className="w-4 h-4 fill-current" /> Stop Process
             </button>
           ) : (
             <button 
               onClick={startTraining}
               className="flex items-center gap-2 px-6 py-2.5 bg-white text-black hover:bg-zinc-200 rounded-full transition-all text-sm font-medium shadow-lg shadow-white/5"
             >
               <Play className="w-4 h-4 fill-current" /> Start Training
             </button>
           )}
        </div>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-12 gap-8 px-8 pb-8 lg:items-start">
        {/* Left Panel: Configuration */}
        <section className="lg:col-span-4 space-y-8">
          <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-2xl p-6 backdrop-blur-sm">
            <div className="flex items-center gap-3 mb-6">
              <Layers className="w-5 h-5 text-zinc-400" />
              <h2 className="text-lg font-semibold tracking-tight">Configuration</h2>
            </div>

            <div className="space-y-6">
              {/* Environment Selection */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                   <label className="text-xs uppercase tracking-wider text-zinc-500 font-semibold">Python Environment</label>
                   <button 
                     onClick={scanCondaEnvs}
                     disabled={scanningEnvs}
                     className="text-xs text-emerald-400 hover:text-emerald-300 transition-colors disabled:opacity-50"
                   >
                     {scanningEnvs ? 'Scanning...' : 'Detect Conda GPU Env'}
                   </button>
                </div>
                <div className="relative">
                  <select 
                    value={selectedEnv}
                    onChange={(e) => setSelectedEnv(e.target.value)}
                    className="w-full appearance-none bg-black border border-zinc-800 rounded-lg py-2.5 px-3 text-sm text-zinc-300 focus:border-zinc-600 focus:outline-none transition-colors"
                  >
                    <option value="system">System Default (python)</option>
                    {condaEnvs.map((env) => (
                      <option key={env.name} value={`conda:${env.name}`}>
                        Conda: {env.name} {env.isGpu ? '(GPU Available)' : '(CPU Only)'}
                      </option>
                    ))}
                  </select>
                  <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none">
                    <svg className="w-4 h-4 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                  </div>
                </div>
              </div>

              {/* Dataset Path */}
              <div className="space-y-2">
                <label className="text-xs uppercase tracking-wider text-zinc-500 font-semibold">Dataset Location</label>
                <div className="flex gap-2">
                  <div className="relative flex-1 group">
                    <input 
                      type="text" 
                      value={datasetPath}
                      readOnly
                      placeholder="No directory selected"
                      className="w-full bg-black border border-zinc-800 rounded-lg py-2.5 px-3 text-sm text-zinc-300 focus:border-zinc-600 focus:outline-none transition-colors font-mono"
                    />
                  </div>
                  <button 
                     onClick={handlePickDataset}
                     className="p-2.5 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 rounded-lg text-zinc-400 hover:text-white transition-colors"
                  >
                    <FolderOpen className="w-5 h-5" />
                  </button>
                  <button 
                     onClick={() => datasetPath && analyzeDataset(datasetPath)}
                     disabled={!datasetPath || analyzingDataset}
                     className="p-2.5 bg-zinc-200 hover:bg-zinc-300 dark:bg-zinc-800 dark:hover:bg-zinc-700 border border-zinc-300 dark:border-zinc-700 rounded-lg text-zinc-600 dark:text-zinc-300 hover:text-zinc-900 dark:hover:text-white transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                     title="Analyze Dataset"
                  >
                    {analyzingDataset ? (
                      <div className="w-5 h-5 border-2 border-zinc-500 dark:border-zinc-400 border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <Search className="w-5 h-5" />
                    )}
                  </button>
                </div>
              </div>

                {/* Dataset Statistics Display */}
                {datasetStats && datasetStats.status === 'success' && (
                <details className="group border-2 border-white rounded-xl overflow-hidden bg-black text-white">
                  <summary className="flex items-center justify-between p-4 cursor-pointer hover:bg-white hover:text-black transition-colors">
                  <div className="flex items-center gap-2">
                    <Database className="w-4 h-4" />
                    <span className="text-sm font-semibold tracking-wide">Dataset Overview</span>
                  </div>
                  <svg className="w-4 h-4 transition-transform group-open:rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" /></svg>
                  </summary>
                  <div className="p-4 pt-2 space-y-6">
                  {/* Stats Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div className="bg-zinc-900/50 rounded-lg p-3 border border-zinc-800">
                    <div className="text-[9px] text-zinc-500 uppercase tracking-wider font-medium">Total Images</div>
                    <div className="text-base font-bold text-white mt-1 truncate">{datasetStats.total_images}</div>
                    </div>
                    <div className="bg-zinc-900/50 rounded-lg p-3 border border-zinc-800">
                    <div className="text-[9px] text-zinc-500 uppercase tracking-wider font-medium">Classes</div>
                    <div className="text-base font-bold text-white mt-1 truncate">{datasetStats.class_count}</div>
                    </div>
                    <div className="bg-zinc-900/50 rounded-lg p-3 border border-zinc-800">
                    <div className="text-[9px] text-zinc-500 uppercase tracking-wider font-medium">Avg Size</div>
                    <div className="text-base font-bold text-white mt-1 truncate">{datasetStats.avg_image_size || 'N/A'}</div>
                    </div>
                    <div className="bg-zinc-900/50 rounded-lg p-3 border border-zinc-800">
                    <div className="text-[9px] text-zinc-500 uppercase tracking-wider font-medium">Splits</div>
                    <div className="text-base font-bold text-white mt-1 truncate">
                      {Object.entries(datasetStats.splits || {}).filter(([, v]: [string, any]) => v > 0).length}
                    </div>
                    </div>
                    </div>
                  
                  {/* Common Sizes */}
                  {datasetStats.common_sizes && datasetStats.common_sizes.length > 0 && (
                    <div className="pt-4 border-t border-zinc-800/50">
                    <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-3 font-semibold">Common Resolutions</div>
                    <div className="flex flex-wrap gap-2">
                      {datasetStats.common_sizes.slice(0, 4).map((s: any, i: number) => (
                      <span key={i} className="text-xs px-3 py-1.5 bg-zinc-900 rounded border border-zinc-800 text-zinc-300 font-mono font-medium hover:border-zinc-700 transition-colors">
                        {s.size}
                      </span>
                      ))}
                    </div>
                    </div>
                  )}
                  </div>
                </details>
                )}

              {/* Model & Hyperparameters Accordion */}
              <details className="group border-2 border-white rounded-xl overflow-hidden bg-black text-white" open>
                <summary className="flex items-center justify-between p-4 cursor-pointer hover:bg-white hover:text-black transition-colors">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4" />
                    <span className="text-sm font-semibold tracking-wide">Model & Hyperparameters</span>
                  </div>
                  <svg className="w-4 h-4 transition-transform group-open:rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" /></svg>
                </summary>
                <div className="p-4 pt-2 space-y-6">
              {/* Model Selection */}
              <div className="space-y-2">
                <label className="text-xs uppercase tracking-wider text-zinc-500 font-semibold">Model Architecture</label>
                <div className="relative">
                  <select 
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="w-full appearance-none bg-black border border-zinc-800 rounded-lg py-2.5 px-3 text-sm text-zinc-300 focus:border-zinc-600 focus:outline-none transition-colors"
                  >
                    <option value="resnet18">ResNet-18 (Standard)</option>
                    <option value="resnet50">ResNet-50 (Deep)</option>
                    <option value="efficientnet_b0">EfficientNet-B0 (Efficient)</option>
                    <option value="dcn">DeepCrossNetwork (DCN)</option>
                    <option value="eva02">EVA-02 (Transformer)</option>
                    <option value="mobilenet_v3">MobileNetV3 (Mobile)</option>
                    <option value="vit_b_16">ViT-B/16 (Vision Transformer)</option>
                    <option value="convnext">ConvNeXt (Modern ConvNet)</option>
                  </select>
                  <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none">
                    <svg className="w-4 h-4 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                  </div>
                </div>
              </div>

              {/* Parameters Grid */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-xs uppercase tracking-wider text-zinc-500 font-semibold">Epochs</label>
                  <input 
                    type="number" 
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
                    className="w-full bg-black border border-zinc-800 rounded-lg py-2.5 px-3 text-sm text-zinc-300 focus:border-zinc-600 focus:outline-none transition-colors font-mono"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs uppercase tracking-wider text-zinc-500 font-semibold">Batch Size</label>
                  <input 
                    type="number" 
                    value={batchSize}
                    onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
                    className="w-full bg-black border border-zinc-800 rounded-lg py-2.5 px-3 text-sm text-zinc-300 focus:border-zinc-600 focus:outline-none transition-colors font-mono"
                  />
                </div>
                <div className="space-y-2 col-span-2">
                  <label className="text-xs uppercase tracking-wider text-zinc-500 font-semibold flex items-center justify-between">
                    <span>DataLoader Workers</span>
                    <span className="text-zinc-600 normal-case font-normal">{numWorkers === -1 ? 'auto-detect' : numWorkers === 0 ? 'disabled (sync)' : `${numWorkers} workers`}</span>
                  </label>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min={-1}
                      max={8}
                      step={1}
                      value={numWorkers}
                      onChange={(e) => setNumWorkers(parseInt(e.target.value))}
                      className="flex-1 accent-white h-1.5 bg-zinc-800 rounded-full appearance-none cursor-pointer"
                    />
                    <input
                      type="number"
                      min={-1}
                      max={16}
                      value={numWorkers}
                      onChange={(e) => setNumWorkers(parseInt(e.target.value))}
                      className="w-16 bg-black border border-zinc-800 rounded-lg py-2.5 px-3 text-sm text-zinc-300 focus:border-zinc-600 focus:outline-none transition-colors font-mono text-center"
                    />
                  </div>
                </div>
              </div>

              {/* Learning Rate */}
              <div className="space-y-2">
                <label className="text-xs uppercase tracking-wider text-zinc-500 font-semibold">Learning Rate</label>
                <input 
                  type="number" 
                  step="0.0001"
                  min="0.00001"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.001)}
                  className="w-full bg-black border border-zinc-800 rounded-lg py-2.5 px-3 text-sm text-zinc-300 focus:border-zinc-600 focus:outline-none transition-colors font-mono"
                />
                <div className="flex gap-2 mt-2">
                  {[0.01, 0.001, 0.0001, 0.00001].map((lr) => (
                    <button
                      key={lr}
                      onClick={() => setLearningRate(lr)}
                      className={`text-xs px-2 py-1 rounded border transition-colors ${
                        learningRate === lr 
                          ? 'bg-white text-black border-white' 
                          : 'bg-zinc-900 text-zinc-400 border-zinc-700 hover:border-zinc-500'
                      }`}
                    >
                      {lr}
                    </button>
                  ))}
                </div>
              </div>
                </div>
              </details>

              {/* Advanced Settings Accordion */}
              <details className="group border-2 border-white rounded-xl overflow-hidden bg-black text-white">
                <summary className="flex items-center justify-between p-4 cursor-pointer hover:bg-white hover:text-black transition-colors">
                  <div className="flex items-center gap-2">
                    <Save className="w-4 h-4" />
                    <span className="text-sm font-semibold tracking-wide">Advanced Settings</span>
                  </div>
                  <svg className="w-4 h-4 transition-transform group-open:rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" /></svg>
                </summary>
                <div className="p-4 pt-2 space-y-6">
              {/* Save Path */}
              <div className="space-y-2">
                <label className="text-xs uppercase tracking-wider text-zinc-500 font-semibold">Checkpoints Output</label>
                <div className="flex gap-2">
                  <input 
                    type="text" 
                    value={savePath}
                    readOnly
                    placeholder="Defaults to dataset dir..."
                    className="w-full bg-black border border-zinc-800 rounded-lg py-2.5 px-3 text-sm text-zinc-300 focus:border-zinc-600 focus:outline-none transition-colors font-mono"
                  />
                  <button 
                     onClick={handlePickSavePath}
                     className="p-2.5 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 rounded-lg text-zinc-400 hover:text-white transition-colors"
                  >
                    <Save className="w-5 h-5" />
                  </button>
                </div>
              </div>

              {/* Early Stopping & Resume */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-xs uppercase tracking-wider text-zinc-500 font-semibold flex items-center justify-between">
                    <span>Early-Stop Patience</span>
                    <span className="text-zinc-600 normal-case font-normal">{patience} ep</span>
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={patience}
                    onChange={(e) => setPatience(parseInt(e.target.value) || 1)}
                    className="w-full bg-black border border-zinc-800 rounded-lg py-2.5 px-3 text-sm text-zinc-300 focus:border-zinc-600 focus:outline-none transition-colors font-mono"
                  />
                </div>
                <div className="space-y-2 flex flex-col justify-end">
                  <span className="text-xs text-zinc-600">Stops if val loss doesn't improve</span>
                </div>
              </div>

              {/* Resume Checkpoint */}
              <div className="space-y-2">
                <label className="text-xs uppercase tracking-wider text-zinc-500 font-semibold">Resume from Checkpoint</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={resumePath}
                    readOnly
                    placeholder="Optional: select a .pth checkpoint..."
                    className="w-full bg-black border border-zinc-800 rounded-lg py-2.5 px-3 text-sm text-zinc-300 focus:border-zinc-600 focus:outline-none transition-colors font-mono"
                  />
                  <button
                    onClick={handlePickResumePath}
                    title="Pick checkpoint file"
                    className="p-2.5 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 rounded-lg text-zinc-400 hover:text-white transition-colors"
                  >
                    <FolderOpen className="w-5 h-5" />
                  </button>
                  {resumePath && (
                    <button
                      onClick={() => setResumePath('')}
                      title="Clear"
                      className="p-2.5 bg-zinc-900 hover:bg-red-900/30 border border-zinc-800 rounded-lg text-zinc-500 hover:text-red-400 transition-colors text-xs font-mono"
                    >
                      ✕
                    </button>
                  )}
                </div>
                {resumePath && (
                  <label className="flex items-center justify-between cursor-pointer group mt-4">
                    <span className="text-sm text-zinc-400 group-hover:text-zinc-200 transition-colors">Test Model Directly (Skip Training)</span>
                    <div className={cn("w-10 h-6 rounded-full border flex items-center px-1 transition-all", evaluateOnly ? "bg-white border-white justify-end" : "bg-zinc-900 border-zinc-700 justify-start")}>
                      <input type="checkbox" className="hidden" checked={evaluateOnly} onChange={e => setEvaluateOnly(e.target.checked)} />
                      <div className={cn("w-3.5 h-3.5 rounded-full transition-colors", evaluateOnly ? "bg-black" : "bg-zinc-500")} />
                    </div>
                  </label>
                )}
              </div>

              {/* Toggles */}
              <div className="pt-6 border-t border-zinc-800/50 space-y-4">
                <label className="flex items-center justify-between cursor-pointer group">
                  <span className="text-sm text-zinc-400 group-hover:text-zinc-200 transition-colors">Generate Dataset Archive</span>
                  <div className={cn("w-10 h-6 rounded-full border flex items-center px-1 transition-all", zipDataset ? "bg-white border-white justify-end" : "bg-zinc-900 border-zinc-700 justify-start")}>
                    <input type="checkbox" className="hidden" checked={zipDataset} onChange={e => setZipDataset(e.target.checked)} />
                    <div className={cn("w-3.5 h-3.5 rounded-full transition-colors", zipDataset ? "bg-black" : "bg-zinc-500")} />
                  </div>
                </label>

                <label className="flex items-center justify-between cursor-pointer group">
                  <span className="text-sm text-zinc-400 group-hover:text-zinc-200 transition-colors">Archive Only Mode</span>
                  <div className={cn("w-10 h-6 rounded-full border flex items-center px-1 transition-all", onlyZip ? "bg-white border-white justify-end" : "bg-zinc-900 border-zinc-700 justify-start")}>
                    <input type="checkbox" className="hidden" checked={onlyZip} onChange={e => setOnlyZip(e.target.checked)} />
                    <div className={cn("w-3.5 h-3.5 rounded-full transition-colors", onlyZip ? "bg-black" : "bg-zinc-500")} />
                  </div>
                </label>
              </div>
                </div>
              </details>

              {/* Data Augmentation Accordion */}
              <details className="group border-2 border-white rounded-xl overflow-hidden bg-black text-white">
                <summary className="flex items-center justify-between p-4 cursor-pointer hover:bg-white hover:text-black transition-colors">
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4" />
                    <span className="text-sm font-semibold tracking-wide">Data Augmentation</span>
                  </div>
                  <svg className="w-4 h-4 transition-transform group-open:rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" /></svg>
                </summary>
                <div className="p-4 pt-2 space-y-4">
                  <p className="text-xs text-zinc-500">Applies random transformations during training only.</p>
  {/* Horizontal Flip */}
  <label className="flex items-center justify-between cursor-pointer group">
    <span className="text-sm text-zinc-400 group-hover:text-zinc-200 transition-colors">
      Random Horizontal Flip
    </span>
    <div
  className={cn(
    "w-10 h-6 rounded-full border flex items-center px-1 transition-all",
    augmentationConfig.horizontalFlip
      ? "bg-white border-white justify-end"
      : "bg-zinc-900 border-zinc-700 justify-start"
  )}
>
  <input
    type="checkbox"
    className="hidden"
    checked={augmentationConfig.horizontalFlip}
    onChange={(e) =>
      setAugmentationConfig({
        ...augmentationConfig,
        horizontalFlip: e.target.checked,
      })
    }
  />
  <div
    className={cn(
      "w-3.5 h-3.5 rounded-full transition-colors",
      augmentationConfig.horizontalFlip ? "bg-black" : "bg-zinc-500"
    )}
  />
</div>
  </label>

  {/* Vertical Flip */}
<label className="flex items-center justify-between cursor-pointer group">
  <span className="text-sm text-zinc-400 group-hover:text-zinc-200 transition-colors">
    Random Vertical Flip
  </span>

  <div
    className={cn(
      "w-10 h-6 rounded-full border flex items-center px-1 transition-all",
      augmentationConfig.verticalFlip
        ? "bg-white border-white justify-end"
        : "bg-zinc-900 border-zinc-700 justify-start"
    )}
  >
    <input
      type="checkbox"
      className="hidden"
      checked={augmentationConfig.verticalFlip}
      onChange={(e) =>
        setAugmentationConfig({
          ...augmentationConfig,
          verticalFlip: e.target.checked,
        })
      }
    />
    <div
      className={cn(
        "w-3.5 h-3.5 rounded-full transition-colors",
        augmentationConfig.verticalFlip ? "bg-black" : "bg-zinc-500"
      )}
    />
  </div>
</label>
{/* Rotation */}
<div className="space-y-3">

  {/* Enable Rotation */}
<label className="flex items-center justify-between cursor-pointer group">
  <span className="text-sm text-zinc-400 group-hover:text-zinc-200 transition-colors">
    Enable Rotation
  </span>

  <div
    className={cn(
      "w-10 h-6 rounded-full border flex items-center px-1 transition-all",
      augmentationConfig.rotation.enabled
        ? "bg-white border-white justify-end"
        : "bg-zinc-900 border-zinc-700 justify-start"
    )}
  >
    <input
      type="checkbox"
      className="hidden"
      checked={augmentationConfig.rotation.enabled}
      onChange={(e) =>
        setAugmentationConfig({
          ...augmentationConfig,
          rotation: {
            ...augmentationConfig.rotation,
            enabled: e.target.checked,
          },
        })
      }
    />
    <div
      className={cn(
        "w-3.5 h-3.5 rounded-full transition-colors",
        augmentationConfig.rotation.enabled ? "bg-black" : "bg-zinc-500"
      )}
    />
  </div>
</label>

  <div>
    <label className="text-xs text-zinc-500">Degrees</label>
    <input
      type="number"
      min={0}
      max={90}
      value={augmentationConfig.rotation.degrees}
      disabled={!augmentationConfig.rotation.enabled}
      onChange={(e) =>
        setAugmentationConfig({
          ...augmentationConfig,
          rotation: {
            ...augmentationConfig.rotation,
            degrees: parseInt(e.target.value) || 0,
          },
        })
      }
      className="w-full bg-black border border-zinc-800 rounded-lg py-2 px-3 text-sm text-zinc-300 disabled:opacity-40 disabled:cursor-not-allowed"
    />
  </div>

</div>
  
                </div>
              </details>

              {/* AutoML Hyperparameter Sweep Accordion */}
              <details className="group border-2 border-white rounded-xl overflow-hidden bg-black text-white">
                <summary className="flex items-center justify-between p-4 cursor-pointer hover:bg-white hover:text-black transition-colors">
                  <div className="flex items-center gap-2">
                    <Zap className="w-4 h-4" />
                    <span className="text-sm font-semibold tracking-wide">AutoML Hyperparameter Sweep</span>
                  </div>
                  <svg className="w-4 h-4 transition-transform group-open:rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" /></svg>
                </summary>
                <div className="p-4 pt-2 space-y-4">
                  <p className="text-xs text-zinc-500">Automatically find the best learning rate, batch size, and optimizer using Optuna.</p>

                <div className="flex items-center gap-3">
                  <div className="flex-1 space-y-1">
                    <label className="text-xs text-zinc-500">Trials</label>
                    <input
                      type="number"
                      min={3}
                      max={50}
                      value={autoMLTrialCount}
                      onChange={(e) => setAutoMLTrialCount(Math.max(3, parseInt(e.target.value) || 10))}
                      disabled={isAutoMLRunning}
                      className="w-full bg-black border border-zinc-800 rounded-lg py-2 px-3 text-sm text-zinc-300 focus:border-zinc-600 focus:outline-none transition-colors font-mono disabled:opacity-50"
                    />
                  </div>
                  <div className="flex-1 pt-5">
                    {isAutoMLRunning ? (
                      <button
                        onClick={() => {
                          autoMLChildRef.current?.kill();
                          addLog('[AutoML] Sweep stopped by user.', 'error');
                          setIsAutoMLRunning(false);
                        }}
                        className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-semibold transition-all bg-red-900/20 hover:bg-red-900/30 text-red-200 border border-red-900/30"
                      >
                        <Square className="w-4 h-4 fill-current" /> Stop Sweep
                      </button>
                    ) : (
                      <button
                        onClick={startAutoML}
                        disabled={isRunning || !datasetPath}
                        className={cn(
                          "w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-semibold transition-all bg-white text-black hover:bg-zinc-200 shadow-lg shadow-white/5",
                          (isRunning || !datasetPath) && "opacity-40 cursor-not-allowed"
                        )}
                      >
                        <Search className="w-4 h-4" /> Find Optimal Config
                      </button>
                    )}
                  </div>
                </div>

                {/* AutoML Progress */}
                {(isAutoMLRunning || autoMLTrials.length > 0) && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-xs text-zinc-500">
                      <span>Progress</span>
                      <span className="font-mono">{autoMLProgress}%</span>
                    </div>
                    <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-white transition-all duration-500 ease-out"
                        style={{ width: `${autoMLProgress}%` }}
                      />
                    </div>

                    {/* Trial Results */}
                    {autoMLTrials.length > 0 && (
                      <div className="max-h-48 overflow-y-auto rounded-lg border border-zinc-800 bg-black/50 scrollbar-thin">
                        <table className="w-full text-xs">
                          <thead className="sticky top-0 bg-zinc-900/95 backdrop-blur-sm">
                            <tr>
                              <th className="px-3 py-2 text-left text-zinc-500 font-medium">#</th>
                              <th className="px-3 py-2 text-left text-zinc-500 font-medium">LR</th>
                              <th className="px-3 py-2 text-left text-zinc-500 font-medium">BS</th>
                              <th className="px-3 py-2 text-left text-zinc-500 font-medium">Opt</th>
                              <th className="px-3 py-2 text-right text-zinc-500 font-medium">Acc</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-zinc-800/50">
                            {autoMLTrials.map((t, i) => {
                              const isBest = autoMLBestParams &&
                                t.params.learning_rate === autoMLBestParams.learning_rate &&
                                t.params.batch_size === autoMLBestParams.batch_size &&
                                t.params.optimizer === autoMLBestParams.optimizer;
                              return (
                                <tr key={i} className={cn(
                                  "transition-colors",
                                  isBest ? "bg-white/5" : "hover:bg-zinc-900/50"
                                )}>
                                  <td className="px-3 py-2 text-zinc-400 font-mono">
                                    {isBest && <span className="mr-1">●</span>}{t.trial}
                                  </td>
                                  <td className="px-3 py-2 text-zinc-300 font-mono">{t.params.learning_rate.toExponential(2)}</td>
                                  <td className="px-3 py-2 text-zinc-300 font-mono">{t.params.batch_size}</td>
                                  <td className="px-3 py-2 text-zinc-300">{t.params.optimizer}</td>
                                  <td className={cn("px-3 py-2 text-right font-mono font-semibold", isBest ? "text-emerald-400" : "text-zinc-300")}>
                                    {(t.val_accuracy * 100).toFixed(2)}%
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    )}

                    {/* Best Result Banner */}
                    {autoMLBestParams && (
                      <div className="p-3 rounded-lg bg-zinc-900/50 border border-zinc-800">
                        <div className="flex items-center gap-2 mb-1">
                          <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                          <span className="text-xs font-semibold text-zinc-300">Best Config Applied</span>
                        </div>
                        <p className="text-xs text-zinc-400">
                          LR: <span className="text-zinc-300 font-mono font-semibold">{autoMLBestParams.learning_rate.toExponential(2)}</span> · 
                          Batch: <span className="text-zinc-300 font-mono font-semibold">{autoMLBestParams.batch_size}</span> · 
                          Optimizer: <span className="text-zinc-300 font-mono font-semibold">{autoMLBestParams.optimizer}</span>
                        </p>
                      </div>
                    )}
                  </div>
                )}
                </div>
              </details>
            </div>
          </div>
        </section>

        {/* Right Panel: Output */}
        <section className="lg:col-span-8 flex flex-col gap-6 lg:sticky lg:top-[120px] lg:h-[calc(100vh-150px)]">
            
           {/* Progress Widget */}
           {(isRunning || progress > 0) && (
             <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-2xl p-6 backdrop-blur-sm">
                <div className="flex justify-between items-end mb-4">
                    <div>
                        <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-widest">Training Progress</h3>
                        <p className="text-2xl font-light text-white mt-1">
                           {isRunning ? `Epoch ${currentStatus?.epoch || 0} / ${currentStatus?.total_epochs || epochs}` : "Complete"}
                        </p>
                        <p className="text-xs text-zinc-500 mt-2 font-mono">
                          Elapsed: {formatTime(elapsedSeconds)}
                        </p>
                        {currentStatus?.learning_rate && (
                          <p className="text-xs text-zinc-500 mt-1">Learning Rate: {currentStatus.learning_rate}</p>
                        )}
                    </div>
                    <div className="text-right flex flex-col items-end">
   <div className="text-3xl font-bold font-mono text-white tracking-tighter">
     {currentStatus?.train_accuracy
  ? `${(parseFloat(currentStatus.train_accuracy)*100).toFixed(2)}%`
  : "0.00%"}
   </div>
   <div className="text-xs text-zinc-500 uppercase tracking-wider mt-1">
     Accuracy
   </div>

   {!isRunning && chartData.length > 0 && (
     <button
       onClick={exportAsJson}
       className="mt-3 flex items-center gap-2 px-4 py-2 bg-white text-black hover:bg-zinc-200 text-xs font-semibold rounded-full transition-all shadow"
     >
       <Download className="w-3 h-3" />
       Export JSON
     </button>
   )}</div> 
                </div>
                <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                    <div className="h-full bg-white transition-all duration-500 ease-out" style={{ width: `${progress}%` }} />
                </div>
             </div>
           )}

           {/* Content Tabs */}
           <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-2xl flex-1 flex flex-col overflow-hidden backdrop-blur-sm">
              <div className="flex border-b border-zinc-800/50 px-2 flex-none">
                 <button 
                   onClick={() => setActiveTab('logs')}
                   className={cn("px-6 py-4 text-sm font-medium border-b-2 transition-all flex items-center gap-2", activeTab === 'logs' ? "border-white text-white" : "border-transparent text-zinc-500 hover:text-zinc-300")}
                 >
                   <Terminal className="w-4 h-4" /> Logs
                 </button>
                 <button 
                   onClick={() => setActiveTab('charts')}
                   className={cn("px-6 py-4 text-sm font-medium border-b-2 transition-all flex items-center gap-2", activeTab === 'charts' ? "border-white text-white" : "border-transparent text-zinc-500 hover:text-zinc-300")}
                 >
                   <BarChart2 className="w-4 h-4" /> Metrics
                 </button>
                 <button 
                   onClick={() => setActiveTab('results')}
                   className={cn("px-6 py-4 text-sm font-medium border-b-2 transition-all flex items-center gap-2", activeTab === 'results' ? "border-white text-white" : "border-transparent text-zinc-500 hover:text-zinc-300")}
                 >
                   <CheckCircle className="w-4 h-4" /> Results
                 </button>
                 <button id="tab-data"
                   onClick={() => setActiveTab('data')}
                   className={cn("px-6 py-4 text-sm font-medium border-b-2 transition-all flex items-center gap-2", activeTab === 'data' ? "border-white text-white" : "border-transparent text-zinc-500 hover:text-zinc-300")}
                 >
                   <Database className="w-4 h-4" /> Data
                 </button>
                 <button 
                onClick={() => setActiveTab('system')}
                className={cn(
                  "px-6 py-4 text-sm font-medium border-b-2 transition-all flex items-center gap-2",
                  activeTab === 'system'
                    ? "border-white text-white"
                    : "border-transparent text-zinc-500 hover:text-zinc-300"
                )}
              >
                <Cpu className="w-4 h-4" /> System
              </button>
              <button
              onClick={() => setActiveTab('compare')}
              className={cn(
                "px-6 py-4 text-sm font-medium border-b-2 transition-all flex items-center gap-2",
                activeTab === 'compare'
                  ? "border-white text-white"
                  : "border-transparent text-zinc-500 hover:text-zinc-300"
              )}
            >
              <BarChart2 className="w-4 h-4" /> Compare
            </button>
            <button
              onClick={() => setActiveTab('insights')}
              className={cn(
                "px-6 py-4 text-sm font-medium border-b-2 transition-all flex items-center gap-2",
                activeTab === 'insights'
                  ? "border-white text-white"
                  : "border-transparent text-zinc-500 hover:text-zinc-300"
              )}
            >
              <Activity className="w-4 h-4" /> Insights
            </button>
            <button
              onClick={() => setActiveTab('runs')}
              className={cn(
                "px-6 py-4 text-sm font-medium border-b-2 transition-all flex items-center gap-2",
                activeTab === 'runs'
                  ? "border-white text-white"
                  : "border-transparent text-zinc-500 hover:text-zinc-300"
              )}
            >
              <History className="w-4 h-4" /> Runs
            </button>
              </div>

              <div className="flex-1 overflow-y-auto bg-black/50 scrollbar-thin">
                 {activeTab === 'logs' && (
                   <div className="p-6 font-mono text-xs space-y-1.5 h-full">
                      {logs.length === 0 && <div className="text-zinc-600 italic text-center mt-20">Waiting for process logs...</div>}
                      {logs.map((log, i) => (
                        <div key={i} className="flex gap-3 text-zinc-300 border-l-2 border-transparent hover:bg-zinc-900/50 pl-2 -ml-2 py-0.5 rounded transition-colors group">
                           <span className="text-zinc-600 select-none w-20 shrink-0 group-hover:text-zinc-500 transition-colors">{log.timestamp}</span>
                           <span className={cn(
                             "break-all",
                             log.type === 'error' ? 'text-red-400' : 
                             log.type === 'success' ? 'text-emerald-400' : 'text-zinc-300'
                           )}>
                             {log.type === 'error' && <span className="mr-2 text-red-500 font-bold">ERR</span>}
                             {log.message}
                           </span>
                        </div>
                      ))}
                      <div ref={logsEndRef} />
                   </div>
                 )}

                 {activeTab === 'charts' && (
                    <div className="p-8 h-full flex flex-col gap-10">
                       {chartData.length > 0 ? (
                         <>
                           <div className="h-[250px] w-full">
                              <div className="flex justify-between mb-4">
                                <h4 className="text-sm font-medium text-zinc-400 uppercase tracking-widest">Model Accuracy</h4>
                                <span className="text-xs text-zinc-600 font-mono">HIGHEST: {Math.max(...chartData.map(d => d.accuracy)).toFixed(4)}</span>
                              </div>
                              <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData}>
                                  <CartesianGrid strokeDasharray="3 3" stroke={isLightMode ? '#e4e4e7' : '#52525b'} vertical={false} />
                                  <XAxis dataKey="epoch" stroke={isLightMode ? '#52525b' : '#a1a1aa'} tick={{fontSize: 12}} tickLine={false} axisLine={false} dy={10} />
                                  <YAxis stroke={isLightMode ? '#52525b' : '#a1a1aa'} tick={{fontSize: 12}} tickLine={false} axisLine={false} dx={-10} domain={[0, 1]} />
                                  <Tooltip 
                                    contentStyle={{ backgroundColor: isLightMode ? '#fff' : '#09090b', borderColor: isLightMode ? '#e4e4e7' : '#27272a', borderRadius: '8px', color: isLightMode ? '#000' : '#fff' }} 
                                    itemStyle={{ color: isLightMode ? '#000' : '#fff' }}
                                  />
                                  <Line type="monotone" dataKey="accuracy" stroke={isLightMode ? '#000' : '#fff'} strokeWidth={2} dot={false} activeDot={{ r: 4, fill: isLightMode ? '#000' : '#fff' }} />
                                </LineChart>
                              </ResponsiveContainer>
                           </div>
                           <div className="h-[250px] w-full">
                              <div className="flex justify-between mb-4">
                                <h4 className="text-sm font-medium text-zinc-400 uppercase tracking-widest">Loss Function</h4>
                                <span className="text-xs text-zinc-600 font-mono">LOWEST: {Math.min(...chartData.map(d => d.loss)).toFixed(4)}</span>
                              </div>
                              <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData}>
                                  <CartesianGrid strokeDasharray="3 3" stroke={isLightMode ? '#e4e4e7' : '#52525b'} vertical={false} />
                                  <XAxis dataKey="epoch" stroke={isLightMode ? '#52525b' : '#a1a1aa'} tick={{fontSize: 12}} tickLine={false} axisLine={false} dy={10} />
                                  <YAxis stroke={isLightMode ? '#52525b' : '#a1a1aa'} tick={{fontSize: 12}} tickLine={false} axisLine={false} dx={-10} />
                                  <Tooltip 
                                    contentStyle={{ backgroundColor: isLightMode ? '#fff' : '#09090b', borderColor: isLightMode ? '#e4e4e7' : '#27272a', borderRadius: '8px', color: isLightMode ? '#000' : '#fff' }} 
                                    itemStyle={{ color: isLightMode ? '#000' : '#fff' }}
                                  />
                                  <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} dot={false} activeDot={{ r: 4, fill: '#ef4444' }} />
                                </LineChart>
                              </ResponsiveContainer>
                           </div>
                         </>
                       ) : (
                         <div className="h-full flex flex-col items-center justify-center text-zinc-600">
                            <BarChart2 className="w-16 h-16 mb-4 opacity-10" />
                            <p className="font-light">Awaiting training metrics...</p>
                         </div>
                       )}
                    </div>
                 )}

                 {activeTab === 'results' && (
                    <div className="p-8">
                       {evalResult ? (
                         <div className="space-y-10">
                            {/* Summary Stats */}
                            <div>
                              <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-widest mb-6">Performance Summary</h3>
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                 {/* Stat Cards */}
                                 {[
                                   { label: "Total Epochs", value: evalResult.total_epochs },
                                   { label: "Test Set Size", value: evalResult.test_size },
                                   { label: "Precision (Avg)", value: evalResult.report['weighted avg']?.precision.toFixed(4) ?? 'N/A', color: 'text-zinc-100' },
                                   { label: "Recall (Avg)", value: evalResult.report['weighted avg']?.recall.toFixed(4) ?? 'N/A', color: 'text-zinc-100' },
                                 ].map((stat, i) => (
                                   <div key={i} className="p-5 bg-zinc-900/50 rounded-xl border border-zinc-800/50">
                                      <div className="text-xs text-zinc-500 uppercase tracking-wider mb-2">{stat.label}</div>
                                      <div className={cn("text-2xl font-bold font-mono", stat.color || "text-white")}>{stat.value}</div>
                                   </div>
                                 ))}
                              </div>
                            </div>
                            
                            <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                              {/* Left: CM */}
                              <div>
                                 <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-widest mb-6">Confusion Matrix</h3>
                                 {evalResult && evalResult.confusion_matrix_path && (
                                    <div className="rounded-xl border border-zinc-800 bg-black p-4">
                                       {matrixImageUrl && !imageLoadError ? (
                                         <div className="bg-white/5 rounded-lg overflow-hidden border border-white/5 mb-4">
                                            <img 
                                              src={matrixImageUrl} 
                                              alt="Confusion Matrix" 
                                              className="w-full h-auto object-contain mix-blend-screen" // Blend helpful for black bg if image has black bg
                                              onError={(e) => {
                                                 console.error('Failed to load confusion matrix:', evalResult.confusion_matrix_path);
                                                 setImageLoadError(true);
                                              }}
                                            />
                                         </div>
                                       ) : imageLoadError ? (
                                         <div className="flex flex-col items-center justify-center p-12 bg-zinc-900 rounded-lg border border-zinc-800 text-zinc-500 mb-4">
                                            <AlertCircle className="w-8 h-8 mb-3 opacity-50" />
                                            <p className="text-sm">Image load failed</p>
                                         </div>
                                       ) : null}
                                       
                                       <div className="flex items-center justify-between gap-4">
                                          <p className="text-xs text-zinc-600 font-mono truncate flex-1" title={evalResult.confusion_matrix_path}>
                                            {evalResult.confusion_matrix_path.split(/[\\/]/).pop()}
                                          </p>
                                          <button
                                            onClick={() => {
                                              if (matrixImageUrl) {
                                                const link = document.createElement('a');
                                                link.href = matrixImageUrl;
                                                link.download = `confusion_matrix_${evalResult.total_epochs}_epochs.png`;
                                                document.body.appendChild(link);
                                                link.click();
                                                document.body.removeChild(link);
                                                addLog('Downloading image...', 'success');
                                              } else {
                                                addLog('Image URL not available', 'error');
                                              }
                                            }}
                                            className="px-4 py-2 bg-white text-black hover:bg-zinc-200 text-xs font-semibold rounded-lg transition-all flex items-center gap-2"
                                          >
                                            <Download className="w-3 h-3" /> Download
                                          </button>
                                       </div>
                                    </div>
                                 )}
                              </div>

                              {/* Right: Detailed Report */}
                              <div>
                                <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-widest mb-6">Classification Report</h3>
                                <div className="overflow-auto max-h-[500px] rounded-xl border border-zinc-800 bg-zinc-900/20 scrollbar-thin">
                                   <table className="w-full min-w-[600px] text-sm text-left relative border-collapse">
                                      <thead className="sticky top-0 z-10 bg-black/90 backdrop-blur-md shadow-sm">
                                         <tr>
                                            <th className="px-5 py-4 whitespace-nowrap text-xs font-semibold text-zinc-400 uppercase tracking-wider border-b border-zinc-800">Class</th>
                                            <th className="px-5 py-4 whitespace-nowrap text-xs font-semibold text-zinc-400 uppercase tracking-wider border-b border-zinc-800 text-right">Precision</th>
                                            <th className="px-5 py-4 whitespace-nowrap text-xs font-semibold text-zinc-400 uppercase tracking-wider border-b border-zinc-800 text-right">Recall</th>
                                            <th className="px-5 py-4 whitespace-nowrap text-xs font-semibold text-zinc-400 uppercase tracking-wider border-b border-zinc-800 text-right">F1-Score</th>
                                            <th className="px-5 py-4 whitespace-nowrap text-xs font-semibold text-zinc-400 uppercase tracking-wider border-b border-zinc-800 text-right">Support</th>
                                         </tr>
                                      </thead>
                                      <tbody className="divide-y divide-zinc-800/50">
                                          {Object.entries(evalResult.report).map(([key, val]: [string, any]) => {
                                             if (typeof val !== 'object') return null;
                                             return (
                                               <tr key={key} className="group hover:bg-white/5 transition-colors">
                                                  <td className="px-5 py-3 font-mono text-zinc-300 font-medium whitespace-nowrap group-hover:text-white border-r border-dashed border-zinc-800/30">{key}</td>
                                                  <td className="px-5 py-3 whitespace-nowrap text-right font-mono text-zinc-400">{val.precision?.toFixed(4)}</td>
                                                  <td className="px-5 py-3 whitespace-nowrap text-right font-mono text-zinc-400">{val.recall?.toFixed(4)}</td>
                                                  <td className="px-5 py-3 whitespace-nowrap text-right font-mono text-zinc-300 font-semibold">{val.f1_score?.toFixed(4) ?? val['f1-score']?.toFixed(4)}</td>
                                                  <td className="px-5 py-3 whitespace-nowrap text-right font-mono text-zinc-500">{val.support}</td>
                                               </tr>
                                             );
                                          })}
                                      </tbody>
                                   </table>
                                </div>
                              </div>
                             </div>
                         </div>
                       ) : (
                         <div className="h-full flex flex-col items-center justify-center text-zinc-600">
                            <p className="font-light">Train a model to see evaluation metrics.</p>
                         </div>
                       )}
                    </div>
                 )}

                 {activeTab === 'data' && (
                   <div className="p-6 overflow-y-auto space-y-4">
                     <div>
                       <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-widest mb-2">Tabular Data Processor</h3>
                       <p className="text-xs text-zinc-500 mb-4">Load or transform CSV / Excel files via <code className="bg-zinc-900 px-1.5 py-0.5 rounded text-zinc-300">tabular_processor.py</code></p>
                     </div>
                     <div className="flex gap-2">
                       <input id="data-file-path" type="text" value={tabFile} onChange={e => setTabFile(e.target.value)}
                         placeholder="Select CSV / Excel file…"
                         className="flex-1 bg-black border border-zinc-800 rounded-lg py-2.5 px-3 text-sm text-zinc-300 placeholder-zinc-600 focus:border-zinc-500 focus:outline-none font-mono"
                       />
                       <button id="data-pick-file" onClick={pickTabFile}
                         className="p-2.5 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 rounded-lg text-zinc-400 hover:text-white transition-colors"
                       ><FolderOpen className="w-5 h-5" /></button>
                     </div>
                     <div className="relative">
                       <select id="data-action" value={tabAction} onChange={e => setTabAction(e.target.value as TabAction)}
                         className="w-full appearance-none bg-black border border-zinc-800 rounded-lg py-2.5 px-3 text-sm text-zinc-300 focus:border-zinc-500 focus:outline-none"
                       >
                         <option value="load">Load &amp; Preview</option>
                         <option value="drop_missing">Drop Missing Rows</option>
                         <option value="fill_missing">Fill Missing Values</option>
                         <option value="label_encode">Label Encode</option>
                         <option value="one_hot_encode">One-Hot Encode</option>
                       </select>
                       <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none"><svg className="w-4 h-4 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg></div>
                     </div>
                     {tabAction === 'fill_missing' && (
                       <div className="relative">
                         <select id="data-fill-method" value={fillMethod} onChange={e => setFillMethod(e.target.value as FillMethod)}
                           className="w-full appearance-none bg-black border border-zinc-800 rounded-lg py-2.5 px-3 text-sm text-zinc-300 focus:border-zinc-500 focus:outline-none"
                         >
                           <option value="mean">Mean</option>
                           <option value="median">Median</option>
                           <option value="mode">Mode</option>
                           <option value="zero">Zero</option>
                         </select>
                         <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none"><svg className="w-4 h-4 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg></div>
                       </div>
                     )}
                     {(tabAction === 'label_encode' || tabAction === 'one_hot_encode') && (
                       <input id="data-encode-columns" type="text" value={encodeColumns} onChange={e => setEncodeColumns(e.target.value)}
                         placeholder="Columns to encode (comma-separated)"
                         className="w-full bg-black border border-zinc-800 rounded-lg py-2.5 px-3 text-sm text-zinc-300 placeholder-zinc-600 focus:border-zinc-500 focus:outline-none"
                       />
                     )}
                     {tabAction !== 'load' && (
                       <input id="data-out-path" type="text" value={tabOutPath} onChange={e => setTabOutPath(e.target.value)}
                         placeholder="Save output to… (optional)"
                         className="w-full bg-black border border-zinc-800 rounded-lg py-2.5 px-3 text-sm text-zinc-300 placeholder-zinc-600 focus:border-zinc-500 focus:outline-none font-mono"
                       />
                     )}
                     <button id="data-run-btn" onClick={runTabular} disabled={!tabFile || tabLoading}
                       className="flex items-center justify-center gap-2 w-full py-2.5 bg-white text-black hover:bg-zinc-200 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm font-semibold transition-all"
                     >
                       {tabLoading ? <><span className="inline-block w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" /> Processing…</> : 'Run'}
                     </button>
                     {tabResult && (
                       <div>
                         {tabResult.status === 'error' ? (
                           <div className="rounded-xl border border-red-900/30 bg-red-900/20 p-3 text-red-400 text-xs"><strong>Error:</strong> {tabResult.message}</div>
                         ) : (
                           <div className="space-y-3">
                             <div className="flex flex-wrap gap-2 text-xs">
                               {tabResult.shape && <span className="rounded-full bg-blue-500/10 border border-blue-500/30 px-3 py-1 text-blue-400">Shape: {tabResult.shape[0]} × {tabResult.shape[1]}</span>}
                               {tabResult.message && <span className="rounded-full bg-emerald-500/10 border border-emerald-500/30 px-3 py-1 text-emerald-400">{tabResult.message}</span>}
                             </div>
                             {tabResult.columns && tabResult.data && (
                               <div className="overflow-x-auto rounded-xl border border-zinc-800">
                                 <table className="min-w-full text-xs">
                                   <thead className="bg-zinc-900"><tr>
                                     {tabResult.columns.map(col => (
                                       <th key={col} className="px-3 py-2 text-left font-semibold text-zinc-300 whitespace-nowrap border-b border-zinc-800">
                                         <div>{col}</div>
                                         {tabResult.dtypes && <div className="font-normal text-zinc-500">{tabResult.dtypes[col]}</div>}
                                       </th>
                                     ))}
                                   </tr></thead>
                                   <tbody className="divide-y divide-zinc-800">
                                     {tabResult.data.map((row, ri) => (
                                       <tr key={ri} className="hover:bg-zinc-900/50 transition-colors">
                                         {row.map((cell, ci) => (
                                           <td key={ci} className={`px-3 py-2 whitespace-nowrap ${cell === null ? 'text-zinc-500 italic' : 'text-zinc-100'}`}>
                                             {cell === null ? 'null' : String(cell)}
                                           </td>
                                         ))}
                                       </tr>
                                     ))}
                                   </tbody>
                                 </table>
                               </div>
                             )}
                           </div>
                         )}
                       </div>
                     )}
                   </div>
                 )}
{activeTab === 'system' && (
  <div className="p-8 space-y-8 animate-in fade-in duration-500">
    {systemLoading && (
      <div className="flex items-center justify-center p-12">
        <span className="inline-block w-8 h-8 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
      </div>
    )}
    

    {systemError && (
      <div className="p-6 bg-red-900/10 border border-red-900/30 rounded-xl text-red-400">
        <div className="flex items-center gap-3 mb-2 text-lg font-semibold">
          <AlertCircle className="w-5 h-5" /> Error Loading System Info
        </div>
        <p className="text-sm opacity-80">{systemError}</p>
      </div>
    )}
    

    {systemInfo && (
      <div className="space-y-8">
         {/* ================= LIVE PERFORMANCE STRIP ================= */}
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

     <div
  className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6 shadow-sm">
  <div className="flex justify-between items-center mb-4">
    <span
      className={cn(
        "text-xs uppercase tracking-wider",
        isLightMode ? "text-zinc-500" : "text-zinc-400"
      )}
    >
      CPU Usage
    </span>
    <span className="text-2xl font-bold">
      {systemInfo.hardware?.cpu_usage_percent ?? 0}%
    </span>
  </div>

  <div
    className={cn(
      "w-full h-2 rounded-full overflow-hidden",
      isLightMode ? "bg-zinc-200" : "bg-zinc-800"
    )}
  >
    <div
      className={cn(
  "h-full transition-all duration-500",
  getUsageColor(systemInfo.hardware?.cpu_usage_percent ?? 0)
)}
      style={{ width: `${systemInfo.hardware?.cpu_usage_percent ?? 0}%` }}
    />
  </div>

  <div
    className={cn(
      "text-xs mt-3",
      isLightMode ? "text-zinc-500" : "text-zinc-400"
    )}
  >
    {systemInfo.hardware?.cpu_freq_mhz ?? "-"} MHz
  </div>
</div>

      {/* RAM CARD */}
      <div className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6 shadow-sm">
  <div className="flex justify-between items-center mb-4">
    <span className="text-xs uppercase tracking-wider text-zinc-500">
      RAM Usage
    </span>
    <span className="text-2xl font-bold">
      {systemInfo.hardware?.ram_used_percent ?? 0}%
    </span>
  </div>

  <div className="w-full h-2 rounded-full overflow-hidden bg-zinc-300 dark:bg-zinc-800">
    <div
      className={cn(
  "h-full transition-all duration-500",
  getUsageColor(systemInfo.hardware?.ram_used_percent ?? 0)
)}
      style={{ width: `${systemInfo.hardware?.ram_used_percent ?? 0}%` }}
    />
  </div>

  <div className="text-xs text-zinc-500 mt-3 space-y-1">
    <div>
      {systemInfo.hardware?.ram_used_gb} GB / {systemInfo.hardware?.ram_total_gb} GB
    </div>
    <div>
      Available: {systemInfo.hardware?.ram_available_gb} GB
    </div>
  </div>
</div>

      {/* DISK CARD */}
       <div
  className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6 shadow-sm">
        <div className="flex justify-between items-center mb-4">
          <span className="text-xs uppercase tracking-wider text-zinc-500">
            Disk Usage
          </span>
          <span className="text-2xl font-bold">
            {systemInfo.hardware?.disk_used_percent ?? 0}%
          </span>
        </div>

        <div className="text-xs text-zinc-500 mt-3 space-y-1">
          <div
  className={cn(
    "w-full h-2 rounded-full overflow-hidden",
    isLightMode ? "bg-zinc-200" : "bg-zinc-800"
  )}
>
  <div
    className={cn(
  "h-full transition-all duration-500",
  getUsageColor(systemInfo.hardware?.disk_used_percent ?? 0)
)}
    style={{ width: `${systemInfo.hardware?.disk_used_percent ?? 0}%` }}
  />
</div>
    <div>
      Free: {systemInfo.hardware?.disk_free_gb} GB
    </div>
    <div>
      Total: {systemInfo.hardware?.disk_total_gb} GB
    </div>
  </div>
      </div>
    </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Hardware Card */}
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-6">
            <Cpu className="w-5 h-5 text-emerald-400" />
            <h3 className="text-lg font-semibold text-zinc-100">Hardware Overview</h3>
          </div>
          <div className="space-y-4">
            <div className="flex justify-between items-center border-b border-zinc-800/50 pb-3">
              <span className="text-zinc-500 text-sm">Processor</span>
              <span className="text-zinc-300 text-sm break-words text-right max-w-[60%] font-medium">{systemInfo.hardware?.cpu || 'Unknown'}</span>
            </div>
            <div className="flex justify-between items-center border-b border-zinc-800/50 pb-3">
              <span className="text-zinc-500 text-sm">Logical Cores</span>
              <span className="text-zinc-300 text-sm font-medium">{systemInfo.hardware?.cores}</span>
            </div>
            <div className="flex justify-between items-center border-b border-zinc-800/50 pb-3">
              <span className="text-zinc-500 text-sm">Total Memory</span>
              <span className="text-zinc-300 text-sm font-medium">{systemInfo.hardware?.ram_total_gb} GB</span>
            </div>
            <div className="flex justify-between items-center pb-1">
              <span className="text-zinc-500 text-sm">Available Memory</span>
              <span className="text-zinc-300 text-sm font-medium">{systemInfo.hardware?.ram_available_gb} GB</span>
            </div>
          </div>
        </div>

        {/* AI & Compute Card */}
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-6">
            <Activity className="w-5 h-5 text-blue-400" />
            <h3 className="text-lg font-semibold text-zinc-100">AI Compute & Python</h3>
          </div>
          <div className="space-y-4">
            <div className="flex justify-between items-center border-b border-zinc-800/50 pb-3">
              <span className="text-zinc-500 text-sm">PyTorch Version</span>
              <span className="text-zinc-300 text-sm font-medium">{systemInfo.torch?.version}</span>
            </div>
            <div className="flex justify-between items-center border-b border-zinc-800/50 pb-3">
              <span className="text-zinc-500 text-sm">CUDA Support</span>
              <span className="text-zinc-300 text-sm font-medium">
                {systemInfo.torch?.cuda_available ? (
                  <span className="text-emerald-400 flex items-center gap-1.5"><CheckCircle className="w-3.5 h-3.5" /> Available ({systemInfo.torch.cuda_version})</span>
                ) : (
                  <span className="text-zinc-500">Not Available</span>
                )}
              </span>
            </div>
            <div className="flex justify-between items-center border-b border-zinc-800/50 pb-3">
              <span className="text-zinc-500 text-sm">Active GPU</span>
              <span className="text-zinc-300 text-sm font-medium max-w-[60%] text-right">{systemInfo.torch?.gpu_name || 'CPU Only'}</span>
            </div>
            <div className="flex justify-between items-center pb-1">
              <span className="text-zinc-500 text-sm">Python Runtime</span>
              <span className="text-zinc-300 text-sm font-medium">v{systemInfo.python?.version}</span>
            </div>
          </div>
        </div>

        {/* Platform Card */}
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6 md:col-span-2 shadow-sm">
          <div className="flex items-center gap-3 mb-6">
            <Terminal className="w-5 h-5 text-purple-400" />
            <h3 className="text-lg font-semibold text-zinc-100">Platform Architecture</h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
             <div className="p-4 bg-zinc-800/20 rounded-xl border border-zinc-800/50">
               <div className="text-zinc-500 text-xs mb-1 uppercase tracking-wider">Operating System</div>
               <div className="text-zinc-300 text-sm font-medium">{systemInfo.platform?.os} {systemInfo.platform?.release}</div>
             </div>
             <div className="p-4 bg-zinc-800/20 rounded-xl border border-zinc-800/50">
               <div className="text-zinc-500 text-xs mb-1 uppercase tracking-wider">Architecture</div>
               <div className="text-zinc-300 text-sm font-medium">{systemInfo.hardware?.architecture}</div>
             </div>
             <div className="p-4 bg-zinc-800/20 rounded-xl border border-zinc-800/50 md:col-span-2 overflow-hidden flex flex-col justify-center">
               <div className="text-zinc-500 text-xs mb-1 uppercase tracking-wider">Python Executable Path</div>
               <div className="text-zinc-400 text-xs font-mono break-all line-clamp-2" title={systemInfo.python?.executable}>{systemInfo.python?.executable}</div>
             </div>
          </div>
        </div>
      </div>
      </div>
    )}
    
  </div>
)}
{activeTab === 'compare' && (
  <div className="p-8 space-y-6">
    {recentExperiments.length < 2 ? (
      <div className="text-center text-zinc-500 mt-20">
        Run at least 2 experiments to compare.
      </div>
    ) : (
      <>
        {/* SELECTORS */}
        <div className="grid grid-cols-2 gap-6">
          <select
            onChange={(e) =>
              setCompareA(
                recentExperiments.find(exp => exp.id === e.target.value) || null
              )
            }
            className="bg-black border border-zinc-800 rounded-lg p-2 text-sm"
          >
            <option value="">Select Experiment A</option>
            {recentExperiments.map(exp => (
              <option key={exp.id} value={exp.id}>
                {exp.model} - {exp.date}
              </option>
            ))}
          </select>

          <select
            onChange={(e) =>
              setCompareB(
                recentExperiments.find(exp => exp.id === e.target.value) || null
              )
            }
            className="bg-black border border-zinc-800 rounded-lg p-2 text-sm"
          >
            <option value="">Select Experiment B</option>
            {recentExperiments.map(exp => (
              <option key={exp.id} value={exp.id}>
                {exp.model} - {exp.date}
              </option>
            ))}
          </select>
        </div>

        {/* GRAPH */}
        {compareData.length > 0 && (
          <div className="h-[350px] w-full mt-6">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={compareData} >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
  dataKey="epoch"
  label={{
    value: "Epoch",
    position: "bottom",
    offset: -10
  }}
/>


<YAxis
  domain={[0, 1]}
  label={{
    value: "Accuracy",
    angle: -90,
    position: "insideLeft"
  }}
/>
                <Tooltip />
                <Legend verticalAlign="bottom"/>
                <Line
  type="monotone"
  dataKey={keyA}
  name={compareA?.model}
  stroke="#22c55e"
  strokeWidth={2}
  dot={false}
/>

<Line
  type="monotone"
  dataKey={keyB}
  name={compareB?.model}
  stroke="#3b82f6"
  strokeWidth={2}
  dot={false}
/>
              </LineChart>
            </ResponsiveContainer>
            
          </div>
        )}
      </>
    )}
  </div>
)}
{activeTab === 'insights' && (
  
  <div className="p-8">

  {!hasMetrics ? (
    <div className="text-center text-zinc-500 mt-20">
      Run training to generate insights.
    </div>
  ) : (
    <>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

        <InsightCard title="Peak Accuracy">
          <span className="text-emerald-400">
            {(peakAccuracy * 100).toFixed(2)}%
          </span>
        </InsightCard>

        <InsightCard title="Best Epoch">
          {bestEpoch}
        </InsightCard>

        <InsightCard title="Lowest Loss">
          <span className="text-red-400">
            {lowestLoss.toFixed(4)}
          </span>
        </InsightCard>

        <InsightCard title="Accuracy Gain">
          <span className={accuracyGain >= 0 ? "text-emerald-400" : "text-red-400"}>
            {(accuracyGain * 100).toFixed(2)}%
          </span>
        </InsightCard>

        <InsightCard title="Training Trend">
          <span className={
            trainingTrend === "Improving"
              ? "text-emerald-400"
              : trainingTrend === "Degrading"
              ? "text-red-400"
              : "text-yellow-400"
          }>
            {trainingTrend}
          </span>
        </InsightCard>

        <InsightCard title="Convergence Epoch">
          {convergenceEpoch ?? "Not Detected"}
        </InsightCard>

        <InsightCard title="Stability Variance">
          {stabilityScore.toFixed(6)}
        </InsightCard>

        <InsightCard title="Efficiency Score">
          {(efficiencyScore * 100).toFixed(4)}%
        </InsightCard>

        <InsightCard title="Overfitting Gap">
          <span className={overfitGap > 0.05 ? "text-red-400" : "text-emerald-400"}>
            {(overfitGap * 100).toFixed(2)}%
          </span>
        </InsightCard>

      </div>

      <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6 mt-8">
        <div className="text-xs text-zinc-500 uppercase tracking-wider mb-3">
          Recommendation
        </div>
        <div className="text-sm text-zinc-300">
          {recommendation}
        </div>
      </div>
    </>
  )}
</div>
)}
{activeTab === 'runs' && (
  <div className="p-8 space-y-6">
    {runs.length === 0 && (
      <div className="text-zinc-500 text-sm">
        No saved runs found.
      </div>
    )}

    {runs.map((run, index) => (
      <div
        key={run.id}
        className={cn(
          "p-6 rounded-2xl border transition-all",
          index === 0
            ? "border-emerald-500 bg-emerald-900/10"
            : "border-zinc-800 bg-zinc-500/40"
        )}
      >
        <div className="flex justify-between items-center">
          <div>
            <div className="text-sm font-mono text-white">
              {run.model}
            </div>
            <div className="text-xs text-zinc-500">
              {run.timestamp}
            </div>
            <div className="text-xs text-zinc-600 mt-1">
              LR: {run.learning_rate} • BS: {run.batch_size} • Epochs: {run.epochs}
            </div>
          </div>

          <div className="text-right">
            <div className="text-2xl font-bold font-mono text-emerald-400">
              {(run.final_validation_accuracy * 100).toFixed(2)}%
            </div>
            <div className="text-xs text-zinc-500">
              Gap: {run.overfitting_gap.toFixed(4)}
            </div>
          </div>
        </div>
      </div>
    ))}
  </div>
)}

              </div>
           </div>
        </section>
      </main>


    </div>
    </>
  );
}

