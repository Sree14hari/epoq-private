// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use futures_util::{SinkExt, StreamExt};
use std::sync::{Arc, Mutex};
use tauri::{Emitter, Manager};
use tauri_plugin_shell::ShellExt;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::process::Command as TokioCommand;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

// ---------------------------------------------------------------------------
// Shared application state
// ---------------------------------------------------------------------------

/// Latest training metrics — updated by the frontend via `update_training_metrics`.
#[derive(Clone, serde::Deserialize, serde::Serialize)]
struct TrainingMetrics {
    epoch: u32,
    loss: f64,
    accuracy: f64,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self { epoch: 0, loss: 0.0, accuracy: 0.0 }
    }
}

/// Handle to the LLM daemon's stdin so any command can write to it.
/// All fields use Arc so they can be mutated after initial `manage()`.
struct LlmDaemon {
    /// Write-end of the daemon's stdin pipe. None when no model is loaded.
    stdin: Arc<tokio::sync::Mutex<Option<tokio::process::ChildStdin>>>,
    /// PID of the running daemon process — used to kill it before reloading.
    child_pid: Arc<Mutex<Option<u32>>>,
    /// Human-readable status string emitted to the frontend.
    /// Possible values: "not_loaded" | "loading" | "ready:<model-name>" | "error:<msg>"
    status: Arc<Mutex<String>>,
    /// Conda env name to use for spawning (empty = system Python).
    conda_env: Arc<Mutex<String>>,
}

struct AppState {
    // WebSocket mobile clients
    clients: Arc<Mutex<Vec<tokio::sync::mpsc::UnboundedSender<Message>>>>,
    token: Arc<Mutex<String>>,
    // Live training metrics (updated by the frontend)
    training_metrics: Arc<Mutex<TrainingMetrics>>,
}

// ---------------------------------------------------------------------------
// Utility: run a one-shot Python script and return stdout
// ---------------------------------------------------------------------------

async fn run_python(app: &tauri::AppHandle, args: &[&str]) -> Result<String, String> {
    let cmds = ["python", "python3", "py"];
    let mut last_err = String::new();

    for cmd in cmds {
        match app.shell().command(cmd).args(args).output().await {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                if output.status.success() {
                    return Ok(stdout);
                } else {
                    last_err = if !stderr.trim().is_empty() { stderr }
                               else if !stdout.trim().is_empty() { stdout }
                               else { format!("Exited with code: {}", output.status.code().unwrap_or(-1)) };
                    continue;
                }
            }
            Err(e) => { last_err = e.to_string(); }
        }
    }
    Err(last_err)
}

// ---------------------------------------------------------------------------
// Existing Tauri commands (unchanged)
// ---------------------------------------------------------------------------

#[tauri::command]
async fn run_tabular_processor(
    app: tauri::AppHandle,
    file: String,
    action: String,
    params: Option<String>,
    out: Option<String>,
) -> Result<String, String> {
    let script_path = app.path().resource_dir().map_err(|e| e.to_string())?
        .join("python_backend").join("tabular_processor.py");
    let script = script_path.to_string_lossy().to_string().replace("\\\\?\\", "");

    let mut args: Vec<String> = vec![script, "--action".into(), action, "--file".into(), file];
    if let Some(p) = params { args.push("--params".into()); args.push(p); }
    if let Some(o) = out   { args.push("--out".into());    args.push(o); }

    let args_ref: Vec<&str> = args.iter().map(String::as_str).collect();
    run_python(&app, &args_ref).await
}

#[tauri::command]
async fn run_check_gpu(app: tauri::AppHandle) -> Result<String, String> {
    let script_path = app.path().resource_dir().map_err(|e| e.to_string())?
        .join("python_backend").join("check_gpu.py");
    let script = script_path.to_string_lossy().to_string().replace("\\\\?\\", "");
    run_python(&app, &[script.as_str()]).await
        .map(|s| s.trim().to_string())
        .map_err(|e| format!("GPU detection failed: {}", e))
}

#[tauri::command]
async fn get_system_info(app: tauri::AppHandle) -> Result<String, String> {
    let script_path = app.path().resource_dir().map_err(|e| e.to_string())?
        .join("python_backend").join("system_info.py");
    let script = script_path.to_string_lossy().to_string().replace("\\\\?\\", "");
    run_python(&app, &[script.as_str()]).await
        .map(|s| s.trim().to_string())
        .map_err(|e| format!("System info failed: {}", e))
}

#[tauri::command]
async fn check_dependencies(app: tauri::AppHandle) -> Result<String, String> {
    println!("DEBUG: Running backend check_dependencies");
    let script = "import sys, json, importlib.util; p = lambda x: importlib.util.find_spec(x) is not None; \
                  print(json.dumps({'python': True, 'executable': sys.executable, 'version': sys.version.split()[0], \
                  'pandas': p('pandas'), 'sklearn': p('sklearn'), 'torch': p('torch'), 'timm': p('timm'), \
                  'optuna': p('optuna'), 'llama_cpp': p('llama_cpp')}))";
    match run_python(&app, &["-c", script]).await {
        Ok(output) => { println!("DEBUG: Python stdout: {}", output); Ok(output.trim().to_string()) }
        Err(e) => {
            println!("DEBUG: Python error: {}", e);
            Ok(format!(
                "{{\"python\": false, \"version\": null, \"pandas\": false, \"sklearn\": false, \
                  \"torch\": false, \"timm\": false, \"optuna\": false, \"llama_cpp\": false, \"error\": \"{}\"}}",
                e.replace('"', "\\\"").replace('\n', " ")
            ))
        }
    }
}

#[tauri::command]
fn fetch_runs(save_path: String) -> Result<String, String> {
    use std::fs;
    use std::path::Path;
    let experiments_dir = Path::new(&save_path).join("experiments");
    if !experiments_dir.exists() { return Ok("[]".to_string()); }

    let mut runs = Vec::new();
    for entry in fs::read_dir(experiments_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            if let Ok(content) = fs::read_to_string(&path) { runs.push(content); }
        }
    }
    Ok(format!("[{}]", runs.join(",")))
}

#[tauri::command]
async fn analyze_dataset(app: tauri::AppHandle, path: String) -> Result<String, String> {
    let script_path = app.path().resource_dir().map_err(|e| e.to_string())?
        .join("python_backend").join("dataset_analyzer.py");
    let script = script_path.to_string_lossy().to_string().replace("\\\\?\\", "");
    run_python(&app, &[script.as_str(), "--path", &path]).await
        .map(|s| s.trim().to_string())
        .map_err(|e| format!("Dataset analysis failed: {}", e))
}

#[tauri::command]
fn get_connection_details(state: tauri::State<AppState>) -> Result<String, String> {
    use local_ip_address::local_ip;
    use std::time::{SystemTime, UNIX_EPOCH};

    let my_local_ip = local_ip().map_err(|e| format!("Failed to get local IP: {}", e))?.to_string();
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
    let token = (now % 900_000) + 100_000;
    *state.token.lock().unwrap() = token.to_string();

    Ok(format!("{{\"ip\": \"{}\", \"port\": 8765, \"token\": \"{}\"}}", my_local_ip, token))
}

#[tauri::command]
fn broadcast_log(log: String, state: tauri::State<AppState>) {
    let mut clients = state.clients.lock().unwrap();
    clients.retain(|client| client.send(Message::Text(log.clone().into())).is_ok());
}

// ---------------------------------------------------------------------------
// NEW Phase 3 commands
// ---------------------------------------------------------------------------

/// Called by the frontend after every epoch to keep shared state current.
/// The aggregator thread reads this every 15 seconds and sends it to the LLM.
#[tauri::command]
fn update_training_metrics(
    epoch: u32,
    loss: f64,
    accuracy: f64,
    state: tauri::State<AppState>,
) {
    let mut m = state.training_metrics.lock().unwrap();
    m.epoch = epoch;
    m.loss = loss;
    m.accuracy = accuracy;
}

/// Sends a user chat message to the LLM daemon.
/// The daemon's response will arrive via the `llm_insight` event.
#[tauri::command]
async fn send_chat_message(
    message: String,
    llm: tauri::State<'_, LlmDaemon>,
) -> Result<(), String> {
    let payload = serde_json::json!({
        "type": "user_chat",
        "message": message
    });
    let line = format!("{}
", payload);

    let mut guard = llm.stdin.lock().await;
    match guard.as_mut() {
        Some(stdin) => {
            stdin.write_all(line.as_bytes()).await.map_err(|e| e.to_string())?;
            stdin.flush().await.map_err(|e| e.to_string())?;
            Ok(())
        }
        None => Err("LLM daemon is not running. Load a model first via the Co-Pilot sidebar.".into()),
    }
}

/// Returns the current LLM daemon status string.
#[tauri::command]
fn get_llm_status(llm: tauri::State<'_, LlmDaemon>) -> String {
    llm.status.lock().unwrap().clone()
}

/// Kills the existing daemon (if any), then respawns with the given model path.
/// `conda_env`: conda environment name (e.g. "nocode_train"). Empty = system Python.
#[tauri::command]
async fn load_llm_model(
    path: String,
    conda_env: Option<String>,
    llm: tauri::State<'_, LlmDaemon>,
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let env = conda_env.unwrap_or_default();
    // Store the conda env for future restarts
    { let mut ce = llm.conda_env.lock().unwrap(); *ce = env.clone(); }

    // Kill old process if alive
    {
        let pid_guard = llm.child_pid.lock().unwrap();
        if let Some(pid) = *pid_guard {
            #[cfg(target_os = "windows")]
            let _ = std::process::Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/F"]).output();
            #[cfg(not(target_os = "windows"))]
            let _ = std::process::Command::new("kill")
                .args(["-9", &pid.to_string()]).output();
            println!("[LLM] Killed old daemon (pid {})", pid);
        }
    }
    // Clear old stdin
    { let mut s = llm.stdin.lock().await; *s = None; }
    // Mark as loading
    { let mut st = llm.status.lock().unwrap(); *st = "loading".to_string(); }
    let model_name = std::path::Path::new(&path)
        .file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "model".to_string());
    let env_display = if env.is_empty() { "system Python".to_string() } else { format!("conda:{}", env) };
    let _ = app.emit("llm_insight", serde_json::json!({
        "tag": "advice", "title": "Loading Model…",
        "body": format!("Loading {} via {}", if path.is_empty() { "auto-discover".to_string() } else { model_name }, env_display)
    }));

    let metrics    = state.training_metrics.clone();
    let stdin_arc  = llm.stdin.clone();
    let pid_arc    = llm.child_pid.clone();
    let status_arc = llm.status.clone();
    let model_path = path.clone();
    let conda      = env.clone();

    tauri::async_runtime::spawn(async move {
        init_llm_daemon(app, metrics, model_path, conda, stdin_arc, pid_arc, status_arc).await;
    });
    Ok(())
}

/// Updates the conda env used by the LLM daemon and immediately respawns it.
/// Call this when the user's GPU conda env changes.
#[tauri::command]
async fn set_llm_conda_env(
    conda_env: String,
    llm: tauri::State<'_, LlmDaemon>,
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    // Preserve the current model path from status if ready
    load_llm_model(String::new(), Some(conda_env), llm, app, state).await
}

// ---------------------------------------------------------------------------
// LLM daemon bootstrap
// ---------------------------------------------------------------------------

/// Resolves the Python executable (same search order as `run_python`).
async fn find_python() -> Option<String> {
    for cmd in ["python", "python3", "py"] {
        let result = TokioCommand::new(cmd).arg("--version").output().await;
        if result.map(|o| o.status.success()).unwrap_or(false) {
            return Some(cmd.to_string());
        }
    }
    None
}

/// Core daemon bootstrap — usable both from initial auto-start and from `load_llm_model`.
async fn init_llm_daemon(
    app_handle: tauri::AppHandle,
    training_metrics: Arc<Mutex<TrainingMetrics>>,
    model_path: String,    // empty = auto-discover
    conda_env:  String,    // empty = system Python
    stdin_arc:  Arc<tokio::sync::Mutex<Option<tokio::process::ChildStdin>>>,
    pid_arc:    Arc<Mutex<Option<u32>>>,
    status_arc: Arc<Mutex<String>>,
) {
    let script_path = match app_handle.path().resource_dir() {
        Ok(d) => d.join("python_backend").join("llm_monitor.py"),
        Err(e) => {
            let mut st = status_arc.lock().unwrap();
            *st = format!("error:{}", e);
            return;
        }
    };
    let script = script_path.to_string_lossy().replace("\\\\?\\", "").to_string();

    // Build Python script args (appended after the interpreter command)
    let mut script_args: Vec<String> = vec![script.clone()];
    if !model_path.is_empty() {
        script_args.push("--model".to_string());
        script_args.push(model_path.clone());
    }

    // Choose spawn strategy: conda env vs system Python
    let mut child = if !conda_env.is_empty() {
        // Use: conda run -n <env> --no-capture-output python <script> [--model <path>]
        let mut conda_args = vec![
            "run".to_string(),
            "-n".to_string(),
            conda_env.clone(),
            "--no-capture-output".to_string(),
            "python".to_string(),
        ];
        conda_args.extend(script_args);
        println!("[LLM] Spawning via conda env '{}'", conda_env);
        match TokioCommand::new("conda")
            .args(&conda_args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                let mut st = status_arc.lock().unwrap();
                *st = format!("error:conda spawn failed: {}", e);
                println!("[LLM] conda spawn failed: {}", e);
                return;
            }
        }
    } else {
        // Fall back to system Python
        let python = match find_python().await {
            Some(p) => p,
            None => {
                let mut st = status_arc.lock().unwrap();
                *st = "error:Python not found".to_string();
                println!("[LLM] Python not found — daemon will not start.");
                return;
            }
        };
        println!("[LLM] Spawning via system Python '{}'", python);
        match TokioCommand::new(&python)
            .args(&script_args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                let mut st = status_arc.lock().unwrap();
                *st = format!("error:{}", e);
                println!("[LLM] Failed to spawn: {}", e);
                return;
            }
        }
    };

    let pid = child.id();
    println!("[LLM] llm_monitor.py spawned (pid {:?})", pid);
    {
        let mut pg = pid_arc.lock().unwrap();
        *pg = pid;
    }

    let child_stdin  = child.stdin.take().expect("piped stdin");
    let child_stdout = child.stdout.take().expect("piped stdout");
    let child_stderr = child.stderr.take().expect("piped stderr");

    // Populate the shared stdin Arc
    {
        let mut s = stdin_arc.lock().await;
        *s = Some(child_stdin);
    }

    // -----------------------------------------------------------------------
    // Task 1: stdout reader
    // -----------------------------------------------------------------------
    {
        let app_h       = app_handle.clone();
        let status_arc2 = status_arc.clone();
        let mp          = model_path.clone();
        tauri::async_runtime::spawn(async move {
            let reader = BufReader::new(child_stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                println!("[LLM out] {}", line);
                if let Ok(mut value) = serde_json::from_str::<serde_json::Value>(&line) {
                    // When the daemon emits "LLM Ready" we update our status
                    if let Some(title) = value.get("title").and_then(|t| t.as_str()) {
                        if title.starts_with("LLM Ready") {
                            let name = std::path::Path::new(&mp)
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_else(|| "unknown".to_string());
                            let mut st = status_arc2.lock().unwrap();
                            *st = format!("ready:{}", name);
                            // Inject the model name into the emitted payload
                            if let Some(obj) = value.as_object_mut() {
                                obj.insert("model".to_string(), serde_json::Value::String(name));
                            }
                        } else if title == "LLM Unavailable" || title == "Model Not Found" || title == "LLM Load Failed" {
                            let mut st = status_arc2.lock().unwrap();
                            *st = format!("error:{}", title);
                        }
                    }
                    let _ = app_h.emit("llm_insight", value);
                }
            }
            println!("[LLM] stdout reader ended.");
        });
    }

    // -----------------------------------------------------------------------
    // Task 2: stderr logger
    // -----------------------------------------------------------------------
    {
        tauri::async_runtime::spawn(async move {
            let reader = BufReader::new(child_stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                println!("[LLM stderr] {}", line);
            }
        });
    }

    // -----------------------------------------------------------------------
    // Task 3: 15-second aggregator
    // -----------------------------------------------------------------------
    {
        let stdin_arc2  = stdin_arc.clone();
        let status_arc3 = status_arc.clone();
        let app_h       = app_handle.clone();

        tauri::async_runtime::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(15));
            interval.tick().await;

            loop {
                interval.tick().await;

                // Skip if not ready
                {
                    let st = status_arc3.lock().unwrap();
                    if !st.starts_with("ready:") { continue; }
                }

                let sys_info: serde_json::Value = {
                    let sp = match app_h.path().resource_dir() {
                        Ok(d) => d.join("python_backend").join("system_info.py"),
                        Err(_) => continue,
                    };
                    let script_str = sp.to_string_lossy().replace("\\\\?\\", "").to_string();
                    let py = match find_python().await { Some(p) => p, None => continue };
                    match TokioCommand::new(&py).arg(&script_str).output().await {
                        Ok(out) if out.status.success() => {
                            let raw = String::from_utf8_lossy(&out.stdout).to_string();
                            serde_json::from_str(&raw).unwrap_or(serde_json::Value::Null)
                        }
                        _ => serde_json::Value::Null,
                    }
                };

                let (epoch, loss, accuracy) = {
                    let m = training_metrics.lock().unwrap();
                    (m.epoch, m.loss, m.accuracy)
                };

                let hw         = sys_info.get("hardware").cloned().unwrap_or(serde_json::Value::Null);
                let torch_info = sys_info.get("torch").cloned().unwrap_or(serde_json::Value::Null);
                let ram_mb     = (hw.get("ram_used_gb").and_then(|v| v.as_f64()).unwrap_or(0.0) * 1024.0) as u64;
                let cuda_ok    = torch_info.get("cuda_available").and_then(|v| v.as_bool()).unwrap_or(false);
                let (vram_mb, vram_total) = if cuda_ok { (0u64, 0u64) } else { (0u64, 0u64) };

                let payload = serde_json::json!({
                    "type":          "system_update",
                    "epoch":         epoch,
                    "loss":          loss,
                    "accuracy":      accuracy,
                    "ram_mb":        ram_mb,
                    "vram_mb":       vram_mb,
                    "vram_total_mb": vram_total
                });
                let line = format!("{}
", payload);

                let mut guard = stdin_arc2.lock().await;
                if let Some(stdin) = guard.as_mut() {
                    if let Err(e) = stdin.write_all(line.as_bytes()).await {
                        println!("[LLM] Aggregator write error: {} — daemon may have crashed.", e);
                        *guard = None;
                    } else {
                        let _ = stdin.flush().await;
                    }
                }
            }
        });
    }

    // Wait for child exit so it isn't dropped
    let status_arc_exit = status_arc.clone();
    let pid_arc_exit    = pid_arc.clone();
    tauri::async_runtime::spawn(async move {
        let _ = child.wait().await;
        println!("[LLM] llm_monitor.py process exited.");
        // Only clear status if it wasn't intentionally restarted (pid would have changed)
        let _ = pid_arc_exit;
        let mut st = status_arc_exit.lock().unwrap();
        if st.starts_with("ready:") || st.as_str() == "loading" {
            *st = "not_loaded".to_string();
        }
    });
}

/// Legacy wrapper used during initial auto-start from .setup().
async fn spawn_llm_daemon(
    app_handle: tauri::AppHandle,
    training_metrics: Arc<Mutex<TrainingMetrics>>,
    daemon: &LlmDaemon,
) {
    let conda = daemon.conda_env.lock().unwrap().clone();
    init_llm_daemon(
        app_handle,
        training_metrics,
        String::new(), // auto-discover model
        conda,
        daemon.stdin.clone(),
        daemon.child_pid.clone(),
        daemon.status.clone(),
    ).await;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let clients: Arc<Mutex<Vec<tokio::sync::mpsc::UnboundedSender<Message>>>> =
        Arc::new(Mutex::new(Vec::new()));
    let token = Arc::new(Mutex::new(String::new()));
    let training_metrics = Arc::new(Mutex::new(TrainingMetrics::default()));

    let clients_clone      = clients.clone();
    let token_clone        = token.clone();
    let metrics_for_daemon = training_metrics.clone();

    tauri::Builder::default()
        .manage(AppState { clients, token, training_metrics })
        // Create LlmDaemon immediately so commands can always access it (no race condition)
        .manage(LlmDaemon {
            stdin:     Arc::new(tokio::sync::Mutex::new(None)),
            child_pid: Arc::new(Mutex::new(None)),
            status:    Arc::new(Mutex::new("not_loaded".to_string())),
            conda_env: Arc::new(Mutex::new(String::new())),
        })
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![
            run_tabular_processor,
            run_check_gpu,
            get_system_info,
            check_dependencies,
            fetch_runs,
            analyze_dataset,
            get_connection_details,
            broadcast_log,
            update_training_metrics,
            send_chat_message,
            get_llm_status,
            load_llm_model,
            set_llm_conda_env,
        ])
        .setup(move |app| {
            let window = app.get_webview_window("main").unwrap();
            let icon = tauri::include_image!("icons/icon.png");
            window.set_icon(icon).unwrap();

            let app_handle = app.handle().clone();

            // Auto-start daemon with model auto-discovery (non-blocking)
            {
                let app_d   = app_handle.clone();
                let metrics = metrics_for_daemon;
                let llm     = app_d.state::<LlmDaemon>();
                let sin     = llm.stdin.clone();
                let pid     = llm.child_pid.clone();
                let stat    = llm.status.clone();
                let conda   = llm.conda_env.lock().unwrap().clone();
                tauri::async_runtime::spawn(async move {
                    init_llm_daemon(app_d, metrics, String::new(), conda, sin, pid, stat).await;
                });
            }

            // ---------------------------------------------------------------
            // WebSocket server for mobile companion app (unchanged)
            // ---------------------------------------------------------------
            tauri::async_runtime::spawn(async move {
                let listener = match TcpListener::bind("0.0.0.0:8765").await {
                    Ok(l) => l,
                    Err(e) => { println!("Failed to bind WebSocket on port 8765: {}", e); return; }
                };
                println!("WebSocket server listening on port 8765");

                while let Ok((stream, _)) = listener.accept().await {
                    let app_handle = app_handle.clone();
                    let clients    = clients_clone.clone();
                    let token      = token_clone.clone();

                    tauri::async_runtime::spawn(async move {
                        if let Ok(ws_stream) = accept_async(stream).await {
                            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
                            clients.lock().unwrap().push(tx.clone());
                            let (mut write, mut read) = ws_stream.split();

                            tauri::async_runtime::spawn(async move {
                                while let Some(msg) = rx.recv().await {
                                    if write.send(msg).await.is_err() { break; }
                                }
                            });

                            while let Some(Ok(Message::Text(text))) = read.next().await {
                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text.to_string()) {
                                    if let Some(action) = json.get("action").and_then(|a| a.as_str()) {
                                        match action {
                                            "auth" => {
                                                let req_token = json.get("token").and_then(|t| t.as_str()).unwrap_or("");
                                                let valid = token.lock().unwrap().clone();
                                                let reply = if req_token == valid && !valid.is_empty() {
                                                    r#"{"status": "authenticated"}"#
                                                } else {
                                                    r#"{"status": "auth_failed"}"#
                                                };
                                                let _ = tx.send(Message::Text(reply.to_string().into()));
                                                if reply.contains("failed") { break; }
                                            }
                                            "stop_training" | "start_training" | "adjust_params" => {
                                                app_handle.emit("mobile_command", text.to_string()).unwrap();
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
