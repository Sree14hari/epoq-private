// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::Manager;
use tauri_plugin_shell::ShellExt;

async fn run_python(
    app: &tauri::AppHandle,
    args: &[&str],
) -> Result<String, String> {
    // Try `python` first, then alternatives including the Windows Python Launcher `py`
    let cmds = ["python", "python3", "py"];
    let mut last_err = String::new();

    for cmd in cmds {
        match app
            .shell()
            .command(cmd)
            .args(args)
            .output()
            .await
        {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                
                // If it succeeds, immediately return the standard output
                if output.status.success() {
                    return Ok(stdout);
                } else {
                    // Record error to return if ALL commands fail
                    let msg = if !stderr.trim().is_empty() {
                        stderr
                    } else if !stdout.trim().is_empty() {
                        stdout
                    } else {
                        format!("Exited with code: {}", output.status.code().unwrap_or(-1))
                    };
                    last_err = msg;
                    continue;
                }
            }
            Err(e) => {
                last_err = e.to_string();
            }
        }
    }
    Err(last_err)
}

/// Runs tabular_processor.py with the given action, file, and optional params.
/// Returns the JSON string printed by the script.
#[tauri::command]
async fn run_tabular_processor(
    app: tauri::AppHandle,
    file: String,
    action: String,
    params: Option<String>,
    out: Option<String>,
) -> Result<String, String> {
    let script_path = app
        .path()
        .resource_dir()
        .map_err(|e| e.to_string())?
        .join("python_backend")
        .join("tabular_processor.py");

    let script = script_path.to_string_lossy().to_string().replace("\\\\?\\", "");

    // Build args list
    let mut args: Vec<String> = vec![
        script,
        "--action".to_string(),
        action,
        "--file".to_string(),
        file,
    ];
    if let Some(p) = params {
        args.push("--params".to_string());
        args.push(p);
    }
    if let Some(o) = out {
        args.push("--out".to_string());
        args.push(o);
    }

    let args_ref: Vec<&str> = args.iter().map(String::as_str).collect();
    run_python(&app, &args_ref).await
}

/// Runs check_gpu.py and returns the stdout lines as a plain string.
#[tauri::command]
async fn run_check_gpu(app: tauri::AppHandle) -> Result<String, String> {
    let script_path = app
        .path()
        .resource_dir()
        .map_err(|e| e.to_string())?
        .join("python_backend")
        .join("check_gpu.py");

    let script = script_path.to_string_lossy().to_string().replace("\\\\?\\", "");

    match run_python(&app, &[script.as_str()]).await {
        Ok(output) => Ok(output.trim().to_string()), // remove extra newline
        Err(e) => Err(format!("GPU detection failed: {}", e)),
    }
}
/// Runs system_info.py and returns structured JSON string.
#[tauri::command]
async fn get_system_info(app: tauri::AppHandle) -> Result<String, String> {
    let script_path = app
        .path()
        .resource_dir()
        .map_err(|e| e.to_string())?
        .join("python_backend")
        .join("system_info.py");

    let script = script_path.to_string_lossy().to_string().replace("\\\\?\\", "");

    match run_python(&app, &[script.as_str()]).await {
        Ok(output) => Ok(output.trim().to_string()),
        Err(e) => Err(format!("System info failed: {}", e)),
    }
}

#[tauri::command]
async fn check_dependencies(app: tauri::AppHandle) -> Result<String, String> {
    println!("DEBUG: Running backend check_dependencies");
    let script = "import sys, json, importlib.util; p = lambda x: importlib.util.find_spec(x) is not None; print(json.dumps({'python': True, 'executable': sys.executable, 'version': sys.version.split()[0], 'pandas': p('pandas'), 'sklearn': p('sklearn'), 'torch': p('torch'), 'timm': p('timm'), 'optuna': p('optuna')}))";
    match run_python(&app, &["-c", script]).await {
        Ok(output) => {
            println!("DEBUG: Python stdout: {}", output);
            Ok(output.trim().to_string())
        },
        Err(e) => {
            println!("DEBUG: Python error: {}", e);
            let error_json = format!(
                "{{\"python\": false, \"version\": null, \"pandas\": false, \"sklearn\": false, \"torch\": false, \"timm\": false, \"optuna\": false, \"error\": \"{}\"}}",
                e.replace("\"", "\\\"").replace("\n", " ")
            );
            Ok(error_json)
        }
    }
}
#[tauri::command]
fn fetch_runs(save_path: String) -> Result<String, String> {
    use std::fs;
    use std::path::Path;

    let experiments_dir = Path::new(&save_path).join("experiments");

    if !experiments_dir.exists() {
        return Ok("[]".to_string());
    }

    let mut runs = Vec::new();

    for entry in fs::read_dir(experiments_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            if let Ok(content) = fs::read_to_string(&path) {
                runs.push(content);
            }
        }
    }

    Ok(format!("[{}]", runs.join(",")))
}
/// Analyzes an image dataset and returns statistics.
#[tauri::command]
async fn analyze_dataset(app: tauri::AppHandle, path: String) -> Result<String, String> {
    let script_path = app
        .path()
        .resource_dir()
        .map_err(|e| e.to_string())?
        .join("python_backend")
        .join("dataset_analyzer.py");

    let script = script_path.to_string_lossy().to_string().replace("\\\\?\\", "");

    match run_python(&app, &[script.as_str(), "--path", &path]).await {
        Ok(output) => Ok(output.trim().to_string()),
        Err(e) => Err(format!("Dataset analysis failed: {}", e)),
    }
}
use futures_util::{StreamExt, SinkExt};
use std::sync::{Arc, Mutex};
use tauri::Emitter;
use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

struct AppState {
    clients: Arc<Mutex<Vec<tokio::sync::mpsc::UnboundedSender<Message>>>>,
    token: Arc<Mutex<String>>,
}

#[tauri::command]
fn get_connection_details(state: tauri::State<AppState>) -> Result<String, String> {
    use local_ip_address::local_ip;
    use std::time::{SystemTime, UNIX_EPOCH};

    let my_local_ip = match local_ip() {
        Ok(ip) => ip.to_string(),
        Err(e) => return Err(format!("Failed to get local IP: {}", e)),
    };

    // Use current time to generate a makeshift 6-digit pin
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    
    // Generate a 6-digit PIN
    let token = (now % 900_000) + 100_000;
    
    // Store token in global state
    *state.token.lock().unwrap() = token.to_string();

    let json = format!(
        "{{\"ip\": \"{}\", \"port\": 8765, \"token\": \"{}\"}}",
        my_local_ip, token
    );

    Ok(json)
}

#[tauri::command]
fn broadcast_log(log: String, state: tauri::State<AppState>) {
    let mut clients = state.clients.lock().unwrap();
    // broadcast and remove dead channels
    clients.retain(|client| client.send(Message::Text(log.clone().into())).is_ok());
}

fn main() {
    let clients: Arc<Mutex<Vec<tokio::sync::mpsc::UnboundedSender<Message>>>> = Arc::new(Mutex::new(Vec::new()));
    let token = Arc::new(Mutex::new(String::new()));
    
    let clients_clone = clients.clone();
    let token_clone = token.clone();

    tauri::Builder::default()
        .manage(AppState { clients, token })
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
            broadcast_log
        ])
        .setup(move |app| {
            let window = app.get_webview_window("main").unwrap();
            let icon = tauri::include_image!("icons/icon.png");
            window.set_icon(icon).unwrap();
            
            let app_handle = app.handle().clone();
            
            tauri::async_runtime::spawn(async move {
                let listener = match TcpListener::bind("0.0.0.0:8765").await {
                    Ok(l) => l,
                    Err(e) => {
                        println!("Failed to bind WebSocket on port 8765: {}", e);
                        return;
                    }
                };
                
                println!("WebSocket server listening on port 8765");
                
                while let Ok((stream, _)) = listener.accept().await {
                    let app_handle = app_handle.clone();
                    let clients = clients_clone.clone();
                    let token = token_clone.clone();
                    
                    tauri::async_runtime::spawn(async move {
                        if let Ok(ws_stream) = accept_async(stream).await {
                            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
                            clients.lock().unwrap().push(tx.clone());
                            
                            let (mut write, mut read) = ws_stream.split();
                            
                            // Task to forward messages from the channel to the websocket
                            tauri::async_runtime::spawn(async move {
                                while let Some(msg) = rx.recv().await {
                                    if write.send(msg).await.is_err() {
                                        break;
                                    }
                                }
                            });

                            // Handle incoming messages from the mobile client
                            while let Some(Ok(Message::Text(text))) = read.next().await {
                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text.to_string()) {
                                    if let Some(action) = json.get("action").and_then(|a| a.as_str()) {
                                        if action == "auth" {
                                            let req_token = json.get("token").and_then(|t| t.as_str()).unwrap_or("");
                                            let valid = token.lock().unwrap().clone();
                                            if req_token == valid && !valid.is_empty() {
                                                let _ = tx.send(Message::Text(r#"{"status": "authenticated"}"#.to_string().into()));
                                            } else {
                                                let _ = tx.send(Message::Text(r#"{"status": "auth_failed"}"#.to_string().into()));
                                                break;
                                            }
                                        } else if action == "stop_training" {
                                            app_handle.emit("mobile_command", "stop_training").unwrap();
                                        } else if action == "start_training" {
                                            // Forward entire JSON payload to desktop frontend
                                            app_handle.emit("mobile_command", text.to_string()).unwrap();
                                        } else if action == "adjust_params" {
                                            // Forward live adjustment command
                                            app_handle.emit("mobile_command", text.to_string()).unwrap();
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
