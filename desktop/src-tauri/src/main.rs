// photolib desktop shell.
//
// Starts the bundled Python server, waits for its readiness handshake, and
// keeps the native window and sidecar lifecycle tied together. A second app
// launch focuses the existing window. Unexpected server exits are restarted
// once; normal app exits ask uvicorn to drain requests before a timed kill.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;
use std::time::Duration;

use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

const READY_PREFIX: &str = "PHOTOLIB_READY ";
const SHUTDOWN_COMMAND: &[u8] = b"PHOTOLIB_SHUTDOWN\n";
const MAX_SERVER_RESTARTS: u8 = 1;

#[derive(serde::Deserialize)]
struct Ready {
    url: String,
}

#[derive(Default)]
struct ServerState {
    child: Option<CommandChild>,
    stopping: bool,
    restarts: u8,
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            app.manage(Mutex::new(ServerState::default()));
            start_server(app.handle().clone())
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building photolib")
        .run(|app, event| match event {
            RunEvent::ExitRequested { api, .. } => {
                if request_server_stop(app) {
                    api.prevent_exit();
                }
            }
            RunEvent::Exit => force_stop_server(app),
            _ => {}
        });
}

fn start_server(handle: tauri::AppHandle) -> Result<(), String> {
    let (mut rx, child) = handle
        .shell()
        .sidecar("photolib-server")
        .map_err(|err| format!("photolib-server sidecar is missing: {err}"))?
        .args(["--no-browser"])
        .spawn()
        .map_err(|err| format!("failed to start the photolib server: {err}"))?;

    {
        let server = handle.state::<Mutex<ServerState>>();
        let mut state = match server.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.child = Some(child);
        state.stopping = false;
    }

    tauri::async_runtime::spawn(async move {
        let mut terminated = false;

        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(line) => {
                    let text = String::from_utf8_lossy(&line);
                    if let Some(payload) = text.trim().strip_prefix(READY_PREFIX) {
                        match serde_json::from_str::<Ready>(payload) {
                            Ok(ready) => open_or_navigate_window(&handle, &ready.url),
                            Err(err) => eprintln!("photolib: bad ready line: {err}"),
                        }
                    }
                }
                CommandEvent::Stderr(line) => {
                    eprintln!("photolib-server: {}", String::from_utf8_lossy(&line));
                }
                CommandEvent::Terminated(status) => {
                    eprintln!("photolib: server exited: {status:?}");
                    terminated = true;
                    break;
                }
                _ => {}
            }
        }

        if terminated {
            handle_server_termination(&handle);
        }
    });

    Ok(())
}

fn handle_server_termination(app: &tauri::AppHandle) {
    let should_restart = {
        let server = app.state::<Mutex<ServerState>>();
        let mut state = match server.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.child.take();

        if state.stopping {
            app.exit(0);
            return;
        }
        if state.restarts >= MAX_SERVER_RESTARTS {
            false
        } else {
            state.restarts += 1;
            true
        }
    };

    if !should_restart {
        eprintln!("photolib: server restart limit reached");
        app.exit(1);
        return;
    }

    if let Some(window) = app.get_webview_window("main") {
        let _ = window.set_title("photolib — restarting…");
    }
    if let Err(err) = start_server(app.clone()) {
        eprintln!("photolib: could not restart server: {err}");
        app.exit(1);
    }
}

fn request_server_stop(app: &tauri::AppHandle) -> bool {
    let write_failed = {
        let server = app.state::<Mutex<ServerState>>();
        let mut state = match server.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if state.child.is_none() {
            return false;
        }
        if state.stopping {
            return true;
        }

        state.stopping = true;
        state
            .child
            .as_mut()
            .is_some_and(|child| child.write(SHUTDOWN_COMMAND).is_err())
    };

    if write_failed {
        force_stop_server(app);
        return false;
    }

    let handle = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(3));
        let still_running = {
            let server = handle.state::<Mutex<ServerState>>();
            let state = match server.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            state.stopping && state.child.is_some()
        };
        if still_running {
            eprintln!("photolib: graceful shutdown timed out; stopping server");
            force_stop_server(&handle);
            handle.exit(0);
        }
    });
    true
}

fn force_stop_server(app: &tauri::AppHandle) {
    let child = {
        let server = app.state::<Mutex<ServerState>>();
        let mut state = match server.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.child.take()
    };

    if let Some(child) = child {
        if let Err(err) = child.kill() {
            eprintln!("photolib: could not stop the server: {err}");
        }
    }
}

fn open_or_navigate_window(app: &tauri::AppHandle, url: &str) {
    let parsed = match url.parse::<tauri::Url>() {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("photolib: server reported an unusable url {url}: {err}");
            app.exit(1);
            return;
        }
    };

    if let Some(window) = app.get_webview_window("main") {
        if let Err(err) = window.navigate(parsed) {
            eprintln!("photolib: could not reconnect the window: {err}");
            app.exit(1);
            return;
        }
        let _ = window.set_title("photolib");
        let _ = window.show();
        let _ = window.set_focus();
        return;
    }

    let result = WebviewWindowBuilder::new(app, "main", WebviewUrl::External(parsed))
        .title("photolib")
        .inner_size(1280.0, 860.0)
        .min_inner_size(760.0, 560.0)
        .resizable(true)
        .build();

    if let Err(err) = result {
        eprintln!("photolib: could not create the window: {err}");
        app.exit(1);
    }
}
