// photolib desktop shell.
//
// Starts the bundled Python server as a sidecar process, waits for it to
// report the port it bound, and opens a window on it. The window is created
// only after the server is ready, so nobody ever sees a connection-refused
// page while the backend is still starting.
//
// The server picks its own free port rather than assuming 8000 — on a
// machine where something else already holds that port, a hardcoded one
// fails in a way that is very hard for a non-technical user to diagnose.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;

use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// Printed by the sidecar on stdout once it is accepting connections.
const READY_PREFIX: &str = "PHOTOLIB_READY ";

#[derive(serde::Deserialize)]
struct Ready {
    url: String,
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let handle = app.handle().clone();

            let (mut rx, child) = app
                .shell()
                .sidecar("photolib-server")
                .expect("photolib-server sidecar is missing from the bundle")
                .args(["--no-browser"])
                .spawn()
                .expect("failed to start the photolib server");

            // CommandChild::kill consumes the handle, so keep it in an
            // Option that the exit callback can take exactly once.
            app.manage(Mutex::new(Some(child)));

            tauri::async_runtime::spawn(async move {
                let mut opened = false;

                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            let text = String::from_utf8_lossy(&line);
                            let text = text.trim();
                            if opened {
                                continue;
                            }
                            if let Some(payload) = text.strip_prefix(READY_PREFIX) {
                                match serde_json::from_str::<Ready>(payload) {
                                    Ok(ready) => {
                                        open_window(&handle, &ready.url);
                                        opened = true;
                                    }
                                    Err(err) => {
                                        eprintln!("photolib: bad ready line: {err}");
                                    }
                                }
                            }
                        }
                        CommandEvent::Stderr(line) => {
                            eprintln!("photolib-server: {}", String::from_utf8_lossy(&line));
                        }
                        CommandEvent::Terminated(status) => {
                            eprintln!("photolib: server exited: {status:?}");
                            if !opened {
                                // The server died before serving anything —
                                // exiting beats leaving an invisible process.
                                handle.exit(1);
                            }
                            break;
                        }
                        _ => {}
                    }
                }
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building photolib")
        .run(|app, event| match event {
            RunEvent::ExitRequested { .. } | RunEvent::Exit => stop_server(app),
            _ => {}
        });
}

fn stop_server(app: &tauri::AppHandle) {
    let server = app.state::<Mutex<Option<CommandChild>>>();
    let child = match server.lock() {
        Ok(mut guard) => guard.take(),
        Err(poisoned) => poisoned.into_inner().take(),
    };

    if let Some(child) = child {
        if let Err(err) = child.kill() {
            eprintln!("photolib: could not stop the server: {err}");
        }
    }
}

fn open_window(app: &tauri::AppHandle, url: &str) {
    let parsed = match url.parse() {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("photolib: server reported an unusable url {url}: {err}");
            app.exit(1);
            return;
        }
    };

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
