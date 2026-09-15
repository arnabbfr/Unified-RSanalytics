//! Supervises the GeoSemanticSat.Daemon child process.
//!
//! The daemon hosts GeoSemanticSat.Core, the same compiled analysis code the Avalonia app
//! calls in-process. Spawning it here is what makes the two frontends comparable: whatever
//! numbers this app shows came out of the identical algorithms.
//!
//! Startup handshake: the daemon binds an ephemeral loopback port and prints one JSON line
//! (`{"ready":true,"port":N,"token":"..."}`) to stdout. We block on that line rather than
//! polling a fixed port, which avoids both a firewall prompt and a port collision.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// How long to wait for the handshake line before giving up.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Deserialize)]
struct Handshake {
    port: u16,
    token: String,
}

/// Where the frontend should send requests, handed over once at startup.
///
/// camelCase matters: Tauri hands this struct straight to JS, and without the rename
/// `base_url` arrives as snake_case, reads as undefined on the TS side, and every fetch
/// silently resolves relative to the app origin - returning index.html instead of JSON.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DaemonEndpoint {
    pub base_url: String,
    pub token: String,
}

pub struct DaemonState {
    pub endpoint: Mutex<Option<DaemonEndpoint>>,
    child: Mutex<Option<Child>>,
}

impl DaemonState {
    pub fn new() -> Self {
        Self {
            endpoint: Mutex::new(None),
            child: Mutex::new(None),
        }
    }

    /// Kills the child. Called on window close so we never strand an orphaned daemon
    /// holding a port after the UI is gone.
    pub fn shutdown(&self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

impl Default for DaemonState {
    fn default() -> Self {
        Self::new()
    }
}

/// Locates the daemon executable.
///
/// Packaged builds ship it next to this binary as a Tauri "external binary". In development
/// it has not been copied yet, so fall back to the .NET build output in the sibling
/// Upgrahan2 tree - that way `npm run desktop` works straight from a fresh checkout without
/// a publish step first.
fn locate_daemon() -> Result<PathBuf, String> {
    let exe_name = if cfg!(windows) { "gss-daemon.exe" } else { "gss-daemon" };

    if let Ok(current) = std::env::current_exe() {
        if let Some(dir) = current.parent() {
            let bundled = dir.join(exe_name);
            if bundled.is_file() {
                return Ok(bundled);
            }
        }
    }

    // Dev fallback: Desktop_App/UpgrahanRust/src-tauri -> Desktop_App/Upgrahan2/src/...
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dev_build = manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|desktop_app| {
            desktop_app
                .join("Upgrahan2/src/GeoSemanticSat.Daemon/bin/Release/net10.0")
                .join(exe_name)
        });

    match dev_build {
        Some(path) if path.is_file() => Ok(path),
        Some(path) => Err(format!(
            "Analysis daemon not found at {}. Build it first:\n  \
             dotnet build Desktop_App/Upgrahan2/src/GeoSemanticSat.Daemon/GeoSemanticSat.Daemon.csproj -c Release",
            path.display()
        )),
        None => Err("Could not resolve the daemon path.".into()),
    }
}

/// Spawns the daemon and blocks until it reports a port and token.
pub fn start(state: &DaemonState) -> Result<DaemonEndpoint, String> {
    // Already running - hand back the existing endpoint rather than spawning a second one.
    if let Ok(guard) = state.endpoint.lock() {
        if let Some(endpoint) = guard.clone() {
            return Ok(endpoint);
        }
    }

    let exe = locate_daemon()?;

    let mut child = Command::new(&exe)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start the analysis daemon at {}: {e}", exe.display()))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "The analysis daemon produced no stdout to read the handshake from.".to_string())?;

    let started = Instant::now();
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();

    let handshake = loop {
        if started.elapsed() > HANDSHAKE_TIMEOUT {
            let _ = child.kill();
            return Err("The analysis daemon did not report a port within 60 seconds.".into());
        }

        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => {
                // stdout closed before the handshake: the daemon died on startup.
                let _ = child.kill();
                return Err("The analysis daemon exited before it was ready.".into());
            }
            Ok(_) => {
                if let Ok(parsed) = serde_json::from_str::<Handshake>(line.trim()) {
                    break parsed;
                }
                // Not the handshake line; keep reading.
            }
            Err(e) => {
                let _ = child.kill();
                return Err(format!("Failed reading the daemon handshake: {e}"));
            }
        }
    };

    let endpoint = DaemonEndpoint {
        base_url: format!("http://127.0.0.1:{}", handshake.port),
        token: handshake.token,
    };

    if let Ok(mut guard) = state.child.lock() {
        *guard = Some(child);
    }
    if let Ok(mut guard) = state.endpoint.lock() {
        *guard = Some(endpoint.clone());
    }

    Ok(endpoint)
}
