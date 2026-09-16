//! Supervises the GeoSemanticSat.Daemon child process.
//!
//! The daemon hosts GeoSemanticSat.Core, the same compiled analysis code the Avalonia app
//! calls in-process. Spawning it here is what makes the two frontends comparable: whatever
//! numbers this app shows came out of the identical algorithms.
//!
//! Startup handshake: the daemon binds an ephemeral loopback port and prints one JSON line
//! (`{"ready":true,"port":N,"token":"..."}`) to stdout. We block on that line rather than
//! polling a fixed port, which avoids both a firewall prompt and a port collision.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, ChildStderr, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// How long to wait for the handshake line before giving up.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(60);

/// Lines of child stderr kept so a death can be explained rather than just observed.
const STDERR_TAIL_LINES: usize = 40;

/// Drains a child pipe for the life of the process, keeping the most recent lines.
///
/// Both halves of this matter. If nobody reads, the OS pipe buffer fills and the child
/// blocks forever on its next write - it looks like a hang with no cause. And if the read
/// end is dropped instead, the pipe closes and the child's next write fails, which on .NET
/// can surface as an unhandled IOException and kill it outright with nothing logged. That
/// is what made the daemon die silently: the handshake reader was a local, so the stdout
/// pipe closed the moment startup finished.
fn drain<R: Read + Send + 'static>(stream: R, tail: Option<Arc<Mutex<VecDeque<String>>>>) {
    thread::spawn(move || {
        let reader = BufReader::new(stream);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            if let Some(tail) = &tail {
                if let Ok(mut buffer) = tail.lock() {
                    if buffer.len() == STDERR_TAIL_LINES {
                        buffer.pop_front();
                    }
                    buffer.push_back(line);
                }
            }
        }
    });
}

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
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
}

impl DaemonState {
    pub fn new() -> Self {
        Self {
            endpoint: Mutex::new(None),
            child: Mutex::new(None),
            stderr_tail: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// The last thing the daemon said before it stopped, for error messages.
    pub fn last_output(&self) -> String {
        self.stderr_tail
            .lock()
            .map(|t| t.iter().cloned().collect::<Vec<_>>().join("
"))
            .unwrap_or_default()
    }

    /// True when a daemon was started and is still running.
    pub fn is_running(&self) -> bool {
        let Ok(mut guard) = self.child.lock() else { return false };
        match guard.as_mut() {
            // try_wait returns Ok(None) while the child is still alive.
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => false,
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
    // Reuse the running daemon, but only if it is actually alive. Returning a cached
    // endpoint for a dead child is what turned a crash into "Failed to fetch" on every
    // later call, with no way back short of restarting the app.
    if state.is_running() {
        if let Ok(guard) = state.endpoint.lock() {
            if let Some(endpoint) = guard.clone() {
                return Ok(endpoint);
            }
        }
    } else if let Ok(mut guard) = state.endpoint.lock() {
        // Stale endpoint from a daemon that has since exited - drop it and start again.
        *guard = None;
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

    // Start draining stderr immediately, so a failure during startup is captured rather
    // than lost, and so the child can never block writing to a pipe nobody reads.
    if let Some(stderr) = child.stderr.take() {
        drain::<ChildStderr>(stderr, Some(Arc::clone(&state.stderr_tail)));
    }

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
                let said = state.last_output();
                return Err(if said.is_empty() {
                    "The analysis daemon exited before it was ready.".to_string()
                } else {
                    format!("The analysis daemon exited before it was ready:
{said}")
                });
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

    // Hand the reader to a drain thread rather than letting it drop here. Dropping it
    // closes the read end, and the daemon's next write to stdout then fails - which is
    // how it came to die silently mid-session.
    drain(reader, None);

    if let Ok(mut guard) = state.child.lock() {
        *guard = Some(child);
    }
    if let Ok(mut guard) = state.endpoint.lock() {
        *guard = Some(endpoint.clone());
    }

    Ok(endpoint)
}
