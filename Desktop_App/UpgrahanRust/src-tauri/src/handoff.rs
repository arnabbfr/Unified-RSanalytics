//! Handing the session over to the Avalonia build and back.
//!
//! Modelled on Cap's tauri <-> GPUI switch (apps/desktop/src-tauri/src/gpui_app.rs). The
//! shape that matters is not the spawn - it is everything around it:
//!
//!   - probe whether the sibling is even installed, so the control is only offered when it
//!     can work rather than failing after the user commits to it;
//!   - write the preference BEFORE launching, so the app that comes up already owns the
//!     session and does not bounce straight back;
//!   - roll that preference back if the launch is refused, so a failure leaves no trace;
//!   - reuse an instance that is already running instead of starting a second one;
//!   - spawn detached, so the child outlives this process exiting.
//!
//! The preference lives in a small file both builds read, because they share nothing else:
//! one is .NET, the other Rust.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde::Serialize;

/// Which frontend should open. Written by whichever build is handing over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Frontend {
    Avalonia,
    Tauri,
}

impl Frontend {
    fn as_str(self) -> &'static str {
        match self {
            Frontend::Avalonia => "avalonia",
            Frontend::Tauri => "tauri",
        }
    }
}

/// Executable name of the Avalonia build, as the release bundle ships it.
fn avalonia_exe_name() -> &'static str {
    if cfg!(windows) {
        "GeoSemanticSat.UI.exe"
    } else {
        "GeoSemanticSat.UI"
    }
}

/// The directory this binary was installed into.
fn install_dir() -> Option<PathBuf> {
    std::env::current_exe().ok()?.parent().map(PathBuf::from)
}

/// Where both builds agree to read the "which frontend" preference from.
///
/// Next to the executables rather than in a user profile: a portable install should carry
/// its own choice, and the two binaries are already guaranteed to share this directory in
/// the combined bundle - which is the only situation where switching is offered at all.
fn preference_path() -> Option<PathBuf> {
    install_dir().map(|dir| dir.join("preferred-frontend"))
}

pub fn write_preference(frontend: Frontend) -> Result<(), String> {
    let path = preference_path().ok_or("Could not resolve the install directory.")?;
    fs::write(&path, frontend.as_str())
        .map_err(|e| format!("Could not record the frontend preference: {e}"))
}

pub fn read_preference() -> Option<Frontend> {
    let raw = fs::read_to_string(preference_path()?).ok()?;
    match raw.trim() {
        "avalonia" => Some(Frontend::Avalonia),
        "tauri" => Some(Frontend::Tauri),
        _ => None,
    }
}

/// Path to the Avalonia build, when it is installed beside this one.
///
/// Separate downloads simply will not find it, which is the intended answer: the switch is
/// a property of the combined bundle, not of either build on its own.
pub fn avalonia_path() -> Option<PathBuf> {
    let candidate = install_dir()?.join(avalonia_exe_name());
    candidate.is_file().then_some(candidate)
}

/// True when the Avalonia build is already running.
///
/// It holds a named single-instance mutex, so a second launch would exit immediately and
/// the user would see nothing happen. Checking first lets us report that honestly instead.
#[cfg(windows)]
fn avalonia_already_running() -> bool {
    // tasklist is already on every Windows install; spawning one is cheaper than taking a
    // dependency on a process-enumeration crate for a single check.
    Command::new("tasklist")
        .args(["/FI", &format!("IMAGENAME eq {}", avalonia_exe_name()), "/NH"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()
        .map(|out| String::from_utf8_lossy(&out.stdout).contains(avalonia_exe_name()))
        .unwrap_or(false)
}

#[cfg(not(windows))]
fn avalonia_already_running() -> bool {
    Command::new("pgrep")
        .args(["-f", avalonia_exe_name()])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Starts a process that keeps running after this one exits.
fn spawn_detached(path: &PathBuf) -> Result<(), String> {
    let mut command = Command::new(path);

    // Inherit nothing. Piping without draining is what made the analysis daemon die
    // silently, and here there is nobody left to drain anything anyway.
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    // Run from the install directory so the child resolves its own assets the same way it
    // would when launched normally.
    if let Some(dir) = path.parent() {
        command.current_dir(dir);
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: no inherited console, and the child
        // is not killed when this process's group goes away.
        command.creation_flags(0x0000_0008 | 0x0000_0200);
    }

    command
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("Could not launch the other build: {e}"))
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HandoffTarget {
    /// Which frontend the shared preference currently names, if either.
    pub preferred: Option<String>,
    /// False when the sibling build is not installed beside this one.
    pub available: bool,
    /// Absolute path, for the message shown when it is missing.
    pub path: Option<String>,
    /// True when it is already running, so this is an activation rather than a launch.
    pub already_running: bool,
}

pub fn avalonia_target() -> HandoffTarget {
    let path = avalonia_path();
    HandoffTarget {
        preferred: read_preference().map(|f| f.as_str().to_string()),
        available: path.is_some(),
        path: path.as_ref().map(|p| p.display().to_string()),
        already_running: path.is_some() && avalonia_already_running(),
    }
}

/// Hands the session to the Avalonia build.
///
/// Returns Err without having changed anything the caller has to undo. The preference is
/// written first and rolled back here on failure, so the frontend only has to revert its
/// own toggle.
pub fn switch_to_avalonia() -> Result<(), String> {
    let path = avalonia_path().ok_or_else(|| {
        "The Avalonia build is not installed alongside this one. Download the combined \
         bundle to switch between them."
            .to_string()
    })?;

    if avalonia_already_running() {
        return Err(
            "The Avalonia build is already running. Switch to that window instead - it holds \
             a single-instance lock, so a second copy would exit immediately."
                .to_string(),
        );
    }

    write_preference(Frontend::Avalonia)?;

    if let Err(error) = spawn_detached(&path) {
        // Nothing was handed over, so the recorded preference is now a lie. Undo it.
        let _ = write_preference(Frontend::Tauri);
        return Err(error);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test, not two: they share a single preference file, and as separate tests they
    /// run in parallel and clobber each other. The product only ever has one writer.
    #[test]
    fn preference_round_trips_and_rejects_unknown_values() {
        let Some(path) = preference_path() else { return };

        // Only meaningful where the install directory is writable, which it is under target/.
        if write_preference(Frontend::Avalonia).is_err() {
            return;
        }
        assert_eq!(read_preference(), Some(Frontend::Avalonia));

        assert!(write_preference(Frontend::Tauri).is_ok());
        assert_eq!(read_preference(), Some(Frontend::Tauri));

        // An unrecognised value is not guessed at - better no preference than a wrong one.
        fs::write(&path, "something-else").expect("preference file should be writable");
        assert_eq!(read_preference(), None);

        let _ = fs::remove_file(&path);
        assert_eq!(read_preference(), None, "a missing file means no preference");
    }

    #[test]
    fn availability_is_reported_not_assumed() {
        let target = avalonia_target();
        // Whatever the answer, it must be self-consistent.
        assert_eq!(target.available, target.path.is_some());
        if !target.available {
            assert!(!target.already_running);
        }
    }
}
