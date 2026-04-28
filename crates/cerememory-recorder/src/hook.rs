use std::fs;
use std::path::{Path, PathBuf};

use crate::RecorderError;

#[derive(Debug, Clone)]
pub struct HookInstallResult {
    pub script_path: PathBuf,
    pub example_path: PathBuf,
}

pub fn install_codex_hook(
    base_dir: &Path,
    server_url: &str,
    force: bool,
) -> Result<HookInstallResult, RecorderError> {
    let hook_dir = base_dir.join(".codex").join("hooks");
    fs::create_dir_all(&hook_dir)?;

    let script_path = hook_dir.join("cerememory-recorder-codex-hook.py");
    let example_path = hook_dir.join("cerememory-recorder.example.json");

    preflight_install_paths(&[script_path.as_path(), example_path.as_path()], force)?;

    write_new_file(&script_path, &codex_hook_script(server_url)?, force)?;
    write_new_file(&example_path, &codex_hook_example(&script_path)?, force)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(&script_path)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script_path, permissions)?;
    }

    Ok(HookInstallResult {
        script_path,
        example_path,
    })
}

fn preflight_install_paths(paths: &[&Path], force: bool) -> Result<(), RecorderError> {
    for path in paths {
        if path_is_symlink(path)? {
            return Err(RecorderError::Config(format!(
                "{} is a symlink; refusing to overwrite hook files through symlinks",
                path.display()
            )));
        }
        if path.exists() && !force {
            return Err(RecorderError::Config(format!(
                "{} already exists; pass --force to overwrite",
                path.display()
            )));
        }
    }
    Ok(())
}

fn write_new_file(path: &Path, content: &str, force: bool) -> Result<(), RecorderError> {
    if !force {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .and_then(|mut file| {
                use std::io::Write;
                file.write_all(content.as_bytes())
            })
            .map_err(|source| RecorderError::File {
                operation: "write file",
                path: path.to_path_buf(),
                source,
            })?;
        return Ok(());
    }

    fs::write(path, content).map_err(|source| RecorderError::File {
        operation: "write file",
        path: path.to_path_buf(),
        source,
    })
}

fn path_is_symlink(path: &Path) -> Result<bool, RecorderError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(metadata.file_type().is_symlink()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(RecorderError::File {
            operation: "inspect file",
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn codex_hook_script(server_url: &str) -> Result<String, RecorderError> {
    let server_url_json = serde_json::to_string(server_url)?;
    Ok(format!(
        r#"#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys

SERVER_URL = {server_url_json}
ALLOWED_EVENT_TYPES = {{
    "user_message",
    "assistant_message",
    "tool_call",
    "tool_result",
    "command",
    "file_change",
    "session_summary",
    "error",
}}
HOOK_EVENT_MAP = {{
    "UserPromptSubmit": "user_message",
    "PreToolUse": "tool_call",
    "PostToolUse": "tool_result",
    "Stop": "session_summary",
    "SubagentStop": "session_summary",
    "Notification": "session_summary",
}}
CONTENT_KEYS = (
    "message",
    "prompt",
    "summary",
    "transcript",
    "tool_response",
    "tool_result",
    "response",
    "result",
    "output",
    "stdout",
    "stderr",
    "tool_input",
    "input",
    "command",
)
METADATA_KEYS = (
    "tool_name",
    "tool_call_id",
    "tool_use_id",
    "action_id",
    "exit_code",
    "status",
    "cwd",
    "transcript_path",
)

def first_payload_value(payload, keys):
    for key in keys:
        value = payload.get(key)
        if value is not None and value != "":
            return value
    return None

def text_or_json(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

def extract_content(payload):
    value = first_payload_value(payload, CONTENT_KEYS)
    if value is not None:
        return text_or_json(value)
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {{}}
    except json.JSONDecodeError:
        payload = {{"raw_stdin": raw}}

    hook_event = str(payload.get("hook_event_name") or payload.get("event") or "")
    event_type = str(payload.get("event_type") or HOOK_EVENT_MAP.get(hook_event) or "session_summary")
    if event_type not in ALLOWED_EVENT_TYPES:
        event_type = "session_summary"

    session_id = (
        os.environ.get("CODEX_SESSION_ID")
        or str(payload.get("session_id") or payload.get("conversation_id") or payload.get("thread_id") or "")
        or "codex-local"
    )
    content = extract_content(payload)
    metadata = {{
        "codex_hook_event": hook_event or None,
        "cwd": os.getcwd(),
    }}
    for key in METADATA_KEYS:
        value = payload.get(key)
        if value is not None and value != "":
            metadata[key] = value

    event = {{
        "session_id": session_id,
        "event_type": event_type,
        "content": content,
        "source": "codex",
        "metadata": metadata,
    }}
    action_id = first_payload_value(payload, ("action_id", "tool_call_id", "tool_use_id"))
    if action_id is not None:
        event["action_id"] = str(action_id)
    turn_id = first_payload_value(payload, ("turn_id", "message_id"))
    if turn_id is not None:
        event["turn_id"] = str(turn_id)
    timestamp = first_payload_value(payload, ("timestamp", "created_at"))
    if timestamp is not None:
        event["timestamp"] = str(timestamp)

    binary = os.environ.get("CEREMEMORY_RECORDER_BIN") or shutil.which("cerememory-recorder")
    if not binary:
        print("cerememory-recorder hook skipped: binary not found", file=sys.stderr)
        return 0

    try:
        subprocess.run(
            [binary, "ingest", "--server-url", SERVER_URL],
            input=json.dumps(event, ensure_ascii=False) + "\n",
            text=True,
            check=False,
            timeout=15,
        )
    except Exception as exc:
        print(f"cerememory-recorder hook skipped: {{exc}}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"#
    ))
}

fn codex_hook_example(script_path: &Path) -> Result<String, RecorderError> {
    let command = script_path.to_string_lossy();
    let command_json = serde_json::to_string(command.as_ref())?;
    Ok(format!(
        r#"{{
  "note": "Copy the command below into the Codex hook configuration you want to use. This installer does not modify existing hook settings.",
  "command": {command_json},
  "env": {{
    "CEREMEMORY_SERVER_API_KEY": "set-this-in-your-shell-or-hook-env"
  }}
}}
"#
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_codex_hook_refuses_overwrite_without_force() {
        let temp = tempfile::tempdir().unwrap();
        let result = install_codex_hook(temp.path(), "http://127.0.0.1:8420", false).unwrap();
        assert!(result.script_path.exists());

        let err = install_codex_hook(temp.path(), "http://127.0.0.1:8420", false).unwrap_err();
        assert!(err.to_string().contains("--force"));
    }

    #[test]
    fn install_codex_hook_preflights_before_partial_write() {
        let temp = tempfile::tempdir().unwrap();
        let hook_dir = temp.path().join(".codex").join("hooks");
        fs::create_dir_all(&hook_dir).unwrap();
        fs::write(
            hook_dir.join("cerememory-recorder.example.json"),
            "existing example",
        )
        .unwrap();

        let err = install_codex_hook(temp.path(), "http://127.0.0.1:8420", false).unwrap_err();
        assert!(err.to_string().contains("--force"));
        assert!(!hook_dir.join("cerememory-recorder-codex-hook.py").exists());
    }

    #[cfg(unix)]
    #[test]
    fn install_codex_hook_refuses_symlink_targets_even_with_force() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let hook_dir = temp.path().join(".codex").join("hooks");
        fs::create_dir_all(&hook_dir).unwrap();
        let outside = temp.path().join("outside.py");
        fs::write(&outside, "outside").unwrap();
        symlink(&outside, hook_dir.join("cerememory-recorder-codex-hook.py")).unwrap();

        let err = install_codex_hook(temp.path(), "http://127.0.0.1:8420", true).unwrap_err();
        assert!(err.to_string().contains("symlink"));
        assert_eq!(fs::read_to_string(outside).unwrap(), "outside");
    }

    #[test]
    fn codex_hook_script_extracts_structured_tool_payloads_best_effort() {
        let script = codex_hook_script("http://127.0.0.1:8420").unwrap();
        assert!(script.contains("CONTENT_KEYS"));
        assert!(script.contains("METADATA_KEYS"));
        assert!(script.contains("tool_call_id"));
        assert!(script.contains("timeout=15"));
    }
}
