use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn mcp_stdio_keeps_logs_off_stdout() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_cerememory"))
        .args(["mcp", "--server-url", "http://127.0.0.1:9"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn cerememory mcp");

    let request = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"cerememory-test","version":"0.0.0"}}}"#;
    child
        .stdin
        .as_mut()
        .expect("open stdin")
        .write_all(format!("{request}\n").as_bytes())
        .expect("write initialize request");
    drop(child.stdin.take());

    let output = child.wait_with_output().expect("wait for cerememory mcp");
    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("stdout is utf-8");
    let stderr = String::from_utf8(output.stderr).expect("stderr is utf-8");

    assert!(
        stdout.starts_with(r#"{"jsonrpc":"2.0","id":1,"result":"#),
        "stdout should start with MCP JSON-RPC initialize response, got: {stdout:?}"
    );
    assert!(
        !stdout.contains("Starting Cerememory MCP HTTP proxy"),
        "stdout must not contain logs: {stdout:?}"
    );
    assert!(
        stderr.contains("Starting Cerememory MCP HTTP proxy"),
        "expected startup log on stderr, got: {stderr:?}"
    );
}
