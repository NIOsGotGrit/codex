use anyhow::Result;
use app_test_support::McpProcess;
use app_test_support::create_exec_command_sse_response;
use app_test_support::create_final_assistant_message_sse_response;
use app_test_support::create_mock_responses_server_sequence;
use app_test_support::create_shell_command_sse_response;
use app_test_support::to_response;
use codex_app_server_protocol::HookPostToolUseFailureNotification;
use codex_app_server_protocol::HookPostToolUseNotification;
use codex_app_server_protocol::HookPreToolUseDecision;
use codex_app_server_protocol::HookPreToolUseResponse;
use codex_app_server_protocol::ItemCompletedNotification;
use codex_app_server_protocol::JSONRPCErrorError;
use codex_app_server_protocol::JSONRPCMessage;
use codex_app_server_protocol::JSONRPCResponse;
use codex_app_server_protocol::RequestId;
use codex_app_server_protocol::ServerRequest;
use codex_app_server_protocol::ThreadItem;
use codex_app_server_protocol::ThreadStartParams;
use codex_app_server_protocol::ThreadStartResponse;
use codex_app_server_protocol::TurnStartParams;
use codex_app_server_protocol::TurnStartResponse;
use codex_app_server_protocol::UserInput as V2UserInput;
use codex_core::features::FEATURES;
use codex_core::features::Feature;
use core_test_support::skip_if_no_network;
use std::collections::BTreeMap;
use std::path::Path;
use tempfile::TempDir;
use tokio::time::timeout;

const DEFAULT_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

#[derive(Clone, Copy)]
struct HookProtocolTestConfig {
    enabled: bool,
    pre_tool_use: bool,
    post_tool_use: bool,
}

#[tokio::test]
async fn hook_pre_allow_emits_post_success_v2() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let call_id = "hook-call-allow";
    let responses = vec![
        create_exec_command_sse_response(call_id)?,
        create_final_assistant_message_sse_response("done")?,
    ];
    let server = create_mock_responses_server_sequence(responses).await;

    let codex_home = TempDir::new()?;
    create_config_toml(
        codex_home.path(),
        &server.uri(),
        "never",
        "danger-full-access",
        &BTreeMap::from([(Feature::UnifiedExec, true)]),
        HookProtocolTestConfig {
            enabled: true,
            pre_tool_use: true,
            post_tool_use: true,
        },
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let thread_id = start_thread(&mut mcp).await?;
    let _turn_id = start_turn(&mut mcp, &thread_id, "run exec command").await?;

    let request = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_request_message(),
    )
    .await??;
    let (request_id, params) = match request {
        ServerRequest::HookPreToolUse { request_id, params } => (request_id, params),
        other => panic!("expected HookPreToolUse request, got {other:?}"),
    };
    assert_eq!(params.item_id, call_id);
    assert_eq!(params.tool_name, "exec_command");

    mcp.send_response(
        request_id,
        serde_json::to_value(HookPreToolUseResponse {
            decision: HookPreToolUseDecision::Allow,
        })?,
    )
    .await?;

    let mut saw_post_success = false;
    let mut saw_turn_completed = false;
    while !saw_turn_completed {
        let message = timeout(DEFAULT_READ_TIMEOUT, mcp.read_next_message()).await??;
        match message {
            JSONRPCMessage::Notification(notification) => match notification.method.as_str() {
                "item/hook/postToolUse" => {
                    let payload: HookPostToolUseNotification = serde_json::from_value(
                        notification
                            .params
                            .expect("item/hook/postToolUse params should be present"),
                    )?;
                    assert_eq!(payload.item_id, call_id);
                    assert_eq!(payload.tool_name, "exec_command");
                    saw_post_success = true;
                }
                "turn/completed" => {
                    saw_turn_completed = true;
                }
                _ => {}
            },
            JSONRPCMessage::Request(request)
                if request.method == "item/commandExecution/requestApproval" =>
            {
                panic!("did not expect command execution approval request in approval_policy=never")
            }
            _ => {}
        }
    }

    assert!(
        saw_post_success,
        "expected item/hook/postToolUse notification for successful tool call"
    );

    Ok(())
}

#[tokio::test]
async fn hook_pre_deny_skips_execution_and_approval_v2() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let call_id = "hook-call-deny";
    let responses = vec![
        create_shell_command_sse_response(
            vec![
                "python3".to_string(),
                "-c".to_string(),
                "print('should not run')".to_string(),
            ],
            None,
            Some(5_000),
            call_id,
        )?,
        create_final_assistant_message_sse_response("done")?,
    ];
    let server = create_mock_responses_server_sequence(responses).await;

    let codex_home = TempDir::new()?;
    create_config_toml(
        codex_home.path(),
        &server.uri(),
        "untrusted",
        "read-only",
        &BTreeMap::default(),
        HookProtocolTestConfig {
            enabled: true,
            pre_tool_use: true,
            post_tool_use: true,
        },
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let thread_id = start_thread(&mut mcp).await?;
    let _turn_id = start_turn(&mut mcp, &thread_id, "run shell command").await?;

    let request = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_request_message(),
    )
    .await??;
    let (request_id, params) = match request {
        ServerRequest::HookPreToolUse { request_id, params } => (request_id, params),
        other => panic!("expected HookPreToolUse request, got {other:?}"),
    };
    assert_eq!(params.item_id, call_id);
    assert_eq!(params.tool_name, "shell_command");

    mcp.send_response(
        request_id,
        serde_json::to_value(HookPreToolUseResponse {
            decision: HookPreToolUseDecision::Deny {
                reason: "blocked by test".to_string(),
            },
        })?,
    )
    .await?;

    let mut saw_post_notification = false;
    let mut saw_approval_request = false;
    let mut saw_turn_completed = false;
    while !saw_turn_completed {
        let message = timeout(DEFAULT_READ_TIMEOUT, mcp.read_next_message()).await??;
        match message {
            JSONRPCMessage::Notification(notification) => match notification.method.as_str() {
                "item/hook/postToolUse" | "item/hook/postToolUseFailure" => {
                    saw_post_notification = true;
                }
                "turn/completed" => {
                    saw_turn_completed = true;
                }
                _ => {}
            },
            JSONRPCMessage::Request(request)
                if request.method == "item/commandExecution/requestApproval" =>
            {
                saw_approval_request = true;
            }
            _ => {}
        }
    }

    assert!(
        !saw_approval_request,
        "pre-hook deny must skip approval flow"
    );
    assert!(
        !saw_post_notification,
        "pre-hook deny should skip dispatch and therefore should not emit post hook notifications"
    );

    Ok(())
}

#[tokio::test]
async fn hook_pre_modify_updates_command_before_execution_v2() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let call_id = "hook-call-modify";
    let responses = vec![
        create_exec_command_sse_response(call_id)?,
        create_final_assistant_message_sse_response("done")?,
    ];
    let server = create_mock_responses_server_sequence(responses).await;

    let codex_home = TempDir::new()?;
    create_config_toml(
        codex_home.path(),
        &server.uri(),
        "never",
        "danger-full-access",
        &BTreeMap::from([(Feature::UnifiedExec, true)]),
        HookProtocolTestConfig {
            enabled: true,
            pre_tool_use: true,
            post_tool_use: false,
        },
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let thread_id = start_thread(&mut mcp).await?;
    let _turn_id = start_turn(&mut mcp, &thread_id, "run exec command").await?;

    let request = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_request_message(),
    )
    .await??;
    let (request_id, params) = match request {
        ServerRequest::HookPreToolUse { request_id, params } => (request_id, params),
        other => panic!("expected HookPreToolUse request, got {other:?}"),
    };
    assert_eq!(params.item_id, call_id);
    assert_eq!(params.tool_name, "exec_command");

    let modified_command = if cfg!(windows) {
        "cmd.exe /d /c echo modified-from-hook".to_string()
    } else {
        "/bin/sh -c 'echo modified-from-hook'".to_string()
    };
    let modified_arguments = serde_json::json!({
        "cmd": modified_command,
        "yield_time_ms": 500,
    });

    mcp.send_response(
        request_id,
        serde_json::to_value(HookPreToolUseResponse {
            decision: HookPreToolUseDecision::Modify {
                new_input: serde_json::json!({
                    "input_type": "function",
                    "arguments": serde_json::to_string(&modified_arguments)?,
                }),
            },
        })?,
    )
    .await?;

    let mut saw_modified_command = false;
    let mut saw_turn_completed = false;
    while !saw_turn_completed {
        let message = timeout(DEFAULT_READ_TIMEOUT, mcp.read_next_message()).await??;
        if let JSONRPCMessage::Notification(notification) = message {
            match notification.method.as_str() {
                "item/completed" => {
                    let payload: ItemCompletedNotification = serde_json::from_value(
                        notification
                            .params
                            .expect("item/completed params should be present"),
                    )?;
                    if let ThreadItem::CommandExecution { id, command, .. } = payload.item
                        && id == call_id
                    {
                        assert!(
                            command.contains("modified-from-hook"),
                            "expected modified command, got: {command}"
                        );
                        saw_modified_command = true;
                    }
                }
                "turn/completed" => {
                    saw_turn_completed = true;
                }
                _ => {}
            }
        }
    }

    assert!(
        saw_modified_command,
        "expected command execution item to use modified pre-hook input"
    );

    Ok(())
}

#[tokio::test]
async fn hook_pre_client_error_defaults_to_allow_v2() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let call_id = "hook-call-client-error";
    let responses = vec![
        create_exec_command_sse_response(call_id)?,
        create_final_assistant_message_sse_response("done")?,
    ];
    let server = create_mock_responses_server_sequence(responses).await;

    let codex_home = TempDir::new()?;
    create_config_toml(
        codex_home.path(),
        &server.uri(),
        "never",
        "danger-full-access",
        &BTreeMap::from([(Feature::UnifiedExec, true)]),
        HookProtocolTestConfig {
            enabled: true,
            pre_tool_use: true,
            post_tool_use: true,
        },
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let thread_id = start_thread(&mut mcp).await?;
    let _turn_id = start_turn(&mut mcp, &thread_id, "run exec command").await?;

    let request = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_request_message(),
    )
    .await??;
    let request_id = match request {
        ServerRequest::HookPreToolUse { request_id, .. } => request_id,
        other => panic!("expected HookPreToolUse request, got {other:?}"),
    };

    mcp.send_error(
        request_id,
        JSONRPCErrorError {
            code: -32000,
            message: "client hook failure".to_string(),
            data: None,
        },
    )
    .await?;

    let mut saw_post_success = false;
    let mut saw_command_completion = false;
    let mut saw_turn_completed = false;
    while !saw_turn_completed {
        let message = timeout(DEFAULT_READ_TIMEOUT, mcp.read_next_message()).await??;
        if let JSONRPCMessage::Notification(notification) = message {
            match notification.method.as_str() {
                "item/hook/postToolUse" => {
                    let payload: HookPostToolUseNotification = serde_json::from_value(
                        notification
                            .params
                            .expect("item/hook/postToolUse params should be present"),
                    )?;
                    if payload.item_id == call_id {
                        saw_post_success = true;
                    }
                }
                "item/completed" => {
                    let payload: ItemCompletedNotification = serde_json::from_value(
                        notification
                            .params
                            .expect("item/completed params should be present"),
                    )?;
                    if let ThreadItem::CommandExecution { id, .. } = payload.item
                        && id == call_id
                    {
                        saw_command_completion = true;
                    }
                }
                "turn/completed" => {
                    saw_turn_completed = true;
                }
                _ => {}
            }
        }
    }

    assert!(
        saw_post_success,
        "client error on pre-hook request should fail-open to allow"
    );
    assert!(
        saw_command_completion,
        "client error on pre-hook request should still execute the tool"
    );

    Ok(())
}

#[tokio::test]
async fn hook_post_failure_emitted_on_tool_execution_failure_v2() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let call_id = "hook-call-runtime-failure";
    let command = if cfg!(windows) {
        vec![
            "cmd.exe".to_string(),
            "/d".to_string(),
            "/c".to_string(),
            "exit".to_string(),
            "7".to_string(),
        ]
    } else {
        vec!["sh".to_string(), "-c".to_string(), "exit 7".to_string()]
    };
    let responses = vec![
        create_shell_command_sse_response(command, None, Some(5_000), call_id)?,
        create_final_assistant_message_sse_response("done")?,
    ];
    let server = create_mock_responses_server_sequence(responses).await;

    let codex_home = TempDir::new()?;
    create_config_toml(
        codex_home.path(),
        &server.uri(),
        "never",
        "read-only",
        &BTreeMap::default(),
        HookProtocolTestConfig {
            enabled: true,
            pre_tool_use: true,
            post_tool_use: true,
        },
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let thread_id = start_thread(&mut mcp).await?;
    let _turn_id = start_turn(&mut mcp, &thread_id, "run failing shell command").await?;

    let request = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_request_message(),
    )
    .await??;
    let request_id = match request {
        ServerRequest::HookPreToolUse { request_id, .. } => request_id,
        other => panic!("expected HookPreToolUse request, got {other:?}"),
    };
    mcp.send_response(
        request_id,
        serde_json::to_value(HookPreToolUseResponse {
            decision: HookPreToolUseDecision::Allow,
        })?,
    )
    .await?;

    let mut saw_post_success = false;
    let mut saw_post_failure = false;
    let mut saw_turn_completed = false;
    while !saw_turn_completed {
        let message = timeout(DEFAULT_READ_TIMEOUT, mcp.read_next_message()).await??;
        if let JSONRPCMessage::Notification(notification) = message {
            match notification.method.as_str() {
                "item/hook/postToolUse" => {
                    let payload: HookPostToolUseNotification = serde_json::from_value(
                        notification
                            .params
                            .expect("item/hook/postToolUse params should be present"),
                    )?;
                    if payload.item_id == call_id {
                        saw_post_success = true;
                    }
                }
                "item/hook/postToolUseFailure" => {
                    let payload: HookPostToolUseFailureNotification = serde_json::from_value(
                        notification
                            .params
                            .expect("item/hook/postToolUseFailure params should be present"),
                    )?;
                    if payload.item_id == call_id {
                        saw_post_failure = true;
                    }
                }
                "turn/completed" => {
                    saw_turn_completed = true;
                }
                _ => {}
            }
        }
    }

    assert!(
        !saw_post_success,
        "did not expect postToolUse on failed command execution"
    );
    assert!(
        saw_post_failure,
        "expected postToolUseFailure notification on failed command execution"
    );

    Ok(())
}

#[tokio::test]
async fn hook_protocol_disabled_emits_no_hook_messages_v2() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let call_id = "hook-call-disabled";
    let responses = vec![
        create_exec_command_sse_response(call_id)?,
        create_final_assistant_message_sse_response("done")?,
    ];
    let server = create_mock_responses_server_sequence(responses).await;

    let codex_home = TempDir::new()?;
    create_config_toml(
        codex_home.path(),
        &server.uri(),
        "never",
        "danger-full-access",
        &BTreeMap::from([(Feature::UnifiedExec, true)]),
        HookProtocolTestConfig {
            enabled: false,
            pre_tool_use: true,
            post_tool_use: true,
        },
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let thread_id = start_thread(&mut mcp).await?;
    let _turn_id = start_turn(&mut mcp, &thread_id, "run exec command").await?;

    let mut saw_command_completion = false;
    let mut saw_turn_completed = false;
    while !saw_turn_completed {
        let message = timeout(DEFAULT_READ_TIMEOUT, mcp.read_next_message()).await??;
        match message {
            JSONRPCMessage::Request(request) => {
                assert_ne!(
                    request.method, "item/hook/preToolUse",
                    "hook pre request must not be emitted when hook_protocol.enabled=false"
                );
            }
            JSONRPCMessage::Notification(notification) => match notification.method.as_str() {
                "item/hook/postToolUse" | "item/hook/postToolUseFailure" => {
                    panic!(
                        "hook post notification {} must not be emitted when hook_protocol.enabled=false",
                        notification.method
                    );
                }
                "item/completed" => {
                    let payload: ItemCompletedNotification = serde_json::from_value(
                        notification
                            .params
                            .expect("item/completed params should be present"),
                    )?;
                    if let ThreadItem::CommandExecution { id, .. } = payload.item
                        && id == call_id
                    {
                        saw_command_completion = true;
                    }
                }
                "turn/completed" => {
                    saw_turn_completed = true;
                }
                _ => {}
            },
            _ => {}
        }
    }

    assert!(
        saw_command_completion,
        "expected command execution to complete normally with hooks disabled"
    );

    Ok(())
}

async fn start_thread(mcp: &mut McpProcess) -> Result<String> {
    let request_id = mcp
        .send_thread_start_request(ThreadStartParams {
            model: Some("mock-model".to_string()),
            ..Default::default()
        })
        .await?;
    let response: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    let ThreadStartResponse { thread, .. } = to_response::<ThreadStartResponse>(response)?;
    Ok(thread.id)
}

async fn start_turn(mcp: &mut McpProcess, thread_id: &str, text: &str) -> Result<String> {
    let request_id = mcp
        .send_turn_start_request(TurnStartParams {
            thread_id: thread_id.to_string(),
            input: vec![V2UserInput::Text {
                text: text.to_string(),
                text_elements: Vec::new(),
            }],
            ..Default::default()
        })
        .await?;
    let response: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    let TurnStartResponse { turn } = to_response::<TurnStartResponse>(response)?;
    Ok(turn.id)
}

fn create_config_toml(
    codex_home: &Path,
    server_uri: &str,
    approval_policy: &str,
    sandbox_mode: &str,
    feature_flags: &BTreeMap<Feature, bool>,
    hook_protocol: HookProtocolTestConfig,
) -> std::io::Result<()> {
    let mut features = BTreeMap::new();
    for (feature, enabled) in feature_flags {
        features.insert(*feature, *enabled);
    }
    let feature_entries = features
        .into_iter()
        .map(|(feature, enabled)| {
            let key = FEATURES
                .iter()
                .find(|spec| spec.id == feature)
                .map(|spec| spec.key)
                .unwrap_or_else(|| panic!("missing feature key for {feature:?}"));
            format!("{key} = {enabled}")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let config_toml = codex_home.join("config.toml");
    std::fs::write(
        config_toml,
        format!(
            r#"
model = "mock-model"
approval_policy = "{approval_policy}"
sandbox_mode = "{sandbox_mode}"

model_provider = "mock_provider"

[features]
{feature_entries}

[hook_protocol]
enabled = {hook_enabled}
pre_tool_use = {hook_pre}
post_tool_use = {hook_post}

[model_providers.mock_provider]
name = "Mock provider for test"
base_url = "{server_uri}/v1"
wire_api = "responses"
request_max_retries = 0
stream_max_retries = 0
"#,
            hook_enabled = hook_protocol.enabled,
            hook_pre = hook_protocol.pre_tool_use,
            hook_post = hook_protocol.post_tool_use,
        ),
    )
}
