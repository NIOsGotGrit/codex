use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;
use tokio_util::either::Either;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::Instrument;
use tracing::instrument;
use tracing::trace_span;

use crate::codex::Session;
use crate::codex::TurnContext;
use crate::error::CodexErr;
use crate::function_tool::FunctionCallError;
use crate::protocol::EventMsg;
use crate::protocol::HookDecision;
use crate::protocol::HookPostToolUseEvent;
use crate::protocol::HookPostToolUseFailureEvent;
use crate::tools::context::SharedTurnDiffTracker;
use crate::tools::context::ToolPayload;
use crate::tools::router::ToolCall;
use crate::tools::router::ToolRouter;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::SandboxPermissions;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

#[derive(Clone)]
pub(crate) struct ToolCallRuntime {
    router: Arc<ToolRouter>,
    session: Arc<Session>,
    turn_context: Arc<TurnContext>,
    tracker: SharedTurnDiffTracker,
    parallel_execution: Arc<RwLock<()>>,
}

impl ToolCallRuntime {
    pub(crate) fn new(
        router: Arc<ToolRouter>,
        session: Arc<Session>,
        turn_context: Arc<TurnContext>,
        tracker: SharedTurnDiffTracker,
    ) -> Self {
        Self {
            router,
            session,
            turn_context,
            tracker,
            parallel_execution: Arc::new(RwLock::new(())),
        }
    }

    #[instrument(level = "trace", skip_all, fields(call = ?call))]
    pub(crate) fn handle_tool_call(
        self,
        call: ToolCall,
        cancellation_token: CancellationToken,
    ) -> impl std::future::Future<Output = Result<ResponseInputItem, CodexErr>> {
        let supports_parallel = self.router.tool_supports_parallel(&call.tool_name);

        let router = Arc::clone(&self.router);
        let session = Arc::clone(&self.session);
        let turn = Arc::clone(&self.turn_context);
        let tracker = Arc::clone(&self.tracker);
        let lock = Arc::clone(&self.parallel_execution);
        let started = Instant::now();

        let dispatch_span = trace_span!(
            "dispatch_tool_call",
            otel.name = call.tool_name.as_str(),
            tool_name = call.tool_name.as_str(),
            call_id = call.call_id.as_str(),
            aborted = false,
        );

        let handle: AbortOnDropHandle<Result<ResponseInputItem, FunctionCallError>> =
            AbortOnDropHandle::new(tokio::spawn(async move {
                tokio::select! {
                    _ = cancellation_token.cancelled() => {
                        let secs = started.elapsed().as_secs_f32().max(0.1);
                        dispatch_span.record("aborted", true);
                        Ok(Self::aborted_response(&call, secs))
                    },
                    res = async {
                        let _guard = if supports_parallel {
                            Either::Left(lock.read().await)
                        } else {
                            Either::Right(lock.write().await)
                        };

                        let mut call_for_dispatch = call.clone();
                        if turn.config.hook_protocol.pre_tool_use_enabled() {
                            let tool_input =
                                Self::payload_to_hook_input(&call_for_dispatch.payload)?;
                            let decision = session
                                .request_pre_tool_use_hook(
                                    turn.as_ref(),
                                    call_for_dispatch.call_id.clone(),
                                    call_for_dispatch.tool_name.clone(),
                                    tool_input,
                                )
                                .await;

                            match decision {
                                HookDecision::Allow => {}
                                HookDecision::Deny { reason } => {
                                    return Ok(Self::pre_hook_denied_response(
                                        &call_for_dispatch,
                                        reason,
                                    ));
                                }
                                HookDecision::Modify { new_input } => {
                                    if let Err(err) = Self::apply_modified_input(
                                        &mut call_for_dispatch.payload,
                                        new_input,
                                    ) {
                                        return Ok(Self::pre_hook_denied_response(
                                            &call_for_dispatch,
                                            format!(
                                                "pre-tool-use hook returned invalid modified input: {err}"
                                            ),
                                        ));
                                    }
                                }
                            }
                        }

                        let dispatch_result = router
                            .dispatch_tool_call(
                                Arc::clone(&session),
                                Arc::clone(&turn),
                                tracker,
                                call_for_dispatch.clone(),
                                crate::tools::router::ToolCallSource::Direct,
                            )
                            .instrument(dispatch_span.clone())
                            .await;

                        if turn.config.hook_protocol.post_tool_use_enabled() {
                            Self::emit_post_tool_use_event(
                                session.as_ref(),
                                turn.as_ref(),
                                &call_for_dispatch,
                                &dispatch_result,
                            )
                            .await;
                        }

                        dispatch_result
                    } => res,
                }
            }));

        async move {
            match handle.await {
                Ok(Ok(response)) => Ok(response),
                Ok(Err(FunctionCallError::Fatal(message))) => Err(CodexErr::Fatal(message)),
                Ok(Err(other)) => Err(CodexErr::Fatal(other.to_string())),
                Err(err) => Err(CodexErr::Fatal(format!(
                    "tool task failed to receive: {err:?}"
                ))),
            }
        }
        .in_current_span()
    }
}

impl ToolCallRuntime {
    fn payload_to_hook_input(payload: &ToolPayload) -> Result<Value, FunctionCallError> {
        serde_json::to_value(HookToolInputPayload::from(payload)).map_err(|err| {
            FunctionCallError::Fatal(format!("failed to serialize tool input: {err}"))
        })
    }

    fn apply_modified_input(payload: &mut ToolPayload, new_input: Value) -> Result<(), String> {
        let original = HookToolInputPayload::from(&*payload);
        let modified = serde_json::from_value::<HookToolInputPayload>(new_input)
            .map_err(|err| format!("failed to deserialize modified input: {err}"))?;

        if original.input_type() != modified.input_type() {
            return Err(format!(
                "modified input type mismatch: expected {}, got {}",
                original.input_type(),
                modified.input_type()
            ));
        }

        *payload = modified.into();
        Ok(())
    }

    fn pre_hook_denied_response(call: &ToolCall, reason: String) -> ResponseInputItem {
        let normalized_reason = if reason.trim().is_empty() {
            "tool blocked by pre-tool-use hook".to_string()
        } else {
            format!("tool blocked by pre-tool-use hook: {reason}")
        };

        match &call.payload {
            ToolPayload::Custom { .. } => ResponseInputItem::CustomToolCallOutput {
                call_id: call.call_id.clone(),
                output: normalized_reason,
            },
            ToolPayload::Mcp { .. } => ResponseInputItem::McpToolCallOutput {
                call_id: call.call_id.clone(),
                result: Err(normalized_reason),
            },
            _ => ResponseInputItem::FunctionCallOutput {
                call_id: call.call_id.clone(),
                output: FunctionCallOutputPayload {
                    body: FunctionCallOutputBody::Text(normalized_reason),
                    success: Some(false),
                },
            },
        }
    }

    async fn emit_post_tool_use_event(
        session: &Session,
        turn: &TurnContext,
        call: &ToolCall,
        dispatch_result: &Result<ResponseInputItem, FunctionCallError>,
    ) {
        let tool_input = match Self::payload_to_hook_input(&call.payload) {
            Ok(tool_input) => tool_input,
            Err(_) => Value::Null,
        };

        match dispatch_result {
            Ok(response) => {
                if Self::response_is_success(response) {
                    let tool_output = serde_json::to_value(response).unwrap_or(Value::Null);
                    session
                        .send_event(
                            turn,
                            EventMsg::HookPostToolUse(HookPostToolUseEvent {
                                turn_id: turn.sub_id.clone(),
                                call_id: call.call_id.clone(),
                                tool_name: call.tool_name.clone(),
                                tool_input,
                                tool_output,
                            }),
                        )
                        .await;
                } else {
                    session
                        .send_event(
                            turn,
                            EventMsg::HookPostToolUseFailure(HookPostToolUseFailureEvent {
                                turn_id: turn.sub_id.clone(),
                                call_id: call.call_id.clone(),
                                tool_name: call.tool_name.clone(),
                                tool_input,
                                error: Self::failure_message_from_response(response),
                            }),
                        )
                        .await;
                }
            }
            Err(err) => {
                session
                    .send_event(
                        turn,
                        EventMsg::HookPostToolUseFailure(HookPostToolUseFailureEvent {
                            turn_id: turn.sub_id.clone(),
                            call_id: call.call_id.clone(),
                            tool_name: call.tool_name.clone(),
                            tool_input,
                            error: err.to_string(),
                        }),
                    )
                    .await;
            }
        }
    }

    fn response_is_success(response: &ResponseInputItem) -> bool {
        match response {
            ResponseInputItem::FunctionCallOutput { output, .. } => output.success.unwrap_or(true),
            ResponseInputItem::McpToolCallOutput { result, .. } => result.is_ok(),
            ResponseInputItem::CustomToolCallOutput { .. } => true,
            _ => true,
        }
    }

    fn failure_message_from_response(response: &ResponseInputItem) -> String {
        match response {
            ResponseInputItem::FunctionCallOutput { output, .. } => output
                .text_content()
                .map(str::to_string)
                .unwrap_or_else(|| "tool execution failed".to_string()),
            ResponseInputItem::McpToolCallOutput { result, .. } => result
                .as_ref()
                .err()
                .cloned()
                .unwrap_or_else(|| "tool execution failed".to_string()),
            ResponseInputItem::CustomToolCallOutput { output, .. } => output.clone(),
            _ => "tool execution failed".to_string(),
        }
    }

    fn aborted_response(call: &ToolCall, secs: f32) -> ResponseInputItem {
        match &call.payload {
            ToolPayload::Custom { .. } => ResponseInputItem::CustomToolCallOutput {
                call_id: call.call_id.clone(),
                output: Self::abort_message(call, secs),
            },
            ToolPayload::Mcp { .. } => ResponseInputItem::McpToolCallOutput {
                call_id: call.call_id.clone(),
                result: Err(Self::abort_message(call, secs)),
            },
            _ => ResponseInputItem::FunctionCallOutput {
                call_id: call.call_id.clone(),
                output: FunctionCallOutputPayload {
                    body: FunctionCallOutputBody::Text(Self::abort_message(call, secs)),
                    ..Default::default()
                },
            },
        }
    }

    fn abort_message(call: &ToolCall, secs: f32) -> String {
        match call.tool_name.as_str() {
            "shell" | "container.exec" | "local_shell" | "shell_command" | "unified_exec" => {
                format!("Wall time: {secs:.1} seconds\naborted by user")
            }
            _ => format!("aborted by user after {secs:.1}s"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "input_type", rename_all = "snake_case")]
enum HookToolInputPayload {
    Function {
        arguments: String,
    },
    Custom {
        input: String,
    },
    LocalShell {
        params: HookToolInputLocalShell,
    },
    Mcp {
        server: String,
        tool: String,
        arguments: String,
    },
}

impl HookToolInputPayload {
    fn input_type(&self) -> &'static str {
        match self {
            HookToolInputPayload::Function { .. } => "function",
            HookToolInputPayload::Custom { .. } => "custom",
            HookToolInputPayload::LocalShell { .. } => "local_shell",
            HookToolInputPayload::Mcp { .. } => "mcp",
        }
    }
}

impl From<&ToolPayload> for HookToolInputPayload {
    fn from(payload: &ToolPayload) -> Self {
        match payload {
            ToolPayload::Function { arguments } => Self::Function {
                arguments: arguments.clone(),
            },
            ToolPayload::Custom { input } => Self::Custom {
                input: input.clone(),
            },
            ToolPayload::LocalShell { params } => Self::LocalShell {
                params: HookToolInputLocalShell {
                    command: params.command.clone(),
                    workdir: params.workdir.clone(),
                    timeout_ms: params.timeout_ms,
                    sandbox_permissions: params.sandbox_permissions,
                    prefix_rule: params.prefix_rule.clone(),
                    justification: params.justification.clone(),
                },
            },
            ToolPayload::Mcp {
                server,
                tool,
                raw_arguments,
            } => Self::Mcp {
                server: server.clone(),
                tool: tool.clone(),
                arguments: raw_arguments.clone(),
            },
        }
    }
}

impl From<HookToolInputPayload> for ToolPayload {
    fn from(value: HookToolInputPayload) -> Self {
        match value {
            HookToolInputPayload::Function { arguments } => Self::Function { arguments },
            HookToolInputPayload::Custom { input } => Self::Custom { input },
            HookToolInputPayload::LocalShell { params } => Self::LocalShell {
                params: codex_protocol::models::ShellToolCallParams {
                    command: params.command,
                    workdir: params.workdir,
                    timeout_ms: params.timeout_ms,
                    sandbox_permissions: params.sandbox_permissions,
                    prefix_rule: params.prefix_rule,
                    justification: params.justification,
                },
            },
            HookToolInputPayload::Mcp {
                server,
                tool,
                arguments,
            } => Self::Mcp {
                server,
                tool,
                raw_arguments: arguments,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HookToolInputLocalShell {
    command: Vec<String>,
    workdir: Option<String>,
    timeout_ms: Option<u64>,
    sandbox_permissions: Option<SandboxPermissions>,
    prefix_rule: Option<Vec<String>>,
    justification: Option<String>,
}
