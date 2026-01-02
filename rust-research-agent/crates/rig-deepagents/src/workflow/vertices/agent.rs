//! AgentVertex: LLM-based agent node with tool calling capabilities
//!
//! Implements the Vertex trait for agent nodes that use LLMs to process
//! messages and can iteratively call tools until a stop condition is met.
//!
//! # Tool Execution
//!
//! Tools are executed through a `ToolRegistry` that maps tool names to
//! implementations. The vertex creates a minimal `ToolRuntime` for each
//! tool execution with the tool_call_id set for tracing.

use async_trait::async_trait;
use std::sync::Arc;

use crate::backends::MemoryBackend;
use crate::llm::{LLMConfig, LLMProvider};
use crate::middleware::{ToolDefinition, ToolRegistry};
use crate::pregel::error::PregelError;
use crate::pregel::message::WorkflowMessage;
use crate::pregel::state::WorkflowState;
use crate::pregel::vertex::{ComputeContext, ComputeResult, StateUpdate, Vertex, VertexId};
use crate::runtime::ToolRuntime;
use crate::state::{AgentState, Message, Role};
use crate::workflow::node::{AgentNodeConfig, StopCondition};

/// An agent vertex that uses an LLM to process messages and call tools
pub struct AgentVertex<S: WorkflowState> {
    id: VertexId,
    config: AgentNodeConfig,
    llm: Arc<dyn LLMProvider>,
    /// Tool registry for looking up and executing tools
    tool_registry: ToolRegistry,
    /// Tool definitions for LLM (cached from registry)
    tool_definitions: Vec<ToolDefinition>,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: WorkflowState> AgentVertex<S> {
    /// Create a new agent vertex with tool registry
    ///
    /// The registry provides both tool definitions (for LLM) and
    /// implementations (for execution).
    pub fn new_with_registry(
        id: impl Into<VertexId>,
        config: AgentNodeConfig,
        llm: Arc<dyn LLMProvider>,
        registry: ToolRegistry,
    ) -> Self {
        let tool_definitions = registry.definitions();
        Self {
            id: id.into(),
            config,
            llm,
            tool_registry: registry,
            tool_definitions,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new agent vertex with tool definitions only (no execution)
    ///
    /// This constructor is for backwards compatibility. Tools will not
    /// be executed - only their definitions are passed to the LLM.
    /// Use `new_with_registry` for full tool execution support.
    pub fn new(
        id: impl Into<VertexId>,
        config: AgentNodeConfig,
        llm: Arc<dyn LLMProvider>,
        tools: Vec<ToolDefinition>,
    ) -> Self {
        Self {
            id: id.into(),
            config,
            llm,
            tool_registry: ToolRegistry::new(),
            tool_definitions: tools,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a minimal ToolRuntime for tool execution
    fn create_tool_runtime(&self, tool_call_id: &str) -> ToolRuntime {
        let backend = Arc::new(MemoryBackend::new());
        let state = AgentState::new();
        ToolRuntime::new(state, backend).with_tool_call_id(tool_call_id)
    }

    /// Execute a tool and return the result
    async fn execute_tool(
        &self,
        tool_name: &str,
        args: serde_json::Value,
        tool_call_id: &str,
    ) -> Result<String, PregelError> {
        if let Some(tool) = self.tool_registry.get(tool_name) {
            let runtime = self.create_tool_runtime(tool_call_id);
            tool.execute(args, &runtime)
                .await
                .map_err(|e| PregelError::VertexError {
                    vertex_id: self.id.clone(),
                    message: format!("Tool '{}' execution failed: {}", tool_name, e),
                    source: None,
                })
        } else {
            // Tool not in registry - return error message as result
            // This allows the LLM to recover and try a different approach
            Ok(format!(
                "Error: Tool '{}' is not available. Available tools: {:?}",
                tool_name,
                self.tool_registry.names()
            ))
        }
    }

    /// Check if any stop condition is met
    ///
    /// # Arguments
    /// * `message` - The latest assistant message
    /// * `iteration` - Current iteration count
    /// * `state_json` - Serialized workflow state for StateMatch conditions
    fn check_stop_conditions(
        &self,
        message: &Message,
        iteration: usize,
        state_json: Option<&serde_json::Value>,
    ) -> bool {
        for condition in &self.config.stop_conditions {
            match condition {
                StopCondition::NoToolCalls => {
                    if message.tool_calls.is_none()
                        || message.tool_calls.as_ref().unwrap().is_empty()
                    {
                        return true;
                    }
                }
                StopCondition::OnTool { tool_name } => {
                    if let Some(tool_calls) = &message.tool_calls {
                        if tool_calls.iter().any(|tc| &tc.name == tool_name) {
                            return true;
                        }
                    }
                }
                StopCondition::ContainsText { pattern } => {
                    if message.content.contains(pattern) {
                        return true;
                    }
                }
                StopCondition::MaxIterations { count } => {
                    if iteration >= *count {
                        return true;
                    }
                }
                StopCondition::StateMatch { field, value } => {
                    // Check if state field matches the expected value
                    if let Some(state) = state_json {
                        if let Some(field_value) = self.get_state_field(state, field) {
                            if &field_value == value {
                                tracing::debug!(
                                    vertex_id = %self.id,
                                    field = %field,
                                    "StateMatch condition met"
                                );
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Get a field value from serialized state using dot notation
    ///
    /// Supports nested paths like "phase" or "research.status"
    fn get_state_field(
        &self,
        state: &serde_json::Value,
        field_path: &str,
    ) -> Option<serde_json::Value> {
        let parts: Vec<&str> = field_path.split('.').collect();
        let mut current = state;

        for part in parts {
            match current {
                serde_json::Value::Object(map) => {
                    current = map.get(part)?;
                }
                _ => return None,
            }
        }

        Some(current.clone())
    }

    /// Filter tools based on allowed list
    fn filter_tools(&self) -> Vec<ToolDefinition> {
        if let Some(allowed) = &self.config.allowed_tools {
            self.tool_definitions
                .iter()
                .filter(|t| allowed.contains(&t.name))
                .cloned()
                .collect()
        } else {
            self.tool_definitions.clone()
        }
    }

    /// Build LLM config from agent config
    fn build_llm_config(&self) -> Option<LLMConfig> {
        self.config.temperature.map(|temp| LLMConfig::new("").with_temperature(temp as f64))
    }
}

#[async_trait]
impl<S: WorkflowState + serde::Serialize> Vertex<S, WorkflowMessage> for AgentVertex<S> {
    fn id(&self) -> &VertexId {
        &self.id
    }

    async fn compute(
        &self,
        ctx: &mut ComputeContext<'_, S, WorkflowMessage>,
    ) -> Result<ComputeResult<S::Update>, PregelError> {
        // Build message history starting with system prompt
        let mut messages = vec![Message {
            role: Role::System,
            content: self.config.system_prompt.clone(),
            tool_calls: None,
            tool_call_id: None,
        }];

        // Add any incoming workflow messages as user messages
        for msg in ctx.messages {
            if let WorkflowMessage::Data { key: _, value } = msg {
                messages.push(Message {
                    role: Role::User,
                    content: value.to_string(),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
        }

        // If no user messages, add a default activation message
        if messages.len() == 1 {
            messages.push(Message {
                role: Role::User,
                content: "Begin processing.".to_string(),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        let filtered_tools = self.filter_tools();
        let llm_config = self.build_llm_config();

        // Serialize state for StateMatch conditions (once, outside the loop)
        let state_json = serde_json::to_value(ctx.state).ok();

        // Agent loop: iterate until stop condition or max iterations
        for iteration in 0..self.config.max_iterations {
            // Call LLM
            let response = self
                .llm
                .complete(&messages, &filtered_tools, llm_config.as_ref())
                .await
                .map_err(|e| PregelError::VertexError {
                    vertex_id: self.id.clone(),
                    message: e.to_string(),
                    source: Some(Box::new(e)),
                })?;

            let assistant_message = response.message.clone();
            messages.push(assistant_message.clone());

            // Check stop conditions (with state for StateMatch)
            if self.check_stop_conditions(&assistant_message, iteration, state_json.as_ref()) {
                // Send final response as output message
                ctx.send_message(
                    "output",
                    WorkflowMessage::Data {
                        key: "response".to_string(),
                        value: serde_json::Value::String(assistant_message.content),
                    },
                );
                return Ok(ComputeResult::halt(S::Update::empty()));
            }

            // If there are tool calls, execute them
            if let Some(tool_calls) = &assistant_message.tool_calls {
                for tool_call in tool_calls {
                    // Execute the tool and get the result
                    let result = self
                        .execute_tool(&tool_call.name, tool_call.arguments.clone(), &tool_call.id)
                        .await?;

                    // Add tool result message to conversation
                    messages.push(Message::tool(&result, &tool_call.id));

                    tracing::debug!(
                        vertex_id = %self.id,
                        tool = %tool_call.name,
                        result_len = result.len(),
                        "Tool executed successfully"
                    );
                }
            } else {
                // No tool calls and no stop condition matched, halt anyway
                ctx.send_message(
                    "output",
                    WorkflowMessage::Data {
                        key: "response".to_string(),
                        value: serde_json::Value::String(assistant_message.content),
                    },
                );
                return Ok(ComputeResult::halt(S::Update::empty()));
            }
        }

        // Max iterations reached
        Err(PregelError::vertex_error(
            self.id.clone(),
            "Max iterations reached",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DeepAgentError;
    use crate::llm::LLMResponse;
    use crate::pregel::state::UnitState;
    use crate::pregel::vertex::VertexState;
    use crate::state::ToolCall;
    use std::sync::Mutex;

    // Mock LLM provider for testing
    struct MockLLMProvider {
        responses: Arc<Mutex<Vec<Message>>>,
    }

    impl MockLLMProvider {
        fn new() -> Self {
            Self {
                responses: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn with_response(self, content: impl Into<String>) -> Self {
            let message = Message {
                role: Role::Assistant,
                content: content.into(),
                tool_calls: None,
                tool_call_id: None,
            };
            self.responses.lock().unwrap().push(message);
            self
        }

        fn with_tool_call(self, content: impl Into<String>, tool_name: impl Into<String>) -> Self {
            let message = Message {
                role: Role::Assistant,
                content: content.into(),
                tool_calls: Some(vec![ToolCall {
                    id: "test_call_1".to_string(),
                    name: tool_name.into(),
                    arguments: serde_json::json!({}),
                }]),
                tool_call_id: None,
            };
            self.responses.lock().unwrap().push(message);
            self
        }
    }

    #[async_trait]
    impl LLMProvider for MockLLMProvider {
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            _config: Option<&LLMConfig>,
        ) -> Result<LLMResponse, DeepAgentError> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                return Err(DeepAgentError::AgentExecution(
                    "No more mock responses".to_string(),
                ));
            }
            let message = responses.remove(0);
            Ok(LLMResponse::new(message))
        }

        fn name(&self) -> &str {
            "mock"
        }

        fn default_model(&self) -> &str {
            "mock-model"
        }
    }

    #[tokio::test]
    async fn test_agent_vertex_single_response() {
        let mock_llm = MockLLMProvider::new().with_response("Hello! How can I help?");

        let vertex = AgentVertex::<UnitState>::new(
            "agent",
            AgentNodeConfig {
                system_prompt: "You are helpful.".into(),
                stop_conditions: vec![StopCondition::NoToolCalls],
                ..Default::default()
            },
            Arc::new(mock_llm),
            vec![],
        );

        let mut ctx =
            ComputeContext::<UnitState, WorkflowMessage>::new("agent".into(), &[], 0, &UnitState);

        let result = vertex.compute(&mut ctx).await.unwrap();

        assert_eq!(result.state, VertexState::Halted);
        assert!(ctx.has_messages() || !ctx.into_outbox().is_empty());
    }

    #[tokio::test]
    async fn test_agent_vertex_stop_on_tool() {
        let mock_llm = MockLLMProvider::new().with_tool_call("Let me search for that", "search");

        let vertex = AgentVertex::<UnitState>::new(
            "agent",
            AgentNodeConfig {
                system_prompt: "You are a researcher.".into(),
                stop_conditions: vec![StopCondition::OnTool {
                    tool_name: "search".to_string(),
                }],
                ..Default::default()
            },
            Arc::new(mock_llm),
            vec![],
        );

        let mut ctx =
            ComputeContext::<UnitState, WorkflowMessage>::new("agent".into(), &[], 0, &UnitState);

        let result = vertex.compute(&mut ctx).await.unwrap();

        assert_eq!(result.state, VertexState::Halted);
    }

    #[tokio::test]
    async fn test_agent_vertex_max_iterations() {
        // Mock LLM that always returns tool calls (would loop forever without limit)
        let mut mock_llm = MockLLMProvider::new();
        for _ in 0..15 {
            mock_llm = mock_llm.with_tool_call("Still thinking...", "think");
        }

        let vertex = AgentVertex::<UnitState>::new(
            "agent",
            AgentNodeConfig {
                system_prompt: "You are helpful.".into(),
                max_iterations: 3,
                stop_conditions: vec![], // No stop conditions, relies on max_iterations
                ..Default::default()
            },
            Arc::new(mock_llm),
            vec![],
        );

        let mut ctx =
            ComputeContext::<UnitState, WorkflowMessage>::new("agent".into(), &[], 0, &UnitState);

        let result = vertex.compute(&mut ctx).await;

        // Should hit max iterations and return error
        assert!(result.is_err());
    }

    #[test]
    fn test_get_state_field_simple() {
        let vertex = AgentVertex::<UnitState>::new(
            "test",
            AgentNodeConfig::default(),
            Arc::new(MockLLMProvider::new()),
            vec![],
        );

        let state = serde_json::json!({
            "phase": "Exploratory",
            "can_continue": true,
            "search_count": 3
        });

        // Test simple field access
        assert_eq!(
            vertex.get_state_field(&state, "phase"),
            Some(serde_json::json!("Exploratory"))
        );
        assert_eq!(
            vertex.get_state_field(&state, "can_continue"),
            Some(serde_json::json!(true))
        );
        assert_eq!(
            vertex.get_state_field(&state, "search_count"),
            Some(serde_json::json!(3))
        );

        // Test missing field
        assert_eq!(vertex.get_state_field(&state, "missing"), None);
    }

    #[test]
    fn test_get_state_field_nested() {
        let vertex = AgentVertex::<UnitState>::new(
            "test",
            AgentNodeConfig::default(),
            Arc::new(MockLLMProvider::new()),
            vec![],
        );

        let state = serde_json::json!({
            "research": {
                "status": "active",
                "depth": 2
            }
        });

        // Test nested field access
        assert_eq!(
            vertex.get_state_field(&state, "research.status"),
            Some(serde_json::json!("active"))
        );
        assert_eq!(
            vertex.get_state_field(&state, "research.depth"),
            Some(serde_json::json!(2))
        );

        // Test missing nested field
        assert_eq!(vertex.get_state_field(&state, "research.missing"), None);
    }

    #[test]
    fn test_check_stop_conditions_state_match() {
        let vertex = AgentVertex::<UnitState>::new(
            "test",
            AgentNodeConfig {
                stop_conditions: vec![StopCondition::StateMatch {
                    field: "phase".to_string(),
                    value: serde_json::json!("Complete"),
                }],
                ..Default::default()
            },
            Arc::new(MockLLMProvider::new()),
            vec![],
        );

        let message = Message {
            role: Role::Assistant,
            content: "Done".to_string(),
            tool_calls: Some(vec![]),
            tool_call_id: None,
        };

        // State with non-matching phase
        let state_exploratory = serde_json::json!({"phase": "Exploratory"});
        assert!(!vertex.check_stop_conditions(&message, 0, Some(&state_exploratory)));

        // State with matching phase
        let state_complete = serde_json::json!({"phase": "Complete"});
        assert!(vertex.check_stop_conditions(&message, 0, Some(&state_complete)));

        // No state provided
        assert!(!vertex.check_stop_conditions(&message, 0, None));
    }
}
