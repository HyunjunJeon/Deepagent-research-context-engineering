// src/middleware/traits.rs
//! AgentMiddleware 트레이트 정의
//!
//! Python Reference: langchain/agents/middleware/types.py

use async_trait::async_trait;
use std::sync::Arc;
use std::collections::HashMap;
use crate::state::{AgentState, Message, Todo, FileData};
use crate::error::MiddlewareError;
use crate::runtime::ToolRuntime;

/// 상태 업데이트 커맨드
/// Python: langgraph.types.Command
#[derive(Debug, Clone)]
pub enum StateUpdate {
    /// 메시지 추가
    AddMessages(Vec<Message>),
    /// Todo 업데이트
    SetTodos(Vec<Todo>),
    /// 파일 업데이트 (None = 삭제)
    UpdateFiles(HashMap<String, Option<FileData>>),
    /// 복합 업데이트
    Batch(Vec<StateUpdate>),
}

/// 도구 정의
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// 도구 인터페이스
#[async_trait]
pub trait Tool: Send + Sync {
    /// 도구 정의 반환
    fn definition(&self) -> ToolDefinition;

    /// 도구 실행
    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError>;
}

/// 동적 도구 타입
pub type DynTool = Arc<dyn Tool>;

/// Tool registry for managing tool implementations
///
/// Maps tool names to their implementations for execution.
///
/// # Example
///
/// ```ignore
/// use rig_deepagents::middleware::{ToolRegistry, DynTool};
/// use rig_deepagents::tools::ThinkTool;
/// use std::sync::Arc;
///
/// let mut registry = ToolRegistry::new();
/// registry.register(Arc::new(ThinkTool));
///
/// // Look up and execute
/// if let Some(tool) = registry.get("think") {
///     let result = tool.execute(args, &runtime).await?;
/// }
/// ```
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, DynTool>,
}

impl ToolRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool implementation
    pub fn register(&mut self, tool: DynTool) {
        let name = tool.definition().name;
        self.tools.insert(name, tool);
    }

    /// Register multiple tools at once
    pub fn register_all(&mut self, tools: Vec<DynTool>) {
        for tool in tools {
            self.register(tool);
        }
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<&DynTool> {
        self.tools.get(name)
    }

    /// Get all tool definitions (schemas) for LLM
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    /// Get all tool names
    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a tool exists
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Number of registered tools
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// AgentMiddleware 트레이트
///
/// Python Reference: AgentMiddleware(Generic[StateT, ContextT])
///
/// 핵심 기능:
/// - tools(): 에이전트에 자동 주입할 도구 목록 반환
/// - modify_system_prompt(): 시스템 프롬프트 수정 (체이닝)
/// - before_agent() / after_agent(): 라이프사이클 훅
#[async_trait]
pub trait AgentMiddleware: Send + Sync {
    /// 미들웨어 이름
    fn name(&self) -> &str;

    /// 이 미들웨어가 제공하는 도구 목록
    fn tools(&self) -> Vec<DynTool> {
        vec![]
    }

    /// 시스템 프롬프트 수정
    fn modify_system_prompt(&self, prompt: String) -> String {
        prompt
    }

    /// 에이전트 실행 전 훅
    /// Python: before_agent(self, state, runtime) -> dict | None
    async fn before_agent(
        &self,
        _state: &mut AgentState,
        _runtime: &ToolRuntime,
    ) -> Result<Option<StateUpdate>, MiddlewareError> {
        Ok(None)
    }

    /// 에이전트 실행 후 훅
    /// Python: after_agent(self, state, runtime) -> dict | None
    async fn after_agent(
        &self,
        _state: &mut AgentState,
        _runtime: &ToolRuntime,
    ) -> Result<Option<StateUpdate>, MiddlewareError> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTool;

    #[async_trait]
    impl Tool for MockTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "mock_tool".to_string(),
                description: "A mock tool for testing".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            }
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
            _runtime: &ToolRuntime,
        ) -> Result<String, MiddlewareError> {
            Ok("mock result".to_string())
        }
    }

    struct MockMiddleware;

    #[async_trait]
    impl AgentMiddleware for MockMiddleware {
        fn name(&self) -> &str {
            "mock"
        }

        fn tools(&self) -> Vec<DynTool> {
            vec![Arc::new(MockTool)]
        }

        fn modify_system_prompt(&self, prompt: String) -> String {
            format!("{}\n\nMock middleware addition.", prompt)
        }
    }

    #[test]
    fn test_middleware_tools() {
        let middleware = MockMiddleware;
        let tools = middleware.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].definition().name, "mock_tool");
    }

    #[test]
    fn test_middleware_prompt_modification() {
        let middleware = MockMiddleware;
        let result = middleware.modify_system_prompt("Base prompt".to_string());
        assert!(result.contains("Base prompt"));
        assert!(result.contains("Mock middleware addition"));
    }
}
