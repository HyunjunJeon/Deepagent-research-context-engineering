//! rig-deepagents: DeepAgents-style middleware system for Rig
//!
//! LangChain DeepAgents의 핵심 패턴을 Rust로 구현합니다.
//! - AgentMiddleware 트레이트: 도구 자동 주입, 프롬프트 수정
//! - Backend 트레이트: 파일시스템 추상화
//! - MiddlewareStack: 미들웨어 조합 및 실행
//! - AgentExecutor: LLM 호출 및 도구 실행 루프
//! - LLMProvider: Provider-agnostic LLM abstraction (OpenAI, Anthropic)
//!
//! # LLM Providers
//!
//! The library provides a unified interface for LLM providers:
//!
//! ```rust,ignore
//! use rig_deepagents::{OpenAIProvider, AnthropicProvider, LLMProvider};
//!
//! // OpenAI (from OPENAI_API_KEY env var)
//! let openai = OpenAIProvider::from_env()?;
//!
//! // Anthropic (from ANTHROPIC_API_KEY env var)
//! let anthropic = AnthropicProvider::from_env()?;
//!
//! // Use with AgentExecutor
//! let executor = AgentExecutor::new(Arc::new(openai), middleware, backend);
//! ```

pub mod error;
pub mod state;
pub mod backends;
pub mod middleware;
pub mod runtime;
pub mod executor;
pub mod tools;
pub mod llm;
pub mod pregel;
pub mod workflow;
pub mod skills;
pub mod research;
pub mod config;

// Re-exports for convenience
pub use error::{BackendError, MiddlewareError, DeepAgentError, WriteResult, EditResult};
pub use state::{AgentState, Message, Role, Todo, TodoStatus, FileData, ToolCall};
pub use backends::{Backend, FileInfo, GrepMatch, MemoryBackend, FilesystemBackend, CompositeBackend};
pub use middleware::{AgentMiddleware, MiddlewareStack, StateUpdate, Tool, ToolDefinition, ToolRegistry, DynTool};
pub use runtime::{ToolRuntime, RuntimeConfig};
pub use tools::{
    ReadFileTool, WriteFileTool, EditFileTool,
    LsTool, GlobTool, GrepTool,
    WriteTodosTool, TaskTool,
    default_tools, all_tools,
    // Domain tools
    TavilySearchTool, TavilyError, SearchDepth, Topic,
    ThinkTool,
    research_tools, research_tools_with_tavily,
};
pub use executor::AgentExecutor;

// Research workflow exports
pub use research::{
    ResearchState, ResearchUpdate, ResearchPhase,
    ResearchDirection, Finding, Source, SourceAgreement,
    ResearchWorkflowBuilder, ResearchConfig,
    ResearchPrompts, PromptBuilder,
    can_continue_research, determine_next_phase, phase_transition_update,
};

// Production configuration exports
pub use config::{ProductionConfig, ProductionSetup, LLMProviderType};

// LLM Provider exports
pub use llm::{
    LLMProvider, LLMResponse, LLMResponseStream, MessageChunk,
    LLMConfig, TokenUsage,
    OpenAIProvider, AnthropicProvider,
    MessageConverter, ToolConverter, convert_messages, convert_tools,
};
