//! 미들웨어 모듈
//!
//! AgentMiddleware 트레이트와 구현체들을 제공합니다.
//!
//! # Available Middleware
//!
//! - [`subagent`]: SubAgent delegation middleware (task tool, registry, execution)
//!
//! # Planned Middleware (Not Yet Implemented)
//!
//! - `summarization`: Token budget management and context summarization
//! - `todolist`: Todo list management middleware
//! - `filesystem`: Filesystem tools middleware
//! - `skills`: Skills progressive disclosure middleware

pub mod traits;
pub mod stack;
pub mod subagent;

// 추후 구현될 미들웨어들
// pub mod todolist;
// pub mod filesystem;
// pub mod summarization;
// pub mod skills;

// Core traits and types
pub use traits::{AgentMiddleware, DynTool, Tool, ToolDefinition, ToolRegistry, StateUpdate};
pub use stack::MiddlewareStack;

// SubAgent types
pub use subagent::{
    SubAgentSpec, SubAgentSpecBuilder, CompiledSubAgent, SubAgentKind,
    SubAgentRegistry, SubAgentResult, IsolatedState, IsolatedStateBuilder,
    EXCLUDED_STATE_KEYS, TASK_SYSTEM_PROMPT,
    // Executor types
    SubAgentExecutorFactory, SubAgentExecutorConfig, DefaultSubAgentExecutorFactory,
    // Task tool
    TaskTool, TaskArgs,
    // Middleware
    SubAgentMiddleware, SubAgentMiddlewareConfig, SubAgentMiddlewareBuilder,
};
