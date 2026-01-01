# Rig-DeepAgents Implementation Tasks

> **For Claude:** Use `superpowers:executing-plans` to implement tasks with TDD approach.
>
> **For Parallel Execution:** Tasks marked with same `[Parallel Group]` can be executed simultaneously by different SubAgents.

---

## Overview

This document breaks down the Pregel-based rig-deepagents implementation into concrete, executable tasks. Each task is designed for **SubAgent delegation** with clear boundaries and verification criteria.

### Task Notation

- `[P#]` - Priority (1=highest)
- `[Parallel Group: X]` - Tasks that can run in parallel
- `[Depends: Task-ID]` - Prerequisites
- `[Est: Xh]` - Estimated hours

---

## Phase 8.1: Pregel Runtime Core

### Task 8.1.1: Create Pregel Module Structure
**[P1] [Parallel Group: A] [Est: 0.5h]**

**Files to create:**
```
src/pregel/
├── mod.rs
├── vertex.rs
├── message.rs
├── config.rs
└── error.rs
```

**Acceptance Criteria:**
- [ ] Module compiles with `cargo check`
- [ ] All submodules exported in `mod.rs`
- [ ] `lib.rs` exports `pub mod pregel`

**Implementation:**
```rust
// src/pregel/mod.rs
pub mod vertex;
pub mod message;
pub mod config;
pub mod error;

pub use vertex::{Vertex, VertexId, VertexState, ComputeContext};
pub use message::{VertexMessage, WorkflowMessage};
pub use config::PregelConfig;
pub use error::PregelError;
```

---

### Task 8.1.2: Implement VertexId and VertexState
**[P1] [Parallel Group: A] [Est: 1h]**

**File:** `src/pregel/vertex.rs`

**Tests to write first:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_id_from_str() {
        let id: VertexId = "planner".into();
        assert_eq!(id.0, "planner");
    }

    #[test]
    fn test_vertex_id_equality() {
        let id1: VertexId = "node1".into();
        let id2: VertexId = "node1".into();
        let id3: VertexId = "node2".into();
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_vertex_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(VertexId::from("a"));
        set.insert(VertexId::from("b"));
        set.insert(VertexId::from("a")); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_vertex_state_default_is_active() {
        // New vertices should start as Active
        assert_eq!(VertexState::default(), VertexState::Active);
    }
}
```

**Acceptance Criteria:**
- [ ] `VertexId` implements `Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize`
- [ ] `VertexId` implements `From<&str>` and `From<String>`
- [ ] `VertexState` enum with `Active, Halted, Completed` variants
- [ ] All tests pass

---

### Task 8.1.3: Implement WorkflowMessage Types
**[P1] [Parallel Group: A] [Est: 1h]**

**File:** `src/pregel/message.rs`

**Tests to write first:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_message_serialization() {
        let msg = WorkflowMessage::Data {
            key: "query".into(),
            value: json!("test query"),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: WorkflowMessage = serde_json::from_str(&json).unwrap();
        // Verify roundtrip
        match deserialized {
            WorkflowMessage::Data { key, value } => {
                assert_eq!(key, "query");
                assert_eq!(value, json!("test query"));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_research_finding_message() {
        let msg = WorkflowMessage::ResearchFinding {
            query: "rust async".into(),
            sources: vec![Source {
                url: "https://example.com".into(),
                title: "Example".into(),
                relevance: 0.95,
            }],
            summary: "Rust async is great".into(),
        };
        assert!(matches!(msg, WorkflowMessage::ResearchFinding { .. }));
    }
}
```

**Acceptance Criteria:**
- [ ] `VertexMessage` trait defined
- [ ] `WorkflowMessage` enum with all variants
- [ ] `Source` and `Priority` types
- [ ] Serialization roundtrip works
- [ ] All tests pass

---

### Task 8.1.4: Implement PregelConfig
**[P1] [Parallel Group: A] [Est: 0.5h]**

**File:** `src/pregel/config.rs`

**Tests to write first:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PregelConfig::default();
        assert_eq!(config.max_supersteps, 100);
        assert!(config.parallelism > 0);
        assert_eq!(config.checkpoint_interval, 10);
    }

    #[test]
    fn test_config_builder() {
        let config = PregelConfig::default()
            .with_max_supersteps(50)
            .with_parallelism(4)
            .with_checkpoint_interval(5);

        assert_eq!(config.max_supersteps, 50);
        assert_eq!(config.parallelism, 4);
        assert_eq!(config.checkpoint_interval, 5);
    }
}
```

**Acceptance Criteria:**
- [ ] `PregelConfig` struct with all fields
- [ ] `RetryPolicy` struct
- [ ] Builder pattern methods
- [ ] Sensible defaults
- [ ] All tests pass

---

### Task 8.1.5: Implement PregelError
**[P1] [Parallel Group: A] [Est: 0.5h]**

**File:** `src/pregel/error.rs`

**Implementation:**
```rust
use thiserror::Error;
use super::vertex::VertexId;

#[derive(Debug, Error)]
pub enum PregelError {
    #[error("Max supersteps exceeded: {0}")]
    MaxSuperstepsExceeded(usize),

    #[error("Vertex timeout: {0:?}")]
    VertexTimeout(VertexId),

    #[error("Vertex error in {vertex_id:?}: {source}")]
    VertexError {
        vertex_id: VertexId,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Routing error in {vertex_id:?}: {decision}")]
    RoutingError { vertex_id: VertexId, decision: String },

    #[error("Recursion limit in {vertex_id:?}: depth {depth}, limit {limit}")]
    RecursionLimit { vertex_id: VertexId, depth: usize, limit: usize },

    #[error("State error: {0}")]
    StateError(String),

    #[error("Checkpoint error: {0}")]
    CheckpointError(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}
```

**Acceptance Criteria:**
- [ ] All error variants implemented
- [ ] `thiserror` derives working
- [ ] Errors are `Send + Sync`

---

### Task 8.1.6: Implement Vertex Trait
**[P2] [Depends: 8.1.2, 8.1.3] [Est: 2h]**

**File:** `src/pregel/vertex.rs` (extend)

**Tests to write first:**
```rust
#[tokio::test]
async fn test_mock_vertex_compute() {
    struct EchoVertex { id: VertexId }

    #[async_trait]
    impl Vertex<TestState, WorkflowMessage> for EchoVertex {
        fn id(&self) -> &VertexId { &self.id }

        async fn compute(
            &self,
            ctx: &mut ComputeContext<'_, TestState, WorkflowMessage>,
        ) -> Result<(TestUpdate, VertexState), PregelError> {
            // Echo back any data messages
            for msg in ctx.messages {
                if let WorkflowMessage::Data { key, value } = msg {
                    ctx.send_message(
                        "output".into(),
                        WorkflowMessage::Data {
                            key: format!("echo_{}", key),
                            value: value.clone(),
                        },
                    );
                }
            }
            Ok((TestUpdate::empty(), VertexState::Halted))
        }
    }

    // Test vertex behavior...
}
```

**Acceptance Criteria:**
- [ ] `Vertex` trait with async `compute` method
- [ ] `ComputeContext` struct with message sending
- [ ] Default `combine_messages` implementation
- [ ] Mock vertex tests pass

---

### Task 8.1.7: Implement WorkflowState Trait
**[P2] [Parallel Group: B] [Est: 1.5h]**

**File:** `src/pregel/state.rs` (new)

**Tests to write first:**
```rust
#[test]
fn test_state_update_merge() {
    #[derive(Clone, Default)]
    struct CounterState { count: i32 }

    #[derive(Clone)]
    struct CounterUpdate { delta: i32 }

    impl StateUpdate<CounterState> for CounterUpdate {
        fn empty() -> Self { CounterUpdate { delta: 0 } }
        fn is_empty(&self) -> bool { self.delta == 0 }
    }

    impl WorkflowState for CounterState {
        type Update = CounterUpdate;

        fn apply_update(&self, update: Self::Update) -> Self {
            CounterState { count: self.count + update.delta }
        }

        fn merge_updates(updates: Vec<Self::Update>) -> Self::Update {
            CounterUpdate {
                delta: updates.iter().map(|u| u.delta).sum(),
            }
        }

        fn is_terminal(&self) -> bool { false }
    }

    let updates = vec![
        CounterUpdate { delta: 5 },
        CounterUpdate { delta: 3 },
        CounterUpdate { delta: -2 },
    ];
    let merged = CounterState::merge_updates(updates);
    assert_eq!(merged.delta, 6);
}
```

**Acceptance Criteria:**
- [ ] `WorkflowState` trait defined
- [ ] `StateUpdate` trait defined
- [ ] Merge semantics work correctly
- [ ] All tests pass

---

### Task 8.1.8: Implement PregelRuntime Core
**[P2] [Depends: 8.1.1-8.1.7] [Est: 4h]**

**File:** `src/pregel/runtime.rs`

**Tests to write first:**
```rust
#[tokio::test]
async fn test_runtime_single_superstep() {
    // Create simple workflow: A -> B -> END
    let runtime = create_test_runtime();
    let initial = TestState::default();

    let result = runtime.run(initial).await.unwrap();

    assert!(result.completed);
}

#[tokio::test]
async fn test_runtime_message_delivery() {
    // Vertex A sends message to B
    // B should receive it in next superstep
    // ...
}

#[tokio::test]
async fn test_runtime_parallel_execution() {
    // Multiple vertices should execute in parallel
    // Verify via timing or execution order tracking
    // ...
}

#[tokio::test]
async fn test_runtime_termination_all_halted() {
    // Workflow terminates when all vertices halted + no messages
    // ...
}

#[tokio::test]
async fn test_runtime_max_supersteps_exceeded() {
    // Infinite loop should hit max_supersteps limit
    // ...
}
```

**Acceptance Criteria:**
- [ ] `PregelRuntime` struct
- [ ] `run()` method executes supersteps
- [ ] Message delivery between supersteps
- [ ] Parallel vertex execution with semaphore
- [ ] Termination conditions work
- [ ] All tests pass

---

## Phase 8.2: Checkpointer Backends

### Task 8.2.1: Checkpointer Trait & Types
**[P2] [Parallel Group: C] [Est: 1h]**

**File:** `src/pregel/checkpoint/mod.rs`

**Acceptance Criteria:**
- [ ] `Checkpoint<S>` struct
- [ ] `Checkpointer<S>` trait
- [ ] `CheckpointerConfig` enum
- [ ] `create_checkpointer` factory function

---

### Task 8.2.2: FileCheckpointer
**[P2] [Parallel Group: C] [Depends: 8.2.1] [Est: 2h]**

**File:** `src/pregel/checkpoint/file.rs`

**Tests to write first:**
```rust
#[tokio::test]
async fn test_file_checkpointer_save_load() {
    let temp_dir = tempfile::tempdir().unwrap();
    let cp = FileCheckpointer::new(temp_dir.path());

    let checkpoint = Checkpoint {
        superstep: 5,
        state: TestState { value: 42 },
        vertex_states: HashMap::new(),
        pending_messages: HashMap::new(),
        timestamp: Utc::now(),
    };

    cp.save(&checkpoint).await.unwrap();
    let loaded = cp.load(5).await.unwrap().unwrap();

    assert_eq!(loaded.superstep, 5);
    assert_eq!(loaded.state.value, 42);
}

#[tokio::test]
async fn test_file_checkpointer_compression() {
    // With compression enabled, files should be smaller
    // ...
}

#[tokio::test]
async fn test_file_checkpointer_prune() {
    // Create 10 checkpoints, prune to keep 3
    // ...
}
```

**Acceptance Criteria:**
- [ ] Save/load works
- [ ] Compression option works
- [ ] Atomic writes via temp file + rename
- [ ] Pruning works
- [ ] All tests pass

---

### Task 8.2.3: SQLiteCheckpointer
**[P3] [Parallel Group: C] [Depends: 8.2.1] [Est: 2h]**

**File:** `src/pregel/checkpoint/sqlite.rs`

**Feature flag:** `checkpointer-sqlite`

**Acceptance Criteria:**
- [ ] Schema auto-creation
- [ ] WAL mode enabled
- [ ] Compression support
- [ ] All CRUD operations
- [ ] In-memory option for testing

---

### Task 8.2.4: RedisCheckpointer
**[P3] [Parallel Group: C] [Depends: 8.2.1] [Est: 2h]**

**File:** `src/pregel/checkpoint/redis.rs`

**Feature flag:** `checkpointer-redis`

**Acceptance Criteria:**
- [ ] Connection pooling
- [ ] TTL support
- [ ] Atomic pipeline operations
- [ ] Sorted set index for listing

---

### Task 8.2.5: PostgreSQLCheckpointer
**[P3] [Parallel Group: C] [Depends: 8.2.1] [Est: 2h]**

**File:** `src/pregel/checkpoint/postgres.rs`

**Feature flag:** `checkpointer-postgres`

**Acceptance Criteria:**
- [ ] Schema auto-creation with indices
- [ ] JSONB metadata
- [ ] Upsert with ON CONFLICT
- [ ] Workflow isolation via workflow_id

---

## Phase 8.3: Node Types & Vertices

### Task 8.3.1: NodeKind Enum
**[P2] [Parallel Group: D] [Est: 1h]**

**File:** `src/workflow/node.rs`

**Acceptance Criteria:**
- [ ] All 7 variants defined
- [ ] Config structs for each variant
- [ ] Serialization works

---

### Task 8.3.2: AgentVertex Implementation
**[P2] [Depends: 8.1.6, 8.3.1] [Est: 3h]**

**File:** `src/workflow/vertices/agent.rs`

**Tests to write first:**
```rust
#[tokio::test]
async fn test_agent_vertex_single_response() {
    let vertex = AgentVertex::new("agent", AgentNodeConfig {
        system_prompt: "You are helpful.".into(),
        stop_conditions: vec![StopCondition::NoToolCalls],
        ..Default::default()
    });

    let mock_llm = MockLLMProvider::new()
        .with_response("Hello! How can I help?");

    // Execute and verify...
}

#[tokio::test]
async fn test_agent_vertex_tool_loop() {
    // Agent calls tool, gets result, responds
    // ...
}

#[tokio::test]
async fn test_agent_vertex_stop_on_tool() {
    // Stop when specific tool is called
    // ...
}
```

**Acceptance Criteria:**
- [ ] LLM calling works
- [ ] Tool filtering by allowed list
- [ ] Stop conditions evaluated
- [ ] Multi-turn iteration
- [ ] All tests pass

---

### Task 8.3.3: ToolVertex Implementation
**[P2] [Parallel Group: D] [Depends: 8.1.6] [Est: 1h]**

**File:** `src/workflow/vertices/tool.rs`

**Acceptance Criteria:**
- [ ] Single tool execution
- [ ] Argument building from static + state
- [ ] Result stored to state path

---

### Task 8.3.4: RouterVertex Implementation
**[P2] [Parallel Group: D] [Depends: 8.1.6] [Est: 2h]**

**File:** `src/workflow/vertices/router.rs`

**Acceptance Criteria:**
- [ ] StateField routing
- [ ] LLMDecision routing
- [ ] Branch conditions (Equals, In, Matches)
- [ ] Default branch fallback

---

### Task 8.3.5: SubAgentVertex Implementation
**[P2] [Depends: 8.3.2, Phase 8 SubAgent] [Est: 2h]**

**File:** `src/workflow/vertices/subagent.rs`

**Acceptance Criteria:**
- [ ] Input/Output mapping
- [ ] Recursion depth tracking
- [ ] Isolated context

---

### Task 8.3.6: FanOut/FanIn Vertices
**[P3] [Parallel Group: D] [Depends: 8.1.6] [Est: 2h]**

**File:** `src/workflow/vertices/parallel.rs`

**Acceptance Criteria:**
- [ ] FanOut broadcasts to targets
- [ ] FanIn waits for all sources
- [ ] Split/Merge strategies

---

## Phase 8.4: Workflow Builder DSL

### Task 8.4.1: WorkflowGraph Builder
**[P2] [Depends: 8.3.1] [Est: 3h]**

**File:** `src/workflow/graph.rs`

**Tests to write first:**
```rust
#[test]
fn test_workflow_builder_basic() {
    let graph = WorkflowGraph::<TestState>::new()
        .name("test")
        .node("a", NodeKind::Agent(Default::default()))
        .node("b", NodeKind::Agent(Default::default()))
        .entry("a")
        .edge("a", "b")
        .edge("b", END);

    assert!(graph.build().is_ok());
}

#[test]
fn test_workflow_builder_missing_entry() {
    let graph = WorkflowGraph::<TestState>::new()
        .node("a", NodeKind::Agent(Default::default()));

    assert!(matches!(
        graph.build(),
        Err(WorkflowBuildError::NoEntryPoint)
    ));
}

#[test]
fn test_workflow_builder_invalid_edge() {
    let graph = WorkflowGraph::<TestState>::new()
        .node("a", NodeKind::Agent(Default::default()))
        .entry("a")
        .edge("a", "nonexistent");

    assert!(matches!(
        graph.build(),
        Err(WorkflowBuildError::UnknownNode(_))
    ));
}
```

**Acceptance Criteria:**
- [ ] Fluent builder API
- [ ] Node and edge management
- [ ] Entry point setting
- [ ] All tests pass

---

### Task 8.4.2: Workflow Validation
**[P2] [Depends: 8.4.1] [Est: 1.5h]**

**File:** `src/workflow/graph.rs` (extend)

**Acceptance Criteria:**
- [ ] Entry point validation
- [ ] Edge reference validation
- [ ] Cycle detection
- [ ] Unreachable node warnings

---

### Task 8.4.3: CompiledWorkflow
**[P2] [Depends: 8.4.1, 8.1.8] [Est: 2h]**

**File:** `src/workflow/compiled.rs`

**Acceptance Criteria:**
- [ ] Compile from WorkflowGraph
- [ ] Create vertices from NodeKind
- [ ] Create edges
- [ ] `run()` delegates to PregelRuntime

---

### Task 8.4.4: Workflow Macros
**[P3] [Depends: 8.4.1] [Est: 1h]**

**File:** `src/workflow/macros.rs`

**Acceptance Criteria:**
- [ ] `workflow!` macro
- [ ] `agent!` macro
- [ ] `router!` macro
- [ ] `hashset!` macro

---

## Phase 8: SubAgent System

### Task 8.5.1: SubAgentKind Enum
**[P2] [Parallel Group: E] [Est: 1h]**

**File:** `src/subagent/definition.rs`

**Acceptance Criteria:**
- [ ] `Simple` and `Compiled` variants
- [ ] `SubAgentDefinition` struct
- [ ] `Capability` enum

---

### Task 8.5.2: SubAgent Registry
**[P2] [Parallel Group: E] [Depends: 8.5.1] [Est: 1.5h]**

**File:** `src/subagent/registry.rs`

**Acceptance Criteria:**
- [ ] Thread-safe registry with `RwLock`
- [ ] Register/get/list operations
- [ ] Pre-defined agents (researcher, explorer, synthesizer)

---

### Task 8.5.3: SubAgent Executor
**[P2] [Depends: 8.5.1, 8.5.2, 8.1.8] [Est: 3h]**

**File:** `src/subagent/executor.rs`

**Tests to write first:**
```rust
#[tokio::test]
async fn test_simple_subagent_execution() {
    let executor = SubAgentExecutor::new(mock_llm(), mock_registry());

    let result = executor.execute(
        "explorer",
        "Find information about Rust",
        SubAgentContext::default(),
    ).await.unwrap();

    assert!(!result.is_empty());
}

#[tokio::test]
async fn test_compiled_subagent_execution() {
    // Multi-turn with tool calls
    // ...
}

#[tokio::test]
async fn test_recursion_limit() {
    // SubAgent calling another SubAgent should hit limit
    // ...
}
```

**Acceptance Criteria:**
- [ ] Simple agent execution (single LLM call)
- [ ] Compiled agent execution (loop with tools)
- [ ] Recursion depth tracking
- [ ] All tests pass

---

## Phase 9a: Skills Middleware

### Task 9a.1: Skill Types
**[P3] [Parallel Group: F] [Est: 0.5h]**

**File:** `src/skills/types.rs`

**Acceptance Criteria:**
- [ ] `SkillMetadata` struct
- [ ] `SkillContent` struct

---

### Task 9a.2: Skill Loader (Lazy)
**[P3] [Parallel Group: F] [Depends: 9a.1] [Est: 2h]**

**File:** `src/skills/loader.rs`

**Acceptance Criteria:**
- [ ] YAML frontmatter parsing
- [ ] Metadata-only listing
- [ ] On-demand full content loading
- [ ] User + Project skills support

---

### Task 9a.3: Skills Middleware
**[P3] [Depends: 9a.2] [Est: 2h]**

**File:** `src/skills/middleware.rs`

**Acceptance Criteria:**
- [ ] Implements `AgentMiddleware`
- [ ] Injects skill list into system prompt
- [ ] Progressive disclosure pattern

---

### Task 9a.4: Skill Validator (CLI Tool)
**[P3] [Depends: 9a.1] [Est: 2h]**

**File:** `tools/skill-validator/src/main.rs`

**Acceptance Criteria:**
- [ ] Validates YAML frontmatter
- [ ] Checks required fields
- [ ] Reports errors and warnings
- [ ] Exit code for CI integration

---

## Phase 9b: Domain Tools

### Task 9b.1: TavilySearchTool
**[P3] [Parallel Group: G] [Est: 2h]**

**File:** `src/tools/tavily.rs`

**Acceptance Criteria:**
- [ ] API client implementation
- [ ] Tool definition with schema
- [ ] Result formatting as markdown
- [ ] Error handling

---

### Task 9b.2: ThinkTool
**[P3] [Parallel Group: G] [Est: 0.5h]**

**File:** `src/tools/think.rs`

**Acceptance Criteria:**
- [ ] Simple reflection recording
- [ ] Tool definition
- [ ] Returns reflection as confirmation

---

## Phase 10: Research Workflow

### Task 10.1: ResearchState Definition
**[P3] [Depends: 8.1.7] [Est: 1h]**

**File:** `src/research/state.rs`

**Acceptance Criteria:**
- [ ] Implements `WorkflowState`
- [ ] Research phases tracking
- [ ] Findings collection
- [ ] Source tracking

---

### Task 10.2: Research Prompts
**[P3] [Parallel Group: H] [Est: 1.5h]**

**File:** `src/research/prompts.rs`

**Acceptance Criteria:**
- [ ] Planner prompt
- [ ] Explorer prompt
- [ ] Researcher prompt
- [ ] Synthesizer prompt

---

### Task 10.3: Pre-built Research Workflow
**[P3] [Depends: 8.4.3, 10.1, 10.2] [Est: 3h]**

**File:** `src/research/workflow.rs`

**Acceptance Criteria:**
- [ ] Three-phase workflow implementation
- [ ] Configurable search limits
- [ ] Coverage checking router
- [ ] End-to-end test with mocks

---

## Parallel Execution Groups

For maximum efficiency, run these groups concurrently:

| Group | Tasks | Total Est. | Dependencies |
|-------|-------|-----------|--------------|
| **A** | 8.1.1, 8.1.2, 8.1.3, 8.1.4, 8.1.5 | 3.5h | None |
| **B** | 8.1.7 | 1.5h | None |
| **C** | 8.2.1, 8.2.2, 8.2.3, 8.2.4, 8.2.5 | 9h | 8.1.x |
| **D** | 8.3.1, 8.3.3, 8.3.4, 8.3.6 | 6h | 8.1.6 |
| **E** | 8.5.1, 8.5.2 | 2.5h | None |
| **F** | 9a.1, 9a.2 | 2.5h | None |
| **G** | 9b.1, 9b.2 | 2.5h | None |
| **H** | 10.2 | 1.5h | None |

### Recommended Execution Order

```
┌─────────────────────────────────────────────────────────────────┐
│  Wave 1 (Parallel): Groups A, B, E, F, G, H                     │
│  Total: ~13.5h across 6 SubAgents = ~2.5h wall time            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Wave 2 (Sequential): 8.1.6 (Vertex Trait) - 2h                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Wave 3 (Parallel): Groups C, D                                 │
│  Total: ~15h across 2 SubAgents = ~7.5h wall time              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Wave 4 (Sequential): 8.1.8 (Runtime), 8.3.2, 8.3.5            │
│  Total: ~9h sequential                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Wave 5 (Sequential): 8.4.x, 8.5.3, 9a.3-4, 10.1, 10.3        │
│  Total: ~14h sequential                                         │
└─────────────────────────────────────────────────────────────────┘
```

**Estimated Total Wall Time:** ~35h (with parallelism)
**Estimated Total Dev Time:** ~50h (sum of all tasks)

---

## Verification Checklist

After each task:
- [ ] All new tests pass: `cargo test`
- [ ] No warnings: `cargo clippy -- -D warnings`
- [ ] Documentation builds: `cargo doc`
- [ ] Commit with descriptive message

After each phase:
- [ ] Integration tests pass
- [ ] README updated if needed
- [ ] CHANGELOG entry added

---

*Task breakdown generated 2026-01-02*
