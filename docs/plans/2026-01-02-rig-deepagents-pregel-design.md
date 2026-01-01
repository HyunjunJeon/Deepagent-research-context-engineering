# Rig-DeepAgents Pregel Runtime Design

> **Status:** Design Complete | **Created:** 2026-01-02 | **Author:** Claude + User Collaboration

## Executive Summary

This document describes the comprehensive design for transforming rig-deepagents into a **graph-based agent orchestration system** inspired by Google's Pregel. The design enables:

- **Superstep-based execution** with deterministic ordering
- **Message passing** between nodes for coordination
- **Fault tolerance** via checkpointing and recovery
- **Distributed-ready design** (single-node first, scalable later)

### Design Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Implementation Approach | Rust-Idiomatic Enhancement | Leverage Rust strengths while porting Python patterns |
| SubAgent Model | Type-Safe Enum | Compile-time guarantees via exhaustive matching |
| Skills Loading | Lazy Loading + Validation | Runtime flexibility with dev-time validation |
| Research Workflow | State Machine + Builder Hybrid | LangGraph-inspired with Pregel runtime |
| Runtime Complexity | Full Pregel | Supersteps, message passing, checkpointing, fault tolerance |

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Core Types & Pregel Primitives](#2-core-types--pregel-primitives)
3. [Pregel Runtime Implementation](#3-pregel-runtime-implementation)
4. [Checkpointer Backends](#4-checkpointer-backends)
5. [Node Types & Vertex Implementations](#5-node-types--vertex-implementations)
6. [Workflow Builder DSL](#6-workflow-builder-dsl)
7. [SubAgent System (Phase 8)](#7-subagent-system-phase-8)
8. [Skills Middleware (Phase 9a)](#8-skills-middleware-phase-9a)
9. [Domain Tools (Phase 9b)](#9-domain-tools-phase-9b)
10. [Research Workflow (Phase 10)](#10-research-workflow-phase-10)
11. [Implementation Roadmap](#11-implementation-roadmap)

---

## 1. Architecture Overview

### High-Level Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Rig-DeepAgents Stack                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Application Layer                                           │   │
│  │  • ResearchWorkflow, ExplorerWorkflow, CustomWorkflows       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ▲                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Workflow Builder (DSL)                                      │   │
│  │  • WorkflowGraph<S>::new().node().edge().build()            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ▲                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Pregel Runtime                                              │   │
│  │  • Superstep orchestration  • Message routing                │   │
│  │  • Checkpoint/Recovery      • Parallel execution             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ▲                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Core Components                                             │   │
│  │  • Nodes (Agent, Tool, SubAgent, Router)                    │   │
│  │  • Edges (Direct, Conditional, MessageBased)                │   │
│  │  • State (WorkflowState trait + checkpointing)              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ▲                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Foundation (Phase 1-7 Complete)                             │   │
│  │  • LLMProvider  • Middleware  • Backends  • Tools           │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
rust-research-agent/crates/rig-deepagents/src/
├── lib.rs
├── pregel/                    # NEW: Pregel Runtime
│   ├── mod.rs
│   ├── runtime.rs             # Superstep orchestration
│   ├── vertex.rs              # Vertex (node) abstraction
│   ├── message.rs             # Message passing types
│   ├── checkpoint/
│   │   ├── mod.rs             # Checkpointer trait & factory
│   │   ├── file.rs            # FileCheckpointer
│   │   ├── redis.rs           # RedisCheckpointer
│   │   ├── sqlite.rs          # SQLiteCheckpointer
│   │   └── postgres.rs        # PostgreSQLCheckpointer
│   └── config.rs              # Runtime configuration
├── workflow/                  # NEW: Workflow Builder
│   ├── mod.rs
│   ├── graph.rs               # WorkflowGraph builder
│   ├── node.rs                # NodeKind enum
│   ├── edge.rs                # Edge types
│   ├── compiled.rs            # CompiledWorkflow executor
│   ├── macros.rs              # Convenience macros
│   └── vertices/
│       ├── mod.rs
│       ├── agent.rs           # AgentVertex
│       ├── tool.rs            # ToolVertex
│       ├── subagent.rs        # SubAgentVertex
│       ├── router.rs          # RouterVertex
│       └── parallel.rs        # FanOut/FanIn vertices
├── subagent/                  # Phase 8: SubAgent System
│   ├── mod.rs
│   ├── definition.rs          # SubAgentKind enum
│   ├── registry.rs            # SubAgent registry
│   └── executor.rs            # SubAgent execution
├── skills/                    # Phase 9a: Skills Middleware
│   ├── mod.rs
│   ├── loader.rs              # Lazy loading
│   ├── middleware.rs          # SkillsMiddleware
│   └── validator.rs           # Validation logic
└── research/                  # Phase 10: Research Workflows
    ├── mod.rs
    ├── workflow.rs            # Pre-built research workflow
    ├── phases.rs              # Phase definitions
    └── state.rs               # ResearchState
```

---

## 2. Core Types & Pregel Primitives

### Concept Mapping

| Pregel Concept | Rust Implementation |
|----------------|---------------------|
| Vertex | `Vertex<S, M>` trait with `compute()` method |
| Edge | `Edge` enum (Direct, Conditional, Message) |
| Message | `VertexMessage<M>` generic over payload |
| Superstep | `SuperstepContext` with iteration tracking |
| Vote to Halt | `VertexState::Halted` vs `Active` |
| Combiner | `MessageCombiner` trait for aggregation |
| Aggregator | `GlobalAggregator` for cross-vertex state |

### Core Type Definitions

```rust
/// Unique identifier for a vertex in the graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VertexId(pub String);

/// Vertex execution state (Pregel's "vote to halt" mechanism)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VertexState {
    Active,     // Will compute in next superstep
    Halted,     // Voted to halt (reactivates on message)
    Completed,  // Will not reactivate
}

/// Context provided to vertex during computation
pub struct ComputeContext<'a, S, M> {
    pub superstep: usize,
    pub messages: &'a [M],
    pub state: &'a S,
    pub(crate) outbox: &'a mut VecDeque<(VertexId, M)>,
    pub(crate) llm: &'a Arc<dyn LLMProvider>,
    pub(crate) runtime: &'a ToolRuntime,
}

/// Core vertex trait
#[async_trait]
pub trait Vertex<S, M>: Send + Sync {
    fn id(&self) -> &VertexId;
    async fn compute(&self, ctx: &mut ComputeContext<'_, S, M>)
        -> Result<(S::Update, VertexState), PregelError>;
    fn combine_messages(&self, messages: Vec<M>) -> Vec<M> { messages }
}
```

### Workflow State Trait

```rust
pub trait WorkflowState: Clone + Send + Sync + Serialize + DeserializeOwned + 'static {
    type Update: StateUpdate<Self>;
    fn apply_update(&self, update: Self::Update) -> Self;
    fn merge_updates(updates: Vec<Self::Update>) -> Self::Update;
    fn is_terminal(&self) -> bool;
}
```

### Message Types

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkflowMessage {
    Activate,
    Data { key: String, value: serde_json::Value },
    Completed { source: VertexId, result: Option<String> },
    Halt,
    ResearchFinding { query: String, sources: Vec<Source>, summary: String },
    ResearchDirection { topic: String, priority: Priority, rationale: String },
}
```

---

## 3. Pregel Runtime Implementation

### Superstep Flow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Superstep   │───▶│  Superstep   │───▶│  Superstep   │──▶ ...
│      0       │    │      1       │    │      2       │
└──────────────┘    └──────────────┘    └──────────────┘

Per-Superstep:
1. Deliver Messages → 2. Parallel Compute → 3. Collect Updates
4. Checkpoint (optional) → 5. Merge Updates → 6. Route Messages
7. Check Termination
```

### Runtime Configuration

```rust
pub struct PregelConfig {
    pub max_supersteps: usize,           // Default: 100
    pub parallelism: usize,              // Default: num_cpus
    pub checkpoint_interval: usize,      // Default: 10 (0 = disabled)
    pub vertex_timeout: Duration,        // Default: 5 min
    pub workflow_timeout: Duration,      // Default: 1 hour
    pub tracing_enabled: bool,           // Default: true
    pub retry_policy: RetryPolicy,
}
```

### Core Runtime

```rust
pub struct PregelRuntime<S, M> {
    config: PregelConfig,
    vertices: HashMap<VertexId, Arc<dyn Vertex<S, M>>>,
    vertex_states: RwLock<HashMap<VertexId, VertexState>>,
    message_queues: RwLock<HashMap<VertexId, Vec<M>>>,
    semaphore: Arc<Semaphore>,  // Parallelism control
    checkpointer: Arc<dyn Checkpointer<S>>,
    llm: Arc<dyn LLMProvider>,
    tool_runtime: Arc<ToolRuntime>,
}

impl<S, M> PregelRuntime<S, M> {
    pub async fn run(&self, initial_state: S) -> Result<S, PregelError>;
    async fn execute_superstep(&self, superstep: usize, state: &S)
        -> Result<SuperstepResult<S, M>, PregelError>;
}
```

---

## 4. Checkpointer Backends

### Storage Options

| Backend | Latency | Durability | Scalability | Use Case |
|---------|---------|------------|-------------|----------|
| File | ~1ms | Disk-dependent | Single node | Development |
| Redis | ~0.5ms | Configurable | Cluster-ready | Fast distributed |
| SQLite | ~2ms | ACID | Single node | Portable embedded |
| PostgreSQL | ~5ms | ACID + replicas | Highly scalable | Production HA |

### Checkpointer Trait

```rust
#[async_trait]
pub trait Checkpointer<S: WorkflowState>: Send + Sync {
    async fn save(&self, checkpoint: &Checkpoint<S>) -> Result<(), PregelError>;
    async fn load_latest(&self) -> Result<Option<Checkpoint<S>>, PregelError>;
    async fn load(&self, superstep: usize) -> Result<Option<Checkpoint<S>>, PregelError>;
    async fn list(&self) -> Result<Vec<usize>, PregelError>;
    async fn prune(&self, keep_count: usize) -> Result<(), PregelError>;
}
```

### Factory Function

```rust
pub enum CheckpointerConfig {
    File { directory: PathBuf, compress: bool },
    Redis { url: String, prefix: String, ttl_secs: Option<u64> },
    SQLite { path: PathBuf },
    PostgreSQL { url: String, workflow_id: uuid::Uuid },
    InMemory,
}

pub async fn create_checkpointer<S: WorkflowState>(
    config: CheckpointerConfig,
) -> Result<Arc<dyn Checkpointer<S>>, PregelError>;
```

---

## 5. Node Types & Vertex Implementations

### NodeKind Type-Safe Enum

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKind {
    Agent(AgentNodeConfig),
    Tool(ToolNodeConfig),
    SubAgent(SubAgentNodeConfig),
    Router(RouterNodeConfig),
    FanOut(FanOutNodeConfig),
    FanIn(FanInNodeConfig),
    Transform(TransformNodeConfig),
}
```

### Node Configurations

**AgentNodeConfig:**
```rust
pub struct AgentNodeConfig {
    pub system_prompt: String,
    pub allowed_tools: HashSet<String>,
    pub max_iterations: usize,
    pub llm_config: Option<LLMConfig>,
    pub stop_conditions: Vec<StopCondition>,
}
```

**RouterNodeConfig:**
```rust
pub struct RouterNodeConfig {
    pub strategy: RoutingStrategy,  // StateField | LLMDecision | Custom
    pub branches: Vec<Branch>,
    pub default_branch: Option<String>,
}
```

**FanOut/FanIn for Parallel:**
```rust
pub struct FanOutNodeConfig {
    pub targets: Vec<VertexId>,
    pub split_strategy: SplitStrategy,  // Broadcast | SplitArray | Custom
}

pub struct FanInNodeConfig {
    pub sources: Vec<VertexId>,
    pub merge_strategy: MergeStrategy,  // CollectArray | DeepMerge | FirstSuccess
}
```

---

## 6. Workflow Builder DSL

### Fluent API

```rust
let workflow = WorkflowGraph::<ResearchState>::new()
    .name("research_workflow")
    .node("planner", NodeKind::Agent(AgentNodeConfig { ... }))
    .node("searcher", NodeKind::Agent(AgentNodeConfig { ... }))
    .node("router", NodeKind::Router(RouterNodeConfig { ... }))
    .entry("planner")
    .edge("planner", "searcher")
    .edge("searcher", "router")
    .conditional_on_field("router", "done", vec![
        (json!(true), "synthesizer"),
        (json!(false), "searcher"),
    ], None)
    .edge("synthesizer", END)
    .build()?;
```

### Convenience Macros

```rust
let workflow = workflow! {
    name: "autonomous_research",
    entry: "planner",
    nodes: {
        "planner" => agent!(PLANNER_PROMPT, tools: ["think"]),
        "explorer" => agent!(EXPLORER_PROMPT, tools: ["tavily_search", "think"]),
        "router" => router!(
            field: "coverage_sufficient",
            branches: { json!(true) => "synthesizer", json!(false) => "explorer" }
        ),
        "synthesizer" => agent!(SYNTH_PROMPT, tools: ["write_file"])
    },
    edges: [
        "planner" => "explorer",
        "explorer" => "router",
        "synthesizer" => END
    ]
}?;
```

### Validation at Build Time

- Entry point existence
- Edge reference validation
- Cycle detection (allowed but warned)
- Unreachable node warnings

---

## 7. SubAgent System (Phase 8)

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SubAgent System                                 │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐     ┌──────────────────┐                     │
│  │  SubAgentKind    │     │  SubAgentRegistry │                     │
│  │  ──────────────  │     │  ───────────────  │                     │
│  │  • Simple        │────▶│  • register()     │                     │
│  │  • Compiled      │     │  • get()          │                     │
│  └──────────────────┘     │  • list()         │                     │
│                           └────────┬─────────┘                      │
│                                    │                                 │
│                                    ▼                                 │
│                      ┌──────────────────────┐                       │
│                      │  SubAgentExecutor    │                       │
│                      │  • execute()         │                       │
│                      │  • execute_simple()  │                       │
│                      │  • execute_compiled()│                       │
│                      └──────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

### SubAgentKind Enum

```rust
/// SubAgent kind - Type-Safe Enum with exhaustive matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubAgentKind {
    /// Single LLM call with system prompt
    Simple {
        system_prompt: String,
        allowed_tools: Vec<String>,
    },

    /// Multi-turn autonomous agent with its own executor
    Compiled {
        system_prompt: String,
        allowed_tools: Vec<String>,
        max_iterations: usize,
        llm_config: Option<LLMConfig>,
    },
}

/// Full SubAgent definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentDefinition {
    pub name: String,
    pub description: String,
    pub kind: SubAgentKind,
    pub capabilities: HashSet<Capability>,
}

/// Type-safe capability markers
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    Research, Explore, Synthesize, Write, Search, ReadOnly, CodeReview, DataAnalysis,
}
```

### Pre-defined SubAgents

| Name | Type | Capabilities | Tools |
|------|------|--------------|-------|
| `explorer` | Simple | Explore, Search, ReadOnly | read_file, ls, glob, grep |
| `researcher` | Compiled (10 iter) | Research, Search, Synthesize | tavily_search, think |
| `synthesizer` | Simple | Synthesize, Write | read_file, write_file, think |

### SubAgent Registry

```rust
pub struct SubAgentRegistry {
    agents: RwLock<HashMap<String, SubAgentDefinition>>,
}

impl SubAgentRegistry {
    pub fn with_predefined() -> Self;
    pub async fn register(&self, definition: SubAgentDefinition);
    pub async fn get(&self, name: &str) -> Option<SubAgentDefinition>;
    pub async fn list(&self) -> Vec<SubAgentDefinition>;
    pub async fn find_by_capability(&self, cap: Capability) -> Vec<SubAgentDefinition>;
}
```

### SubAgent Executor

```rust
pub struct SubAgentExecutor {
    registry: Arc<SubAgentRegistry>,
    llm: Arc<dyn LLMProvider>,
    tool_runtime: Arc<ToolRuntime>,
}

impl SubAgentExecutor {
    /// Execute a SubAgent by name
    pub async fn execute(
        &self,
        agent_name: &str,
        input: &str,
        context: SubAgentContext,
    ) -> Result<String, DeepAgentError>;
}

/// Execution context with recursion tracking
pub struct SubAgentContext {
    pub recursion_depth: usize,
    pub max_recursion: usize,  // Default: 5
    pub parent_request_id: Option<String>,
    pub isolated_root: Option<PathBuf>,
}
```

### Execution Flow

```
execute(agent_name, input, context)
    │
    ├─── Check recursion limit
    │
    ├─── Lookup definition in registry
    │
    └─── Match SubAgentKind
            │
            ├─── Simple ──────────────────┐
            │    • Single LLM call        │
            │    • No tools               │
            │    • Return response        │
            │                             │
            └─── Compiled ────────────────┤
                 • Multi-turn loop        │
                 • Tool execution         │
                 • Nested task handling   │
                 • Max iterations check   │
                                          ▼
                                    Return result
```

---

## 8. Skills Middleware (Phase 9a)

### Design Philosophy: Progressive Disclosure

Skills follow a **lazy loading** pattern - only metadata is injected into the system prompt at startup, while full instructions are loaded on-demand via `read_file`.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Development Time                    Runtime                         │
│  ┌─────────────────────┐            ┌─────────────────────┐         │
│  │  skill-validator    │            │  SkillsMiddleware   │         │
│  │  ─────────────────  │            │  ─────────────────  │         │
│  │  • YAML frontmatter │            │  • Metadata only at │         │
│  │  • Required fields  │            │    startup          │         │
│  │  • File structure   │            │  • Lazy load full   │         │
│  │  • Markdown syntax  │            │    content via      │         │
│  │                     │            │    read_file tool   │         │
│  └──────────┬──────────┘            └─────────────────────┘         │
│             │                                                        │
│             ▼                                                        │
│  ┌─────────────────────┐                                            │
│  │  skills/            │◄─── Validated before commit                │
│  │  ├── web-research/  │                                            │
│  │  │   └── SKILL.md   │                                            │
│  │  └── data-synthesis/│                                            │
│  │      └── SKILL.md   │                                            │
│  └─────────────────────┘                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Skill Types

```rust
/// Skill metadata (parsed from YAML frontmatter)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMetadata {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
    pub source: SkillSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SkillSource {
    User,     // ~/.deepagents/{assistant}/skills/
    Project,  // {PROJECT_ROOT}/skills/
}

/// Full skill content (loaded on demand)
#[derive(Debug, Clone)]
pub struct SkillContent {
    pub metadata: SkillMetadata,
    pub instructions: String,
}
```

### Skill Loader

```rust
pub struct SkillLoader {
    user_skills_dir: PathBuf,
    project_skills_dir: Option<PathBuf>,
}

impl SkillLoader {
    /// List all skills (metadata only - progressive disclosure)
    pub async fn list_skills(&self) -> Result<Vec<SkillMetadata>, DeepAgentError>;

    /// Load full skill content by name (on-demand)
    pub async fn load_skill(&self, name: &str) -> Result<SkillContent, DeepAgentError>;

    /// Parse YAML frontmatter from SKILL.md
    fn parse_frontmatter(content: &str) -> Result<(String, String, String), DeepAgentError>;
}
```

### Skills Middleware

```rust
pub struct SkillsMiddleware {
    loader: SkillLoader,
    cached_metadata: RwLock<Vec<SkillMetadata>>,
}

impl AgentMiddleware for SkillsMiddleware {
    /// Inject skill list into system prompt
    fn modify_prompt(&self, prompt: &str, state: &AgentState) -> String {
        let skills_section = format!(r#"
## Skills System

**Available Skills:**
{skills_list}

**How to Use Skills (Progressive Disclosure):**
1. Identify when a skill applies
2. Read the skill's full instructions via `read_file` with the path shown
3. Follow the skill's instructions

Skills are self-documenting - each SKILL.md tells you exactly what to do.
"#, skills_list = self.format_skills_list());

        format!("{}\n\n{}", prompt, skills_section)
    }
}
```

### Skill Validator (CLI Tool)

```rust
// tools/skill-validator/src/main.rs

pub struct SkillValidation {
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
}

#[derive(Debug)]
pub enum ValidationError {
    MissingFrontmatter { path: PathBuf },
    MissingRequiredField { path: PathBuf, field: &'static str },
    InvalidYaml { path: PathBuf, error: String },
    EmptyInstructions { path: PathBuf },
}

#[derive(Debug)]
pub enum ValidationWarning {
    NoExamples { path: PathBuf },
    LongDescription { path: PathBuf, chars: usize },
    MissingHelperFiles { path: PathBuf },
}

fn main() {
    let skills_dir = std::env::args().nth(1).unwrap_or("skills".into());
    let validation = validate_skills_directory(&skills_dir);

    for error in &validation.errors {
        eprintln!("ERROR: {:?}", error);
    }
    for warning in &validation.warnings {
        eprintln!("WARNING: {:?}", warning);
    }

    std::process::exit(if validation.errors.is_empty() { 0 } else { 1 });
}
```

---

## 9. Domain Tools (Phase 9b)

### TavilySearchTool

```rust
/// Tavily web search tool
pub struct TavilySearchTool {
    client: reqwest::Client,
    api_key: String,
}

impl TavilySearchTool {
    pub fn from_env() -> Result<Self, DeepAgentError> {
        let api_key = std::env::var("TAVILY_API_KEY")
            .map_err(|_| DeepAgentError::Config("TAVILY_API_KEY not set".into()))?;
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
        })
    }
}

#[derive(Debug, Serialize)]
struct TavilyRequest {
    api_key: String,
    query: String,
    max_results: usize,
    search_depth: String,  // "basic" or "advanced"
    include_raw_content: bool,
    include_answer: bool,
}

#[derive(Debug, Deserialize)]
struct TavilyResponse {
    results: Vec<TavilyResult>,
    answer: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TavilyResult {
    title: String,
    url: String,
    content: String,
    raw_content: Option<String>,
    score: f32,
}

#[async_trait]
impl Tool for TavilySearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "tavily_search".to_string(),
            description: "Search the web using Tavily API. Returns relevant content with sources.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (default: 5)",
                        "default": 5
                    },
                    "topic": {
                        "type": "string",
                        "enum": ["general", "news"],
                        "default": "general"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn execute(&self, args: Value, _runtime: &ToolRuntime) -> Result<String, MiddlewareError> {
        let query = args["query"].as_str().unwrap_or("");
        let max_results = args["max_results"].as_u64().unwrap_or(5) as usize;
        let search_depth = if args["topic"] == "news" { "advanced" } else { "basic" };

        let request = TavilyRequest {
            api_key: self.api_key.clone(),
            query: query.to_string(),
            max_results,
            search_depth: search_depth.to_string(),
            include_raw_content: true,
            include_answer: true,
        };

        let response: TavilyResponse = self.client
            .post("https://api.tavily.com/search")
            .json(&request)
            .send()
            .await?
            .json()
            .await?;

        // Format as markdown
        let mut output = String::new();

        if let Some(answer) = response.answer {
            output.push_str(&format!("## Summary\n\n{}\n\n", answer));
        }

        output.push_str("## Sources\n\n");
        for result in response.results {
            output.push_str(&format!(
                "### [{}]({})\n\n{}\n\n",
                result.title, result.url, result.content
            ));
        }

        Ok(output)
    }
}
```

### ThinkTool

```rust
/// Think tool for explicit reasoning
pub struct ThinkTool;

#[async_trait]
impl Tool for ThinkTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "think".to_string(),
            description: "Use this tool for explicit reflection and reasoning before making decisions. \
                         This helps structure your thought process.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "reflection": {
                        "type": "string",
                        "description": "Your current thinking, reasoning, or reflection"
                    }
                },
                "required": ["reflection"]
            }),
        }
    }

    async fn execute(&self, args: Value, _runtime: &ToolRuntime) -> Result<String, MiddlewareError> {
        let reflection = args["reflection"].as_str().unwrap_or("");

        // Think tool just acknowledges the reflection - it's for prompting explicit reasoning
        Ok(format!(
            "Reflection recorded. Continue with your analysis based on this reasoning:\n\n> {}",
            reflection
        ))
    }
}
```

### Tool Registration

```rust
// In tools/mod.rs

pub fn create_domain_tools() -> Vec<Box<dyn Tool>> {
    let mut tools: Vec<Box<dyn Tool>> = vec![
        Box::new(ThinkTool),
    ];

    // Add Tavily if API key is available
    if let Ok(tavily) = TavilySearchTool::from_env() {
        tools.push(Box::new(tavily));
    }

    tools
}
```

---

## 10. Research Workflow (Phase 10)

### Research State

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResearchState {
    /// Current research phase
    pub phase: ResearchPhase,

    /// Original research query
    pub query: String,

    /// Identified research directions
    pub directions: Vec<ResearchDirection>,

    /// Collected findings
    pub findings: Vec<ResearchFinding>,

    /// Source tracking
    pub sources: Vec<Source>,

    /// Final synthesized report
    pub report: Option<String>,

    /// Metadata
    pub search_count: usize,
    pub max_searches: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum ResearchPhase {
    #[default]
    Planning,
    Exploratory,
    Directed,
    Synthesis,
    Complete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchDirection {
    pub topic: String,
    pub priority: Priority,
    pub rationale: String,
    pub explored: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchFinding {
    pub direction: String,
    pub query: String,
    pub summary: String,
    pub sources: Vec<Source>,
    pub confidence: f32,
}

impl WorkflowState for ResearchState {
    type Update = ResearchStateUpdate;

    fn apply_update(&self, update: Self::Update) -> Self { /* ... */ }
    fn merge_updates(updates: Vec<Self::Update>) -> Self::Update { /* ... */ }
    fn is_terminal(&self) -> bool {
        self.phase == ResearchPhase::Complete
    }
}
```

### Three-Phase Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Research Workflow                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐                                                       │
│  │ PLANNING │  Understand query, identify initial directions        │
│  └────┬─────┘                                                       │
│       │                                                              │
│       ▼                                                              │
│  ┌────────────────┐                                                 │
│  │  EXPLORATORY   │  1-2 broad searches to map the landscape       │
│  │  Phase 1       │  → Identify 2-4 research directions            │
│  └────────┬───────┘                                                 │
│           │                                                          │
│           ▼                                                          │
│  ┌────────────────┐                                                 │
│  │   DIRECTED     │  1-2 searches per direction                    │
│  │   Phase 2      │  → Deep dive into each area                    │
│  │                │  → Loop until coverage sufficient              │
│  └────────┬───────┘                                                 │
│           │         ◄─────────────────────────────┐                 │
│           │                                        │                 │
│           ▼                                        │                 │
│  ┌────────────────┐     ┌──────────────────────┐  │                 │
│  │ Coverage Check │────▶│ Need more research?  │──┘                 │
│  └────────┬───────┘     │ (< 3 directions)     │                    │
│           │ sufficient  └──────────────────────┘                    │
│           ▼                                                          │
│  ┌────────────────┐                                                 │
│  │   SYNTHESIS    │  Combine findings                               │
│  │   Phase 3      │  → Source agreement analysis                   │
│  │                │  → Generate structured report                  │
│  └────────┬───────┘                                                 │
│           │                                                          │
│           ▼                                                          │
│  ┌────────────────┐                                                 │
│  │   COMPLETE     │  Write final report to filesystem              │
│  └────────────────┘                                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Pre-built Research Workflow

```rust
/// Create the autonomous research workflow
pub fn create_research_workflow() -> Result<CompiledWorkflow<ResearchState>, WorkflowBuildError> {
    WorkflowGraph::<ResearchState>::new()
        .name("autonomous_research")
        .description("Three-phase research workflow: Exploratory → Directed → Synthesis")

        // Planning node
        .node("planner", NodeKind::Agent(AgentNodeConfig {
            system_prompt: include_str!("prompts/planner.md").into(),
            allowed_tools: hashset!["think".into()],
            max_iterations: 3,
            stop_conditions: vec![StopCondition::NoToolCalls],
            ..Default::default()
        }))

        // Exploratory search node
        .node("explorer", NodeKind::Agent(AgentNodeConfig {
            system_prompt: include_str!("prompts/explorer.md").into(),
            allowed_tools: hashset!["tavily_search".into(), "think".into()],
            max_iterations: 5,
            stop_conditions: vec![
                StopCondition::NoToolCalls,
                StopCondition::MaxIterations(5),
            ],
            ..Default::default()
        }))

        // Directed research node
        .node("researcher", NodeKind::Agent(AgentNodeConfig {
            system_prompt: include_str!("prompts/researcher.md").into(),
            allowed_tools: hashset!["tavily_search".into(), "think".into()],
            max_iterations: 8,
            stop_conditions: vec![
                StopCondition::NoToolCalls,
                StopCondition::MaxIterations(8),
            ],
            ..Default::default()
        }))

        // Coverage check router
        .node("coverage_check", NodeKind::Router(RouterNodeConfig {
            strategy: RoutingStrategy::StateField {
                path: "coverage_sufficient".into(),
            },
            branches: vec![
                Branch {
                    name: "continue".into(),
                    target: "researcher".into(),
                    condition: BranchCondition::Equals(json!(false)),
                },
                Branch {
                    name: "synthesize".into(),
                    target: "synthesizer".into(),
                    condition: BranchCondition::Equals(json!(true)),
                },
            ],
            default_branch: Some("synthesizer".into()),
        }))

        // Synthesis node
        .node("synthesizer", NodeKind::Agent(AgentNodeConfig {
            system_prompt: include_str!("prompts/synthesizer.md").into(),
            allowed_tools: hashset!["think".into(), "write_file".into()],
            max_iterations: 5,
            stop_conditions: vec![
                StopCondition::ToolCalled("write_file".into()),
            ],
            ..Default::default()
        }))

        // Edges
        .entry("planner")
        .edge("planner", "explorer")
        .edge("explorer", "researcher")
        .edge("researcher", "coverage_check")
        .edge("synthesizer", END)

        // Config
        .with_pregel_config(PregelConfig::default()
            .with_max_supersteps(50)
            .with_checkpoint_interval(5))

        .build()
}
```

### Usage Example

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create workflow
    let workflow = create_research_workflow()?;

    // Create providers
    let llm = Arc::new(OpenAIProvider::from_env()?);
    let tool_runtime = Arc::new(ToolRuntime::with_domain_tools());

    // Initial state
    let initial_state = ResearchState {
        query: "What are the latest developments in Rust async runtime?".into(),
        max_searches: 6,
        ..Default::default()
    };

    // Execute with checkpointing
    let checkpointer = create_checkpointer(CheckpointerConfig::SQLite {
        path: "research_checkpoints.db".into(),
    }).await?;

    let result = workflow
        .run_with_recovery(initial_state, llm, tool_runtime, checkpointer)
        .await?;

    println!("Research complete in {} supersteps", result.supersteps);
    println!("Report:\n{}", result.state.report.unwrap_or_default());

    Ok(())
}

---

## 11. Implementation Roadmap

### Phase Summary

| Phase | Description | Status | Est. Effort |
|-------|-------------|--------|-------------|
| 7 | LLM Provider Abstraction | ✅ Complete | - |
| 8 | SubAgent System | 📝 Designed | 4-6 hours |
| 8.1 | Pregel Runtime Core | 📝 Designed | 6-8 hours |
| 8.2 | Checkpointer Backends | 📝 Designed | 4-6 hours |
| 8.3 | Node Types & Vertices | 📝 Designed | 4-6 hours |
| 8.4 | Workflow Builder DSL | 📝 Designed | 3-4 hours |
| 9a | Skills Middleware | 📝 Designed | 3-4 hours |
| 9b | Domain Tools | 📝 Designed | 2-3 hours |
| 10 | Research Workflow | 📝 Designed | 4-6 hours |

**Total Estimated Effort:** 30-43 hours

### Dependency Graph

```
Phase 7 (Complete)
    │
    ├──▶ Phase 8.1: Pregel Runtime Core
    │         │
    │         ├──▶ Phase 8.2: Checkpointer Backends
    │         │
    │         └──▶ Phase 8.3: Node Types & Vertices
    │                   │
    │                   └──▶ Phase 8.4: Workflow Builder DSL
    │
    ├──▶ Phase 8: SubAgent System
    │
    ├──▶ Phase 9a: Skills Middleware
    │
    └──▶ Phase 9b: Domain Tools
              │
              └──▶ Phase 10: Research Workflow
```

### TDD Verification Process

For each implementation phase:
1. Write failing tests
2. Implement minimum code to pass
3. Refactor if needed
4. Run full test suite
5. Run clippy
6. Commit with descriptive message

---

## Appendix A: Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum PregelError {
    #[error("Max supersteps exceeded: {0}")]
    MaxSuperstepsExceeded(usize),

    #[error("Vertex timeout: {0:?}")]
    VertexTimeout(VertexId),

    #[error("Vertex error in {vertex_id:?}: {source}")]
    VertexError { vertex_id: VertexId, source: Box<dyn std::error::Error + Send + Sync> },

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

---

## Appendix B: Cargo Dependencies

```toml
[dependencies]
# Existing
rig-core = { version = "0.27", features = ["openai", "anthropic"] }
tokio = { version = "1", features = ["full"] }
async-trait = "0.1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "1"
tracing = "0.1"
futures = "0.3"

# New for Pregel
uuid = { version = "1", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
flate2 = "1"  # Compression
regex = "1"

# Checkpointer backends (optional features)
redis = { version = "0.25", features = ["aio", "tokio-comp"], optional = true }
sqlx = { version = "0.7", features = ["runtime-tokio", "sqlite", "postgres"], optional = true }

[features]
default = ["checkpointer-file"]
checkpointer-file = []
checkpointer-redis = ["redis"]
checkpointer-sqlite = ["sqlx/sqlite"]
checkpointer-postgres = ["sqlx/postgres"]
checkpointer-all = ["checkpointer-redis", "checkpointer-sqlite", "checkpointer-postgres"]
```

---

*Document generated through collaborative design session.*
