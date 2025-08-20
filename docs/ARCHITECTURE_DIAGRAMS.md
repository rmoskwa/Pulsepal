# PulsePal Architecture Diagrams

## System Architecture

```mermaid
graph TB
    subgraph "User Interfaces"
        CLI[CLI Interface<br/>run_pulsepal.py]
        WEB[Web Interface<br/>chainlit_app.py]
    end

    subgraph "Core Agent"
        AGENT[PulsePal Agent<br/>main_agent.py]
        DEPS[Dependencies<br/>dependencies.py]
        TOOLS[Tools<br/>tools.py]
    end

    subgraph "Services"
        RAG[RAG Service<br/>rag_service.py]
        SESSION[Session Manager]
        SETTINGS[Settings<br/>settings.py]
    end

    subgraph "External Services"
        GEMINI[Google Gemini<br/>2.5 Flash]
        SUPABASE[Supabase<br/>Vector DB]
        EMBEDDINGS[Google<br/>Embeddings API]
    end

    CLI --> AGENT
    WEB --> AGENT
    AGENT --> DEPS
    AGENT --> TOOLS
    TOOLS --> RAG
    DEPS --> SESSION
    AGENT --> SETTINGS
    RAG --> SUPABASE
    RAG --> EMBEDDINGS
    AGENT --> GEMINI
    SETTINGS --> GEMINI
```

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Interface as CLI/Web
    participant Agent as PulsePal Agent
    participant Router as Semantic Router
    participant RAG as RAG Service
    participant Gemini as Google Gemini
    participant Supabase as Supabase DB

    User->>Interface: Query
    Interface->>Agent: Process Query
    Agent->>Router: Analyze Query Type

    alt Needs Documentation
        Router->>Agent: Force RAG Search
        Agent->>RAG: Search Knowledge
        RAG->>Supabase: Vector Search
        Supabase-->>RAG: Results
        RAG-->>Agent: Formatted Results
    else Built-in Knowledge
        Router->>Agent: Skip RAG
    end

    Agent->>Gemini: Generate Response
    Gemini-->>Agent: Response
    Agent-->>Interface: Final Answer
    Interface-->>User: Display Result
```

## Module Relationships

```mermaid
graph LR
    subgraph "Entry Points"
        CLI[run_pulsepal.py]
        WEB[chainlit_app.py]
    end

    subgraph "Core Modules"
        MAIN[main_agent.py]
        TOOLS[tools.py]
        DEPS[dependencies.py]
    end

    subgraph "Service Modules"
        RAG[rag_service.py]
        SETTINGS[settings.py]
        PROVIDERS[providers.py]
        LOGGER[conversation_logger.py]
    end

    subgraph "Support Modules"
        PROMPTS[system_prompts.py]
        ROUTER[semantic_router.py]
        UTILS[utils.py]
    end

    CLI --> MAIN
    WEB --> MAIN
    MAIN --> TOOLS
    MAIN --> DEPS
    MAIN --> SETTINGS
    TOOLS --> RAG
    DEPS --> RAG
    RAG --> PROVIDERS
    MAIN --> PROMPTS
    MAIN --> ROUTER
    MAIN --> LOGGER
    RAG --> UTILS
```

## Session Management Flow

```mermaid
stateDiagram-v2
    [*] --> New: User starts conversation
    New --> Active: Session created
    Active --> Active: Query processed
    Active --> Inactive: No activity (2h)
    Inactive --> Active: User returns
    Active --> Expired: 24 hours passed
    Inactive --> Expired: 24 hours passed
    Expired --> Archived: Cleanup process
    Archived --> Deleted: 7 days passed
    Deleted --> [*]

    note right of Active
        - Stores conversation history
        - Tracks language preference
        - Maintains code examples
    end note

    note right of Archived
        - Compressed storage
        - Historical reference
        - Can be restored
    end note
```

## RAG Service Architecture

```mermaid
graph TD
    subgraph "Query Processing"
        QUERY[User Query]
        ROUTER[Query Router]
        OPTIMIZE[Query Optimizer]
    end

    subgraph "Search Methods"
        DOC[Documentation Search]
        FUNC[Function Search]
        EXAMPLE[Example Search]
        CODE[Code Search]
    end

    subgraph "Search Strategy"
        VECTOR[Vector Search]
        KEYWORD[Keyword Search]
        HYBRID[Hybrid Search]
    end

    subgraph "Results"
        FORMAT[Result Formatter]
        CACHE[Result Cache]
        RESPONSE[Final Response]
    end

    QUERY --> ROUTER
    ROUTER --> OPTIMIZE

    OPTIMIZE --> DOC
    OPTIMIZE --> FUNC
    OPTIMIZE --> EXAMPLE
    OPTIMIZE --> CODE

    DOC --> HYBRID
    FUNC --> HYBRID
    EXAMPLE --> VECTOR
    CODE --> KEYWORD

    HYBRID --> FORMAT
    VECTOR --> FORMAT
    KEYWORD --> FORMAT

    FORMAT --> CACHE
    CACHE --> RESPONSE
```

## Decision Flow - Built-in vs RAG Search

```mermaid
flowchart TD
    START[User Query] --> ANALYZE[Analyze Query]
    ANALYZE --> PHYSICS{Pure Physics?}

    PHYSICS -->|Yes| BUILTIN[Use Built-in Knowledge]
    PHYSICS -->|No| SPECIFIC{Specific Pulseq?}

    SPECIFIC -->|Yes| CHECKFUNC{Function Name?}
    SPECIFIC -->|No| GENERAL{General MRI?}

    CHECKFUNC -->|Yes| RAGSEARCH[Force RAG Search]
    CHECKFUNC -->|No| CHECKEXAMPLE{Code Example?}

    CHECKEXAMPLE -->|Yes| RAGSEARCH
    CHECKEXAMPLE -->|No| BUILTIN

    GENERAL -->|Yes| BUILTIN
    GENERAL -->|No| RAGSEARCH

    BUILTIN --> GEMINI[Generate with Gemini]
    RAGSEARCH --> SEARCH[Search Supabase]
    SEARCH --> AUGMENT[Augment Context]
    AUGMENT --> GEMINI

    GEMINI --> RESPONSE[Final Response]
```

## Component Interaction Diagram

```mermaid
graph TB
    subgraph "Application Layer"
        UI[User Interface]
    end

    subgraph "Agent Layer"
        AGENT[PulsePal Agent]
        TOOLS[Agent Tools]
        CONTEXT[Conversation Context]
    end

    subgraph "Service Layer"
        RAG[RAG Service]
        SESSION[Session Service]
        CONFIG[Configuration Service]
    end

    subgraph "Data Layer"
        VECTORDB[(Supabase<br/>Vector DB)]
        SESSIONS[(Session<br/>Storage)]
        LOGS[(Conversation<br/>Logs)]
    end

    subgraph "External APIs"
        LLM[Google Gemini API]
        EMBED[Embeddings API]
    end

    UI <--> AGENT
    AGENT <--> TOOLS
    AGENT <--> CONTEXT
    TOOLS <--> RAG
    CONTEXT <--> SESSION
    AGENT <--> CONFIG
    RAG <--> VECTORDB
    SESSION <--> SESSIONS
    AGENT --> LOGS
    RAG <--> EMBED
    AGENT <--> LLM
```

## Deployment Architecture

```mermaid
graph LR
    subgraph "Development"
        DEV[Local Development]
        DEVDB[(Local Supabase)]
    end

    subgraph "Production"
        PROD[Production Server]
        PRODDB[(Production Supabase)]

        subgraph "Services"
            API[API Server]
            WEB[Web Server]
            WORKER[Background Workers]
        end
    end

    subgraph "External"
        GEMINI[Google Gemini]
        EMBED[Google Embeddings]
    end

    DEV --> DEVDB
    DEV --> GEMINI
    DEV --> EMBED

    PROD --> PRODDB
    API --> GEMINI
    API --> EMBED
    WEB --> API
    WORKER --> PRODDB

    style DEV fill:#e1f5fe
    style PROD fill:#c8e6c9
    style GEMINI fill:#fff3e0
    style EMBED fill:#fff3e0
```

## Error Handling Flow

```mermaid
flowchart TD
    REQUEST[User Request] --> TRY{Try Process}

    TRY -->|Success| RESPONSE[Return Response]
    TRY -->|API Error| APIHANDLE{API Handler}
    TRY -->|Session Error| SESSIONHANDLE{Session Handler}
    TRY -->|RAG Error| RAGHANDLE{RAG Handler}

    APIHANDLE -->|Rate Limit| WAIT[Wait & Retry]
    APIHANDLE -->|Auth Error| FAIL[Return Error]

    SESSIONHANDLE -->|Not Found| CREATE[Create New Session]
    SESSIONHANDLE -->|Corrupted| REBUILD[Rebuild Session]

    RAGHANDLE -->|No Results| FALLBACK[Use Built-in Knowledge]
    RAGHANDLE -->|Timeout| CACHE[Check Cache]

    WAIT --> TRY
    CREATE --> TRY
    REBUILD --> TRY
    FALLBACK --> RESPONSE
    CACHE --> RESPONSE
    FAIL --> ERROR[Error Message]
```

## Performance Optimization Strategy

```mermaid
graph TD
    subgraph "Query Processing"
        Q1[Query Analysis<br/>~50ms]
        Q2[Semantic Routing<br/>~100ms]
    end

    subgraph "Knowledge Decision"
        D1{Use RAG?}
        D2[Built-in Knowledge<br/>~0ms]
        D3[RAG Search<br/>~500-1000ms]
    end

    subgraph "Response Generation"
        G1[Gemini Processing<br/>~1000-2000ms]
        G2[Format Response<br/>~50ms]
    end

    subgraph "Optimizations"
        O1[Session Cache]
        O2[Result Cache]
        O3[Query Batching]
    end

    Q1 --> Q2
    Q2 --> D1
    D1 -->|No 90%| D2
    D1 -->|Yes 10%| D3
    D2 --> G1
    D3 --> G1
    G1 --> G2

    O1 -.-> Q1
    O2 -.-> D3
    O3 -.-> G1

    style D2 fill:#c8e6c9
    style D3 fill:#ffccbc
```

## Testing Coverage Map

```mermaid
graph LR
    subgraph "Unit Tests"
        UT1[main_agent.py<br/>90% coverage]
        UT2[rag_service.py<br/>85% coverage]
        UT3[dependencies.py<br/>95% coverage]
        UT4[tools.py<br/>100% coverage]
    end

    subgraph "Integration Tests"
        IT1[Agent + RAG<br/>80% coverage]
        IT2[Session Flow<br/>85% coverage]
        IT3[API Integration<br/>75% coverage]
    end

    subgraph "E2E Tests"
        E2E1[CLI Workflow<br/>70% coverage]
        E2E2[Web Workflow<br/>60% coverage]
        E2E3[Full Query Flow<br/>80% coverage]
    end

    UT1 --> IT1
    UT2 --> IT1
    UT3 --> IT2
    UT4 --> IT3

    IT1 --> E2E3
    IT2 --> E2E3
    IT3 --> E2E1
    IT3 --> E2E2

    style UT4 fill:#c8e6c9
    style UT3 fill:#c8e6c9
    style UT1 fill:#dcedc8
    style E2E2 fill:#ffccbc
```

---

These diagrams use Mermaid syntax and can be rendered in:
- GitHub markdown files
- VS Code with Mermaid extension
- Online at mermaid.live
- MkDocs documentation

To update diagrams, edit the Mermaid code blocks above.
