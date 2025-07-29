## FEATURE:

Build Pulsepal, an AI-powered assistant system for Pulseq MRI sequence programming that serves as both a coding expert and MRI physics educator. The system consists of two specialized agents:

1. **Pulsepal (Main Agent)**: Expert in Pulseq programming across MATLAB, Octave, and Python (pypulseq), capable of generating complete pulse sequences, debugging code, and providing detailed explanations of code functionality
2. **MRI Expert (Sub-agent)**: Specialist in MRI physics and pulse sequence theory, explains how Pulseq code relates to real-world MRI principles and scanner behavior

The system targets graduate-level researchers and programmers working with Pulseq v1.5.0 across all supported platforms (MATLAB, Octave, Python). It leverages a comprehensive Supabase database containing scraped documentation from 25+ authoritative sources through a custom MCP server.

Key capabilities:
- Generate complete Pulseq sequences in MATLAB, Octave, or Python (pypulseq)
- Debug existing sequences and troubleshoot scanner issues across all platforms
- Explain code functionality line-by-line for any Pulseq implementation
- Connect code implementations to MRI physics principles
- Provide tutorials and examples for learning Pulseq in user's preferred language
- Build upon previous code examples within a session
- Generate visualization code for sequence timing and k-space trajectories
- Convert sequences between MATLAB/Octave and Python implementations when needed

## TOOLS:

### MCP Server Integration (crawl4ai-rag)
The system must integrate with the custom MCP server providing these tools:

1. **perform_rag_query**
   - Purpose: Primary tool for retrieving Pulseq documentation and knowledge
   - Arguments: 
     - query (string, required): Semantic search query
     - source (string, optional): Filter by specific source
     - match_count (int, optional, default: 5): Number of results
   - Returns: JSON with content, metadata, similarity scores, and rerank scores

2. **search_code_examples**
   - Purpose: Find relevant Pulseq code examples
   - Arguments:
     - query (string, required): Description of needed code
     - source_id (string, optional): Filter by source
     - match_count (int, optional, default: 5): Number of examples
   - Returns: JSON with code examples, summaries, metadata, and similarity scores

3. **get_available_sources**
   - Purpose: List all available documentation sources
   - Arguments: None
   - Returns: JSON with sources list, summaries, word counts, timestamps

4. **analyze_github_repo** (if needed for new repositories)
   - Purpose: Analyze structure of GitHub repositories
   - Arguments: github_url (string, required)
   - Returns: Repository analysis with file counts, structure

### Agent Communication Tools
- **delegate_to_mri_expert**: Tool for Pulsepal to send queries to the MRI Expert sub-agent
  - Arguments: 
    - question (string): MRI physics question
    - context (dict): Relevant code context
  - Returns: Expert explanation of MRI principles

## DEPENDENCIES

1. **LLM Configuration**:
   - Model: gemini-2.0-flash-lite
   - API Key: GOOGLE_API_KEY (environment variable)

2. **MCP Server Connection**:
   - Server name: crawl4ai-rag
   - Connection handled by Pydantic AI MCP integration

3. **Session Management**:
   - In-memory conversation history
   - Code example tracking within session
   - No persistence between sessions

4. **Environment Variables**:
   - GOOGLE_API_KEY: For Gemini API access
   - Any MCP server configuration variables

## SYSTEM PROMPT(S)

### Pulsepal (Main Agent) System Prompt:
```
You are Pulsepal, an expert Pulseq programming assistant specializing in MRI pulse sequence development across MATLAB, Octave, and Python (pypulseq) implementations. You have extensive knowledge of Pulseq v1.5.0 and access to comprehensive documentation through your RAG system.

Your primary responsibilities:
1. Generate complete, functional Pulseq sequences in the user's preferred language (MATLAB, Octave, or Python)
2. Debug and fix existing Pulseq code across all platforms
3. Explain code functionality in detail
4. Provide tutorials and examples for learning Pulseq
5. Generate visualization code for sequence analysis
6. Convert sequences between MATLAB/Octave and Python when requested

Key behaviors:
- Always search the knowledge base first using perform_rag_query for relevant documentation
- Use search_code_examples to find similar implementations before generating new code
- Detect which language the user is working with (MATLAB/Octave/Python) from context
- When users ask about MRI physics principles, delegate to the MRI Expert sub-agent
- Build upon previous code examples in the current session
- Provide clear, well-commented code in the appropriate language
- Include error handling and scanner compatibility considerations
- Be aware of syntax differences between MATLAB/Octave and pypulseq
- Target graduate-level users but remain approachable

Language-specific considerations:
- MATLAB/Octave: Use standard MATLAB conventions, handle .seq file paths appropriately
- Python/pypulseq: Follow PEP 8 style guide, use numpy arrays appropriately, handle imports correctly

Response format:
- Use markdown with proper code blocks (```matlab or ```python)
- Include inline comments in generated code
- Provide step-by-step explanations when teaching
- Reference specific Pulseq documentation when available
- Clearly indicate which language/platform the code is for
```

### MRI Expert (Sub-agent) System Prompt:
```
You are the MRI Expert, a specialist in magnetic resonance imaging physics and pulse sequence theory. You work alongside Pulsepal to explain how Pulseq code implementations relate to fundamental MRI principles.

Your expertise includes:
- RF pulse design and slice selection
- Gradient timing and k-space trajectories
- Echo formation and signal evolution
- Sequence timing constraints
- Scanner hardware limitations
- Image contrast mechanisms
- Safety considerations (SAR, PNS)

When receiving queries from Pulsepal:
1. Analyze the provided code context
2. Explain the underlying MRI physics principles
3. Connect code elements to real scanner behavior
4. Identify potential issues that could arise on actual scanners
5. Suggest physics-based optimizations

Use clear explanations with appropriate technical depth for graduate-level students and researchers. Include equations when helpful but focus on intuitive understanding.
```

## EXAMPLES:

Reference these example patterns from the examples/ folder:
- examples/basic_chat_agent - For session memory implementation
- examples/tool_enabled_agent - For MCP tool integration patterns
- examples/main_agent_reference - For agent configuration best practices

Additional Pulsepal-specific interaction patterns:

### Example 1: Code Generation Request
```
User: "Create a spin echo sequence with TR=500ms, TE=20ms, 256x256 matrix in Python"
Pulsepal: 
1. Searches knowledge base for pypulseq spin echo implementations
2. Retrieves relevant code examples
3. Generates complete Python code with proper timing
4. Includes matplotlib visualization code
5. Explains key sequence elements
```

### Example 2: Language Conversion
```
User: "Can you convert this MATLAB gradient echo sequence to Python?"
Pulsepal:
1. Analyzes the MATLAB code structure
2. Identifies equivalent pypulseq functions
3. Handles syntax differences (0-based vs 1-based indexing)
4. Provides complete Python implementation
5. Notes any platform-specific considerations
```

### Example 3: Debugging Request
```
User: "My gradient echo sequence crashes the scanner with a gradient error"
Pulsepal:
1. Detects language from provided code
2. Requests the problematic code if not provided
3. Analyzes gradient timing and amplitude
4. Checks against hardware limits
5. Identifies the issue
6. Provides corrected code with explanation
```

### Example 4: MRI Physics Question
```
User: "Why do we need this spoiler gradient after the readout?"
Pulsepal:
1. Identifies this as an MRI physics question
2. Delegates to MRI Expert with code context
3. MRI Expert explains transverse magnetization spoiling
4. Relates back to the specific code implementation
```

## DOCUMENTATION:

### Core Resources:
- Pulseq Official Documentation (stored in Supabase)
- PulsePy Documentation (stored in Supabase)
- Pulseq GitHub Examples (stored in Supabase)
- Pydantic AI Documentation: https://ai.pydantic.dev/

### Database Content:
- **sources table**: Metadata about all documentation sources
- **crawled_pages table**: Chunked documentation with embeddings
- **code_examples table**: Code snippets with AI summaries

### Available Knowledge Base Sources:
The Supabase database contains comprehensive documentation from these sources:

**Core Pulseq Resources:**
- pulseq.github.io - Official Pulseq documentation
- github.com/pulseq/pulseq - Main Pulseq repository
- github.com/pulseq/tutorials - Official tutorials
- github.com/pulseq/MR-Physics-with-Pulseq - Physics explanations
- github.com/pulseq/ISMRM-Virtual-Meeting--November-15-17-2023 - Recent developments

**Python Implementation:**
- github.com/imr-framework/pypulseq - Python implementation
- pypulseq.readthedocs.io - PyPulseq documentation

**HarmonizedMRI Organization Repositories:**
- github.com/HarmonizedMRI/SequenceExamples-GE - GE scanner examples
- github.com/HarmonizedMRI/3DEPI - 3D EPI sequences
- github.com/HarmonizedMRI/SMS-EPI - Simultaneous multi-slice EPI
- github.com/HarmonizedMRI/Pulseq-diffusion - Diffusion sequences
- github.com/HarmonizedMRI/Functional - fMRI sequences
- github.com/HarmonizedMRI/B0shimming - B0 shimming implementations
- github.com/HarmonizedMRI/Calibration - Calibration sequences
- github.com/HarmonizedMRI/qualityAssurance - QA protocols
- github.com/HarmonizedMRI/PulCeq - Additional tools
- github.com/HarmonizedMRI/SOSP3d - 3D spiral sequences
- github.com/HarmonizedMRI/wave-haste - WAVE-HASTE sequences
- harmonizedmri.github.io - Organization documentation

**Specialized Sequences:**
- github.com/asgaspar/OpenMOLLI - MOLLI T1 mapping
- github.com/fmrifrey/lps - LPS sequences
- github.com/rextlfung/Fast-fMRI - Fast fMRI acquisitions
- github.com/RitaSchmidt/Shuffle3DGRE - 3D GRE shuffling
- github.com/shoheifujitaSF/Pulseq-qalas - QALAS sequences
- github.com/yohan-jun/PRIME - PRIME sequences
- github.com/yutingchen11/MIMOSA - MIMOSA implementation

**Additional Resources:**
- mrzero-core.readthedocs.io - MRzero documentation

### Key Pulseq Concepts to Reference:
- Sequence object structure (MATLAB: `mr.Sequence()`, Python: `Sequence()`)
- RF pulse design:
  - MATLAB: `mr.makeBlockPulse()`, `mr.makeSincPulse()`, etc.
  - Python: `make_block_pulse()`, `make_sinc_pulse()`, etc.
- Gradient waveforms:
  - MATLAB: `mr.makeTrapezoid()`, `mr.makeArbitraryGrad()`
  - Python: `make_trapezoid()`, `make_arbitrary_grad()`
- Timing calculations and delays
- k-space trajectory planning
- Scanner limits and system objects
- File I/O differences between platforms

## OTHER CONSIDERATIONS:

### Technical Implementation:
- Use gemini-2.0-flash-lite for cost-effective, fast responses
- Implement proper error handling for MCP server connections
- Cache frequently accessed documentation chunks within session
- Use streaming responses for long code generations

### User Experience:
- Automatically detect programming language from user's code or context
- Ask for language preference if unclear (MATLAB, Octave, or Python)
- Always search knowledge base before generating code from scratch
- Provide working code first, then explain if needed
- Include helpful comments but avoid over-commenting
- Suggest appropriate visualization code (MATLAB plots or matplotlib)
- Be specific about Pulseq version and platform compatibility
- Highlight key differences between MATLAB and Python implementations when relevant

### Code Quality:
- Follow language-specific best practices:
  - MATLAB/Octave: Standard MATLAB naming conventions (camelCase)
  - Python: PEP 8 style guide (snake_case)
- Include proper error checking in generated sequences
- Consider scanner hardware limits in all implementations
- Provide modular, reusable code components
- Include timing diagrams in comments when helpful
- Handle platform-specific differences (e.g., file I/O, array indexing)

### Safety and Best Practices:
- Always consider SAR and PNS limits in sequence design
- Warn about potential scanner compatibility issues
- Include gradient moment nulling where appropriate
- Check for proper k-space coverage
- Validate timing constraints

### Session Management:
- Track all code examples shown in current session
- Allow users to reference "the previous code" or "the spin echo example"
- Build complexity gradually when teaching
- Don't persist any information between sessions

### Error Handling:
- Gracefully handle MCP server connection issues
- Provide helpful fallbacks if knowledge base is unavailable
- Clear error messages for debugging assistance
- Suggest common fixes for typical Pulseq errors

## CLAUDE CODE DEVELOPMENT SUBAGENTS:

Use these Claude Code subagents during development with the /agents command:

### 1. Testing Agent (/agents testing)
**When to use**: Throughout development, especially for:
- Testing all MCP server tools (perform_rag_query, search_code_examples, etc.)
- Verifying agent delegation between Pulsepal and MRI Expert works correctly
- Testing session memory persistence within conversations
- Validating multi-language code generation (MATLAB, Python, Octave)
- Creating test cases for all 4 expected question categories:
  1. "Can you explain what this code does?"
  2. "Why is this code necessary for MRI?"
  3. "Help me learn Pulseq"
  4. "Debug my scanner error"

**Key test scenarios**:
- MCP server connection failures
- Empty or irrelevant search results
- Language detection and conversion
- Code generation accuracy
- Session state management

### 2. Documentation Agent (/agents documentation)
**When to use**: After implementing major features to create:
- User guide with example queries and expected responses
- API documentation for agent tools and MCP integration
- Setup instructions for connecting to the crawl4ai-rag server
- Examples showing MATLAB vs Python code generation
- Troubleshooting guide for common issues

**Documentation priorities**:
- How to phrase questions for best results
- Language-specific examples
- MCP server configuration
- Session management behavior

### 3. Code Review Agent (/agents code-review)
**When to use**: Before committing major features, review:
- RAG query optimization and search strategies
- Agent communication patterns and delegation logic
- Error handling robustness
- Code organization and modularity
- Performance optimizations for tool calls
- Session memory implementation

**Focus areas**:
- Minimize redundant MCP server calls
- Ensure clean separation between agents
- Validate error messages are helpful
- Check for edge cases in language detection

### 4. Debugging Agent (/agents debugging)
**When to use**: When encountering complex issues with:
- MCP server connection or authentication
- Vector similarity search returning poor results
- Agent delegation not working as expected
- Session state corruption or memory leaks
- Performance bottlenecks in tool calls

**Common debugging scenarios**:
- Why RAG queries return irrelevant results
- Agent communication failures
- Slow response times
- Memory persistence issues

### Development Workflow:
1. Implement core features with main Claude Code
2. Use `/agents testing` to create comprehensive test suites early
3. Call `/agents documentation` after each major feature
4. Run `/agents code-review` before significant commits
5. Invoke `/agents debugging` for troubleshooting complex issues

### Subagent Usage Tips:
- Testing Agent: Run after every major function implementation
- Documentation Agent: Update docs incrementally, not all at once
- Code Review Agent: Use for architectural decisions, not just syntax
- Debugging Agent: Provide detailed context about the issue

This multi-agent approach ensures robust, well-documented, and thoroughly tested implementation of Pulsepal.