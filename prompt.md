# Prompt for Claude Code:

**Enhance the Chainlit UI for my Pulsepal MRI sequence programming assistant with better visual appeal and markdown rendering, ensuring support for all Pulseq language implementations in the knowledge base.**

### Critical First Task:
**Query the Supabase database to determine ALL supported languages:**
```python
# In pulsepal/rag_service.py or supabase_client.py, check:
result = supabase_client.client.table('api_reference').select('*').execute()
# Extract unique programming languages and their characteristics
# The user mentions C/C++ is in there, but current code only handles MATLAB/Python/Octave
```

### Current State (Verified):
- Working Chainlit UI in `chainlit_app.py` using PydanticAI agents directly
- `dependencies.py` currently only tracks MATLAB, Python, and Octave
- Knowledge base in Supabase project "crawl4ai-mcp-local-embed" 
- **The `api_reference` table contains additional languages including C/C++**

### Required Updates Based on Database:
1. **Update `dependencies.py`** to support all languages found in database
2. **Extend language detection** to include C/C++ patterns
3. **Add syntax highlighting** for all discovered languages

### Confirmed Minimum Language Support:
- **MATLAB** (`.m` files) - Currently default
- **Python** (`.py` files) - pypulseq  
- **Octave** (`.m` files) - MATLAB-compatible
- **C** (`.c`, `.h` files) - Per user confirmation
- **C++** (`.cpp`, `.cc`, `.hpp` files) - Per user confirmation
- **Any additional languages found in `api_reference` table**

### Primary Goals:
1. **Perfect markdown and code rendering** for ALL languages in database
2. **Professional medical/MRI themed interface**
3. **Dynamic language support** based on database contents

### Specific Implementation Tasks:

#### Task 1: Database Query and Language Discovery
```python
# First, update dependencies.py to support dynamic languages:
SUPPORTED_LANGUAGES = {
    'matlab': {'extensions': ['.m'], 'keywords': ['function', 'end']},
    'python': {'extensions': ['.py'], 'keywords': ['import', 'def']},
    'octave': {'extensions': ['.m'], 'keywords': ['function', 'endfunction']},
    'c': {'extensions': ['.c', '.h'], 'keywords': ['#include', 'int main']},
    'cpp': {'extensions': ['.cpp', '.cc', '.hpp'], 'keywords': ['#include', 'class', 'namespace']},
    # Add more based on database query
}
```

#### Task 2: Enhanced Code Block Rendering
- Ensure Chainlit properly highlights:
  - MATLAB/Octave with `matlab` syntax
  - Python with `python` syntax
  - C with `c` syntax
  - C++ with `cpp` syntax
  - Any additional languages from database

#### Task 3: Visual Improvements
Create `public/pulsepal.css`:
```css
/* Language-specific code block styling */
.language-matlab { border-left: 4px solid #FF6B00; }
.language-python { border-left: 4px solid #3776AB; }
.language-c { border-left: 4px solid #00599C; }
.language-cpp { border-left: 4px solid #004482; }

/* Professional medical theme */
:root {
    --primary-color: #00a8cc;  /* Medical blue */
    --accent-color: #00d4aa;   /* MRI green */
}
```

#### Task 4: Update Language Detection
In `dependencies.py`, enhance `detect_language_preference()`:
```python
def detect_language_preference(self, content: str) -> Optional[str]:
    """Detect language including C/C++."""
    content_lower = content.lower()
    
    # Add C/C++ detection
    if any(indicator in content for indicator in ['#include', 'int main', 'void', 'struct']):
        if 'class' in content or 'namespace' in content or '::' in content:
            return 'cpp'
        return 'c'
    # ... existing MATLAB/Python detection
```

#### Task 5: File Upload Support
Enable upload for all language files:
- `.m` (MATLAB/Octave)
- `.py` (Python)
- `.c` (C files)
- `.cpp`, `.cc`, `.cxx` (C++ files)  
- `.h`, `.hpp` (Header files)
- `.seq` (Pulseq sequence files)

#### Task 6: Update UI Language Selector
Add comprehensive language selector:
```python
# In chainlit_app.py
languages = ['MATLAB', 'Python', 'Octave', 'C', 'C++']  # Plus any from database
```

### Testing Requirements:
1. Verify all languages from `api_reference` table are supported
2. Test syntax highlighting for each language
3. Ensure language detection works for uploaded files
4. Confirm copy buttons work for all code blocks

### Priority Order:
1. **CRITICAL**: Query database and update language support accordingly
2. **MUST HAVE**: Perfect rendering for all discovered languages
3. **MUST HAVE**: Professional visual styling
4. **SHOULD HAVE**: Interactive features and file upload
5. **NICE TO HAVE**: Visualization features

**Note**: The current implementation is missing C/C++ support that exists in the knowledge base. This MUST be fixed based on what's actually in the Supabase `api_reference` table.
