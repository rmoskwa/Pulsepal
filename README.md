# 🔬 PulsePal - AI Assistant for MRI Sequence Programming

**An advanced PydanticAI multi-agent system for Pulseq MRI sequence development with modern web interface**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PydanticAI](https://img.shields.io/badge/PydanticAI-latest-green.svg)](https://ai.pydantic.dev/)
[![Chainlit](https://img.shields.io/badge/Chainlit-2.6.3-orange.svg)](https://chainlit.io/)

## 🚀 What is PulsePal?

PulsePal is an intelligent AI assistant designed to help researchers and programmers work with **Pulseq v1.5.0** for MRI sequence development. It combines the power of PydanticAI agents with advanced RAG (Retrieval Augmented Generation) to provide expert assistance in:

- **🧪 Code Generation** - Create MRI sequences in MATLAB, Python, and Octave
- **🐛 Debugging** - Fix sequence errors and optimize performance
- **📚 Documentation** - Search comprehensive Pulseq knowledge base
- **⚛️ Physics** - Get expert MRI physics explanations
- **🔄 Conversion** - Translate between programming languages

## ✨ Key Features

### 🤖 **PydanticAI Multi-Agent Architecture**
- **Main Agent**: Handles programming questions and code generation
- **MRI Expert**: Provides specialized physics explanations
- **Seamless Delegation**: Automatic routing to appropriate expertise

### 🌐 **Modern Web Interface**
- **Chainlit Integration**: Beautiful, responsive web UI
- **Real-time Processing**: Live typing indicators and structured responses
- **Markdown Rendering**: Syntax highlighting for MATLAB/Python/Octave
- **Session Memory**: Maintains context across conversations

### 🔍 **Advanced RAG System**
- **Comprehensive Knowledge**: Pulseq documentation, tutorials, and examples
- **Vector Search**: BGE embeddings with Supabase backend
- **Hybrid Queries**: Semantic + keyword search for optimal results
- **Source Attribution**: Clear references to original documentation

### 🎯 **Multi-Language Support**
- **MATLAB** (default): Primary Pulseq environment with .seq file generation
- **Python**: Using pulseq-python package
- **Octave**: Open-source MATLAB alternative

## 🏃‍♂️ Quick Start

### 1. **Installation**
```bash
git clone https://github.com/your-username/pulsepal.git
cd pulsepal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Configuration**
Create a `.env` file with your API keys:
```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key_here
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_role_key

# Optional
BGE_MODEL_PATH=/path/to/local/bge/model
```

### 3. **Launch Web Interface**
```bash
chainlit run chainlit_app.py
```
Open http://localhost:8000 in your browser

### 4. **Or Use CLI Interface**
```bash
# Single question
python run_pulsepal.py "How do I create a spin echo sequence?"

# Interactive mode
python run_pulsepal.py --interactive
```

## 💬 Example Interactions

### 🧪 **Code Generation**
```
You: "Create a gradient echo sequence with TE=5ms, TR=100ms in MATLAB"

PulsePal: Here's a gradient echo sequence implementation:

```matlab
% Gradient Echo Sequence - TE=5ms, TR=100ms
seq = mr.Sequence();

% System limits
sys = mr.opts('MaxGrad', 32, 'GradUnit', 'mT/m', ...
              'MaxSlew', 130, 'SlewUnit', 'T/m/s');

% Timing parameters
TE = 5e-3;  % Echo time
TR = 100e-3; % Repetition time

% RF pulse
rf = mr.makeBlockPulse(pi/6, 'Duration', 1e-3, 'system', sys);

% Gradients
gx = mr.makeTrapezoid('x', 'Area', 1e-3, 'Duration', 2e-3, 'system', sys);
gr = mr.makeTrapezoid('x', 'Area', -1e-3, 'Duration', 2e-3, 'system', sys);

% Add to sequence
seq.addBlock(rf);
seq.addBlock(gx);
seq.addBlock(mr.makeDelay(TE - 2e-3));
seq.addBlock(gr);
seq.addBlock(mr.makeDelay(TR - TE - 2e-3));

% Write sequence
seq.write('gre_sequence.seq');
```
```

### ⚛️ **Physics Explanation**
```
You: "Why do we need spoiler gradients in gradient echo sequences?"

PulsePal: Spoiler gradients are essential in gradient echo sequences for several reasons:

**1. Transverse Magnetization Dephasing**
- After each RF pulse, residual transverse magnetization remains
- Without spoiling, this creates unwanted signal contamination
- Spoiler gradients dephase these residual spins across voxels

**2. T1 Contrast Enhancement**
- Proper spoiling ensures pure T1-weighted contrast
- Eliminates T2* contamination from previous excitations
- Results in cleaner, more predictable image contrast

**3. Steady State Control**
- Prevents formation of unwanted steady states
- Ensures predictable signal behavior across repetitions
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐
│  Chainlit Web   │    │   CLI Interface  │
│     Interface   │    │                  │
└─────────┬───────┘    └─────────┬────────┘
          │                      │
          └──────────┬───────────┘
                     │
        ┌────────────▼─────────────┐
        │     Pulsepal Agent       │
        │   (PydanticAI Core)      │
        └─────────┬─────────┬──────┘
                  │         │
        ┌─────────▼─────┐   │
        │  RAG Service  │   │
        │   (Supabase   │   │
        │  + BGE Model) │   │
        └───────────────┘   │
                            │
                   ┌────────▼──────────┐
                   │   MRI Expert      │
                   │    Sub-Agent      │
                   └───────────────────┘
```

## 🧪 Testing

```bash
# Quick validation
python test_chainlit_quick.py

# Run comprehensive test suite
python run_chainlit_tests.py

# Specific test categories
python run_chainlit_tests.py smoke    # Fast essential tests
python run_chainlit_tests.py session  # Session management
python run_chainlit_tests.py tools    # RAG and delegation
```

## 📁 Project Structure

```
pulsepal/
├── chainlit_app.py           # Web interface (Chainlit)
├── run_pulsepal.py          # CLI interface
├── requirements.txt         # Dependencies
├── pulsepal/               # Core package
│   ├── main_agent.py       # Main PydanticAI agent
│   ├── mri_expert_agent.py # Physics expert sub-agent
│   ├── tools.py           # RAG tools & delegation
│   ├── dependencies.py    # Session management
│   ├── rag_service.py     # RAG integration
│   └── settings.py        # Configuration
├── examples/              # Example agents & patterns
├── pulsepal/tests/       # Comprehensive test suite
└── public/               # Web assets (CSS, etc.)
```

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY` - Gemini API key for LLM inference
- `SUPABASE_URL` - Supabase project URL for RAG database
- `SUPABASE_KEY` - Supabase service role key
- `BGE_MODEL_PATH` - Local BGE embedding model path (optional)

### Settings
See `pulsepal/settings.py` for full configuration options including session timeouts, RAG parameters, and model settings.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python run_chainlit_tests.py`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📖 Documentation

- **[Complete Documentation](PULSEPAL-DOCUMENTATION.md)** - Full system architecture and API reference
- **[Chainlit Integration Guide](CHAINLIT_INTEGRATION_FIXES.md)** - Web interface setup and troubleshooting

## 🏷️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[PydanticAI](https://ai.pydantic.dev/)** - Powerful agent framework
- **[Pulseq](https://pulseq.github.io/)** - Open MRI sequence development
- **[Chainlit](https://chainlit.io/)** - Beautiful chat interfaces
- **[Supabase](https://supabase.com/)** - Backend and vector database

---

**Ready to revolutionize your MRI sequence development? Start with PulsePal today!** 🚀