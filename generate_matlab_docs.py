#!/usr/bin/env python3
"""
Generate MATLAB API documentation from Supabase database.
Creates MkDocs-compatible markdown files for hosting online.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Set
import re
import sys

from supabase import create_client, Client
from dotenv import load_dotenv

# Add parent directory to path to import function_index
sys.path.append(str(Path(__file__).parent))
from pulsepal.function_index import MATLAB_FUNCTIONS

# Load environment variables
load_dotenv()

class MatlabDocGenerator:
    def __init__(self):
        """Initialize Supabase client and setup directories."""
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
            
        self.supabase: Client = create_client(url, key)
        self.docs_dir = Path("docs/matlab_api")
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Get valid function names from function_index.py
        self.valid_functions = self._get_valid_function_names()
    
    def _get_valid_function_names(self) -> Set[str]:
        """Extract all valid MATLAB function names from function_index."""
        valid_names = set()
        
        # Add direct calls
        valid_names.update(MATLAB_FUNCTIONS.get("direct_calls", set()))
        
        # Add class methods (just the method names, not the class prefix)
        for class_name, methods in MATLAB_FUNCTIONS.get("class_methods", {}).items():
            valid_names.update(methods)
            # Also add the class constructor
            valid_names.add(class_name)
        
        # Add other function groups
        valid_names.update(MATLAB_FUNCTIONS.get("mr_aux_functions", set()))
        valid_names.update(MATLAB_FUNCTIONS.get("mr_aux_quat_functions", set()))
        valid_names.update(MATLAB_FUNCTIONS.get("tra_functions", set()))
        
        return valid_names
        
    def fetch_matlab_functions(self) -> List[Dict[str, Any]]:
        """Fetch only valid MATLAB functions from the api_reference table."""
        print("Fetching MATLAB functions from Supabase...")
        
        response = self.supabase.table('api_reference').select(
            "name,signature,description,parameters,returns,calling_pattern,usage_examples,related_functions"
        ).eq('language', 'matlab').execute()
        
        all_functions = response.data
        
        # Filter to only include valid functions from function_index.py
        valid_functions = [
            func for func in all_functions 
            if func.get('name') in self.valid_functions
        ]
        
        print(f"Found {len(all_functions)} total MATLAB functions in database")
        print(f"Filtered to {len(valid_functions)} valid functions from function_index.py")
        
        # Clean up related_functions in each function
        for func in valid_functions:
            if func.get('related_functions'):
                func['related_functions'] = self._clean_related_functions(func['related_functions'])
        
        return valid_functions
    
    def _clean_related_functions(self, related_functions):
        """Remove invalid functions from related_functions list."""
        if not related_functions:
            return None
            
        if isinstance(related_functions, list):
            # Filter to only valid functions
            cleaned = [
                func for func in related_functions 
                if func.split('.')[-1] in self.valid_functions  # Get base function name
            ]
            return cleaned if cleaned else None
        
        return related_functions
    
    def _fix_examples_namespace(self, examples, function_name: str, calling_pattern: str) -> list:
        """Fix examples to use correct namespace based on calling pattern."""
        if not examples or not calling_pattern:
            return examples
            
        # Extract namespace from calling pattern (e.g., "mr." from "mr.makeAdc(...)")
        namespace_match = re.match(r'^([^.]+\.).*\(', calling_pattern)
        if not namespace_match:
            return examples  # No namespace in calling pattern
            
        namespace = namespace_match.group(1)  # e.g., "mr." or "seq."
        
        fixed_examples = []
        for example in examples:
            # Check if function is called without namespace
            # Pattern: function_name( but not already with namespace
            pattern = rf'(?<![.\w]){re.escape(function_name)}\s*\('
            
            # Replace with namespace.function_name(
            fixed_example = re.sub(pattern, f'{namespace}{function_name}(', example)
            fixed_examples.append(fixed_example)
            
        return fixed_examples
    
    def format_parameters(self, parameters: Dict) -> str:
        """Format parameters section for markdown."""
        if not parameters:
            return "*No parameters*"
        
        sections = []
        
        # Required parameters
        if 'required' in parameters and parameters['required']:
            sections.append("### Required Parameters\n")
            sections.append("| Name | Type | Description | Example | Units |")
            sections.append("|------|------|-------------|---------|-------|")
            
            for param in parameters['required']:
                name = f"`{param.get('name', 'unknown')}`"
                param_type = param.get('type', '')
                description = param.get('description', '')
                example = f"`{param.get('example', '')}`" if param.get('example') else ''
                units = param.get('units', '')
                if units == 'none':
                    units = ''
                
                sections.append(f"| {name} | {param_type} | {description} | {example} | {units} |")
        
        # Optional parameters
        if 'optional' in parameters and parameters['optional']:
            sections.append("\n### Optional Parameters\n")
            sections.append("| Name | Type | Default | Description | Example |")
            sections.append("|------|------|---------|-------------|---------|")
            
            for param in parameters['optional']:
                name = f"`{param.get('name', 'unknown')}`"
                param_type = param.get('type', '')
                default = f"`{param.get('default', '')}`" if param.get('default') is not None else ''
                description = param.get('description', '')
                
                # Add valid values to description if present
                if param.get('valid_values'):
                    description += f" Valid values: {param['valid_values']}"
                
                # Add units to description if present and not 'none'
                if param.get('units') and param['units'] != 'none':
                    description += f" (Units: {param['units']})"
                
                example = f"`{param.get('example', '')}`" if param.get('example') else ''
                
                sections.append(f"| {name} | {param_type} | {default} | {description} | {example} |")
        
        return '\n'.join(sections) if sections else "*No parameters*"
    
    def clean_signature(self, signature: str) -> str:
        """Clean up function signature for display."""
        if not signature:
            return ""
        
        # Remove extra whitespace and newlines
        signature = ' '.join(signature.split())
        
        # Ensure it starts with 'function' if it doesn't
        if not signature.strip().startswith('function'):
            signature = 'function ' + signature
            
        return signature
    
    def fix_function_links(self, text: str, existing_functions: set) -> str:
        """Fix function links to only link to existing documentation."""
        if not text:
            return text
            
        # Pattern to match markdown links
        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\.md\)'
        
        def replace_link(match):
            link_text = match.group(1)
            link_target = match.group(2)
            
            # Extract just the function name from paths like 'mr.funcname' or 'Sequence.funcname'
            func_name = link_target.split('.')[-1]
            
            # Check if this function exists in our documentation
            if func_name in existing_functions:
                return f'[{link_text}]({func_name}.md)'
            else:
                # Return just the text without link for non-existent functions
                return f'`{link_text}`'
        
        return re.sub(link_pattern, replace_link, text)
    
    def generate_function_doc(self, func: Dict[str, Any], existing_functions: set) -> str:
        """Generate markdown documentation for a single function."""
        name = func.get('name', 'Unknown')
        signature = self.clean_signature(func.get('signature', ''))
        description = func.get('description', 'No description available.')
        parameters = func.get('parameters', {})
        returns = func.get('returns', None)
        calling_pattern = func.get('calling_pattern', None)
        usage_examples = func.get('usage_examples', None)
        related_functions = func.get('related_functions', None)
        
        # Fix examples to match calling pattern namespace
        if usage_examples and isinstance(usage_examples, list):
            usage_examples = self._fix_examples_namespace(usage_examples, name, calling_pattern)
        
        # Build the markdown content
        content = [f"# {name}\n"]
        
        # Add description
        content.append(f"{description}\n")
        
        # Add signature
        if signature:
            content.append("## Syntax\n")
            content.append("```matlab")
            content.append(signature)
            content.append("```\n")
        
        # Add calling pattern if different from signature
        if calling_pattern and calling_pattern != signature:
            content.append("## Calling Pattern\n")
            content.append("```matlab")
            content.append(calling_pattern)
            content.append("```\n")
        
        # Add parameters
        content.append("## Parameters\n")
        content.append(self.format_parameters(parameters))
        content.append("")
        
        # Add returns section
        if returns:
            content.append("## Returns\n")
            
            # Handle different return formats
            if isinstance(returns, list):
                # List of return values
                if len(returns) > 0:
                    content.append("| Output | Type | Description |")
                    content.append("|--------|------|-------------|")
                    for ret in returns:
                        if isinstance(ret, dict):
                            name = ret.get('name', 'output')
                            ret_type = ret.get('type', '')
                            desc = ret.get('description', '')
                            content.append(f"| `{name}` | {ret_type} | {desc} |")
                    content.append("")
            elif isinstance(returns, dict):
                # Single dict or dict of returns
                if 'name' in returns:
                    # Single return value as dict
                    content.append("| Output | Type | Description |")
                    content.append("|--------|------|-------------|")
                    content.append(f"| `{returns.get('name', 'output')}` | {returns.get('type', '')} | {returns.get('description', '')} |")
                    content.append("")
                else:
                    # Multiple returns as dict
                    content.append("| Output | Type | Description |")
                    content.append("|--------|------|-------------|")
                    for key, value in returns.items():
                        if isinstance(value, dict):
                            content.append(f"| `{key}` | {value.get('type', '')} | {value.get('description', '')} |")
                        else:
                            content.append(f"| `{key}` | | {value} |")
                    content.append("")
            elif isinstance(returns, str):
                # Simple string description
                content.append(returns)
                content.append("")
            else:
                # Fallback - convert to string
                content.append(str(returns))
                content.append("")
        
        # Add usage examples
        if usage_examples:
            content.append("## Examples\n")
            content.append("```matlab")
            if isinstance(usage_examples, list):
                content.append('\n'.join(usage_examples))
            else:
                content.append(str(usage_examples))
            content.append("```\n")
        
        # Add related functions
        if related_functions:
            content.append("## See Also\n")
            if isinstance(related_functions, list):
                # Only link to functions that exist in our documentation
                links = []
                for func_ref in related_functions:
                    func_name = func_ref.split('.')[-1]  # Get just the function name
                    if func_name in existing_functions:
                        links.append(f"[{func_ref}]({func_name}.md)")
                    else:
                        links.append(f"`{func_ref}`")
                content.append(", ".join(links))
            else:
                # Fix any links in the text
                related_text = self.fix_function_links(str(related_functions), existing_functions)
                content.append(related_text)
            content.append("")
        
        return '\n'.join(content)
    
    def generate_index_page(self, functions: List[Dict[str, Any]]) -> str:
        """Generate index page with all functions listed."""
        content = ["# MATLAB API Reference\n"]
        content.append("Complete reference for PulsePal MATLAB functions.\n")
        
        # Group functions by first letter
        grouped = {}
        for func in sorted(functions, key=lambda x: x.get('name', '').lower()):
            name = func.get('name', '')
            if name:
                first_letter = name[0].upper()
                if not first_letter.isalpha():
                    first_letter = '#'
                    
                if first_letter not in grouped:
                    grouped[first_letter] = []
                grouped[first_letter].append(func)
        
        # Create alphabetical sections
        for letter in sorted(grouped.keys()):
            content.append(f"## {letter}\n")
            
            for func in grouped[letter]:
                name = func.get('name', '')
                description = func.get('description', '')
                
                # Truncate description for index
                if description and len(description) > 100:
                    description = description[:97] + "..."
                
                content.append(f"- [{name}]({name}.md) - {description}")
            content.append("")
        
        return '\n'.join(content)
    
    def generate_mkdocs_config(self, functions: List[Dict[str, Any]]) -> None:
        """Generate MkDocs configuration file."""
        print("Generating MkDocs configuration...")
        
        # Sort functions alphabetically
        sorted_funcs = sorted(functions, key=lambda x: x.get('name', '').lower())
        
        config = {
            'site_name': 'Pulseq MATLAB API Documentation',
            'site_description': 'Complete API reference for Pulseq MATLAB functions',
            'site_url': 'https://rmoskwa.github.io/Pulsepal/',
            
            'theme': {
                'name': 'material',
                'palette': {
                    'primary': 'blue',
                    'accent': 'light blue'
                },
                'features': [
                    'navigation.tabs',
                    'navigation.sections',
                    'navigation.expand',
                    'navigation.top',
                    'search.suggest',
                    'search.highlight',
                    'content.code.copy',
                    'content.code.annotate'
                ],
                'font': {
                    'text': 'Roboto',
                    'code': 'Roboto Mono'
                }
            },
            
            'plugins': [
                {
                    'search': {
                        'lang': 'en',
                        'separator': r'[\s\-\.]+'
                    }
                }
            ],
            
            'markdown_extensions': [
                'pymdownx.highlight',
                'pymdownx.superfences',
                'pymdownx.tabbed',
                'pymdownx.details',
                'pymdownx.snippets',
                'admonition',
                'tables',
                'toc',
                {'toc': {'permalink': True}}
            ],
            
            'nav': [
                {'Home': 'index.md'},
                {
                    'API Reference': [
                        {'Overview': 'matlab_api/index.md'},
                        {'Functions': [
                            {func.get('name', 'Unknown'): f"matlab_api/{func.get('name', 'unknown')}.md"}
                            for func in sorted_funcs
                        ]}
                    ]
                }
            ],
            
            'extra': {
                'social': [
                    {
                        'icon': 'fontawesome/brands/github',
                        'link': 'https://github.com/pulseq/pulseq'
                    }
                ]
            }
        }
        
        # Write as YAML
        import yaml
        with open('mkdocs.yml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print("Created mkdocs.yml")
    
    def generate_all_docs(self):
        """Main function to generate all documentation."""
        # Fetch functions from database
        functions = self.fetch_matlab_functions()
        
        if not functions:
            print("No MATLAB functions found in database")
            return
        
        # Create set of existing function names for cross-reference validation
        existing_functions = {func.get('name', '') for func in functions if func.get('name')}
        
        # Generate individual function pages
        print(f"\nGenerating documentation for {len(functions)} functions...")
        for func in functions:
            name = func.get('name', 'unknown')
            content = self.generate_function_doc(func, existing_functions)
            
            # Save markdown file
            filepath = self.docs_dir / f"{name}.md"
            filepath.write_text(content, encoding='utf-8')
            print(f"  Created: {filepath}")
        
        # Generate index page
        print("\nGenerating index page...")
        index_content = self.generate_index_page(functions)
        index_path = self.docs_dir / "index.md"
        index_path.write_text(index_content, encoding='utf-8')
        print(f"  Created: {index_path}")
        
        # Generate MkDocs config
        self.generate_mkdocs_config(functions)
        
        # Create main index focused on API only
        main_index = """# PulsePal MATLAB API Documentation

## MATLAB Function Reference

This site provides complete documentation for all MATLAB functions in the Pulseq framework.

### Quick Links

- [Browse All Functions](matlab_api/index.md) - Complete alphabetical listing
- [Official Pulseq Repository](https://github.com/pulseq/pulseq) - Source code and examples

### Getting Started

1. Browse the [complete function list](matlab_api/index.md)
2. Click on any function for detailed documentation
3. Each function page includes syntax, parameters, returns, and examples
"""
        
        docs_index = Path("docs/index.md")
        docs_index.parent.mkdir(exist_ok=True)
        docs_index.write_text(main_index, encoding='utf-8')
        print(f"  Created: {docs_index}")
        
        print("\n✅ Documentation generation complete!")
        print("\nNext steps:")
        print("1. Install MkDocs: pip install mkdocs-material")
        print("2. Preview locally: mkdocs serve")
        print("3. Build static site: mkdocs build")
        print("4. Deploy to GitHub Pages: mkdocs gh-deploy")


if __name__ == "__main__":
    try:
        generator = MatlabDocGenerator()
        generator.generate_all_docs()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()