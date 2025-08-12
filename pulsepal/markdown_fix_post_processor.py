#\!/usr/bin/env python3
"""
Post-processor to fix common Gemini markdown formatting issues.
Specifically addresses the problem where Gemini wraps explanatory text in code blocks.
"""

import re
import logging

logger = logging.getLogger(__name__)


def fix_markdown_code_blocks(text: str) -> str:
    """
    Fix common markdown formatting issues from Gemini responses.
    
    Issues addressed:
    1. Code blocks that don't close properly
    2. Explanatory text wrapped in code blocks
    3. Multiple consecutive code blocks without proper separation
    
    Args:
        text: Raw response text from Gemini
        
    Returns:
        Fixed markdown text
    """
    
    # Pattern to detect problematic code blocks
    # Look for code blocks that contain markdown headers or bullet points
    # which indicate they should be regular text
    
    lines = text.split('\n')
    fixed_lines = []
    in_code_block = False
    code_lang = ""
    code_buffer = []
    
    for i, line in enumerate(lines):
        # Check for code block delimiter
        if line.strip().startswith('```'):
            if not in_code_block:
                # Starting a code block
                in_code_block = True
                # Extract language if specified
                code_lang = line.strip()[3:].strip()
                code_buffer = []
                fixed_lines.append(line)
            else:
                # Ending a code block
                in_code_block = False
                
                # Check if the code buffer contains markdown that shouldn't be in a code block
                code_content = '\n'.join(code_buffer)
                
                # Heuristics to detect if this is actually explanatory text, not code
                looks_like_markdown = any([
                    '###' in code_content and not code_lang,  # Markdown headers
                    re.search(r'^\*\s+\*\*', code_content, re.MULTILINE),  # Bullet points with bold
                    re.search(r'^\d+\.\s+\*\*', code_content, re.MULTILINE),  # Numbered lists with bold
                    'Args:' in code_content and 'Returns:' in code_content and not code_lang,  # Docstring-like
                    code_content.count('**') > 4 and not code_lang,  # Multiple bold markers
                ])
                
                if looks_like_markdown and not code_lang:
                    # This was incorrectly wrapped in a code block
                    logger.warning(f"Found explanatory text wrapped in code block at line {i}")
                    # Don't add the opening ``` we already added
                    fixed_lines = fixed_lines[:-1]
                    # Add the content as regular text
                    fixed_lines.extend(code_buffer)
                    # Don't add the closing ```
                else:
                    # This is a legitimate code block
                    fixed_lines.append(line)
                
                code_buffer = []
        else:
            if in_code_block:
                code_buffer.append(line)
            else:
                fixed_lines.append(line)
    
    # Handle unclosed code block at end
    if in_code_block:
        logger.warning("Found unclosed code block at end of response")
        # Check if it's actually markdown content
        code_content = '\n'.join(code_buffer)
        if any(['###' in code_content, '**' in code_content, 'Args:' in code_content]):
            # Remove the opening ``` and treat as regular text
            fixed_lines = fixed_lines[:-1]
            fixed_lines.extend(code_buffer)
        else:
            # Close the code block
            fixed_lines.append('```')
    
    return '\n'.join(fixed_lines)


def validate_markdown_structure(text: str) -> list[str]:
    """
    Validate markdown structure and return list of issues found.
    
    Args:
        text: Markdown text to validate
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    # Count code block delimiters
    code_blocks = text.count('```')
    if code_blocks % 2 != 0:
        issues.append(f"Unmatched code block delimiters (found {code_blocks})")
    
    # Check for nested code blocks (simplified check)
    lines = text.split('\n')
    in_code_block = False
    for i, line in enumerate(lines):
        if line.strip().startswith('```'):
            if not in_code_block:
                in_code_block = True
            else:
                # Check if next non-empty line is also ```
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip():
                        if not lines[j].strip().startswith('```'):
                            # Found content between ``` markers
                            break
                        else:
                            issues.append(f"Possible nested code blocks at line {i+1}")
                        break
                in_code_block = False
    
    return issues


def apply_markdown_fixes(response: str) -> tuple[str, bool]:
    """
    Apply markdown fixes to a Gemini response.
    
    Args:
        response: Raw response from Gemini
        
    Returns:
        Tuple of (fixed_response, was_modified)
    """
    # First validate
    issues = validate_markdown_structure(response)
    
    if not issues:
        return response, False
    
    logger.info(f"Found markdown issues: {issues}")
    
    # Apply fixes
    fixed = fix_markdown_code_blocks(response)
    
    # Validate again
    remaining_issues = validate_markdown_structure(fixed)
    
    if remaining_issues:
        logger.warning(f"Some issues remain after fixing: {remaining_issues}")
    else:
        logger.info("All markdown issues resolved")
    
    return fixed, True
