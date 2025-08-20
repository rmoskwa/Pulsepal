"""
Re-crawl pages from GitHub with correct content fetching.
Fetches actual content from URLs, generates simple summaries, and updates database.
"""

import os
import sys
import json
import time
import asyncio
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Google Gemini
import google.generativeai as genai

# Supabase
from supabase import create_client, Client
import numpy as np

# Progress tracking
from tqdm import tqdm

# Configure logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'recrawl_log_{timestamp}.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CrawlConfig:
    """Configuration for the re-crawling process."""
    test_mode: bool = True
    test_entries: int = 20
    batch_size: int = 5
    checkpoint_file: str = "recrawl_checkpoint.json"
    gemini_rate_limit: int = 60  # requests per minute
    embedding_rate_limit: int = 1500  # requests per minute
    
class DatabaseManager:
    """Manages Supabase database operations."""
    
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        if not self.url or not self.key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment")
        
        self.client: Client = create_client(self.url, self.key)
        logger.info("Connected to Supabase")
    
    def get_diverse_test_entries(self, limit: int = 20) -> List[Dict]:
        """Get diverse test entries with various file extensions."""
        try:
            # Get a mix of different file types
            entries = []
            
            # Scale the distribution based on limit
            scale = max(1, limit // 20)
            
            # Define file types to test (scaled for larger limits)
            file_types = [
                ('%.m', 5 * scale),      # MATLAB files
                ('%.py', 3 * scale),     # Python files
                ('%.md', 2 * scale),     # Markdown files
                ('%.ipynb', 2 * scale),  # Jupyter notebooks
                ('%.pdf', 2 * scale),    # PDF files
                ('%.yml', 1 * scale),    # YAML files
                ('%.xml', 1 * scale),    # XML files
                ('%.cpp', 1 * scale),    # C++ files
                ('%.h', 1 * scale),      # Header files
                ('%.css', 1 * scale),    # CSS files
                ('%.tex', 1 * scale),    # LaTeX files
            ]
            
            for pattern, count in file_types:
                query = self.client.table('crawled_pages').select('*')
                query = query.neq('metadata->>source', 'github.com/imr-framework/pypulseq')
                query = query.like('url', pattern)
                query = query.limit(count)
                result = query.execute()
                
                if result.data:
                    entries.extend(result.data[:count])
                    logger.info(f"Found {len(result.data)} entries for pattern {pattern}")
            
            # If we don't have enough diverse entries, fill with any non-PyPulseq entries
            if len(entries) < limit:
                remaining = limit - len(entries)
                existing_ids = {e['id'] for e in entries}
                
                query = self.client.table('crawled_pages').select('*')
                query = query.neq('metadata->>source', 'github.com/imr-framework/pypulseq')
                query = query.limit(remaining + 20)  # Get extra to filter
                result = query.execute()
                
                for entry in result.data:
                    if entry['id'] not in existing_ids:
                        entries.append(entry)
                        if len(entries) >= limit:
                            break
            
            logger.info(f"Selected {len(entries)} diverse test entries")
            
            # Log file type distribution
            extensions = {}
            for entry in entries:
                ext = Path(entry['url']).suffix
                extensions[ext] = extensions.get(ext, 0) + 1
            logger.info(f"File type distribution: {extensions}")
            
            return entries[:limit]
            
        except Exception as e:
            logger.error(f"Failed to fetch diverse test entries: {e}")
            return []
    
    def get_all_entries_to_process(self, offset: int = 0) -> List[Dict]:
        """Get all entries to process, excluding PyPulseq."""
        try:
            query = self.client.table('crawled_pages').select('*')
            query = query.neq('metadata->>source', 'github.com/imr-framework/pypulseq')
            query = query.range(offset, offset + 999)  # Supabase limit
            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to fetch entries: {e}")
            return []
    
    def update_entry(self, entry_id: int, content: str, metadata: Dict, embedding: List[float]) -> bool:
        """Update a single entry with new content, metadata, and embedding."""
        try:
            self.client.table('crawled_pages').update({
                'content': content,
                'metadata': metadata,
                'embedding': embedding
            }).eq('id', entry_id).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to update entry {entry_id}: {e}")
            return False

class ContentFetcher:
    """Fetches content from GitHub URLs using multiple methods."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PulsePal-Crawler/1.0'
        })
    
    def fetch_from_url(self, url: str) -> Tuple[Optional[str], Optional[bytes], str]:
        """
        Fetch content from GitHub URL.
        Returns: (text_content, binary_content, error_message)
        """
        try:
            # Convert blob URL to raw URL
            if 'github.com' in url and '/blob/' in url:
                raw_url = url.replace('github.com', 'raw.githubusercontent.com')
                raw_url = raw_url.replace('/blob/', '/')
            else:
                raw_url = url
            
            logger.debug(f"Fetching from: {raw_url}")
            
            # Fetch content
            response = self.session.get(raw_url, timeout=30)
            
            if response.status_code == 200:
                # Try to decode as text
                try:
                    text_content = response.content.decode('utf-8')
                    return text_content, None, ""
                except UnicodeDecodeError:
                    # Return as binary
                    return None, response.content, ""
            else:
                error_msg = f"HTTP {response.status_code}: {response.reason}"
                logger.warning(f"Failed to fetch {url}: {error_msg}")
                return None, None, error_msg
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error fetching {url}: {error_msg}")
            return None, None, error_msg
    
    def process_special_formats(self, content: str, file_ext: str) -> str:
        """Process special file formats to extract meaningful content."""
        if file_ext == '.ipynb':
            try:
                notebook = json.loads(content)
                extracted = []
                
                # Extract markdown and code cells
                for cell in notebook.get('cells', []):
                    cell_type = cell.get('cell_type', '')
                    source = cell.get('source', [])
                    
                    if isinstance(source, list):
                        source = ''.join(source)
                    
                    # Remove base64 embedded images to save space and improve RAG
                    # They provide no value for search and take up huge amounts of space
                    import re
                    source = re.sub(
                        r'!\[([^\]]*)\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)',
                        r'![\\1](image-removed-for-storage)',
                        source
                    )
                    
                    # Add content without cell type markers for better RAG search
                    # The cell type doesn't add value for searching code/documentation
                    if cell_type == 'markdown':
                        # Add markdown content directly
                        extracted.append(source)
                    elif cell_type == 'code':
                        # Add code content directly
                        extracted.append(source)
                        
                        # Include text outputs only (skip image outputs)
                        outputs = cell.get('outputs', [])
                        for output in outputs:
                            if 'text' in output:
                                output_text = output['text']
                                if isinstance(output_text, list):
                                    output_text = ''.join(output_text)
                                # Only include if not too large
                                if len(output_text) < 1000:
                                    extracted.append(f"# Output:\n{output_text}")
                
                return '\n\n'.join(extracted)
            except:
                return content
        
        return content
    
    def sanitize_text(self, text: str) -> str:
        """Remove problematic Unicode characters that can't be stored in database."""
        # Remove null bytes and other control characters
        # Keep newlines, tabs, and other common whitespace
        import unicodedata
        
        # Remove null bytes first
        text = text.replace('\x00', '')
        
        # Remove other problematic control characters (except \t, \n, \r)
        cleaned = []
        for char in text:
            if ord(char) < 32 and char not in '\t\n\r':
                # Skip control characters
                continue
            elif ord(char) == 127:  # DEL character
                continue
            else:
                cleaned.append(char)
        
        result = ''.join(cleaned)
        
        # Also normalize Unicode to prevent other issues
        result = unicodedata.normalize('NFKC', result)
        
        return result
    
    def process_pdf_content(self, binary_content: bytes) -> str:
        """Extract text from PDF binary content."""
        try:
            import pdfplumber
            import io
            
            # Create a BytesIO object from the binary content
            pdf_file = io.BytesIO(binary_content)
            
            extracted_text = []
            with pdfplumber.open(pdf_file) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract text from each page
                    text = page.extract_text()
                    if text:
                        # Sanitize the extracted text
                        text = self.sanitize_text(text)
                        extracted_text.append(f"# Page {i+1}\n\n{text}")
            
            if extracted_text:
                full_text = '\n\n'.join(extracted_text)
                logger.info(f"Extracted {len(full_text)} chars from PDF ({len(pdf.pages)} pages)")
                return full_text
            else:
                logger.warning("No text could be extracted from PDF")
                return "PDF document with no extractable text content."
                
        except Exception as e:
            logger.error(f"Failed to extract PDF content: {e}")
            return f"PDF extraction failed: {str(e)}"

class GeminiProcessor:
    """Handles Gemini API operations for summary generation."""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.last_request_time = 0
        self.rate_limit_delay = 60 / 60  # 60 requests per minute
        logger.info("Initialized Gemini 2.5 Flash model")
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def generate_summary(self, content: str, metadata: Dict, is_binary: bool = False) -> Tuple[str, Dict]:
        """Generate simple summary for content."""
        self._rate_limit()
        
        try:
            file_path = metadata.get('file_path', '')
            language = metadata.get('language', 'unknown')
            file_ext = Path(file_path).suffix
            
            # Handle binary files
            if is_binary:
                if file_ext == '.pdf':
                    summary = f"PDF document from {file_path}. Binary content ({len(content)} bytes). This PDF file contains documentation, specifications, or research papers related to the Pulseq MRI sequence programming framework."
                else:
                    summary = f"Binary file from {file_path}. File type: {file_ext}. Size: {len(content)} bytes."
                
                return summary, metadata
            
            # Create appropriate prompt based on file type
            if file_ext in ['.m', '.py', '.cpp', '.h', '.js']:
                # Code files
                prompt = f"""Analyze this {language} code file from {file_path}.

Create a simple, focused summary (150-300 words) that describes:
1. The primary purpose and functionality of this code
2. Key functions or classes defined
3. Main algorithms or techniques implemented
4. Dependencies and libraries used
5. How this fits into the larger Pulseq/MRI programming context

Code:
```{language}
{content[:50000]}
```

Provide ONLY the summary text, no JSON or formatting."""

            elif file_ext in ['.md', '.txt', '.tex']:
                # Documentation files
                prompt = f"""Analyze this documentation file from {file_path}.

Create a simple summary (150-300 words) that describes:
1. The main topics covered
2. Key concepts explained
3. Important information or instructions provided
4. Target audience or use case
5. Relevance to Pulseq/MRI sequence programming

Content:
{content[:50000]}

Provide ONLY the summary text, no JSON or formatting."""

            elif file_ext == '.ipynb':
                # Jupyter notebooks
                prompt = f"""Analyze this Jupyter notebook content from {file_path}.

Create a simple summary (150-300 words) that describes:
1. The educational or research purpose of this notebook
2. Key concepts demonstrated or explained
3. Main code examples and their purposes
4. Expected learning outcomes or results
5. Relevance to MRI sequence programming with Pulseq

Notebook content:
{content[:50000]}

Provide ONLY the summary text, no JSON or formatting."""

            else:
                # Generic files
                prompt = f"""Analyze this {file_ext} file from {file_path}.

Create a simple summary (150-300 words) that describes:
1. The purpose and content of this file
2. Key information or functionality provided
3. How this relates to the Pulseq project
4. Any notable features or configurations

Content:
{content[:50000]}

Provide ONLY the summary text, no JSON or formatting."""

            # Ensure correct API key is configured
            genai.configure(api_key=self.api_key)
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            
            # Update metadata
            new_metadata = metadata.copy()
            new_metadata['char_count'] = len(content)
            new_metadata['word_count'] = len(content.split())
            new_metadata['file_extension'] = file_ext
            new_metadata['summary_generated'] = datetime.now().isoformat()
            
            # Remove old notebook_context if it exists (contains old base64 data)
            if 'notebook_context' in new_metadata:
                del new_metadata['notebook_context']
            
            # Remove contextual_embedding if it exists (outdated)
            if 'contextual_embedding' in new_metadata:
                del new_metadata['contextual_embedding']
            
            # Extract function information for code files
            if file_ext in ['.m', '.py']:
                functions = self._extract_functions(content, language)
                if functions:
                    new_metadata['function_name'] = functions[0]['name'] if functions else ""
                    new_metadata['function_signature'] = functions[0]['signature'] if functions else ""
                    new_metadata['all_functions'] = [f['name'] for f in functions]
                else:
                    new_metadata['function_name'] = ""
                    new_metadata['function_signature'] = ""
                    new_metadata['all_functions'] = []
            
            return summary, new_metadata
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            # Return a fallback summary
            return f"Content from {metadata.get('file_path', 'unknown')}. Summary generation failed.", metadata
    
    def _extract_functions(self, code: str, language: str) -> List[Dict]:
        """Extract function definitions from code."""
        functions = []
        
        if language == "matlab":
            pattern = r'^function\s+(?:\[?[\w,\s\[\]]+\]?\s*=\s*)?(\w+)\s*(\([^)]*\))'
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                functions.append({
                    'name': match.group(1),
                    'signature': match.group(0)
                })
        
        elif language == "python":
            pattern = r'^def\s+(\w+)\s*(\([^)]*\)):'
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                functions.append({
                    'name': match.group(1),
                    'signature': f"def {match.group(1)}{match.group(2)}"
                })
        
        return functions

class EmbeddingGenerator:
    """Handles Google Embeddings API operations."""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY_EMBEDDING")
        if not self.api_key:
            # Fall back to main API key
            self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Missing embedding API key")
        
        # Store the original configuration
        self.last_request_time = 0
        self.rate_limit_delay = 60 / 1500  # 1500 requests per minute
        logger.info("Initialized Google Embeddings API")
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text."""
        self._rate_limit()
        
        try:
            original_size = len(text.encode('utf-8'))
            
            # Truncate text if too long (API limit is ~36KB)
            # Assuming UTF-8, limit to ~30KB to be safe
            if original_size > 30000:
                logger.debug(f"Content too large ({original_size} bytes), truncating...")
                
                # Keep summary and truncate content
                if '---' in text:
                    parts = text.split('---', 1)
                    summary = parts[0]
                    content = parts[1] if len(parts) > 1 else ''
                    # Calculate how much content we can keep
                    summary_bytes = len(summary.encode('utf-8'))
                    content_limit = 29000 - summary_bytes  # Leave room for separator
                    if content_limit > 0:
                        # Truncate content to fit
                        truncated_content = content.encode('utf-8')[:content_limit].decode('utf-8', errors='ignore')
                        text = f"{summary}---{truncated_content}"
                        logger.debug(f"Truncated from {original_size} to {len(text.encode('utf-8'))} bytes")
                    else:
                        # Summary alone is too large
                        text = summary.encode('utf-8')[:29000].decode('utf-8', errors='ignore')
                        logger.warning(f"Summary alone exceeds limit, truncating summary")
                else:
                    # No separator, just truncate
                    text = text.encode('utf-8')[:30000].decode('utf-8', errors='ignore')
                    logger.debug(f"No separator found, truncated to {len(text.encode('utf-8'))} bytes")
            
            # Final size check
            final_size = len(text.encode('utf-8'))
            if final_size > 35000:  # Leave some margin below 36KB
                logger.error(f"Content still too large after truncation: {final_size} bytes")
                # Force truncate more aggressively
                text = text.encode('utf-8')[:28000].decode('utf-8', errors='ignore')
                logger.info(f"Force truncated to {len(text.encode('utf-8'))} bytes")
            
            # Configure API key just for this call
            genai.configure(api_key=self.api_key)
            
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

class RecrawlProcessor:
    """Main processor for re-crawling pages with correct content."""
    
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.db = DatabaseManager()
        self.fetcher = ContentFetcher()
        self.gemini = GeminiProcessor()
        self.embedder = EmbeddingGenerator()
        self.checkpoint = self._load_checkpoint()
        
    def _load_checkpoint(self) -> Dict:
        """Load checkpoint from file if exists."""
        if os.path.exists(self.config.checkpoint_file):
            try:
                with open(self.config.checkpoint_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'processed_ids': [],
            'failed_ids': [],
            'fetch_errors': {},
            'summary_errors': {}
        }
    
    def _save_checkpoint(self):
        """Save checkpoint to file."""
        with open(self.config.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def process_entry(self, entry: Dict) -> bool:
        """Process a single entry: fetch content, generate summary, create embedding."""
        entry_id = entry['id']
        url = entry['url']
        
        logger.info(f"Processing entry {entry_id}: {url}")
        
        # Step 1: Fetch content from URL
        text_content, binary_content, error = self.fetcher.fetch_from_url(url)
        
        if error:
            logger.error(f"Failed to fetch content for {entry_id}: {error}")
            self.checkpoint['fetch_errors'][str(entry_id)] = error
            self.checkpoint['failed_ids'].append(entry_id)
            return False
        
        # Determine content type and process
        is_binary = binary_content is not None
        file_ext = Path(url).suffix
        
        if is_binary:
            # Check if it's a PDF
            if file_ext == '.pdf':
                # Extract text from PDF
                content = self.fetcher.process_pdf_content(binary_content)
                raw_content = content
                is_binary = False  # We now have text content
            else:
                # For other binary files, we'll store metadata about them
                content = f"Binary file: {Path(url).name}"
                raw_content = content
        else:
            # Process special formats
            content = self.fetcher.process_special_formats(text_content, file_ext)
            raw_content = content
        
        # Step 2: Generate summary
        summary, updated_metadata = self.gemini.generate_summary(
            content if not is_binary else str(len(binary_content)),
            entry['metadata'],
            is_binary
        )
        
        # Step 3: Format content with summary
        # Sanitize the final content to remove problematic Unicode characters
        final_content = f"{summary}\n---\n{raw_content}"
        final_content = self.fetcher.sanitize_text(final_content)
        
        # Step 4: Generate embedding
        embedding = self.embedder.generate_embedding(final_content)
        if not embedding:
            logger.error(f"Failed to generate embedding for {entry_id}")
            self.checkpoint['failed_ids'].append(entry_id)
            return False
        
        # Step 5: Update database
        if self.db.update_entry(entry_id, final_content, updated_metadata, embedding):
            logger.info(f"Successfully updated entry {entry_id}")
            self.checkpoint['processed_ids'].append(entry_id)
            return True
        else:
            logger.error(f"Failed to update database for entry {entry_id}")
            self.checkpoint['failed_ids'].append(entry_id)
            return False
    
    def run(self):
        """Run the re-crawling process."""
        logger.info("="*60)
        logger.info("Starting re-crawling process")
        logger.info(f"Mode: {'TEST' if self.config.test_mode else 'FULL'}")
        
        # Get entries to process
        if self.config.test_mode:
            entries = self.db.get_diverse_test_entries(self.config.test_entries)
            logger.info(f"Test mode: Processing {len(entries)} diverse entries")
        else:
            # Get all entries in batches
            all_entries = []
            offset = 0
            while True:
                batch = self.db.get_all_entries_to_process(offset)
                if not batch:
                    break
                all_entries.extend(batch)
                offset += len(batch)
            entries = all_entries
            logger.info(f"Full mode: Processing {len(entries)} entries (excluding PyPulseq)")
        
        # Filter out already processed entries
        entries = [e for e in entries if e['id'] not in self.checkpoint['processed_ids']]
        if self.checkpoint['processed_ids']:
            logger.info(f"Skipping {len(self.checkpoint['processed_ids'])} already processed entries")
        
        # Process entries
        success_count = 0
        failed_count = 0
        
        with tqdm(total=len(entries), desc="Re-crawling pages") as pbar:
            for entry in entries:
                success = self.process_entry(entry)
                
                if success:
                    success_count += 1
                else:
                    failed_count += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'success': success_count,
                    'failed': failed_count
                })
                
                # Save checkpoint after each entry
                self._save_checkpoint()
        
        # Generate report
        self._generate_report(success_count, failed_count)
    
    def _generate_report(self, success_count: int, failed_count: int):
        """Generate a detailed report of the re-crawling process."""
        report_file = f"recrawl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("RE-CRAWLING REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Mode: {'TEST' if self.config.test_mode else 'FULL'}\n")
            f.write(f"Total Processed: {success_count + failed_count}\n")
            f.write(f"Successful: {success_count}\n")
            f.write(f"Failed: {failed_count}\n\n")
            
            if self.checkpoint['fetch_errors']:
                f.write("FETCH ERRORS:\n")
                f.write("-"*40 + "\n")
                for entry_id, error in self.checkpoint['fetch_errors'].items():
                    f.write(f"Entry {entry_id}: {error}\n")
                f.write("\n")
            
            if self.checkpoint['failed_ids']:
                f.write("FAILED ENTRY IDs:\n")
                f.write("-"*40 + "\n")
                for entry_id in self.checkpoint['failed_ids']:
                    f.write(f"- {entry_id}\n")
                f.write("\n")
            
            f.write("SUMMARY:\n")
            f.write("-"*40 + "\n")
            success_rate = (success_count / (success_count + failed_count) * 100) if (success_count + failed_count) > 0 else 0
            f.write(f"Success Rate: {success_rate:.1f}%\n")
            
            if self.config.test_mode:
                f.write("\nThis was a TEST run. To process all entries, run without --test flag.\n")
        
        logger.info(f"Report saved to {report_file}")
        logger.info(f"Processing complete: {success_count} successful, {failed_count} failed")
        logger.info(f"Success rate: {success_rate:.1f}%")

def main():
    """Main entry point."""
    # Parse command line arguments
    test_mode = '--test' not in sys.argv or '--test' in sys.argv  # Default to test mode
    
    if '--full' in sys.argv:
        test_mode = False
    
    # Check for custom test count
    test_count = 20  # default
    for i, arg in enumerate(sys.argv):
        if arg == '--count' and i + 1 < len(sys.argv):
            try:
                test_count = int(sys.argv[i + 1])
            except ValueError:
                pass
    
    config = CrawlConfig(test_mode=test_mode, test_entries=test_count)
    processor = RecrawlProcessor(config)
    
    try:
        processor.run()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Checkpoint saved.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()