"""
Regenerate summaries and embeddings for all entries in crawled_pages table.
Excludes PyPulseq entries (github.com/imr-framework/pypulseq).
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re

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
        logging.FileHandler(f'regeneration_log_{timestamp}.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for the regeneration process."""
    batch_size: int = 10
    test_mode: bool = False
    test_entries: int = 10
    checkpoint_file: str = "regeneration_checkpoint.json"
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
    
    def verify_backup_exists(self) -> bool:
        """Verify that backup table exists."""
        try:
            # Check if backup table has data
            result = self.client.table('crawled_pages_backup').select('id').limit(1).execute()
            if result.data:
                count_result = self.client.table('crawled_pages_backup').select('id', count='exact').execute()
                logger.info(f"Backup table exists with {count_result.count} entries")
                return True
            else:
                logger.warning("Backup table exists but is empty")
                return False
        except Exception as e:
            logger.error(f"Backup table not found or inaccessible: {e}")
            logger.info("Please create backup with: CREATE TABLE crawled_pages_backup AS SELECT * FROM crawled_pages;")
            return False
    
    def get_entries_to_process(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict]:
        """Get entries to process, excluding PyPulseq."""
        try:
            query = self.client.table('crawled_pages').select('*')
            
            # Exclude PyPulseq entries
            query = query.neq('metadata->>source', 'github.com/imr-framework/pypulseq')
            
            if limit:
                query = query.range(offset, offset + limit - 1)
            
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

class GeminiProcessor:
    """Handles Gemini API operations for summary and metadata generation."""
    
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment")
        
        genai.configure(api_key=api_key)
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
    
    def _extract_function_info(self, code: str, language: str) -> Tuple[str, str, str]:
        """Extract function name, signature, and headers from code."""
        function_name = ""
        function_signature = ""
        headers = ""
        
        if language == "matlab":
            # Look for MATLAB function definition
            func_pattern = r'^function\s+(?:\[?[\w,\s\[\]]+\]?\s*=\s*)?(\w+)\s*\([^)]*\)'
            match = re.search(func_pattern, code, re.MULTILINE)
            if match:
                function_name = match.group(1) if match.group(1) else ""
                function_signature = match.group(0)
                
                # Extract help text (comments immediately after function definition)
                lines = code.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('function'):
                        # Get subsequent comment lines
                        help_lines = []
                        for j in range(i+1, min(i+20, len(lines))):
                            if lines[j].strip().startswith('%'):
                                help_lines.append(lines[j].strip())
                            elif lines[j].strip() and not lines[j].strip().startswith('%'):
                                break
                        headers = '\n'.join(help_lines[:10])  # Limit to first 10 lines
                        break
        
        elif language == "python":
            # Look for Python function definition
            func_pattern = r'^def\s+(\w+)\s*\([^)]*\):'
            match = re.search(func_pattern, code, re.MULTILINE)
            if match:
                function_name = match.group(1)
                function_signature = match.group(0).rstrip(':')
                
                # Extract docstring
                docstring_pattern = r'"""(.*?)"""'
                doc_match = re.search(docstring_pattern, code[match.end():], re.DOTALL)
                if doc_match:
                    headers = doc_match.group(1).strip()[:500]  # Limit length
        
        return function_name, function_signature, headers
    
    def process_entry(self, entry: Dict) -> Optional[Dict]:
        """Process a single entry to generate summary and extract metadata."""
        self._rate_limit()
        
        try:
            content = entry['content']
            metadata = entry['metadata']
            
            # Split content at separator
            if '---' in content:
                parts = content.split('---', 1)
                code = parts[1].strip() if len(parts) > 1 else content
            else:
                code = content
            
            # Determine file type
            language = metadata.get('language', 'unknown')
            file_path = metadata.get('file_path', '')
            
            # Create prompt for Gemini
            prompt = f"""Analyze this {language} code file from {file_path}.

Provide:
1. A comprehensive summary (200-500 words) that includes:
   - Primary purpose and functionality
   - Key algorithms, methods, or techniques used
   - Main dependencies and libraries used
   - Typical use cases and context within the project
   - Any notable implementation details

2. Extract all function/method dependencies (list of function names called)

3. If this is a function file (not a script), identify the main function name and its signature.

Code:
```{language}
{code[:50000]}  # Limit to prevent extreme cases
```

Return as JSON:
{{
  "summary": "...",
  "dependencies": ["func1", "func2", ...],
  "is_function": true/false,
  "main_function_name": "..." (if is_function is true),
  "function_signature": "..." (if is_function is true)
}}"""

            response = self.model.generate_content(prompt)
            result_text = response.text
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                logger.warning(f"Could not extract JSON from Gemini response for entry {entry['id']}")
                return None
            
            # Build new content
            new_summary = result.get('summary', 'No summary available')
            new_content = f"{new_summary}\n---\n{code}"
            
            # Update metadata
            new_metadata = metadata.copy()
            
            # Extract function info if applicable
            if result.get('is_function', False):
                function_name, function_signature, headers = self._extract_function_info(code, language)
                new_metadata['function_name'] = result.get('main_function_name', function_name)
                new_metadata['function_signature'] = result.get('function_signature', function_signature)
                new_metadata['headers'] = headers
            else:
                new_metadata['function_name'] = ""
                new_metadata['function_signature'] = ""
                new_metadata['headers'] = ""
            
            # Update dependencies
            new_metadata['dependencies'] = result.get('dependencies', [])
            
            # Update counts
            new_metadata['char_count'] = len(new_content)
            new_metadata['word_count'] = len(new_content.split())
            
            return {
                'id': entry['id'],
                'content': new_content,
                'metadata': new_metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to process entry {entry['id']}: {e}")
            return None

class EmbeddingGenerator:
    """Handles Google Embeddings API operations."""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY_EMBEDDING")
        if not self.api_key:
            raise ValueError("Missing GOOGLE_API_KEY_EMBEDDING in environment")
        
        genai.configure(api_key=self.api_key)
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
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

class RegenerationProcessor:
    """Main processor for regenerating summaries and embeddings."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.db = DatabaseManager()
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
        return {'processed_ids': [], 'failed_ids': []}
    
    def _save_checkpoint(self):
        """Save checkpoint to file."""
        with open(self.config.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f)
    
    def run(self):
        """Run the regeneration process."""
        logger.info("Starting regeneration process")
        
        # Verify backup exists
        if not self.config.test_mode:
            if not self.db.verify_backup_exists():
                logger.error("Backup verification failed. Aborting.")
                return
        
        # Get entries to process
        if self.config.test_mode:
            entries = self.db.get_entries_to_process(limit=self.config.test_entries)
            logger.info(f"Test mode: Processing {len(entries)} entries")
        else:
            entries = self.db.get_entries_to_process()
            logger.info(f"Processing {len(entries)} entries (excluding PyPulseq)")
        
        # Filter out already processed entries
        entries = [e for e in entries if e['id'] not in self.checkpoint['processed_ids']]
        logger.info(f"Skipping {len(self.checkpoint['processed_ids'])} already processed entries")
        
        # Process in batches
        success_count = 0
        failed_count = 0
        
        with tqdm(total=len(entries), desc="Processing entries") as pbar:
            for i in range(0, len(entries), self.config.batch_size):
                batch = entries[i:i + self.config.batch_size]
                
                for entry in batch:
                    # Process with Gemini
                    processed = self.gemini.process_entry(entry)
                    if not processed:
                        logger.warning(f"Failed to process entry {entry['id']} with Gemini")
                        self.checkpoint['failed_ids'].append(entry['id'])
                        failed_count += 1
                        pbar.update(1)
                        continue
                    
                    # Generate embedding
                    embedding = self.embedder.generate_embedding(processed['content'])
                    if not embedding:
                        logger.warning(f"Failed to generate embedding for entry {entry['id']}")
                        self.checkpoint['failed_ids'].append(entry['id'])
                        failed_count += 1
                        pbar.update(1)
                        continue
                    
                    # Update database
                    if self.db.update_entry(
                        processed['id'],
                        processed['content'],
                        processed['metadata'],
                        embedding
                    ):
                        self.checkpoint['processed_ids'].append(entry['id'])
                        success_count += 1
                        logger.debug(f"Successfully updated entry {entry['id']}")
                    else:
                        self.checkpoint['failed_ids'].append(entry['id'])
                        failed_count += 1
                    
                    pbar.update(1)
                    
                    # Save checkpoint after EACH entry for better resume capability
                    self._save_checkpoint()
                
                # Also save checkpoint after each batch (redundant but safe)
                self._save_checkpoint()
        
        # Final report
        logger.info(f"Processing complete: {success_count} successful, {failed_count} failed")
        
        # Save list of PyPulseq entries for future reference
        self._save_pypulseq_list()
    
    def _save_pypulseq_list(self):
        """Save list of skipped PyPulseq entries."""
        try:
            query = self.db.client.table('crawled_pages').select('id, url')
            query = query.eq('metadata->>source', 'github.com/imr-framework/pypulseq')
            result = query.execute()
            
            with open('skipped_pypulseq_entries.txt', 'w') as f:
                f.write(f"PyPulseq entries skipped ({len(result.data)} total):\n\n")
                for entry in result.data:
                    f.write(f"ID: {entry['id']}, URL: {entry['url']}\n")
            
            logger.info(f"Saved {len(result.data)} PyPulseq entries to skipped_pypulseq_entries.txt")
        except Exception as e:
            logger.error(f"Failed to save PyPulseq list: {e}")

def main():
    """Main entry point."""
    # Parse command line arguments
    test_mode = '--test' in sys.argv
    
    config = ProcessingConfig(test_mode=test_mode)
    processor = RegenerationProcessor(config)
    
    try:
        processor.run()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Checkpoint saved.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()