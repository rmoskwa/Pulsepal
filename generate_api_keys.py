#!/usr/bin/env python3
"""
Generate API keys for PulsePal users.

This script creates API keys for alpha/beta testing users and outputs them
in a format suitable for the ALPHA_API_KEYS environment variable.
"""

import json
import secrets
import sys
from datetime import datetime
from typing import Dict, List, Optional


def generate_api_key(prefix: str = "pulsepal") -> str:
    """Generate a secure API key."""
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}-{random_part}"


def create_user_entry(name: str, email: str, limit: int = 1000) -> Dict:
    """Create a user entry for the API key database."""
    return {
        "name": name,
        "email": email,
        "limit": limit,
        "created": datetime.now().strftime("%Y-%m-%d"),
        "usage": 0
    }


def add_api_key(
    existing_keys: Dict,
    name: str,
    email: str,
    limit: int = 1000,
    custom_key: Optional[str] = None
) -> tuple[Dict, str]:
    """Add a new API key to the existing keys dictionary."""
    api_key = custom_key if custom_key else generate_api_key()
    
    if api_key in existing_keys:
        raise ValueError(f"API key '{api_key}' already exists!")
    
    existing_keys[api_key] = create_user_entry(name, email, limit)
    return existing_keys, api_key


def load_existing_keys(filename: str = "alpha_keys.json") -> Dict:
    """Load existing API keys from a file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No existing keys file found at {filename}, starting fresh.")
        return {}
    except json.JSONDecodeError:
        print(f"Error reading {filename}, starting fresh.")
        return {}


def save_keys_to_file(keys: Dict, filename: str = "alpha_keys.json"):
    """Save API keys to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(keys, f, indent=2)
    print(f"Keys saved to {filename}")


def main():
    """Main function for interactive API key generation."""
    print("=== PulsePal API Key Generator ===\n")
    
    # Load existing keys
    keys = load_existing_keys()
    print(f"Currently {len(keys)} API keys in database.\n")
    
    while True:
        print("\nOptions:")
        print("1. Add new API key")
        print("2. List all keys")
        print("3. Export for environment variable")
        print("4. Remove a key")
        print("5. Save and exit")
        print("6. Exit without saving")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            # Add new key
            name = input("User name: ").strip()
            email = input("User email: ").strip()
            limit_str = input("Query limit (default 1000): ").strip()
            limit = int(limit_str) if limit_str else 1000
            
            custom = input("Custom API key (leave blank to auto-generate): ").strip()
            custom_key = custom if custom else None
            
            try:
                keys, new_key = add_api_key(keys, name, email, limit, custom_key)
                print(f"\n✓ Created API key: {new_key}")
                print(f"  User: {name} ({email})")
                print(f"  Limit: {limit} queries")
            except ValueError as e:
                print(f"\n✗ Error: {e}")
        
        elif choice == "2":
            # List all keys
            if not keys:
                print("\nNo API keys in database.")
            else:
                print(f"\n{'='*60}")
                for api_key, info in keys.items():
                    print(f"\nKey: {api_key}")
                    print(f"  Name: {info['name']}")
                    print(f"  Email: {info['email']}")
                    print(f"  Limit: {info['limit']}")
                    print(f"  Created: {info['created']}")
                    if 'usage' in info:
                        print(f"  Usage: {info['usage']}")
                print(f"\n{'='*60}")
        
        elif choice == "3":
            # Export for environment variable
            json_str = json.dumps(keys)
            print("\n=== Environment Variable Format ===")
            print(f"ALPHA_API_KEYS='{json_str}'")
            print("\nCopy the above line to your .env file or set it as an environment variable.")
            
            # Also save to a file for backup
            backup_file = f"alpha_keys_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_keys_to_file(keys, backup_file)
        
        elif choice == "4":
            # Remove a key
            if not keys:
                print("\nNo keys to remove.")
            else:
                print("\nExisting keys:")
                for i, key in enumerate(keys.keys(), 1):
                    print(f"{i}. {key} ({keys[key]['name']})")
                
                key_to_remove = input("\nEnter key to remove (or number): ").strip()
                
                # Check if input is a number
                try:
                    idx = int(key_to_remove) - 1
                    if 0 <= idx < len(keys):
                        key_to_remove = list(keys.keys())[idx]
                except ValueError:
                    pass
                
                if key_to_remove in keys:
                    user_info = keys[key_to_remove]
                    del keys[key_to_remove]
                    print(f"\n✓ Removed key for {user_info['name']} ({user_info['email']})")
                else:
                    print(f"\n✗ Key '{key_to_remove}' not found.")
        
        elif choice == "5":
            # Save and exit
            save_keys_to_file(keys)
            print("\n✓ Keys saved. Exiting.")
            break
        
        elif choice == "6":
            # Exit without saving
            confirm = input("\nAre you sure you want to exit without saving? (y/n): ").strip().lower()
            if confirm == 'y':
                print("Exiting without saving.")
                break
        
        else:
            print("\n✗ Invalid choice. Please enter 1-6.")


if __name__ == "__main__":
    # Check for command line arguments for batch processing
    if len(sys.argv) > 1:
        if sys.argv[1] == "--batch":
            # Batch mode for adding multiple users
            keys = load_existing_keys()
            
            print("Batch mode: Enter user details (empty line to finish)")
            while True:
                line = input("Name,Email,Limit (or empty to finish): ").strip()
                if not line:
                    break
                
                try:
                    parts = line.split(',')
                    name = parts[0].strip()
                    email = parts[1].strip()
                    limit = int(parts[2].strip()) if len(parts) > 2 else 1000
                    
                    keys, new_key = add_api_key(keys, name, email, limit)
                    print(f"✓ Created: {new_key} for {name}")
                except Exception as e:
                    print(f"✗ Error processing line: {e}")
            
            save_keys_to_file(keys)
            print(f"\n✓ Saved {len(keys)} keys to alpha_keys.json")
            
            # Also print environment variable format
            print("\nEnvironment variable format:")
            print(f"ALPHA_API_KEYS='{json.dumps(keys)}'")
        else:
            print("Usage: python generate_api_keys.py [--batch]")
    else:
        main()