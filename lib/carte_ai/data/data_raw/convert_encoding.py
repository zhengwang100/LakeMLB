#!/usr/bin/env python
"""Convert CSV files to UTF-8 encoding."""

import os
import chardet
from pathlib import Path

def detect_encoding(file_path):
    """Detect file encoding."""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']

def convert_to_utf8(file_path):
    """Convert file to UTF-8."""
    try:
        # Detect original encoding
        encoding = detect_encoding(file_path)
        print(f"Detected encoding for {file_path.name}: {encoding}")
        
        if encoding and encoding.lower() != 'utf-8':
            # Read with detected encoding
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # Write as UTF-8
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"  ✓ Converted {file_path.name} to UTF-8")
            return True
        else:
            print(f"  - {file_path.name} already UTF-8")
            return False
            
    except Exception as e:
        print(f"  ✗ Error converting {file_path.name}: {e}")
        return False

def main():
    """Convert all CSV files in current directory to UTF-8."""
    current_dir = Path(__file__).parent
    csv_files = list(current_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found.")
        return
    
    print(f"Found {len(csv_files)} CSV files\n")
    
    converted_count = 0
    for csv_file in csv_files:
        if convert_to_utf8(csv_file):
            converted_count += 1
    
    print(f"\nSummary: Converted {converted_count}/{len(csv_files)} files")

if __name__ == "__main__":
    main()

