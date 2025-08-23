#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union
import os
import sys
import argparse
import re
import time
import heapq
from threading import Lock
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from tabulate import tabulate
from colorama import init, Fore, Style
import mimetypes
import json
import csv
import glob
import chardet  # Added for encoding detection

# Initialize colorama for cross-platform color support
init()

# Configurable stopwords - can be loaded from file
DEFAULT_STOPWORDS = {
    "the", "and", "of", "to", "in", "a", "is", "for", "on", "with", "by", "at", 
    "this", "that", "are", "be", "was", "were", "an", "it", "as", "from", "or",
    "but", "not", "what", "all", "were", "when", "we", "your", "can", "said",
    "there", "use", "each", "which", "she", "how", "their", "will", "up", "other",
    "about", "out", "many", "then", "them", "these", "so", "some", "her", "would",
    "make", "like", "him", "into", "time", "has", "look", "two", "more", "write",
    "go", "see", "number", "no", "way", "could", "people", "my", "than", "first",
    "water", "been", "call", "who", "oil", "its", "now", "find", "long", "down",
    "day", "did", "get", "come", "made", "may", "part"
}

# Common text file extensions for fallback detection
TEXT_EXTENSIONS = {
    '.txt', '.csv', '.json', '.xml', '.html', '.htm', '.md', '.rst', '.py', '.js',
    '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.php', '.rb', '.go', '.rs', '.ts',
    '.css', '.scss', '.less', '.sql', '.sh', '.bash', '.zsh', '.yaml', '.yml',
    '.toml', '.ini', '.cfg', '.conf', '.log'
}

COLORS = {
    "red": Fore.RED,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "blue": Fore.BLUE,
    "reset": Style.RESET_ALL,
}

def load_stopwords(stopwords_file: Optional[Path] = None) -> Set[str]:
    """Load stopwords from file or use defaults."""
    if stopwords_file and stopwords_file.exists():
        try:
            with stopwords_file.open('r', encoding='utf-8') as f:
                return {line.strip().lower() for line in f if line.strip()}
        except IOError:
            print(colorize(f"Warning: Could not load stopwords from {stopwords_file}, using defaults", "yellow"))
    
    return DEFAULT_STOPWORDS

def colorize(text: str, color: str) -> str:
    """Colorize text for console output if supported."""
    return f"{COLORS[color]}{text}{COLORS['reset']}" if sys.stdout.isatty() else text

def detect_encoding(file_path: Path, sample_size: int = 1024) -> str:
    """Detect file encoding using chardet."""
    try:
        with file_path.open('rb') as f:
            raw_data = f.read(sample_size)
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'
    except IOError:
        return 'utf-8'

class FileAnalyzer:
    """Analyzes a text file to count lines, words, characters, bytes, and word frequencies."""

    def __init__(self, file_path: Path, show_progress: bool = False, 
                 encoding: Optional[str] = None, top_words: int = 10, 
                 exclude_stopwords: bool = False, stopwords: Optional[Set[str]] = None,
                 max_memory_usage: int = 1000000):
        self.file_path: Path = Path(file_path)
        self.show_progress: bool = show_progress
        self.encoding: str = encoding or detect_encoding(file_path)
        self.top_words: int = max(1, top_words)
        self.exclude_stopwords: bool = exclude_stopwords
        self.stopwords: Set[str] = stopwords or DEFAULT_STOPWORDS
        self.max_memory_usage: int = max_memory_usage  # Max unique words to track
        self.lock: Lock = Lock()
        self.results: Dict[str, any] = {
            "lines": 0,
            "words": 0,
            "chars": 0,
            "bytes": 0,
            "longest_line": 0,
            "word_freq": [],
            "encoding": self.encoding,
        }
        self.word_counts: Counter = Counter()
        self.error: Optional[str] = None
        self._unique_words_count: int = 0

    def analyze(self, pbar: Optional[tqdm] = None) -> bool:
        """Analyze the file and update results."""
        try:
            file_size = self.file_path.stat().st_size
            if file_size == 0:
                self.results.update({"empty": True})
                return True
            
            batch_size = 0
            with self.file_path.open('rb') as f:
                for line in f:
                    self._process_line(line)
                    batch_size += len(line)
                    if pbar and batch_size >= 1024 * 1024:  # Update every 1MB
                        with self.lock:
                            pbar.update(batch_size)
                        batch_size = 0
                if pbar and batch_size > 0:
                    with self.lock:
                        pbar.update(batch_size)
            
            self._finalize_word_freq()
            return True
        
        except IOError as e:
            self.error = f"IOError analyzing {self.file_path}: {e}"
            return False
        except UnicodeDecodeError as e:
            self.error = f"Encoding error in {self.file_path}: {e}"
            return False
        except MemoryError as e:
            self.error = f"Memory error analyzing {self.file_path}: {e}"
            return False
        except Exception as e:
            self.error = f"Unexpected error analyzing {self.file_path}: {e}"
            return False
        finally:
            if pbar:
                pbar.close()

    def _process_line(self, line: bytes) -> None:
        """Process a single line of the file."""
        try:
            decoded_line = line.decode(self.encoding, errors='replace')
            with self.lock:
                self.results["bytes"] += len(line)
                self.results["chars"] += len(decoded_line)
                self.results["lines"] += 1
                self.results["longest_line"] = max(self.results["longest_line"], len(decoded_line.rstrip()))
                
                # Extract words with improved regex for various languages
                words = re.findall(r"\b[\w\u00C0-\u017F]+(?:['-][\w\u00C0-\u017F]+)*\b", decoded_line.lower())
                self.results["words"] += len(words)
                
                if self.exclude_stopwords:
                    words = [w for w in words if w not in self.stopwords]
                
                # Memory management: stop counting if we hit the limit
                if self._unique_words_count < self.max_memory_usage:
                    for word in words:
                        if word not in self.word_counts:
                            self._unique_words_count += 1
                            if self._unique_words_count >= self.max_memory_usage:
                                break
                        self.word_counts[word] += 1
        
        except UnicodeDecodeError as e:
            if not self.error:  # Only set first error
                self.error = f"Encoding error processing line in {self.file_path}: {e}"
        except MemoryError as e:
            if not self.error:
                self.error = f"Memory error processing line in {self.file_path}: {e}"
        except Exception as e:
            if not self.error:
                self.error = f"Unexpected error processing line in {self.file_path}: {e}"

    def _finalize_word_freq(self) -> None:
        """Compute the top N word frequencies."""
        with self.lock:
            if self.word_counts:
                self.results["word_freq"] = self.word_counts.most_common(self.top_words)
            # Clear counter to free memory
            self.word_counts.clear()
            self._unique_words_count = 0

def is_valid_file(file_path: Path, strict_mime: bool = False) -> bool:
    """Check if a file is a readable text file with multiple fallback methods."""
    path = Path(file_path)
    
    # Basic checks
    if not (path.is_file() and os.access(path, os.R_OK)):
        return False
    
    # Method 1: MIME type detection
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type and mime_type.startswith('text'):
        return True
    
    # Method 2: File extension fallback
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    
    # Method 3: Content sampling for files without clear indicators
    if not strict_mime:
        try:
            with path.open('rb') as f:
                sample = f.read(1024)
                # Check if sample contains mostly text characters
                text_ratio = sum(1 for byte in sample if 32 <= byte <= 126 or byte in (9, 10, 13)) / len(sample)
                if text_ratio > 0.7:  # 70% printable ASCII
                    return True
        except (IOError, UnicodeDecodeError):
            pass
    
    return False

def get_files(args: argparse.Namespace) -> List[Path]:
    """Get a sorted list of valid text files from the provided patterns."""
    valid_files = set()
    for pattern in args.files:
        try:
            matches = glob.glob(pattern, recursive=args.recursive)
            for file_path in matches:
                if is_valid_file(file_path, args.strict_mime):
                    valid_files.add(Path(file_path))
        except (ValueError, OSError) as e:
            print(colorize(f"Error processing pattern '{pattern}': {e}", "red"))
    
    return sorted(valid_files)

def print_individual_results(analyzer: FileAnalyzer, show_word_freq: bool = False) -> None:
    """Print analysis results for a single file."""
    print(f"\nFile: {colorize(str(analyzer.file_path), 'green')}")
    if analyzer.error:
        print(colorize(analyzer.error, "red"))
        return
    
    stats = [
        ("Lines", analyzer.results["lines"]),
        ("Words", analyzer.results["words"]),
        ("Chars", analyzer.results["chars"]),
        ("Bytes", analyzer.results["bytes"]),
        ("Longest Line", f"{analyzer.results['longest_line']} chars"),
        ("Encoding", analyzer.results.get("encoding", "unknown")),
    ]
    
    for label, value in stats:
        print(f"  {label}: {value}")
    
    if show_word_freq and analyzer.results["word_freq"]:
        word_count = len(analyzer.results["word_freq"])
        print("\n" + colorize(f"--- Top {word_count} Words ---", "yellow"))
        for word, count in analyzer.results["word_freq"]:
            print(f"  {word}: {count}")

def save_results(analyzers: List[FileAnalyzer], args: argparse.Namespace) -> bool:
    """Save results to JSON or CSV file."""
    output_path = Path(f"{args.output_file}.{args.output}")
    
    try:
        if args.output == 'json':
            data = []
            for analyzer in analyzers:
                file_data = {
                    "file": str(analyzer.file_path),
                    "error": analyzer.error,
                    "encoding": analyzer.results.get("encoding", "unknown"),
                    **{k: v for k, v in analyzer.results.items() if k != 'encoding'}
                }
                data.append(file_data)
            
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        else:  # CSV
            with output_path.open('w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                headers = ["File", "Lines", "Words", "Chars", "Bytes", "Longest Line", "Encoding", "Error"]
                if args.words:
                    headers.extend([f"Word_{i+1}" for i in range(args.top_words)])
                    headers.extend([f"Count_{i+1}" for i in range(args.top_words)])
                
                writer.writerow(headers)
                
                for analyzer in analyzers:
                    row = [
                        str(analyzer.file_path),
                        analyzer.results["lines"],
                        analyzer.results["words"],
                        analyzer.results["chars"],
                        analyzer.results["bytes"],
                        analyzer.results["longest_line"],
                        analyzer.results.get("encoding", "unknown"),
                        analyzer.error or ""
                    ]
                    
                    if args.words:
                        if analyzer.results["word_freq"]:
                            for word, count in analyzer.results["word_freq"]:
                                row.extend([word, count])
                            # Pad with empty values if fewer than top_words
                            row.extend([''] * 2 * (args.top_words - len(analyzer.results["word_freq"])))
                        else:
                            row.extend([''] * 2 * args.top_words)
                    
                    writer.writerow(row)
        
        print(colorize(f"Results saved to {output_path}", "green"))
        return True
    
    except IOError as e:
        print(colorize(f"Error saving to {output_path}: {e}", "red"))
        return False

def main() -> None:
    """Main function to process files and display a single summary table."""
    parser = argparse.ArgumentParser(
        description="Advanced wc+ with analytics and threading.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("files", nargs="+", help="Files or patterns to analyze")
    parser.add_argument("--progress", action="store_true", help="Show progress bars")
    parser.add_argument("--words", action="store_true", help="Include top word frequencies in summary")
    parser.add_argument("--top-words", type=int, default=10, help="Top N words to display")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1), help="Threads to use")
    parser.add_argument("--encoding", help="Force specific encoding (auto-detected by default)")
    parser.add_argument("--recursive", action="store_true", help="Recursive file search")
    parser.add_argument("--output", choices=['json', 'csv'], help="Save results as JSON or CSV")
    parser.add_argument("--output-file", default="wc_analysis_results", help="Output file base name")
    parser.add_argument("--sort-by", choices=['lines', 'words', 'chars', 'bytes', 'longest_line'], help="Sort summary")
    parser.add_argument("--exclude-stopwords", action="store_true", help="Exclude common stopwords")
    parser.add_argument("--stopwords-file", type=Path, help="Custom stopwords file (one per line)")
    parser.add_argument("--strict-mime", action="store_true", help="Use strict MIME type detection only")
    parser.add_argument("--max-words", type=int, default=1000000, help="Maximum unique words to track per file")
    parser.add_argument("--individual", action="store_true", help="Show individual file results")
    
    args = parser.parse_args()

    # Load stopwords if specified
    stopwords = load_stopwords(args.stopwords_file) if args.exclude_stopwords else None

    valid_files = get_files(args)
    if not valid_files:
        print(colorize("No valid or readable text files found.", "red"))
        sys.exit(1)

    print(f"Analyzing {len(valid_files)} file(s) with {args.threads} thread(s)...")

    analyzers = [
        FileAnalyzer(
            f, 
            args.progress, 
            args.encoding, 
            args.top_words, 
            args.exclude_stopwords, 
            stopwords,
            args.max_words
        )
        for f in valid_files
    ]

    errors = []
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        for analyzer in analyzers:
            try:
                file_size = analyzer.file_path.stat().st_size
                pbar = tqdm(
                    total=file_size,
                    unit='B',
                    unit_scale=True,
                    disable=not args.progress,
                    desc=f"Analyzing {analyzer.file_path.name}",
                    mininterval=0.5,
                    leave=False
                ) if file_size > 0 and args.progress else None
                futures.append(executor.submit(analyzer.analyze, pbar))
            except OSError as e:
                analyzer.error = f"OSError: {e}"
                errors.append(analyzer.error)
        
        for analyzer, future in zip(analyzers, futures):
            if not future.result():
                errors.append(analyzer.error)

    # Print errors, if any
    for error in errors:
        if error:
            print(colorize(error, "red"))

    # Show individual results if requested
    if args.individual:
        for analyzer in analyzers:
            if not analyzer.error:
                print_individual_results(analyzer, args.words)

    # Prepare summary table
    successful_analyzers = [a for a in analyzers if not a.error]
    if not successful_analyzers:
        print(colorize("No files were successfully analyzed.", "red"))
        sys.exit(1)

    # Build table
    table_headers = ["File", "Lines", "Words", "Chars", "Bytes", "Longest Line", "Encoding"]
    table = []
    for analyzer in successful_analyzers:
        row = [
            analyzer.file_path.name,
            analyzer.results["lines"],
            analyzer.results["words"],
            analyzer.results["chars"],
            analyzer.results["bytes"],
            f"{analyzer.results['longest_line']} chars",
            analyzer.results.get("encoding", "unknown")
        ]
        table.append(row)
    
    if args.words:
        table_headers.append("Top Words")
        for row, analyzer in zip(table, successful_analyzers):
            if analyzer.results["word_freq"]:
                top_words = ", ".join(f"{word}:{count}" for word, count in analyzer.results["word_freq"][:3])
                if len(analyzer.results["word_freq"]) > 3:
                    top_words += "..."
                row.append(top_words)
            else:
                row.append("N/A")

    # Sort table if requested
    if args.sort_by:
        sort_index = {
            'lines': 1,
            'words': 2,
            'chars': 3,
            'bytes': 4,
            'longest_line': 5
        }[args.sort_by]
        
        table.sort(key=lambda x: float(x[sort_index]) if isinstance(x[sort_index], (int, float)) 
                  else float(x[sort_index].replace(" chars", "")), reverse=True)

    # Print summary table
    print("\n" + colorize("=== Summary ===", "green"))
    print(tabulate(table, headers=table_headers, tablefmt="grid", numalign="right"))

    # Print totals
    totals = [sum(col) for col in zip(*[
        [row[1], row[2], row[3], row[4]] for row in table
    ])]
    print(f"\nTotal: {totals[0]} lines, {totals[1]} words, {totals[2]} chars, {totals[3]} bytes")

    # Save to file if requested
    if args.output:
        save_results(analyzers, args)

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print(colorize("\nAnalysis interrupted by user", "red"))
        sys.exit(1)
    except Exception as e:
        print(colorize(f"Unexpected error: {e}", "red"))
        sys.exit(1)
    finally:
        print(f"\nTime taken: {time.time() - start_time:.2f}s")
