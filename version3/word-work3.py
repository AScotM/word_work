#!/usr/bin/env python3
import os
import sys
import argparse
import re
import time
import heapq
from threading import Lock
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from tabulate import tabulate
from colorama import init, Fore, Style

# Initialize colorama for cross-platform color support
init()

# ANSI colors for output
COLORS = {
    "red": Fore.RED,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "reset": Style.RESET_ALL,
}

def colorize(text, color):
    """Colorize text for terminal output."""
    return f"{COLORS[color]}{text}{COLORS['reset']}" if sys.stdout.isatty() else text

class FileAnalyzer:
    def __init__(self, file_path, show_progress=False):
        """
        Initialize a file analyzer.

        Args:
            file_path (str): Path to the file to analyze.
            show_progress (bool): Whether to show a progress bar.
        """
        self.file_path = file_path
        self.show_progress = show_progress
        self.lock = Lock()
        self.results = {
            "lines": 0,
            "words": 0,
            "chars": 0,
            "bytes": 0,
            "longest_line": 0,
            "word_freq": [],  # Will store top words at the end
        }
        self.word_counts = defaultdict(int)  # Track all word counts first
        self.max_words = 10  # Limit for word frequency tracking

    def analyze(self, pbar=None):
        """
        Analyze the file and update statistics.

        Args:
            pbar (tqdm.tqdm, optional): Progress bar for updating progress.
        """
        try:
            file_size = os.path.getsize(self.file_path)
            with open(self.file_path, 'rb') as f:
                for line in f:
                    self._process_line(line)
                    if pbar:
                        pbar.update(len(line))
            # Finalize word frequency analysis after processing all lines
            self._finalize_word_freq()
        except (IOError, UnicodeDecodeError) as e:
            print(colorize(f"Error analyzing {self.file_path}: {e}", "red"))
            self.results["error"] = str(e)
        except Exception as e:
            print(colorize(f"Unexpected error analyzing {self.file_path}: {e}", "red"))
            self.results["error"] = str(e)

    def _process_line(self, line):
        """
        Process a single line and update statistics.

        Args:
            line (bytes): A line of bytes from the file.
        """
        decoded_line = line.decode('utf-8', errors='replace')
        with self.lock:
            # Update basic statistics
            line_bytes = len(line)
            line_chars = len(decoded_line)
            line_length = len(decoded_line.rstrip())
            
            self.results["bytes"] += line_bytes
            self.results["chars"] += line_chars
            self.results["lines"] += 1
            self.results["longest_line"] = max(
                self.results["longest_line"], line_length
            )
            
            # Extract words and update counts
            words = re.findall(r'\w+', decoded_line.lower())
            self.results["words"] += len(words)
            
            # Update word frequency counts
            for word in words:
                self.word_counts[word] += 1

    def _finalize_word_freq(self):
        """Process word counts to get top N words using heapq.nlargest."""
        with self.lock:
            if self.word_counts:
                self.results["word_freq"] = heapq.nlargest(
                    self.max_words,
                    self.word_counts.items(),
                    key=lambda item: (item[1], item[0])
                )
            # Clear word_counts to free memory
            self.word_counts.clear()

def print_results(analyzer, show_word_freq=False):
    """
    Print formatted results for a single file.

    Args:
        analyzer (FileAnalyzer): The analyzer instance with results.
        show_word_freq (bool): Whether to show word frequency.
    """
    print(f"\nFile: {colorize(analyzer.file_path, 'green')}")
    if "error" in analyzer.results:
        print(colorize(f"Failed to analyze: {analyzer.results['error']}", "red"))
        return
    
    # Basic statistics
    stats = [
        ("Lines", analyzer.results["lines"]),
        ("Words", analyzer.results["words"]),
        ("Chars", analyzer.results["chars"]),
        ("Bytes", analyzer.results["bytes"]),
        ("Longest Line", f"{analyzer.results['longest_line']} chars"),
    ]
    
    for label, value in stats:
        print(f"  {label}: {value}")

    # Word frequency analysis
    if show_word_freq and analyzer.results["word_freq"]:
        word_count = len(analyzer.results["word_freq"])
        title = f"--- Top {min(word_count, 10)} Words ---"
        print("\n" + colorize(title, "yellow"))
        for word, count in analyzer.results["word_freq"]:
            print(f"  {word}: {count}")

def print_summary(analyzers):
    """
    Print a summary table for all files.

    Args:
        analyzers (list): List of FileAnalyzer instances.
    """
    if not analyzers:
        return
    
    # Prepare table data
    table = []
    headers = ["File", "Lines", "Words", "Chars", "Bytes", "Longest Line"]
    
    for analyzer in analyzers:
        if "error" not in analyzer.results:
            table.append([
                os.path.basename(analyzer.file_path),
                analyzer.results["lines"],
                analyzer.results["words"],
                analyzer.results["chars"],
                analyzer.results["bytes"],
                f"{analyzer.results['longest_line']} chars",
            ])
    
    if table:
        print("\n" + colorize("=== Summary ===", "green"))
        print(tabulate(table, headers=headers, tablefmt="grid"))

def is_valid_file(file_path):
    """Check if file exists and is readable, and looks like a text file."""
    if not os.path.isfile(file_path) or not os.access(file_path, os.R_OK):
        return False
    
    # Simple check for binary files (not perfect but helpful)
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:
                return False
    except IOError:
        return False
    
    return True

def main():
    """Main function to parse arguments and analyze files."""
    parser = argparse.ArgumentParser(
        description="Enhanced word count (wc) with analytics, progress bars, and threading.",
        epilog="Example: %(prog)s file1.txt file2.txt --progress --words --threads 4"
    )
    parser.add_argument("files", nargs="+", help="Files to analyze")
    parser.add_argument("--progress", action="store_true", help="Show unified progress bar")
    parser.add_argument("--words", action="store_true", help="Show top 10 word frequencies")
    parser.add_argument(
        "--threads",
        type=int,
        default=min(4, os.cpu_count() or 1),
        choices=range(1, (os.cpu_count() or 1) + 1),
        help=f"Number of threads (1 to {os.cpu_count() or 1})"
    )
    args = parser.parse_args()

    # Validate files
    valid_files = [f for f in args.files if is_valid_file(f)]
    if not valid_files:
        print(colorize("No valid or readable text files provided.", "red"))
        sys.exit(1)

    analyzers = [FileAnalyzer(f, args.progress) for f in valid_files]

    # Calculate total size for unified progress bar
    total_size = sum(os.path.getsize(f) for f in valid_files if os.path.getsize(f) > 0)
    pbar = tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        disable=not args.progress,
        desc="Analyzing files",
        mininterval=0.5  # Update at most twice per second
    ) if args.progress else None

    # Analyze files using thread pool
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        executor.map(lambda a: a.analyze(pbar), analyzers)

    if pbar:
        pbar.close()

    # Print results for each file
    for analyzer in analyzers:
        print_results(analyzer, args.words)

    # Print summary table
    print_summary(analyzers)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTime taken: {time.time() - start_time:.2f}s")
