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
import mimetypes
import json
import csv
import glob

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
    """Colorize text for terminal output.

    Args:
        text (str): Text to colorize.
        color (str): Color key from COLORS dictionary.

    Returns:
        str: Colorized text if output is a terminal, else plain text.
    """
    return f"{COLORS[color]}{text}{COLORS['reset']}" if sys.stdout.isatty() else text

class FileAnalyzer:
    def __init__(self, file_path, show_progress=False, encoding='utf-8', top_words=10):
        """
        Initialize a file analyzer.

        Args:
            file_path (str): Path to the file to analyze.
            show_progress (bool): Whether to show a progress bar.
            encoding (str): File encoding to use for reading.
            top_words (int): Number of top words to track for frequency analysis.
        """
        self.file_path = file_path
        self.show_progress = show_progress
        self.encoding = encoding
        self.top_words = max(1, top_words)  # Ensure at least 1
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
        self.error = None

    def analyze(self, pbar=None):
        """
        Analyze the file and update statistics.

        Args:
            pbar (tqdm.tqdm, optional): Progress bar for updating progress.

        Returns:
            bool: True if analysis succeeded, False if an error occurred.
        """
        try:
            file_size = os.path.getsize(self.file_path)
            if file_size == 0:
                return True  # Empty file, no processing needed
            with open(self.file_path, 'rb') as f:
                for line in f:
                    self._process_line(line)
                    if pbar:
                        with self.lock:  # Ensure thread-safe progress bar updates
                            pbar.update(len(line))
            # Finalize word frequency analysis after processing all lines
            self._finalize_word_freq()
            return True
        except (IOError, UnicodeDecodeError) as e:
            self.error = f"Error analyzing {self.file_path}: {e}"
            return False
        except Exception as e:
            self.error = f"Unexpected error analyzing {self.file_path}: {e}"
            return False
        finally:
            if pbar:
                pbar.close()

    def _process_line(self, line):
        """
        Process a single line and update statistics.

        Args:
            line (bytes): A line of bytes from the file.
        """
        try:
            decoded_line = line.decode(self.encoding, errors='replace')
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
                
                # Extract words and update counts (improved word regex)
                words = re.findall(r'\b[\w\'-]+(?:-[\w\'-]+)*\b', decoded_line.lower())
                self.results["words"] += len(words)
                
                # Update word frequency counts
                for word in words:
                    self.word_counts[word] += 1
        except Exception as e:
            self.error = f"Error processing line in {self.file_path}: {e}"

    def _finalize_word_freq(self):
        """Process word counts to get top N words using heapq.nlargest."""
        with self.lock:
            if self.word_counts:
                self.results["word_freq"] = heapq.nlargest(
                    self.top_words,
                    self.word_counts.items(),
                    key=lambda item: (item[1], item[0])
                )
            # Clear word_counts to free memory
            self.word_counts.clear()

def print_results(analyzer, show_word_freq=False):
    """
    Print formatted results for a single file.

 interference
    Args:
        analyzer (FileAnalyzer): The analyzer instance with results.
        show_word_freq (bool): Whether to show word frequency.
    """
    print(f"\nFile: {colorize(analyzer.file_path, 'green')}")
    if analyzer.error:
        print(colorize(analyzer.error, "red"))
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
        title = f"--- Top {min(word_count, analyzer.top_words)} Words ---"
        print("\n" + colorize(title, "yellow"))
        for word, count in analyzer.results["word_freq"]:
            print(f"  {word}: {count}")

def save_results_json(analyzers, output_file):
    """
    Save results to a JSON file.

    Args:
        analyzers (list): List of FileAnalyzer instances.
        output_file (str): Path to output JSON file.
    """
    results = []
    for analyzer in analyzers:
        result = {
            "file": analyzer.file_path,
            "error": analyzer.error,
            **analyzer.results
        }
        results.append(result)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(colorize(f"Results saved to {output_file}", "green"))
    except IOError as e:
        print(colorize(f"Error saving JSON to {output_file}: {e}", "red"))

def save_results_csv(analyzers, output_file):
    """
    Save results to a CSV file.

    Args:
        analyzers (list): List of FileAnalyzer instances.
        output_file (str): Path to output CSV file.
    """
    headers = ["File", "Lines", "Words", "Chars", "Bytes", "Longest Line", "Error"]
    rows = []
    for analyzer in analyzers:
        rows.append([
            analyzer.file_path,
            analyzer.results["lines"],
            analyzer.results["words"],
            analyzer.results["chars"],
            analyzer.results["bytes"],
            f"{analyzer.results['longest_line']} chars",
            analyzer.error or ""
        ])
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(colorize(f"Results saved to {output_file}", "green"))
    except IOError as e:
        print(colorize(f"Error saving CSV to {output_file}: {e}", "red"))

def print_summary(analyzers, sort_by=None):
    """
    Print a summary table for all files.

    Args:
        analyzers (list): List of FileAnalyzer instances.
        sort_by (str, optional): Field to sort by ('lines', 'words', 'chars', 'bytes', 'longest_line').
    """
    if not analyzers:
        return
    
    # Prepare table data
    table = []
    headers = ["File", "Lines", "Words", "Chars", "Bytes", "Longest Line"]
    
    for analyzer in analyzers:
        if not analyzer.error:
            table.append([
                os.path.basename(analyzer.file_path),
                analyzer.results["lines"],
                analyzer.results["words"],
                analyzer.results["chars"],
                analyzer.results["bytes"],
                f"{analyzer.results['longest_line']} chars",
            ])
    
    # Sort table if requested
    if sort_by and table:
        sort_index = {
            'lines': 1,
            'words': 2,
            'chars': 3,
            'bytes': 4,
            'longest_line': 5
        }.get(sort_by, 0)
        if sort_index:
            table.sort(key=lambda x: float(x[sort_index].replace(' chars', '') if sort_index == 5 else x[sort_index]), reverse=True)
    
    if table:
        print("\n" + colorize("=== Summary ===", "green"))
        print(tabulate(table, headers=headers, tablefmt="grid"))

def is_valid_file(file_path):
    """Check if file exists, is readable, and is likely a text file."""
    if not os.path.isfile(file_path) or not os.access(file_path, os.R_OK):
        return False
    
    # Use mimetypes to check for text files
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type and mime_type.startswith('text')

def get_files(args):
    """
    Get list of valid files to analyze, including recursive scanning if enabled.

    Args:
        args: Parsed command-line arguments.

    Returns:
        list: List of valid file paths.
    """
    valid_files = set()
    for pattern in args.files:
        if args.recursive:
            # Expand wildcards and recurse directories
            for file_path in glob.glob(pattern, recursive=True):
                if is_valid_file(file_path):
                    valid_files.add(file_path)
        elif is_valid_file(pattern):
            valid_files.add(pattern)
    
    return sorted(valid_files)

def main():
    """Main function to parse arguments and analyze files."""
    parser = argparse.ArgumentParser(
        description="Enhanced word count (wc) with analytics, progress bars, and threading.",
        epilog="""
        Examples:
            %(prog)s file1.txt file2.txt --progress --words --threads 4
            %(prog)s *.txt --recursive --sort-by words --output json --output-file results
            %(prog)s file1.txt --encoding latin-1 --words --top-words 5
        """
    )
    parser.add_argument("files", nargs="+", help="Files or patterns to analyze (supports wildcards with --recursive)")
    parser.add_argument("--progress", action="store_true", help="Show per-file progress bars")
    parser.add_argument("--words", action="store_true", help="Show word frequencies")
    parser.add_argument("--top-words", type=int, default=10, help="Number of top words to show (default: 10)")
    parser.add_argument(
        "--threads",
        type=int,
        default=min(4, os.cpu_count() or 1),
        choices=range(1, (os.cpu_count() or 1) + 1),
        help=f"Number of threads (1 to {os.cpu_count() or 1}, default: 4)"
    )
    parser.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directories for files")
    parser.add_argument("--output", choices=['json', 'csv'], help="Output results in specified format")
    parser.add_argument("--output-file", default="results", help="Base name for output file (default: results)")
    parser.add_argument("--sort-by", choices=['lines', 'words', 'chars', 'bytes', 'longest_line'],
                        help="Sort summary table by specified field")
    args = parser.parse_args()

    # Get valid files
    valid_files = get_files(args)
    if not valid_files:
        print(colorize("No valid or readable text files provided.", "red"))
        sys.exit(1)

    # Decide whether to use threading based on file count and size
    total_size = sum(os.path.getsize(f) for f in valid_files)
    use_threading = len(valid_files) > 1 or total_size > 1024 * 1024  # Use threading for >1MB or multiple files

    analyzers = [FileAnalyzer(f, args.progress, args.encoding, args.top_words) for f in valid_files]
    errors = []

    if use_threading:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            if args.progress:
                for analyzer in analyzers:
                    file_size = os.path.getsize(analyzer.file_path)
                    pbar = tqdm(
                        total=file_size,
                        unit='B',
                        unit_scale=True,
                        disable=not args.progress,
                        desc=f"Analyzing {os.path.basename(analyzer.file_path)}",
                        mininterval=0.5
                    ) if file_size > 0 else None
                    if not analyzer.analyze(pbar):
                        errors.append(analyzer.error)
            else:
                for analyzer in analyzers:
                    if not analyzer.analyze():
                        errors.append(analyzer.error)
    else:
        for analyzer in analyzers:
            pbar = tqdm(
                total=os.path.getsize(analyzer.file_path),
                unit='B',
                unit_scale=True,
                disable=not args.progress,
                desc=f"Analyzing {os.path.basename(analyzer.file_path)}",
                mininterval=0.5
            ) if args.progress and os.path.getsize(analyzer.file_path) > 0 else None
            if not analyzer.analyze(pbar):
                errors.append(analyzer.error)

    # Print errors in a controlled manner
    for error in errors:
        if error:
            print(colorize(error, "red"))

    # Print results for each file
    for analyzer in analyzers:
        print_results(analyzer, args.words)

    # Print summary table
    print_summary(analyzers, args.sort_by)

    # Save results to file if requested
    if args.output:
        output_file = f"{args.output_file}.{args.output}"
        if args.output == 'json':
            save_results_json(analyzers, output_file)
        elif args.output == 'csv':
            save_results_csv(analyzers, output_file)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTime taken: {time.time() - start_time:.2f}s")
