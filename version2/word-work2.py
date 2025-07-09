#!/usr/bin/env python3
import os
import sys
import argparse
import re
import time
from threading import Lock
from collections import defaultdict
from heapq import heappush, heappop, heapify
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
            "word_freq": [],  # Heap for top 10 words
        }
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
        except (IOError, UnicodeDecodeError) as e:
            print(colorize(f"Error analyzing {self.file_path}: {e}", "red"))
            self.results["error"] = str(e)

    def _process_line(self, line):
        """
        Process a single line and update statistics.

        Args:
            line (bytes): A line of bytes from the file.
        """
        decoded_line = line.decode('utf-8', errors='replace')
        with self.lock:
            self.results["bytes"] += len(line)
            self.results["chars"] += len(decoded_line)
            self.results["lines"] += 1
            self.results["longest_line"] = max(
                self.results["longest_line"], len(decoded_line.rstrip())
            )
            # Use regex to extract words (alphanumeric, ignoring punctuation)
            words = re.findall(r'\b\w+\b', decoded_line.lower())
            self.results["words"] += len(words)
            # Update word frequency using a heap to keep top 10
            word_counts = defaultdict(int)
            for word in words:
                word_counts[word] += 1
            for word, count in word_counts.items():
                heappush(self.results["word_freq"], (count, word))
                if len(self.results["word_freq"]) > self.max_words:
                    heappop(self.results["word_freq"])

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
    print(f"  Lines: {analyzer.results['lines']}")
    print(f"  Words: {analyzer.results['words']}")
    print(f"  Chars: {analyzer.results['chars']}")
    print(f"  Bytes: {analyzer.results['bytes']}")
    print(f"  Longest Line: {analyzer.results['longest_line']} chars")

    if show_word_freq and analyzer.results["word_freq"]:
        print("\n" + colorize("--- Top 10 Words ---", "yellow"))
        # Sort heap by count (descending) and word (ascending)
        sorted_words = sorted(
            [(count, word) for count, word in analyzer.results["word_freq"]],
            key=lambda x: (-x[0], x[1])
        )
        for count, word in sorted_words[:10]:
            print(f"  {word}: {count}")

def print_summary(analyzers):
    """
    Print a summary table for all files.

    Args:
        analyzers (list): List of FileAnalyzer instances.
    """
    if not analyzers:
        return
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
                analyzer.results["longest_line"],
            ])
    if table:
        print("\n" + colorize("=== Summary ===", "green"))
        print(tabulate(table, headers=headers, tablefmt="grid"))

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
    valid_files = [f for f in args.files if os.path.isfile(f) and os.access(f, os.R_OK)]
    if not valid_files:
        print(colorize("No valid or readable files provided.", "red"))
        sys.exit(1)

    analyzers = [FileAnalyzer(f, args.progress) for f in valid_files]

    # Calculate total size for unified progress bar
    total_size = sum(os.path.getsize(f) for f in valid_files if os.path.getsize(f) > 0)
    pbar = tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        disable=not args.progress,
        desc="Analyzing files"
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
