#!/usr/bin/env python3
import os
import sys
import argparse
from threading import Thread, Lock
from collections import defaultdict
import time
from tqdm import tqdm  # Progress bar (install with: pip install tqdm)

# ANSI colors for output
COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "reset": "\033[0m",
}

def colorize(text, color):
    """Colorize text if output is a terminal."""
    return f"{COLORS[color]}{text}{COLORS['reset']}" if sys.stdout.isatty() else text

class FileAnalyzer:
    def __init__(self, file_path, show_progress=False):
        self.file_path = file_path
        self.show_progress = show_progress
        self.lock = Lock()
        self.results = {
            "lines": 0,
            "words": 0,
            "chars": 0,
            "bytes": 0,
            "longest_line": 0,
            "word_freq": defaultdict(int),
        }

    def analyze(self):
        """Analyze file with optional progress bar."""
        try:
            file_size = os.path.getsize(self.file_path)
            with open(self.file_path, 'rb') as f:
                with tqdm(
                    total=file_size,
                    unit='B',
                    unit_scale=True,
                    disable=not self.show_progress,
                    desc=f"Reading {os.path.basename(self.file_path)}"
                ) as pbar:
                    for line in f:
                        self._process_line(line)
                        pbar.update(len(line))
        except Exception as e:
            print(colorize(f"Error analyzing {self.file_path}: {e}", "red"))

    def _process_line(self, line):
        """Process a single line and update stats."""
        decoded_line = line.decode('utf-8', errors='replace')
        with self.lock:
            self.results["bytes"] += len(line)
            self.results["chars"] += len(decoded_line)
            self.results["lines"] += 1
            self.results["longest_line"] = max(
                self.results["longest_line"], len(decoded_line.rstrip())
            )
            words = decoded_line.split()
            self.results["words"] += len(words)
            for word in words:
                self.results["word_freq"][word.lower()] += 1

def print_results(results, show_word_freq=False):
    """Print formatted results."""
    print("\n" + colorize("=== File Analysis Report ===", "green"))
    print(f"  Lines: {results['lines']}")
    print(f"  Words: {results['words']}")
    print(f"  Chars: {results['chars']}")
    print(f"  Bytes: {results['bytes']}")
    print(f"  Longest Line: {results['longest_line']} chars")

    if show_word_freq:
        print("\n" + colorize("--- Top 10 Words ---", "yellow"))
        sorted_words = sorted(
            results["word_freq"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for word, count in sorted_words:
            print(f"  {word}: {count}")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced word count (wc) with analytics and progress bars.",
        epilog="Example: %(prog)s file.txt --progress --words"
    )
    parser.add_argument("files", nargs="+", help="Files to analyze")
    parser.add_argument("--progress", action="store_true", help="Show progress bar")
    parser.add_argument("--words", action="store_true", help="Show word frequency")
    parser.add_argument("--threads", type=int, default=2, help="Threads for large files")
    args = parser.parse_args()

    analyzers = [FileAnalyzer(f, args.progress) for f in args.files]
    threads = []

    for analyzer in analyzers:
        t = Thread(target=analyzer.analyze)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for analyzer in analyzers:
        print(f"\nFile: {colorize(analyzer.file_path, 'green')}")
        print_results(analyzer.results, args.words)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTime taken: {time.time() - start_time:.2f}s")
