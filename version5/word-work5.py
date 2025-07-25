#!/usr/bin/env python3
from pathlib import Path
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

# Optional: add common stopwords
STOPWORDS = {"the", "and", "of", "to", "in", "a", "is"}  # Add more as needed

# Initialize colorama for cross-platform color support
init()

COLORS = {
    "red": Fore.RED,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "reset": Style.RESET_ALL,
}

def colorize(text, color):
    return f"{COLORS[color]}{text}{COLORS['reset']}" if sys.stdout.isatty() else text

class FileAnalyzer:
    def __init__(self, file_path, show_progress=False, encoding='utf-8', top_words=10, exclude_stopwords=False):
        self.file_path = Path(file_path)
        self.show_progress = show_progress
        self.encoding = encoding
        self.top_words = max(1, top_words)
        self.exclude_stopwords = exclude_stopwords
        self.lock = Lock()
        self.results = {
            "lines": 0,
            "words": 0,
            "chars": 0,
            "bytes": 0,
            "longest_line": 0,
            "word_freq": [],
        }
        self.word_counts = defaultdict(int)
        self.error = None

    def analyze(self, pbar=None):
        try:
            file_size = self.file_path.stat().st_size
            if file_size == 0:
                return True
            with self.file_path.open('rb') as f:
                for line in f:
                    self._process_line(line)
                    if pbar:
                        with self.lock:
                            pbar.update(len(line))
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
        try:
            decoded_line = line.decode(self.encoding, errors='replace')
            with self.lock:
                line_bytes = len(line)
                line_chars = len(decoded_line)
                line_length = len(decoded_line.rstrip())
                self.results["bytes"] += line_bytes
                self.results["chars"] += line_chars
                self.results["lines"] += 1
                self.results["longest_line"] = max(self.results["longest_line"], line_length)
                words = re.findall(r"\b[\w'-]+(?:-[\w'-]+)*\b", decoded_line.lower())
                self.results["words"] += len(words)
                for word in words:
                    if self.exclude_stopwords and word in STOPWORDS:
                        continue
                    self.word_counts[word] += 1
        except Exception as e:
            self.error = f"Error processing line in {self.file_path}: {e}"

    def _finalize_word_freq(self):
        with self.lock:
            if self.word_counts:
                self.results["word_freq"] = heapq.nlargest(
                    self.top_words, self.word_counts.items(), key=lambda item: (item[1], item[0])
                )
            self.word_counts.clear()

def print_results(analyzer, show_word_freq=False):
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
    ]
    for label, value in stats:
        print(f"  {label}: {value}")
    if show_word_freq and analyzer.results["word_freq"]:
        word_count = len(analyzer.results["word_freq"])
        print("\n" + colorize(f"--- Top {word_count} Words ---", "yellow"))
        for word, count in analyzer.results["word_freq"]:
            print(f"  {word}: {count}")

def is_valid_file(file_path):
    path = Path(file_path)
    if not path.is_file() or not os.access(path, os.R_OK):
        return False
    try:
        with path.open('r', encoding='utf-8') as f:
            f.read(1024)
        return True
    except UnicodeDecodeError:
        return False

def get_files(args):
    valid_files = set()
    for pattern in args.files:
        matches = glob.glob(pattern, recursive=args.recursive)
        for file_path in matches:
            if is_valid_file(file_path):
                valid_files.add(Path(file_path))
    return sorted(valid_files)

def main():
    parser = argparse.ArgumentParser(description="Advanced wc+ with analytics and threading.")
    parser.add_argument("files", nargs="+", help="Files or patterns to analyze")
    parser.add_argument("--progress", action="store_true", help="Show progress bars")
    parser.add_argument("--words", action="store_true", help="Show top word frequencies")
    parser.add_argument("--top-words", type=int, default=10, help="Top N words")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1), help="Threads to use")
    parser.add_argument("--encoding", default="utf-8", help="Encoding to use")
    parser.add_argument("--recursive", action="store_true", help="Recursive file search")
    parser.add_argument("--output", choices=['json', 'csv'], help="Save results as JSON or CSV")
    parser.add_argument("--output-file", default="results", help="Output file base name")
    parser.add_argument("--sort-by", choices=['lines', 'words', 'chars', 'bytes', 'longest_line'], help="Sort summary")
    parser.add_argument("--exclude-stopwords", action="store_true", help="Exclude common stopwords")
    args = parser.parse_args()

    valid_files = get_files(args)
    if not valid_files:
        print(colorize("No valid or readable text files found.", "red"))
        sys.exit(1)

    analyzers = [
        FileAnalyzer(f, args.progress, args.encoding, args.top_words, args.exclude_stopwords)
        for f in valid_files
    ]

    errors = []
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        for analyzer in analyzers:
            file_size = analyzer.file_path.stat().st_size
            pbar = tqdm(
                total=file_size,
                unit='B',
                unit_scale=True,
                disable=not args.progress,
                desc=f"Analyzing {analyzer.file_path.name}",
                mininterval=0.5
            ) if file_size > 0 else None
            futures.append(executor.submit(analyzer.analyze, pbar))
        for analyzer, future in zip(analyzers, futures):
            if not future.result():
                errors.append(analyzer.error)

    for error in errors:
        if error:
            print(colorize(error, "red"))

    for analyzer in analyzers:
        print_results(analyzer, args.words)

    def sort_key(row):
        index = {
            'lines': 1,
            'words': 2,
            'chars': 3,
            'bytes': 4,
            'longest_line': 5
        }.get(args.sort_by)
        if index is not None:
            val = row[index]
            return float(val.replace(" chars", "")) if index == 5 else float(val)
        return 0

    table = [
        [analyzer.file_path.name, analyzer.results["lines"], analyzer.results["words"],
         analyzer.results["chars"], analyzer.results["bytes"],
         f"{analyzer.results['longest_line']} chars"]
        for analyzer in analyzers if not analyzer.error
    ]
    if args.sort_by:
        table.sort(key=sort_key, reverse=True)
    if table:
        print("\n" + colorize("=== Summary ===", "green"))
        print(tabulate(table, headers=["File", "Lines", "Words", "Chars", "Bytes", "Longest Line"], tablefmt="grid"))

    if args.output:
        output_path = Path(f"{args.output_file}.{args.output}")
        try:
            with output_path.open('w', encoding='utf-8', newline='') as f:
                if args.output == 'json':
                    json.dump([{
                        "file": str(a.file_path),
                        "error": a.error,
                        **a.results
                    } for a in analyzers], f, indent=2)
                else:
                    writer = csv.writer(f)
                    writer.writerow(["File", "Lines", "Words", "Chars", "Bytes", "Longest Line", "Error"])
                    for a in analyzers:
                        writer.writerow([
                            str(a.file_path), a.results["lines"], a.results["words"],
                            a.results["chars"], a.results["bytes"],
                            f"{a.results['longest_line']} chars", a.error or ""
                        ])
            print(colorize(f"Results saved to {output_path}", "green"))
        except IOError as e:
            print(colorize(f"Error saving to {output_path}: {e}", "red"))

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTime taken: {time.time() - start:.2f}s")
