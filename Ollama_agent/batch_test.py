#!/usr/bin/env python
# batch_test.py — run the full test suite and print a structured summary.
#
# Usage
#     python batch_test.py            # run all tests
#     python batch_test.py -v         # verbose (show each test name)
#     python batch_test.py -k chat    # filter to tests matching 'chat'
#     python batch_test.py --no-header  # skip the banner
#
# Internally this calls pytest programmatically so you don't need to remember
# any pytest flags for the common case.

from __future__ import annotations

import sys
import argparse
import textwrap
import time

import pytest

# -- TEST SUITES --

# Each entry is (label, path, description).
# Add new test files here — batch_test will pick them up automatically.
SUITES: list[tuple[str, str, str]] = [
    (
        "chat_types",
        "tests/test_chat_types.py",
        "TypedDict shape and ThinkingMode literal values",
    ),
    (
        "config",
        "tests/test_config.py",
        "Constant sanity checks — ranges, defaults, registry keys",
    ),
    (
        "models",
        "tests/test_models.py",
        "Thinking-mode detection and options builder validation",
    ),
    (
        "chat",
        "tests/test_chat.py",
        "History trimming, payload building, tag parsing, chunk handlers",
    ),
]

# -- HELPERS --

PASS = "\033[32m PASS \033[0m"
FAIL = "\033[31m FAIL \033[0m"
SKIP = "\033[33m SKIP \033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"

def _banner() -> None:
    print(
        f"\n{BOLD}{'═' * 60}{RESET}\n"
        f"  Ollama Chatbot — Test Suite\n"
        f"{BOLD}{'═' * 60}{RESET}\n"
    )

def _run_suite(
    label: str,
    path: str,
    description: str,
    extra_args: list[str],
) -> tuple[str, int, float]:
    """Run one suite and return (label, exit_code, elapsed_seconds)."""
    print(f"{BOLD}▶ {label}{RESET}  {description}")
    t0 = time.perf_counter()
    code = pytest.main([path, "--tb=short", "-q"] + extra_args)
    elapsed = time.perf_counter() - t0
    status = PASS if code == 0 else FAIL
    print(f"  {status}  {elapsed:.2f}s\n")
    return label, code, elapsed

# -- MAIN --

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Ollama chatbot test suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python batch_test.py            run everything
              python batch_test.py -v         verbose output
              python batch_test.py -k config  only run config suite
              python batch_test.py --no-header skip the banner
        """),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose pytest output")
    parser.add_argument("-k", "--filter", metavar="EXPR", help="Only run suites whose label matches EXPR")
    parser.add_argument("--no-header", action="store_true", help="Skip the banner")
    args = parser.parse_args(argv)

    if not args.no_header:
        _banner()

    extra_args: list[str] = []
    if args.verbose:
        extra_args.append("-v")

    suites = SUITES
    if args.filter:
        suites = [(l, p, d) for l, p, d in SUITES if args.filter.lower() in l.lower()]
        if not suites:
            print(f"No suites matched filter '{args.filter}'. Available: {[l for l,_,_ in SUITES]}")
            return 1

    results: list[tuple[str, int, float]] = []
    for label, path, description in suites:
        results.append(_run_suite(label, path, description, extra_args))

    # --- Summary table ---
    total_time = sum(e for _, _, e in results)
    passed = [l for l, c, _ in results if c == 0]
    failed = [l for l, c, _ in results if c != 0]

    print(f"{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  Summary{RESET}")
    print(f"{'─' * 60}")
    for label, code, elapsed in results:
        status = PASS if code == 0 else FAIL
        print(f"  {status}  {label:<20}  {elapsed:.2f}s")
    print(f"{'─' * 60}")
    print(
        f"  {BOLD}{len(passed)}/{len(results)} suites passed{RESET}  "
        f"total {total_time:.2f}s"
    )
    if failed:
        print(f"\n  {BOLD}Failed:{RESET} {', '.join(failed)}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")

    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())