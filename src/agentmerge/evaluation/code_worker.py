"""Constrained child-process runner for HumanEval and sanitized MBPP.

This module is invoked with ``python -I -S`` by :class:`PythonCodeScorer`.
It intentionally has no AgentMerge imports so startup stays small.
"""

from __future__ import annotations

import ast
import json
import re
import resource
import signal
import sys


_ALLOWED_IMPORTS = {
    "bisect",
    "collections",
    "copy",
    "cmath",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "hashlib",
    "heapq",
    "itertools",
    "math",
    "operator",
    "array",
    "random",
    "re",
    "statistics",
    "string",
    "sys",
    "typing",
}
_BLOCKED_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}


def _limit_resources() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    memory = 512 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    signal.alarm(3)


def _extract_code(prediction: str, entry_point: str = "") -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", prediction, re.DOTALL)
    if fenced:
        if entry_point:
            definition = re.compile(
                r"(?:^|\n)\s*(?:async\s+)?def\s+{}\s*\(".format(
                    re.escape(entry_point)
                )
            )
            matching = [item for item in fenced if definition.search(item)]
            if matching:
                return _trim_script_suffix(matching[-1].strip())
        return _trim_script_suffix(fenced[-1].strip())
    stripped = prediction.strip()
    try:
        ast.parse(stripped)
        return _trim_script_suffix(stripped)
    except SyntaxError:
        pass
    lines = stripped.splitlines()
    for index, line in enumerate(lines):
        if line.lstrip().startswith(("def ", "class ", "from ", "import ", "@")):
            return _trim_script_suffix("\n".join(lines[index:]).strip())
    return _trim_script_suffix(stripped)


def _trim_script_suffix(code: str) -> str:
    """Drop common model-added demo runners without altering function bodies."""

    lines = code.splitlines()
    for index, line in enumerate(lines):
        if (
            line.startswith('if __name__ == "__main__"')
            or line.startswith("if __name__ == '__main__'")
            or line.startswith("# Example usage")
            or line.startswith("# Test examples")
        ):
            return "\n".join(lines[:index]).rstrip()
    return code.strip()


def _audit(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(name.split(".", 1)[0] not in _ALLOWED_IMPORTS for name in names):
                raise ValueError("Import is not permitted")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not permitted")
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
            and node.attr not in {"getsizeof", "maxsize"}
        ):
            raise ValueError("This sys attribute is not permitted")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
                raise ValueError("Blocked runtime primitive")


def _candidate_sources(prediction: str, target: dict):
    prompt = str(target.get("prompt", ""))
    entry_point = str(target.get("entry_point", ""))
    code = _extract_code(prediction, entry_point)
    yield code
    if prompt and entry_point and "def {}".format(entry_point) not in code:
        fenced = re.findall(
            r"```(?:python)?\s*(.*?)```", prediction, re.DOTALL
        )
        completion = (
            fenced[-1].strip("\n")
            if fenced
            else prediction.strip("\n")
        )
        yield prompt + completion
        if completion and not completion[0].isspace():
            yield prompt + "\n".join(
                "    " + line if line.strip() else line
                for line in completion.splitlines()
            )


def _run(payload: dict) -> None:
    target = payload["target"]
    tests = str(target.get("test", ""))
    if not tests:
        tests = "\n".join(str(item) for item in target.get("tests", ()))
    setup = str(target.get("test_setup", ""))
    entry_point = str(target["entry_point"])
    last_error = None
    for source in _candidate_sources(str(payload["prediction"]), target):
        try:
            tree = ast.parse(source)
            _audit(tree)
            namespace = {"__name__": "__agentmerge_candidate__"}
            exec(compile(tree, "<candidate>", "exec"), namespace, namespace)
            if entry_point not in namespace or not callable(namespace[entry_point]):
                raise ValueError("Entry point is missing")
            if setup:
                exec(compile(setup, "<trusted-setup>", "exec"), namespace, namespace)
            if target.get("kind") == "humaneval":
                exec(compile(tests, "<trusted-tests>", "exec"), namespace, namespace)
                namespace["check"](namespace[entry_point])
            else:
                exec(compile(tests, "<trusted-tests>", "exec"), namespace, namespace)
            return
        except BaseException as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("No candidate source")


def main() -> int:
    _limit_resources()
    payload = json.loads(sys.stdin.read())
    _run(payload)
    return 0


if __name__ == "__main__":
    try:
        exit_status = main()
    except BaseException:
        exit_status = 1
    raise SystemExit(exit_status)
