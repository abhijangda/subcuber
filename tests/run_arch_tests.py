"""Run existing architecture-specific test executables with live output."""

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass


TEST_DIRECTORY = Path(__file__).resolve().parent
ARCHITECTURES = {"ampere": (8, 0), "hopper": (9, 0), "blackwell": (10, 0)}


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    capability: tuple[int, int]


def supports_architecture(architecture: str, capability: tuple[int, int]) -> bool:
    required = ARCHITECTURES[architecture]
    return capability >= required if architecture == "ampere" else capability == required


def get_test_targets(architecture: str) -> list[str]:
    variable = architecture.upper() + "_TESTS"
    target = "__python_runner_test_targets"
    result = subprocess.run(
        [
            "make", "--no-print-directory", "-s", "-C", str(TEST_DIRECTORY),
            "--eval=" + target + ':;@printf "%s\\n" $(' + variable + ")",
            target,
        ],
        capture_output=True, text=True, check=True,
    )
    targets = list(dict.fromkeys(result.stdout.split()))
    if not targets:
        raise RuntimeError(f"No tests are listed in {variable}.")
    return targets


def get_gpus() -> list[GPU]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    devices = []
    for fields in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        index, uuid, name, capability = (field.strip() for field in fields)
        major, minor = capability.split(".")
        devices.append(GPU(int(index), uuid, name, (int(major), int(minor))))
    return devices


def select_gpu(architecture: str, requested: str | None) -> GPU:
    devices = get_gpus()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selectors = [requested] if requested is not None else (
        visible.split(",") if visible is not None else None
    )
    if selectors is not None:
        selected = []
        for selector in selectors:
            selector = selector.strip()
            matches = [
                device for device in devices
                if selector and (str(device.index) == selector or device.uuid.startswith(selector))
            ]
            if len(matches) != 1:
                raise RuntimeError(f"GPU selector {selector!r} does not identify one GPU.")
            selected.append(matches[0])
        devices = selected
    for device in devices:
        if supports_architecture(architecture, device.capability):
            return device
    required = ".".join(map(str, ARCHITECTURES[architecture]))
    raise RuntimeError(f"No visible GPU supports {architecture} tests (required SM {required}).")


def positive_timeout(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive, finite number")
    return seconds


def main(architecture: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Run existing {architecture} test binaries with live output and a final summary.",
        epilog="Pass executable options after --, for example -- --gtest_filter='*512*'.",
    )
    parser.add_argument("--gpu", help="physical GPU index or UUID; otherwise choose a compatible visible GPU")
    parser.add_argument("--timeout", type=positive_timeout, default=120.0, help="seconds per executable (default: 120)")
    parser.add_argument("--list", action="store_true", help="list existing test binaries without running them")
    parser.add_argument("test_arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    try:
        targets = get_test_targets(architecture)
        if options.list:
            print("\n".join(targets), flush=True)
            return 0
        gpu = select_gpu(architecture, options.gpu)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        print(f"{architecture.upper()} OVERALL: FAILED (tests were not started)", flush=True)
        return 2

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu.uuid
    arguments = options.test_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    print(
        f"Running {len(targets)} {architecture} test executables on GPU {gpu.index}: "
        f"{gpu.name} (SM {gpu.capability[0]}.{gpu.capability[1]}).",
        flush=True,
    )
    failures: list[tuple[str, str]] = []
    completed = 0
    interrupted = False
    for index, target in enumerate(targets, 1):
        print(f"\n== [{index}/{len(targets)}] {target} ==", flush=True)
        try:
            result = subprocess.run(
                [str(TEST_DIRECTORY / target), *arguments],
                cwd=TEST_DIRECTORY, env=environment, timeout=options.timeout, check=False,
            )
            if result.returncode:
                reason = f"exit code {result.returncode}"
            else:
                print(f"PASS: {target}", flush=True)
                completed += 1
                continue
        except subprocess.TimeoutExpired:
            reason = f"timed out after {options.timeout:g} seconds"
        except OSError as error:
            reason = str(error)
        except KeyboardInterrupt:
            failures.append((target, "interrupted"))
            interrupted = True
            break
        failures.append((target, reason))
        print(f"FAIL: {target} ({reason})", flush=True)

    print(f"\n{architecture.upper()} SUMMARY: {completed} passed, {len(failures)} failed.", flush=True)
    for target, reason in failures:
        print(f"  {target}: {reason}", flush=True)
    if interrupted:
        print(f"  {len(targets) - completed - len(failures)} executables not run.", flush=True)
    print(f"{architecture.upper()} OVERALL: {'FAILED' if failures else 'PASSED'}", flush=True)
    return 130 if interrupted else int(bool(failures))