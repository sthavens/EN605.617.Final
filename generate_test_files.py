#!/usr/bin/env python3
import sys
import os
import random
import string
import multiprocessing as mp

# Printable ASCII range
ASCII_START = 32
ASCII_END = 127
ASCII_CHARS = ''.join(chr(i) for i in range(ASCII_START, ASCII_END))

def parse_size(s):
    """Parse size like 4k, 10m, 2g into bytes."""
    s = s.strip().lower()
    multiplier = 1

    if s.endswith(("kb", "k")):
        multiplier = 1024
        s = s.rstrip("kb").rstrip("k")
    elif s.endswith(("mb", "m")):
        multiplier = 1024 ** 2
        s = s.rstrip("mb").rstrip("m")
    elif s.endswith(("gb", "g")):
        multiplier = 1024 ** 3
        s = s.rstrip("gb").rstrip("g")
    else:
        raise ValueError(f"Invalid size '{s}'. Use k/m/g.")

    return int(s) * multiplier


def generate_chunk(size, compressibility):
    """
    Generate an ASCII chunk with a given compressibility.
    0   = fully random
    100 = fully repetitive
    """
    if compressibility == 100:
        # One repeated character for maximum compressibility
        c = random.choice(ASCII_CHARS)
        return (c * size).encode("ascii")

    if compressibility == 0:
        # Fully random
        return ''.join(random.choice(ASCII_CHARS) for _ in range(size)).encode("ascii")

    # Mixed entropy:
    # Weighted blend of patterns and noise
    pattern_length = max(1, int((compressibility / 100) * 64))  # repeating block size
    pattern = ''.join(random.choice(ASCII_CHARS) for _ in range(pattern_length))

    res = []
    for _ in range(size):
        if random.random() * 100 < compressibility:
            # Use a character from the repeating pattern
            res.append(random.choice(pattern))
        else:
            # Random ASCII char
            res.append(random.choice(ASCII_CHARS))

    return ''.join(res).encode("ascii")


def write_file(size_bytes, filename, compressibility):
    """Worker function to generate a single file in parallel."""
    print(f"[PID {os.getpid()}] Creating {filename} ({size_bytes} bytes, compress={compressibility}%)")

    chunk_size = 1024 * 1024  # 1MB chunks
    full_chunks = size_bytes // chunk_size
    remainder = size_bytes % chunk_size

    with open(filename, "wb") as f:
        for _ in range(full_chunks):
            f.write(generate_chunk(chunk_size, compressibility))
        if remainder:
            f.write(generate_chunk(remainder, compressibility))

    print(f"[PID {os.getpid()}] Finished {filename}")


def main():
    import sys
    import multiprocessing as mp
    import random, os

    args = sys.argv[1:]

    # Default compressibility
    compressibility = 0

    # Check for optional --compressibility
    if "--compressibility" in args:
        idx = args.index("--compressibility")
        compressibility = float(args[idx + 1]) * 100  # convert 0.0–1.0 to 0–100
        # Remove the option and value from args
        args = args[:idx] + args[idx + 2:]

    # Remaining args are sizes
    sizes = args
    if not sizes:
        raise ValueError("No sizes specified!")

    jobs = []
    for s in sizes:
        size_bytes = parse_size(s)   # now only real sizes remain
        filename = f"{s}.txt"
        jobs.append((size_bytes, filename))

    print(f"\nGenerating {len(jobs)} file(s) with compressibility={compressibility}%...\n")

    # Multiprocessing: one file per process
    with mp.Pool(processes=min(len(jobs), mp.cpu_count())) as pool:
        pool.starmap(write_file, [(sz, fn, compressibility) for sz, fn in jobs])

    print("\nAll files complete.\n")


if __name__ == "__main__":
    main()
