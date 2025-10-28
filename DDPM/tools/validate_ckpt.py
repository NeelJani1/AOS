#!/usr/bin/env python3
import argparse, torch, math, sys, collections

def tensor_stats(t):
    return {
        "nan": bool(torch.isnan(t).any().item()) if isinstance(t, torch.Tensor) else False,
        "inf": bool(torch.isinf(t).any().item()) if isinstance(t, torch.Tensor) else False,
        "min": float(t.min().item()) if isinstance(t, torch.Tensor) else None,
        "max": float(t.max().item()) if isinstance(t, torch.Tensor) else None,
        "mean": float(t.mean().item()) if isinstance(t, torch.Tensor) else None,
        "std": float(t.std().item()) if isinstance(t, torch.Tensor) else None
    }

def inspect_state_dict(sd, huge_threshold=1e3, top_k=10):
    corrupted = False
    count = 0
    low_var = []
    extremes = []
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        count += 1
        s = tensor_stats(v)
        if s["nan"] or s["inf"] or (s["mean"] is not None and (math.isnan(s["mean"]) or math.isinf(s["mean"]))):
            print(f"[CORRUPT] {k}: NaN/Inf detected -> {s}")
            corrupted = True
        if s["std"] is not None and s["std"] < 1e-12:
            if len(low_var) < top_k:
                low_var.append((k, s))
        if (s["mean"] is not None and abs(s["mean"]) > huge_threshold) or (s["max"] is not None and abs(s["max"]) > huge_threshold):
            extremes.append((k, s))
    return corrupted, count, low_var, extremes

def find_state_dicts(obj):
    """
    Yield (path, candidate) for dict-like objects that look like state_dicts,
    scanning nested lists/tuples/dicts.
    """
    stack = [("", obj)]
    while stack:
        path, cur = stack.pop()
        if isinstance(cur, dict):
            # Heuristic: if values are tensors, probably a state_dict
            some_tensor = any(isinstance(v, torch.Tensor) for v in cur.values())
            if some_tensor:
                yield path or "<root>", cur
            else:
                # scan nested dicts for candidates
                for k, v in cur.items():
                    stack.append((f"{path}.{k}" if path else k, v))
        elif isinstance(cur, (list, tuple)):
            for i, v in enumerate(cur):
                stack.append((f"{path}[{i}]", v))
        # else ignore other types
    return

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--huge-threshold", type=float, default=1e3)
    args = p.parse_args()

    raw = torch.load(args.ckpt, map_location="cpu")
    print("Loaded checkpoint type:", type(raw).__name__)

    # If the top-level is a dict with common keys, report them
    if isinstance(raw, dict):
        print("Top-level keys:", list(raw.keys()))
    else:
        print("Top-level repr:", repr(raw)[:200])

    found = list(find_state_dicts(raw))
    if not found:
        print("No candidate state_dict-like objects found. If this file contains tensors directly,")
        print("it may be saved differently. Displaying top-level items (if any)...")
        if isinstance(raw, dict):
            for k,v in raw.items():
                print(f" - {k}: {type(v).__name__}")
        else:
            print("Top-level object is", type(raw).__name__)
        sys.exit(1)

    print(f"Found {len(found)} candidate state_dict(s). Validating each...")

    overall_corrupt = False
    for path, sd in found:
        print("\n=== Candidate:", path, " (entries:", len(sd), ") ===")
        corrupted, count, low_var, extremes = inspect_state_dict(sd, huge_threshold=args.huge_threshold)
        print("Tensors inspected:", count)
        if low_var:
            print("-- low-variance tensors (sample):")
            for k,s in low_var[:10]:
                print("  ", k, s)
        if extremes:
            print("-- tensors with huge values (sample):")
            for k,s in extremes[:10]:
                print("  ", k, s)
        if corrupted:
            print("-> Candidate appears CORRUPT (NaN/Inf).")
            overall_corrupt = True
        else:
            print("-> Candidate looks OK (no NaN/Inf detected).")

    if overall_corrupt:
        print("\n>>> At least one candidate state_dict contained NaN/Inf. Do NOT resume from those.")
        sys.exit(2)
    else:
        print("\n>>> No NaN/Inf found in any candidate state_dict.")
        sys.exit(0)

if __name__ == '__main__':
    main()
