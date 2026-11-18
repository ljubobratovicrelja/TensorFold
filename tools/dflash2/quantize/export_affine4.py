#!/usr/bin/env python3
"""Turn a fitted affine4 weight file into the raw little-endian artifact the engine's loader reads.

The fitter (`quantize/nvfp4_body.py`) decides the numbers: one 4-bit code per weight, one bf16 scale
and one bf16 bias per group of 64 input columns. This tool owns only the **bytes**, which are a
contract with `zig/src/core/affine4_host.zig` and nothing else:

  <name>.words.u32   n*k/8 uint32, k/8 words per row, 8 codes a word, the LOW nibble the LOWEST index
  <name>.scale.bf16  n * k/64 uint16 (bf16 bits), row-major [n][k/64]
  <name>.bias.bf16   n * k/64 uint16, same shape and order
  manifest.json      format/group/code_max plus per tensor n, k and the three sha256 digests

The packing rule is not a choice. `dequantize` (affine4_host.zig:15-20) reads value i as
`(words[i / 8] >> 4 * (i % 8)) & 0xf` and takes its scale and bias from `scales[i / 64]` over a
row-major flattening, so a row's first eight codes fill word 0 from the bottom up and every 32-bit
word belongs to exactly one row; the parameters are indexed `[row][col / 64]`. `quantizeMlx` (lines
41-66) writes the mirror image of that read, `words[i / 8] |= q << 4 * (i % 8)` (line 62), and rounds
scale and bias through `bf16of` = `bf16_round.bits` (line 57-58, ties to even) -- so the parameters
here are rounded, never truncated.

`verify` re-reads the artifact from disk and proves it: file sizes, sha256 against the manifest, the
codes recovered byte by byte equal the fit's codes, and the dequantised values equal the fit's.

Nothing here touches a model, a GPU or a build: it reads an .npz and writes files.

  export --fit fitted.npz --out artifacts/ [--dedupe | --no-dedupe]
  verify --fit fitted.npz --out artifacts/
  synthetic --out stand_in.npz [--checkpoint SNAPSHOT_DIR | --safetensors FILE]   (test support only)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

GROUP = 64
CODE_MAX = 15
FORMAT = "affine4"

#: The order a manifest lists tensors in: `fc`, then the five layers' seven linears each.
LINEAR_ORDER = ("q", "k", "v", "o", "gate", "up", "down")


def tensor_order(name: str) -> tuple:
    """Sort key for a tensor name: fc, then L0_q ... L4_down, then anything unrecognised, alphabetically."""
    if name == "fc":
        return (0, 0, 0, name)
    if "_" in name and name[0] == "L" and name[1:].split("_")[0].isdigit():
        layer, _, rest = name[1:].partition("_")
        rank = LINEAR_ORDER.index(rest) if rest in LINEAR_ORDER else len(LINEAR_ORDER)
        return (1, int(layer), rank, name)
    return (2, 0, 0, name)


def load_nvfp4():
    """`nvfp4.py` from this directory: its packer and its bf16 rounding are the ones this artifact needs."""
    path = Path(__file__).resolve().with_name("nvfp4.py")
    spec = importlib.util.spec_from_file_location("nvfp4_for_export", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"{path}: cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NV = load_nvfp4()


def bf16_value(bits: np.ndarray) -> np.ndarray:
    """The f32 a bf16 bit pattern means: `f32of` (affine4_host.zig:7-9), the top half of the word."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def as_bf16_bits(x: np.ndarray, what: str) -> np.ndarray:
    """A fit parameter array as bf16 bits, whichever way the fitter recorded it.

    `nvfp4_body.py` stores the parameters as the bits the engine writes (`bf16of(s, b)`), which is what
    an artifact carries verbatim; a float array is accepted too and rounded the way the engine rounds
    it, so a fitter that keeps f32 cannot silently ship a different number than the one it scored.
    """
    a = np.asarray(x)
    if a.dtype == np.uint16:
        return a
    if a.dtype.kind == "f":
        return NV.affine4_bf16_bits(a.astype(np.float32))
    raise SystemExit(f"{what}: {a.dtype} is neither uint16 bf16 bits nor a float array")


def write_binary(path: Path, payload: bytes, dedupe: bool, stats: dict) -> str:
    """Write `payload` unless the file already holds exactly these bytes; returns its sha256.

    `--dedupe` (the default) is what makes a re-export free: a tensor whose file already hashes to the
    same bytes is left alone rather than rewritten, so an interrupted run finishes without touching
    250 MB of words that are already correct.
    """
    digest = hashlib.sha256(payload).hexdigest()
    if dedupe and path.is_file() and path.stat().st_size == len(payload):
        if hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            stats["skipped"] += 1
            return digest
    path.write_bytes(payload)
    stats["written"] += 1
    return digest


# ---------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------

def fit_tensor_names(npz) -> list[str]:
    """Every tensor the file carries codes for, in manifest order."""
    names = sorted({key[: -len(".code")] for key in npz.files if key.endswith(".code")},
                   key=tensor_order)
    if not names:
        raise SystemExit("no `<name>.code` arrays in the fit file")
    for name in names:
        for part in ("code", "scale", "bias"):
            if f"{name}.{part}" not in npz.files:
                raise SystemExit(f"{name}: has .code but no .{part}")
    return names


def tensor_arrays(npz, name: str) -> dict:
    """One tensor's fit as `codes` uint8 [n][k] and bf16 bits `scales`/`biases` [n][k/64]."""
    codes = np.asarray(npz[f"{name}.code"])
    if codes.dtype != np.uint8:
        raise SystemExit(f"{name}.code: {codes.dtype}, expected uint8")
    if codes.ndim != 2:
        raise SystemExit(f"{name}.code: {codes.shape}, expected (n, k)")
    n, k = codes.shape
    if k % GROUP:
        raise SystemExit(f"{name}: k={k} is not a multiple of the group {GROUP}")
    if int(codes.max(initial=0)) > CODE_MAX:
        raise SystemExit(f"{name}.code: {int(codes.max())} > {CODE_MAX}")
    nb = k // GROUP
    params = {}
    for part, key in (("scales", "scale"), ("biases", "bias")):
        bits = as_bf16_bits(npz[f"{name}.{key}"], f"{name}.{key}")
        if bits.size != n * nb:
            raise SystemExit(f"{name}.{key}: {bits.size} values, expected {n * nb} ({n} x {k // GROUP})")
        params[part] = bits.reshape(n, nb)
    return {"codes": codes, "scales": params["scales"], "biases": params["biases"], "n": n, "k": k}


_ATTN = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}
_MLP = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}


def engine_name(short: str) -> str:
    """`L0_q` -> `layers.0.self_attn.q_proj.weight`: the key the checkpoint and the loader both use.

    The fit keys are compact; the engine asks the manifest for the checkpoint's own tensor name, so the
    manifest and the file names must speak checkpoint, not fit.
    """
    if short == "fc":
        return "fc.weight"
    if not short.startswith("L") or "_" not in short:
        raise SystemExit(f"{short}: not a fit key (fc or L<layer>_<proj>)")
    layer, key = short[1:].split("_", 1)
    if not layer.isdigit():
        raise SystemExit(f"{short}: layer is not a number")
    if key in _ATTN:
        return f"layers.{layer}.self_attn.{_ATTN[key]}.weight"
    if key in _MLP:
        return f"layers.{layer}.mlp.{_MLP[key]}.weight"
    raise SystemExit(f"{short}: no checkpoint name for projection {key!r}")


def cmd_export(args) -> int:
    # the engine's loader reads <checkpoint>/affine4/<name>.* and <checkpoint>/affine4/manifest.json,
    # so --out is the checkpoint directory and the files live one level down
    fit_path, out_dir = Path(args.fit), Path(args.out) / "affine4"
    if not fit_path.is_file():
        raise SystemExit(f"{fit_path}: no such file")
    out_dir.mkdir(parents=True, exist_ok=True)
    dedupe = not args.no_dedupe
    stats = {"written": 0, "skipped": 0}
    tensors, total = [], 0

    with np.load(fit_path) as npz:
        names = fit_tensor_names(npz)
        for name in names:
            t = tensor_arrays(npz, name)
            name = engine_name(name)
            n, k = t["n"], t["k"]
            # The packer rounds nothing twice: codes go in as they were fitted, parameters as bf16 bits.
            packed = NV.affine4_pack(t["codes"], bf16_value(t["biases"]), bf16_value(t["scales"]), GROUP)
            if not np.array_equal(NV.affine4_bf16_bits(bf16_value(t["scales"])), t["scales"]):
                raise SystemExit(f"{name}: bf16 scale bits do not survive the packer's rounding")
            words = np.ascontiguousarray(packed["words"], dtype="<u4")
            scales = np.ascontiguousarray(packed["scales"], dtype="<u2")
            biases = np.ascontiguousarray(packed["biases"], dtype="<u2")
            if words.shape != (n, k // 8) or scales.shape != (n, k // GROUP):
                raise SystemExit(f"{name}: packer produced {words.shape}/{scales.shape} for {(n, k)}")
            files = {
                "words": (out_dir / f"{name}.words.u32", words.tobytes()),
                "scale": (out_dir / f"{name}.scale.bf16", scales.tobytes()),
                "bias": (out_dir / f"{name}.bias.bf16", biases.tobytes()),
            }
            digests = {key: write_binary(path, payload, dedupe, stats)
                       for key, (path, payload) in files.items()}
            size = sum(len(payload) for _, payload in files.values())
            total += size
            tensors.append({"name": name, "n": n, "k": k,
                            "words_sha256": digests["words"], "scale_sha256": digests["scale"],
                            "bias_sha256": digests["bias"]})
            print(f"  {name:12s} {n:5d} x {k:6d}  {size / 2**20:8.2f} MiB  "
                  f"{int(t['codes'].min())}..{int(t['codes'].max())} codes")

    manifest = {"format": FORMAT, "group": GROUP, "code_max": CODE_MAX, "tensors": tensors}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"{len(tensors)} tensors, {sum(t['n'] * t['k'] for t in tensors):,} weights, "
          f"{total:,} bytes ({total / 2**20:.2f} MiB) "
          f"-> {out_dir}")
    print(f"files written {stats['written']}, already correct {stats['skipped']}")
    return 0


# ---------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------

def unpack_words(raw: bytes, n: int, k: int) -> np.ndarray:
    """Codes from the artifact's bytes alone: four bytes a word, low nibble the earlier index.

    Deliberately not `nvfp4.py`'s packer run backwards -- it walks the file byte by byte, so it also
    catches a byte order or a word stride that a symmetric numpy round-trip would share the bug with.
    """
    if len(raw) != n * k // 2:
        raise SystemExit(f"words file is {len(raw)} bytes, expected {n * k // 2} for {(n, k)}")
    quads = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4)
    codes = np.empty((quads.shape[0], 8), dtype=np.uint8)
    for j in range(4):  # byte j of the word holds codes 2j (low nibble) and 2j + 1 (high nibble)
        codes[:, 2 * j] = quads[:, j] & 0x0F
        codes[:, 2 * j + 1] = quads[:, j] >> 4
    return codes.reshape(n, k)


def dequant_from_codes(codes: np.ndarray, scales_bits: np.ndarray, biases_bits: np.ndarray) -> np.ndarray:
    """The fit's own dequantisation, straight from its codes and parameter bits, with no packing involved.

    Same arithmetic as the engine (`code * scale + bias` in f32, one rounding each) but it never goes
    near a word, so a packing error cannot hide behind it.
    """
    n, k = codes.shape
    c = np.asarray(codes, dtype=np.float32)
    s = np.repeat(bf16_value(np.asarray(scales_bits, dtype=np.uint16).reshape(n, k // GROUP)), GROUP, axis=1)
    b = np.repeat(bf16_value(np.asarray(biases_bits, dtype=np.uint16).reshape(n, k // GROUP)), GROUP, axis=1)
    return (c * s + b).astype(np.float32)


def dequant_engine(words: np.ndarray, scales_bits: np.ndarray, biases_bits: np.ndarray,
                   n: int, k: int) -> np.ndarray:
    """`dequantize` (affine4_host.zig:15-20), value for value: f32 `scale * code + bias` per group."""
    codes = np.empty((n, k), dtype=np.float32)
    q = np.asarray(words, dtype=np.uint32).reshape(n, k // 8)
    for j in range(8):
        codes[:, j::8] = ((q >> np.uint32(4 * j)) & np.uint32(0xF)).astype(np.float32)
    s = bf16_value(np.asarray(scales_bits, dtype=np.uint16).reshape(n, k // GROUP))
    b = bf16_value(np.asarray(biases_bits, dtype=np.uint16).reshape(n, k // GROUP))
    return (codes * np.repeat(s, GROUP, axis=1) + np.repeat(b, GROUP, axis=1)).astype(np.float32)


def cmd_verify(args) -> int:
    fit_path, out_dir = Path(args.fit), Path(args.out) / "affine4"
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{manifest_path}: no such file")
    manifest = json.loads(manifest_path.read_text())
    failures = []
    if manifest.get("format") != FORMAT:
        failures.append(f"manifest format {manifest.get('format')!r} != {FORMAT!r}")
    if manifest.get("group") != GROUP:
        failures.append(f"manifest group {manifest.get('group')} != {GROUP}")
    if manifest.get("code_max") != CODE_MAX:
        failures.append(f"manifest code_max {manifest.get('code_max')} != {CODE_MAX}")
    listed = manifest.get("tensors", [])

    rows = []
    with np.load(fit_path) as npz:
        short_of = {engine_name(s): s for s in fit_tensor_names(npz)}
        names = list(short_of)
        names_seen = [t["name"] for t in listed]
        if names_seen != names:
            failures.append(f"manifest tensors {names_seen} != fit's {names}")
        for entry in listed:
            name = entry["name"]
            if name not in short_of:
                failures.append(f"{name}: in the manifest, not in the fit")
                continue
            t = tensor_arrays(npz, short_of[name])
            n, k = t["n"], t["k"]
            if (entry.get("n"), entry.get("k")) != (n, k):
                failures.append(f"{name}: manifest says {(entry.get('n'), entry.get('k'))}, fit says {(n, k)}")
            path = {p: out_dir / f"{name}.{p}.bf16" if p != "words" else out_dir / f"{name}.words.u32"
                    for p in ("words", "scale", "bias")}
            missing = [str(p) for p in path.values() if not p.is_file()]
            if missing:
                failures.append(f"{name}: missing {missing}")
                continue

            raw = {key: p.read_bytes() for key, p in path.items()}
            want = {"words": n * k // 2, "scale": n * k // GROUP * 2, "bias": n * k // GROUP * 2}
            size_ok = all(len(raw[key]) == want[key] for key in want)
            if not size_ok:
                failures.append(f"{name}: sizes " + ", ".join(f"{key} {len(raw[key])} != {want[key]}"
                                                              for key in want))
                rows.append((name, n, k, False, False, False, False, False, False, float("nan")))
                continue

            sha_ok = all(hashlib.sha256(raw[key]).hexdigest() == entry.get(f"{key}_sha256")
                         for key in ("words", "scale", "bias"))
            if not sha_ok:
                failures.append(f"{name}: sha256 does not match the manifest")

            words = np.frombuffer(raw["words"], dtype="<u4").reshape(n, k // 8)
            scales = np.frombuffer(raw["scale"], dtype="<u2").reshape(n, k // GROUP)
            biases = np.frombuffer(raw["bias"], dtype="<u2").reshape(n, k // GROUP)

            codes_back = unpack_words(raw["words"], n, k)
            codes_ok = np.array_equal(codes_back, t["codes"])
            if not codes_ok:
                failures.append(f"{name}: {int(np.count_nonzero(codes_back != t['codes']))} codes "
                                f"differ between the words file and the fit")

            params_ok = np.array_equal(scales, t["scales"]) and np.array_equal(biases, t["biases"])
            if not params_ok:
                failures.append(f"{name}: scale/bias bits differ from the fit's")

            got = dequant_engine(words, scales, biases, n, k)
            if f"{name}.dequant" in npz.files:
                stored = np.asarray(npz[f"{name}.dequant"])
                if stored.dtype == np.uint16:  # the fitter stored bf16 bits: compare on bits
                    ref = bf16_value(stored.reshape(n, k))
                    exact = np.array_equal(NV.affine4_bf16_bits(got), stored.reshape(n, k))
                else:
                    ref = stored.astype(np.float32).reshape(n, k)
                    exact = np.array_equal(got, ref)
            else:
                ref = dequant_from_codes(t["codes"], t["scales"], t["biases"])
                exact = np.array_equal(got, ref)
            bitwise = np.array_equal(NV.affine4_bf16_bits(got), NV.affine4_bf16_bits(ref))
            if not exact:
                failures.append(f"{name}: dequantised values are not the fit's")
            worst = float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64))))
            rows.append((name, n, k, size_ok, sha_ok, codes_ok, params_ok, exact, bitwise, worst))

    head = ("tensor", "n", "k", "size", "sha256", "codes", "params", "dequant", "bf16-bits", "max|d|")
    widths = (12, 5, 6, 4, 6, 5, 6, 7, 9, 9)
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*head))
    for row in rows:
        *fixed, worst = row
        print(fmt.format(*[str(v) if not isinstance(v, bool) else ("ok" if v else "FAIL") for v in fixed],
                         f"{worst:.3g}"))

    bad = [r[0] for r in rows if not all(r[3:9])]
    print()
    if failures:
        for line in failures:
            print(f"FAIL {line}")
    if len(rows) != len(listed):
        print(f"FAIL {len(rows)} of {len(listed)} manifest tensors were checked")
    ok = not failures and len(rows) == len(listed) and not bad
    print(f"{'PASS' if ok else 'FAIL'}: {len(rows)} tensors, "
          f"{sum(r[1] * r[2] for r in rows):,} weights, "
          f"{sum(r[1] * r[2] // 2 + r[1] * r[2] // GROUP * 4 for r in rows):,} bytes "
          f"in {out_dir}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------
# synthetic: a stand-in fit, so the byte path can be finished before the fitter writes one
# ---------------------------------------------------------------------------------------------

def safetensors_header(path: Path) -> dict:
    """The JSON header of a .safetensors file: names to dtype and shape."""
    with path.open("rb") as f:
        (length,) = np.frombuffer(f.read(8), dtype="<u8")
        return json.loads(f.read(int(length)))


def find_safetensors(args) -> Path:
    if args.safetensors:
        return Path(args.safetensors)
    root = Path(args.checkpoint).expanduser() if args.checkpoint else Path(
        "~/.cache/huggingface/hub/models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots").expanduser()
    hits = sorted(root.glob("*/model.safetensors")) or sorted(root.glob("*.safetensors"))
    if not hits:
        raise SystemExit(f"{root}: no model.safetensors")
    return hits[0]


SYNTHETIC_SUFFIX = {key: f"layers.{{i}}.self_attn.{key}_proj.weight" for key in ("q", "k", "v", "o")}
SYNTHETIC_SUFFIX |= {key: f"layers.{{i}}.mlp.{key}_proj.weight" for key in ("gate", "up", "down")}


def cmd_synthetic(args) -> int:
    """A stand-in `.npz` with the real checkpoint's shapes and a legal affine4 fit of random weights.

    Test support, not part of the artifact contract: it exists so the byte path can be finished and
    proved before `nvfp4_body.py` writes a fit, and so a shape change in the checkpoint is caught here
    (`k` not a multiple of 64, a layer count that is not five) rather than at load time.
    """
    header = safetensors_header(find_safetensors(args))
    rng = np.random.default_rng(args.seed)
    n_layers = len([k for k in header if k.endswith("self_attn.q_proj.weight")])
    plan = [("fc", "fc.weight", None)]
    for layer in range(n_layers):
        plan += [(key, suffix, layer) for key, suffix in SYNTHETIC_SUFFIX.items()]
    store, total = {}, 0
    for key, suffix, layer in plan:
        name = f"L{layer}_{key}" if layer is not None else key
        shape = header[suffix.format(i=layer)]["shape"]
        n, k = int(shape[0]), int(shape[1])
        # bf16-valued weights, like the checkpoint: an f32 already on the bf16 grid
        bits = NV.affine4_bf16_bits(rng.standard_normal((n, k), dtype=np.float32))
        w = (bits.astype(np.uint32) << np.uint32(16)).view(np.float32) * np.float32(args.spread)
        codes, bias, scale = NV.affine4_rtn_params(w, GROUP)
        nb = k // GROUP
        store[f"{name}.code"] = codes
        store[f"{name}.scale"] = NV.affine4_bf16_bits(scale).reshape(n, nb)
        store[f"{name}.bias"] = NV.affine4_bf16_bits(bias).reshape(n, nb)
        if args.with_dequant:
            store[f"{name}.dequant"] = dequant_engine(
                NV.affine4_words(codes), store[f"{name}.scale"], store[f"{name}.bias"], n, k)
        total += n * k
        print(f"  {name:12s} {n:5d} x {k:6d}")
    np.savez(args.out, **store)
    print(f"synthetic fit: {len(store) // (3 + args.with_dequant)} tensors, {total:,} weights -> {args.out}")
    return 0


# ---------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export", help="write the raw artifact for every tensor in the fit file")
    p.add_argument("--fit", required=True, help=".npz from quantize/nvfp4_body.py")
    p.add_argument("--out", required=True, help="artifact directory")
    p.add_argument("--dedupe", dest="no_dedupe", action="store_false",
                   help="leave a file alone when it already holds the same bytes (default)")
    p.add_argument("--no-dedupe", dest="no_dedupe", action="store_true",
                   help="rewrite every file even when its bytes are unchanged")
    p.set_defaults(no_dedupe=False, func=cmd_export)

    p = sub.add_parser("verify", help="prove the artifact dequantises to the fit's weights")
    p.add_argument("--fit", required=True, help="the .npz the artifact was exported from")
    p.add_argument("--out", required=True, help="artifact directory")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("synthetic", help="write a stand-in fit at the checkpoint's real shapes")
    p.add_argument("--out", required=True, help=".npz to write")
    p.add_argument("--safetensors", help="a .safetensors file to read shapes from")
    p.add_argument("--checkpoint", help="a snapshot directory holding model.safetensors")
    p.add_argument("--seed", type=int, default=0, help="random seed for the stand-in weights")
    p.add_argument("--spread", type=float, default=1.0, help="weight scale, to reach other code ranges")
    p.add_argument("--with-dequant", action="store_true", help="also store the f32 dequantised weights")
    p.set_defaults(func=cmd_synthetic)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
