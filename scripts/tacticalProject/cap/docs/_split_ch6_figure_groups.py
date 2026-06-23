from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps


LAYOUTS: Dict[str, Tuple[str, int, int]] = {
    "overview.png": ("grid", 6, 1),
    "fig02_detection_tracking_pipeline.png": ("grid", 5, 1),
    "fig02b_cooperative_chain_timeline.png": ("grid", 3, 1),
    "fig03_state_tactic_phase_timeline.png": ("grid", 8, 1),
    "fig03b_control_distance_timeline.png": ("grid", 2, 1),
    "fig04_intent_analysis_trace.png": ("grid", 6, 1),
    "fig04b_intent_target_timeline.png": ("grid", 6, 1),
    "fig05_gate_and_relay.png": ("grid", 2, 1),
    "fig05_engagement_chain.png": ("grid", 5, 1),
    "fig06_survival_distance_targeting.png": ("grid", 4, 1),
    "fig07_missile_end_reasons.png": ("grid", 1, 2),
    "fig08_scan_target_timeline.png": ("grid", 2, 1),
    "fig04_alive_and_missiles.png": ("grid", 2, 1),
    "fig09_flight_safety_envelope.png": ("grid", 3, 1),
    "comparison_effectiveness.png": ("grid", 2, 3),
    "comparison_timing.png": ("grid", 2, 3),
}

FIG_RE = re.compile(r"^图6-(\d+)\s+(.+?)\s*$")
IMG_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
SUBFIG_LINE_RE = re.compile(r"^图6-(\d+)\(([a-z])\)")
SUBFIG_PARSE_TEMPLATE = r"图6-{num}\(([a-z])\)\s*(.*?)(?=(?:；\s*图6-{num}\(|$))"


def _find_single_path(pattern: str) -> Path:
    matches = sorted(Path(".").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one match for {pattern!r}, got {len(matches)}")
    return matches[0]


def _nonwhite_ratio(image: Image.Image, axis: int) -> np.ndarray:
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.uint8)
    return (arr < 245).mean(axis=axis)


def _axis_boundaries(length: int, parts: int, ratios: np.ndarray) -> List[int]:
    if parts <= 1:
        return [0, length]
    boundaries = [0]
    span = length / parts
    window = max(10, int(span * 0.22))
    for part_idx in range(1, parts):
        center = int(round(span * part_idx))
        start = max(boundaries[-1] + 1, center - window)
        end = min(length - 2, center + window)
        if end <= start:
            cut = center
        else:
            local = ratios[start : end + 1]
            cut = start + int(np.argmin(local))
        boundaries.append(cut)
    boundaries.append(length)
    return boundaries


def _white_runs(
    ratios: np.ndarray,
    *,
    start: int = 0,
    end: int | None = None,
    threshold: float = 0.003,
    min_len: int = 4,
) -> List[Tuple[int, int]]:
    upper = len(ratios) if end is None else min(len(ratios), end)
    runs: List[Tuple[int, int]] = []
    run_start: int | None = None
    for idx in range(max(0, start), upper):
        if float(ratios[idx]) <= threshold:
            if run_start is None:
                run_start = idx
            continue
        if run_start is not None and idx - run_start >= min_len:
            runs.append((run_start, idx))
        run_start = None
    if run_start is not None and upper - run_start >= min_len:
        runs.append((run_start, upper))
    return runs


def _crop_top(
    source_name: str,
    row_idx: int,
    row_start: int,
    row_end: int,
    row_ratios: np.ndarray,
    *,
    rows: int,
    cols: int,
) -> int:
    row_h = row_end - row_start

    # 1x2 missile-end figures include a global suptitle above the two subplot titles.
    if rows == 1 and cols > 1:
        runs = _white_runs(row_ratios, start=18, end=min(row_end, 180), min_len=8)
        if runs:
            best = max(runs, key=lambda item: (item[1] - item[0], item[0]))
            if best[1] - best[0] >= 20:
                return best[1]
        return row_start

    # 2x3 comparison figures keep the upper-row x tick labels between the two rows.
    if rows > 1 and cols > 1 and row_idx > 0:
        search_end = min(row_end, row_start + max(80, int(row_h * 0.20)))
        runs = _white_runs(row_ratios, start=row_start + 8, end=search_end, min_len=6)
        if runs:
            best = max(runs, key=lambda item: (item[1] - item[0], item[0]))
            if best[1] - best[0] >= 10:
                return best[1]

    return row_start + (int(row_h * 0.02) if row_idx > 0 else 0)


def _split_image(source: Path, rows: int, cols: int) -> List[Path]:
    image = Image.open(source).convert("RGB")
    row_ratios = _nonwhite_ratio(image, axis=1)
    col_ratios = _nonwhite_ratio(image, axis=0)
    row_edges = _axis_boundaries(image.height, rows, row_ratios)
    col_edges = _axis_boundaries(image.width, cols, col_ratios)

    outputs: List[Path] = []
    letter_idx = 0
    for row_idx, (row_start, row_end) in enumerate(zip(row_edges[:-1], row_edges[1:])):
        row_h = row_end - row_start
        top = _crop_top(
            source.name,
            row_idx,
            row_start,
            row_end,
            row_ratios,
            rows=rows,
            cols=cols,
        )
        if row_idx < rows - 1:
            if rows >= 5:
                trim_ratio = 0.38
            elif rows >= 3:
                trim_ratio = 0.30
            else:
                trim_ratio = 0.18
        else:
            trim_ratio = 0.0
        trim_bottom = int(row_h * trim_ratio)
        for col_start, col_end in zip(col_edges[:-1], col_edges[1:]):
            bottom = min(image.height, row_end - trim_bottom)
            top = max(0, min(top, bottom - 1))
            left = max(0, col_start)
            right = min(image.width, col_end)
            crop = image.crop((left, top, right, bottom))
            crop = ImageOps.expand(crop, border=8, fill="white")
            letter = chr(ord("a") + letter_idx)
            out_path = source.with_name(f"{source.stem}_{letter}{source.suffix}")
            crop.save(out_path)
            outputs.append(out_path)
            letter_idx += 1
    return outputs


def _parse_subfigs(num: str, lines: Sequence[str]) -> List[Tuple[str, str]]:
    joined = " ".join(line.strip().rstrip("。").rstrip() for line in lines)
    pattern = re.compile(SUBFIG_PARSE_TEMPLATE.format(num=re.escape(num)))
    result: List[Tuple[str, str]] = []
    for letter, caption in pattern.findall(joined):
        clean = caption.strip().rstrip("；").strip()
        result.append((letter, clean))
    return result


def _rewrite_markdown(md_path: Path) -> None:
    md_path = md_path.resolve()
    lines = md_path.read_text(encoding="utf-8").splitlines()
    rewritten: List[str] = []
    subfig_counts: Dict[str, int] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        fig_match = FIG_RE.match(line)
        if fig_match:
            num = fig_match.group(1)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and SUBFIG_LINE_RE.match(lines[j]):
                i += 1
                continue
        if not fig_match or i + 2 >= len(lines):
            rewritten.append(line)
            i += 1
            continue

        num = fig_match.group(1)
        image_line = lines[i + 1]
        image_match = IMG_RE.match(image_line)
        if not image_match:
            rewritten.append(line)
            i += 1
            continue

        subfig_lines: List[str] = []
        j = i + 2
        while j < len(lines) and SUBFIG_LINE_RE.match(lines[j]):
            subfig_lines.append(lines[j])
            j += 1
        if not subfig_lines:
            rewritten.append(line)
            i += 1
            continue

        rel_path = image_match.group(2)
        source_path = (md_path.parent / rel_path).resolve()
        layout = LAYOUTS.get(source_path.name)
        if layout is None:
            rewritten.append(line)
            i += 1
            continue

        _, rows, cols = layout
        captions = _parse_subfigs(num, subfig_lines)
        expected = rows * cols
        if len(captions) != expected:
            raise RuntimeError(
                f"Caption count mismatch for 图6-{num}: expected {expected}, got {len(captions)} from {source_path.name}"
            )

        outputs = _split_image(source_path, rows=rows, cols=cols)
        subfig_counts[num] = expected
        for (letter, caption), out_path in zip(captions, outputs):
            rel_out = Path(".") / out_path.resolve().relative_to(md_path.parent)
            rel_out_text = str(rel_out).replace("\\", "/")
            alt = f"图6-{num}({letter}) {caption}"
            rewritten.append(f"图6-{num}({letter}) {caption}  ")
            rewritten.append(f"![{alt}]({rel_out_text})")
            rewritten.append("")
        i = j
    range_map: Dict[str, str] = {}
    for num, count in subfig_counts.items():
        last_letter = chr(ord("a") + count - 1)
        if count == 2:
            range_map[num] = f"图6-{num}(a)和图6-{num}(b)"
        else:
            range_map[num] = f"图6-{num}(a)至图6-{num}({last_letter})"

    normalized: List[str] = []
    for line in rewritten:
        if FIG_RE.match(line) or SUBFIG_LINE_RE.match(line) or line.startswith("!["):
            normalized.append(line)
            continue
        updated = line
        for num, replacement in range_map.items():
            updated = re.sub(rf"图6-{num}(?!\()", replacement, updated)
        normalized.append(updated)

    md_path.write_text("\n".join(normalized) + "\n", encoding="utf-8")


def main() -> None:
    md_path = _find_single_path("*0511promote.md")
    _rewrite_markdown(md_path)
    print(f"updated {md_path}")


if __name__ == "__main__":
    main()
