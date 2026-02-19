from __future__ import annotations

from dataclasses import dataclass


MM_TO_PT = 72.0 / 25.4


@dataclass(frozen=True)
class CheckerboardSpec:
    inside_cols: int
    inside_rows: int
    square_size_m: float
    paper: str = "A4"

    @property
    def square_size_mm(self) -> float:
        return self.square_size_m * 1000.0

    @property
    def squares_x(self) -> int:
        return self.inside_cols + 1

    @property
    def squares_y(self) -> int:
        return self.inside_rows + 1


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _a4_page_points() -> tuple[float, float]:
    return 210.0 * MM_TO_PT, 297.0 * MM_TO_PT


def generate_checkerboard_pdf(spec: CheckerboardSpec) -> bytes:
    if spec.inside_cols <= 0 or spec.inside_rows <= 0:
        raise ValueError("chessboard inside corners must be positive")
    if spec.square_size_m <= 0:
        raise ValueError("square size must be positive")
    if spec.paper.upper() != "A4":
        raise ValueError("only A4 is supported in this version")

    square_pt = spec.square_size_mm * MM_TO_PT
    board_w = spec.squares_x * square_pt
    board_h = spec.squares_y * square_pt
    margin = 8.0 * MM_TO_PT

    p_w, p_h = _a4_page_points()
    orientations = [(p_w, p_h, "portrait"), (p_h, p_w, "landscape")]
    chosen = None
    for page_w, page_h, orientation in orientations:
        if board_w <= (page_w - (2 * margin)) and board_h <= (page_h - (2 * margin)):
            chosen = (page_w, page_h, orientation)
            break

    if chosen is None:
        raise ValueError("checkerboard does not fit on A4 with current square size")
    page_w, page_h, orientation = chosen

    board_x = (page_w - board_w) / 2.0
    board_y = (page_h - board_h) / 2.0

    commands: list[str] = []
    # White background
    commands.append("1 g")
    commands.append(f"0 0 {page_w:.3f} {page_h:.3f} re f")
    # Black squares
    commands.append("0 g")
    for y in range(spec.squares_y):
        for x in range(spec.squares_x):
            if (x + y) % 2 != 0:
                continue
            rx = board_x + (x * square_pt)
            ry = board_y + ((spec.squares_y - 1 - y) * square_pt)
            commands.append(f"{rx:.3f} {ry:.3f} {square_pt:.3f} {square_pt:.3f} re f")

    annotation = (
        f"Checkerboard {spec.inside_cols}x{spec.inside_rows} inside corners | "
        f"{spec.square_size_mm:.1f} mm squares | A4 {orientation} | Print 100%"
    )
    text_x = margin
    text_y = page_h - (12.0 * MM_TO_PT)
    commands.append("BT")
    commands.append("/F1 11 Tf")
    commands.append(f"{text_x:.3f} {text_y:.3f} Td")
    commands.append(f"({_pdf_escape(annotation)}) Tj")
    commands.append("ET")

    content = "\n".join(commands).encode("ascii")

    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.3f} {page_h:.3f}] "
        f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".encode("ascii")
    )
    objects.append(
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    header = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"
    out = bytearray(header)
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")

    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )
    return bytes(out)
