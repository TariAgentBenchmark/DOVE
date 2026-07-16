import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Build DOVE pure-algorithm comparison sheets")
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--candidate_dir", type=Path, required=True)
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", default=["000", "002", "005", "007"])
    parser.add_argument("--frames", nargs="+", type=int, default=[5, 15, 25])
    parser.add_argument("--cell_width", type=int, default=600)
    return parser.parse_args()


def read_frame(path, frame_index):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def resolve_video(directory, sample):
    for suffix in (".mp4", ".mkv", ".mov", ".avi"):
        path = directory / f"{sample}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No video found for sample {sample} in {directory}")


def get_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def build_sheet(args, sample):
    sources = [
        ("Original DOVE", resolve_video(args.baseline_dir, sample)),
        ("Pure algorithm", resolve_video(args.candidate_dir, sample)),
        ("GT", resolve_video(args.gt_dir, sample)),
    ]
    first = read_frame(sources[0][1], args.frames[0])
    cell_height = round(args.cell_width * first.height / first.width)
    header_height = 52
    row_label_width = 92
    canvas = Image.new(
        "RGB",
        (row_label_width + args.cell_width * len(sources), header_height + cell_height * len(args.frames)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    header_font = get_font(23, bold=True)
    row_font = get_font(18, bold=True)
    for column, (label, _) in enumerate(sources):
        x0 = row_label_width + column * args.cell_width
        bbox = draw.textbbox((0, 0), label, font=header_font)
        draw.text(
            (x0 + (args.cell_width - (bbox[2] - bbox[0])) / 2, 12),
            label,
            fill="#1f2937",
            font=header_font,
        )
    for row, frame_index in enumerate(args.frames):
        y0 = header_height + row * cell_height
        label = f"F{frame_index}"
        bbox = draw.textbbox((0, 0), label, font=row_font)
        draw.text(
            ((row_label_width - (bbox[2] - bbox[0])) / 2, y0 + (cell_height - (bbox[3] - bbox[1])) / 2),
            label,
            fill="#475467",
            font=row_font,
        )
        for column, (_, video_path) in enumerate(sources):
            image = read_frame(video_path, frame_index).resize(
                (args.cell_width, cell_height), Image.Resampling.LANCZOS
            )
            x0 = row_label_width + column * args.cell_width
            canvas.paste(image, (x0, y0))
            draw.rectangle(
                (x0, y0, x0 + args.cell_width - 1, y0 + cell_height - 1),
                outline="#d0d5dd",
                width=1,
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output_dir / f"{sample}_comparison.png", quality=95)


def main():
    args = parse_args()
    for sample in args.samples:
        build_sheet(args, sample)
        print(f"Built {sample}_comparison.png", flush=True)


if __name__ == "__main__":
    main()
