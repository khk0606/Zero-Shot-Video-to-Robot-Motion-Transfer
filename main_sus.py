import cv2
import numpy as np
import base64
import os
from pathlib import Path
import argparse
from yt_dlp import YoutubeDL
from agent_gemini import SUSGenerator

# 설정
cfg = {
    "model": "gemini-2.5-flash"
}

def download_video(url, output_path="input_video.mp4"):
    print(f"Downloading video from {url}...")
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': output_path,
        'quiet': True,
        'overwrites': True
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Download complete: {output_path}")
    return output_path

def create_frame_grid(video_path, output_dir: str, grid_size=(2, 4)):
    print("Extracting frames...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames-10, grid_size[0]*grid_size[1], dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (320, 240))
            frames.append(frame)
    cap.release()

    rows = []
    for r in range(grid_size[0]):
        row_frames = frames[r*grid_size[1] : (r+1)*grid_size[1]]
        if row_frames:
            rows.append(np.hstack(row_frames))

    if not rows:
        raise ValueError("No frames extracted!")

    grid_image = np.vstack(rows)
    save_path = os.path.join(output_dir, f"{Path(video_path).stem}_frame_grid.png")
    cv2.imwrite(save_path, grid_image)
    print(f"Frame grid saved to: {save_path}")

    return grid_image

def encode_image_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, help="Input video path or URL")
    parser.add_argument("--output", required=False, help="Output SUS report path")
    parser.add_argument(
        "--output-dir",
        required=False,
        default=None,
        help="Directory for generated artifacts (default: <repo>/output)",
    )
    args = parser.parse_args()

    try:
        base_dir = Path(__file__).resolve().parent
        output_dir = Path(args.output_dir).resolve() if args.output_dir else (base_dir / "output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # input 처리
        if args.input is None:
            raise RuntimeError("Missing --input. Provide a local file or URL.")
        if args.input.startswith("http"):
            video_path = os.path.join(output_dir, "target_motion.mp4")
            download_video(args.input, video_path)
        else:
            video_path = args.input
            print("USING VIDEO PATH:", video_path)


        # 그리드 이미지 생성
        grid_img = create_frame_grid(video_path, output_dir=str(output_dir))
        encoded_grid = encode_image_base64(grid_img)

        # SUS 파이프라인 실행
        print("\n>>> Starting SUS Analysis Pipeline <<<")
        sus_gen = SUSGenerator(cfg)
        final_report = sus_gen.generate_sus_prompt(encoded_grid)

        # 결과 저장
        report_path = args.output or os.path.join(output_dir, "final_sus_report.txt")
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(final_report)

        print("\n" + "="*50)
        print(f" Analysis Complete! Report saved to: {report_path}")
        print("="*50)
        print(final_report)

    except Exception as e:
        print(f"\n Error occurred: {e}")
