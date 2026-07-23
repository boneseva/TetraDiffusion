#!/usr/bin/env python3
"""
pipeline/dashboard.py — Interactive Web Browser Suite for TetraDiffusion

Provides a visual control desk and diagnostic dashboard:
  - Real-time Pipeline Health & Status Matrix
  - Interactive 3D WebGL Mesh Inspector
  - One-Click Stage Execution (Registration, Training, Inference, Evaluation)
  - Embedded HTML Shape Space Plot Viewer
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    print("ERROR: Gradio is required for the visual dashboard. Install via: pip install gradio", file=sys.stderr)
    sys.exit(1)


# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Helper Functions ──────────────────────────────────────────────────────────
def count_files(pattern: str | Path) -> int:
    return len(glob.glob(str(pattern), recursive=True))


def get_status_data():
    """Build status rows for Gradio Dataframe."""
    rows = []
    datasets = [
        ("OpenOrganelle", REPO_ROOT / "data", REPO_ROOT / "lib" / "all.csv", ["Lysosome", "Mitochondria", "Golgi", "ER"]),
        ("UroCell", REPO_ROOT / "data_urocell", REPO_ROOT / "lib" / "all_urocell.csv", ["lyso", "mito", "fv"]),
    ]

    for name, root, csv_file, categories in datasets:
        csv_counts = {}
        if csv_file.exists():
            with csv_file.open(newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    cat = r.get("synsetId", r.get("subSynsetId", "unknown")).strip()
                    csv_counts[cat] = csv_counts.get(cat, 0) + 1

        raw_base = root / "organelles_raw"
        pre_base = root / "preprocessed"
        runs_base = REPO_ROOT / "runs"

        for cat in categories:
            raw_cnt = count_files(raw_base / cat / "**" / "*.obj")
            if raw_cnt == 0:
                raw_cnt = count_files(root / "organelles" / cat / "*.obj")

            pre_cnt = count_files(pre_base / cat / "**" / "sample.pth")
            csv_cnt = csv_counts.get(cat, 0)

            ckpt_cnt = 0
            inf_cnt = 0
            if runs_base.exists():
                for run_dir in runs_base.iterdir():
                    if run_dir.is_dir() and cat.lower() in run_dir.name.lower():
                        ckpt_cnt += count_files(run_dir / "model-*.pt")
                        inf_cnt += count_files(run_dir / "**" / "*.obj")

            rows.append([name, cat, raw_cnt, pre_cnt, csv_cnt, ckpt_cnt, inf_cnt])

    return rows


def find_available_obj_files() -> list[str]:
    """Collect available OBJ files across dataset and runs for the 3D viewer."""
    objs = glob.glob(str(REPO_ROOT / "runs" / "**" / "*.obj"), recursive=True)
    objs.extend(glob.glob(str(REPO_ROOT / "data_urocell" / "**" / "*.obj"), recursive=True))
    objs.extend(glob.glob(str(REPO_ROOT / "data" / "**" / "*.obj"), recursive=True))
    return sorted(objs)[:100]  # cap list for dropdown


def run_register_action(dataset: str):
    all_csv = "lib/all_urocell.csv" if dataset == "UroCell" else "lib/all.csv"
    data_root = "data_urocell/preprocessed" if dataset == "UroCell" else "data/preprocessed"
    cats = ["lyso", "mito", "fv"] if dataset == "UroCell" else ["Lysosome", "Mitochondria", "Golgi", "ER"]

    cmd = [sys.executable, str(REPO_ROOT / "register_category.py"), *cats, "--data_root", data_root, "--all_csv", all_csv]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout + "\n" + res.stderr


def run_training_action(dataset: str, dry_run: bool):
    script = "launch_production_runs_urocell.sh" if dataset == "UroCell" else "launch_production_runs.sh"
    cmd = ["bash", str(REPO_ROOT / script)]
    if dry_run:
        cmd.append("--dry_run")
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout + "\n" + res.stderr


def run_inference_action(run_name: str):
    cmd = ["bash", str(REPO_ROOT / "launch_inference.sh")]
    if run_name.strip():
        cmd.extend(["--run_name", run_name.strip()])
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout + "\n" + res.stderr


def run_eval_action(dataset: str):
    csv_path = "lib/all_urocell.csv" if dataset == "UroCell" else "lib/all.csv"
    data_path = "data_urocell/preprocessed" if dataset == "UroCell" else "data/preprocessed"

    cmd1 = [sys.executable, str(REPO_ROOT / "evaluation" / "compare.py"), "--data_path", data_path, "--csv_path", csv_path]
    cmd2 = [sys.executable, str(REPO_ROOT / "evaluation" / "plot_shape_space_html.py"), "--data_path", data_path, "--csv_path", csv_path]

    r1 = subprocess.run(cmd1, capture_output=True, text=True)
    r2 = subprocess.run(cmd2, capture_output=True, text=True)

    return f"--- Compare Metrics ---\n{r1.stdout}\n{r1.stderr}\n\n--- Shape Space Plot ---\n{r2.stdout}\n{r2.stderr}"


def load_html_shape_space():
    html_files = glob.glob(str(REPO_ROOT / "*.html")) + glob.glob(str(REPO_ROOT / "evaluation" / "*.html"))
    if not html_files:
        return "<p style='padding:20px;'>No shape-space HTML plots generated yet. Click 'Run Quantitative Evaluation' to generate one!</p>"
    latest_html = max(html_files, key=os.path.getmtime)
    with open(latest_html, "r", encoding="utf-8") as f:
        return f.read()


# ── Gradio App Layout ────────────────────────────────────────────────────────
def create_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="cyan",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=theme, title="TetraDiffusion Studio") as app:
        gr.Markdown(
            """
            # 🔬 TetraDiffusion — Unified 3D Organelle Pipeline Studio
            Manage datasets, launch production training runs, sample 3D meshes, and inspect geometric shape-spaces.
            """
        )

        with gr.Tab("📊 Pipeline Health & Status"):
            gr.Markdown("### Real-time Dataset & Training Matrix")
            status_df = gr.Dataframe(
                headers=["Dataset", "Category", "Raw OBJs", "Preprocessed", "CSV Reg.", "Checkpoints", "Inferred OBJs"],
                value=get_status_data(),
                interactive=False,
            )
            refresh_btn = gr.Button("🔄 Refresh Status", variant="primary")
            refresh_btn.click(fn=get_status_data, outputs=status_df)

        with gr.Tab("🧊 3D Mesh Inspector"):
            gr.Markdown("### Interactive 3D WebGL Viewport")
            obj_list = find_available_obj_files()
            with gr.Row():
                obj_dropdown = gr.Dropdown(choices=obj_list, label="Select 3D Mesh OBJ File", value=obj_list[0] if obj_list else None)
                reload_objs_btn = gr.Button("🔄 Refresh File List")

            model_3d = gr.Model3D(value=obj_list[0] if obj_list else None, label="Interactive 3D Render", height=450)

            obj_dropdown.change(fn=lambda path: path, inputs=obj_dropdown, outputs=model_3d)
            reload_objs_btn.click(fn=lambda: gr.update(choices=find_available_obj_files()), outputs=obj_dropdown)

        with gr.Tab("🚀 Action Control Desk"):
            gr.Markdown("### Trigger Pipeline Stages")
            with gr.Row():
                target_dataset = gr.Radio(["UroCell", "OpenOrganelle"], label="Target Dataset", value="UroCell")

            with gr.Accordion("1. Audit & Register CSV", open=True):
                reg_btn = gr.Button("▶ Run register_category.py", variant="secondary")
                reg_out = gr.Textbox(label="Registration Output", lines=5)
                reg_btn.click(fn=run_register_action, inputs=target_dataset, outputs=reg_out)

            with gr.Accordion("2. Production Training", open=True):
                dry_run_chk = gr.Checkbox(label="Dry Run (Preview commands without submitting)", value=True)
                train_btn = gr.Button("🚀 Launch / Resume Production Training", variant="primary")
                train_out = gr.Textbox(label="Launcher Output", lines=5)
                train_btn.click(fn=run_training_action, inputs=[target_dataset, dry_run_chk], outputs=train_out)

            with gr.Accordion("3. Mesh Inference & Sampling", open=True):
                run_name_input = gr.Textbox(label="Specific Run Name (optional, leave blank for all ready runs)", placeholder="e.g. urocell_fv_final_prod")
                infer_btn = gr.Button("⚡ Launch Inference", variant="secondary")
                infer_out = gr.Textbox(label="Inference Output", lines=5)
                infer_btn.click(fn=run_inference_action, inputs=run_name_input, outputs=infer_out)

            with gr.Accordion("4. Quantitative Evaluation", open=True):
                eval_btn = gr.Button("📈 Run Quantitative Evaluation & Plot Shape Space", variant="secondary")
                eval_out = gr.Textbox(label="Evaluation Output", lines=6)
                eval_btn.click(fn=run_eval_action, inputs=target_dataset, outputs=eval_out)

        with gr.Tab("🌐 Interactive Shape Space Plot"):
            gr.Markdown("### Morphological Feature Space (PCA / t-SNE)")
            render_html_btn = gr.Button("🔄 Load Latest Interactive Shape Space HTML")
            html_view = gr.HTML(value=load_html_shape_space())
            render_html_btn.click(fn=load_html_shape_space, outputs=html_view)

    return app


# ── Entry Point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="TetraDiffusion Visual Dashboard")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    args = parser.parse_args()

    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
