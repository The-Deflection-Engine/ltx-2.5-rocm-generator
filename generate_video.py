#!/usr/bin/env python3
"""Tk control panel for LTX-2.5. All generation logic lives in ltx_engine."""
import os
import sys
import json
import random
import threading
import multiprocessing as mp
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog

import ltx_engine as eng
# Read-only constants and functions can be bound directly. `cancel_flag` and
# `debug_flag` deliberately are NOT: the GUI mutates them and the worker reads
# them, so they must be touched as eng.<name> or the two modules would end up
# with independent copies.
from ltx_engine import (
    CONFIG_FILE,
    EMBED_CACHE_DIR,
    ENHANCER_PATH,
    MIN_DIMENSION,
    SPATIAL_COMPRESSION,
    VRAM_BASE_GB,
    VRAM_GB_PER_TOKEN,
    align_frames,
    auto_duration_cap_s,
    free_resident_models,
    generation_worker,
    guidance_pass_count,
    hw_monitor,
    latent_tokens,
    run_subprocess_logged,
    set_diffusers_verbosity,
    snap_dimension,
    token_warn_threshold,
)

def tooltip(widget, text, delay=500):
    """Hover tip. ttk has no tooltip widget, and `idlelib.tooltip.Hovertip`
    (which would do) isn't dependable -- Debian/Ubuntu ship idlelib in a
    separate idle-python3.x package that often isn't installed. This is the
    whole feature in a few lines, so it needs neither."""
    state = {"win": None, "after": None}

    def show():
        state["after"] = None
        if state["win"] or not widget.winfo_viewable():
            return
        win = tk.Toplevel(widget)
        win.wm_overrideredirect(True)      # no title bar / border
        win.wm_attributes("-topmost", True)  # else it can hide behind the app
        tk.Label(win, text=text, justify=tk.LEFT, background="#ffffe0",
                 foreground="#000000", relief=tk.SOLID, borderwidth=1,
                 font=("Arial", 9), padx=6, pady=4, wraplength=380).pack()

        # Measure before placing: controls in the bottom button row would
        # otherwise draw their tip below the screen edge and appear "broken".
        win.update_idletasks()
        tw, th = win.winfo_reqwidth(), win.winfo_reqheight()
        x = widget.winfo_rootx() + 12
        y = widget.winfo_rooty() + widget.winfo_height() + 4
        if y + th > widget.winfo_screenheight():
            y = widget.winfo_rooty() - th - 4        # flip above
        if x + tw > widget.winfo_screenwidth():
            x = widget.winfo_screenwidth() - tw - 8  # nudge left
        win.wm_geometry(f"+{max(0, x)}+{max(0, y)}")
        state["win"] = win

    def enter(_e=None):
        state["after"] = widget.after(delay, show)

    def leave(_e=None):
        if state["after"]:
            widget.after_cancel(state["after"])
            state["after"] = None
        if state["win"]:
            state["win"].destroy()
            state["win"] = None

    widget.bind("<Enter>", enter, add="+")
    widget.bind("<Leave>", leave, add="+")
    # A click shouldn't leave a tip orphaned over the window.
    widget.bind("<ButtonPress>", leave, add="+")
    return widget


class TextRedirector:
    def __init__(self, widget):
        self.widget = widget
    def write(self, text):
        self.widget.after(0, self._write, text)
    def _write(self, text):
        # Only follow the tail if the view is already at the bottom. Otherwise a
        # running generation yanks the viewport away mid-selection, which makes
        # the log impossible to read or copy from while it's still writing.
        at_bottom = self.widget.yview()[1] >= 0.999
        self.widget.configure(state="normal")
        # tqdm-style progress (checkpoint shards, "Loading pipeline
        # components...", ffmpeg's encode progress) writes bare '\r' to
        # overwrite the same line, the way a real terminal renders it. A plain
        # insert() treats '\r' as just another character, so repeated updates
        # piled up jammed together instead of overwriting -- this splits on
        # '\r' and erases the current line before each subsequent part, which
        # is what a terminal does. Text with no '\r' (the common case) takes
        # the exact same single-insert path as before.
        for i, part in enumerate(text.split("\r")):
            if i > 0:
                self.widget.delete("end-1c linestart", "end-1c")
            self.widget.insert(tk.END, part)
        if at_bottom:
            self.widget.see(tk.END)
        self.widget.configure(state="disabled")
    def flush(self):
        pass

# =========================================================================
# 5. Main Application / GUI
# =========================================================================
def main():
    def load_saved_config():
        defaults = {
            "prompt": "A high-quality cinematic shot of a classic sports car driving along a coastal highway at sunset, vibrant orange horizon, clear ocean view.",
            "negative_prompt": "",
            # Reference LTX-2.5 two-stage base resolution; stage 2 emits 1920x1088.
            "width": 960,
            "height": 544,
            "frames": 121,
            "fps": 24.0,
            "seed": "42",
            "upscale": False,
            "mode": "text2video",
            "image_path": "",
            # Auto Duration: let the model pick clip length from the prompt.
            # Capped by auto_duration_cap_s() regardless of what's set here.
            "auto_duration": False,
            # CFG quality mode: much better prompt adherence, ~7-8x the compute
            # and ~2x activation VRAM. See the warning on the GUI checkbox.
            "cfg_mode": False,
            # STG: structural guidance, no negative prompt. ~2x cost.
            "stg_mode": False,
            "stg_scale": 1.0,
            # 0 = no cap on the enhanced prompt length.
            "enhance_max_words": 0,
            "cfg_steps": 30,
            "cfg_scale": 3.0,
            # Separate from cfg_scale: the reference recipe runs audio CFG
            # much stronger than video (7.0 vs 3.0). No GUI control -- costs
            # nothing extra since it rides the same doubled pass as video CFG,
            # so there's nothing to trade off by exposing it; edit the config
            # directly if you want to experiment.
            "audio_cfg_scale": 7.0,
            # Modality-isolation guidance: costs a THIRD full transformer pass
            # per step (unlike audio_cfg_scale, not free), so off (1.0) by
            # default even with CFG on. Reference recipe runs this at 3.0
            # alongside CFG; opt in here once you have VRAM/time to spare.
            "cfg_modality_scale": 1.0,
            "auto_min_seconds": 2.0,
            "auto_max_seconds": 5.0,
            # Conditioning-image compression for image-to-video. null = model
            # default (CRF 18 on LTX-2.5), 0 = keep full source detail.
            "image_crf": None,
            # 48 transformer blocks. 4 => 12 offload groups; raise to 6-8 for a bit
            # more speed at the cost of VRAM, drop to 2 if stage 2 OOMs.
            "blocks_per_group": 4
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    defaults.update(json.load(f))
            except Exception as exc:
                # A silently-discarded prompt/settings load looks like "the
                # app forgot everything" with no clue why. A corrupt file or a
                # read-only directory both land here; both are worth a line.
                print(f"[!] Could not load {CONFIG_FILE}, using defaults: {exc}")
        return defaults

    def save_config(config):
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
        except Exception as exc:
            print(f"[!] Could not save {CONFIG_FILE}: {exc}")

    config = load_saved_config()
    
    root = tk.Tk()
    root.title("🎬 LTX-2.5 Control Panel")
    # Opening size: 1080 tall pays for the 7-row prompt box (Arial-10 is 16px
    # per row). Resizable both ways; spare height goes to the log pane, the only
    # widget packed with expand=True, and spare width to everything on fill=X.
    # minsize is set once the layout has been measured, further down -- below it
    # the bottom button row starts clipping.
    WIN_W, WIN_H = 740, 1080
    root.geometry(f"{WIN_W}x{WIN_H}")
    root.resizable(True, True)
    
    style = ttk.Style(root)
    style.theme_use('clam')

    # messagebox/showerror dialogs are Tk's own (tk::MessageBox) on X11, and
    # take their font from the option database rather than any per-call
    # argument. Default is a large serif face, which makes the multi-paragraph
    # token/CFG warnings enormous. wrapLength has to grow as the font shrinks,
    # or the dialog just gets tall and narrow instead.
    root.option_add("*Dialog.msg.font", "Arial 9")
    root.option_add("*Dialog.msg.wrapLength", "560")


    # Telemetry Panel
    telemetry_frame = tk.Frame(root, bg="#111111", pady=8, padx=12)
    telemetry_frame.pack(fill=tk.X)
    
    primary_metrics = tk.Frame(telemetry_frame, bg="#111111")
    primary_metrics.pack(fill=tk.X)
    
    lbl_cpu = tk.Label(primary_metrics, text="CPU: --%", bg="#111111", fg="#ffffff", font=("Consolas", 10, "bold"), width=15, anchor=tk.W)
    lbl_cpu.pack(side=tk.LEFT)
    
    lbl_ram = tk.Label(primary_metrics, text="RAM: -- / -- GB", bg="#111111", fg="#ffffff", font=("Consolas", 10, "bold"), width=20)
    lbl_ram.pack(side=tk.LEFT, expand=True)
    
    lbl_gpu = tk.Label(primary_metrics, text="GPU: --%", bg="#111111", fg="#00ffcc", font=("Consolas", 10, "bold"), width=12)
    lbl_gpu.pack(side=tk.LEFT, expand=True)
    
    lbl_vram = tk.Label(primary_metrics, text="VRAM: -- / -- GB", bg="#111111", fg="#00ffcc", font=("Consolas", 10, "bold"), width=22, anchor=tk.E)
    lbl_vram.pack(side=tk.RIGHT)

    cores_box = tk.Frame(telemetry_frame, bg="#222222", padx=6, pady=5, relief=tk.SUNKEN, bd=1)
    cores_box.pack(fill=tk.X, pady=(6, 0))
    
    # One label per core so each can be coloured independently. Built lazily on
    # the first sample, because the core count comes from /proc/stat, not us.
    lbl_cores_text = tk.Label(
        cores_box,
        text="Reading CPU Core states...",
        bg="#222222",
        fg="#00ff88",
        font=("Consolas", 8, "bold"),
        justify=tk.LEFT,
        anchor=tk.W
    )
    lbl_cores_text.pack(fill=tk.X)
    core_labels = []

    def core_colour(pct):
        if pct >= 75:
            return "#ff4444"   # red    - saturated
        if pct >= 50:
            return "#ffaa00"   # orange - working hard
        return "#00ff88"       # green  - idle/light

    # The primary readouts are white/cyan by default rather than green, so they
    # get their own "normal" colour but share the 50/75 thresholds.
    def load_colour(pct, normal):
        if pct >= 76:
            return "#ff4444"
        if pct >= 50:
            return "#ffaa00"
        return normal

    def update_telemetry():
        cpu_avg, cores = hw_monitor.get_cpu_stats()
        ram_used, ram_total = hw_monitor.get_ram_stats()
        gpu_usage, vram_used, vram_total = hw_monitor.get_gpu_stats()

        lbl_cpu.config(text=f"CPU: {cpu_avg:.0f}% (Avg)",
                       fg=load_colour(cpu_avg, "#ffffff"))
        ram_pct = (ram_used / ram_total * 100) if ram_total else 0
        lbl_ram.config(text=f"RAM: {ram_used:.1f}/{ram_total:.1f} GB",
                       fg=load_colour(ram_pct, "#ffffff"))

        if gpu_usage is None:
            lbl_gpu.config(text="GPU: N/A", fg="#00ffcc")
            lbl_vram.config(text="VRAM: N/A", fg="#00ffcc")
        else:
            lbl_gpu.config(text=f"GPU: {gpu_usage}%",
                           fg=load_colour(gpu_usage, "#00ffcc"))
            vram_pct = (vram_used / vram_total * 100) if vram_total else 0
            lbl_vram.config(text=f"VRAM: {vram_used:.1f}/{vram_total:.1f} GB",
                            fg=load_colour(vram_pct, "#00ffcc"))
            
        if cores:
            if not core_labels:
                # pack() and grid() can't share a parent, so the placeholder goes
                # before the grid of per-core labels is built.
                lbl_cores_text.destroy()
                for i in range(len(cores)):
                    lab = tk.Label(cores_box, bg="#222222", font=("Consolas", 8, "bold"),
                                   width=8, anchor=tk.W)
                    lab.grid(row=i // 8, column=i % 8, sticky=tk.W, padx=(0, 4))
                    core_labels.append(lab)
            for i, pct in enumerate(cores):
                core_labels[i].config(text=f"C{i:02d}:{int(pct):2d}%", fg=core_colour(pct))
            
        root.after(500, update_telemetry)

    update_telemetry()
    
    # Main Form
    main_frame = ttk.Frame(root, padding="15")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # --- Generation mode ---
    # Unload Models sits inside this box, at the right of the mode row -- not
    # among the Generate/Cancel buttons at the bottom, where it used to be one
    # misplaced click away from Generate and cost an 18GB reload.
    mode_frame = ttk.LabelFrame(main_frame, text=" Generation Mode ", padding="8")
    mode_frame.pack(fill=tk.X, pady=(0, 12))

    mode_var = tk.StringVar(value=config.get("mode", "text2video"))
    image_path_var = tk.StringVar(value=config.get("image_path", ""))

    mode_row = ttk.Frame(mode_frame)
    mode_row.pack(fill=tk.X)
    ttk.Radiobutton(mode_row, text="Text → Video", variable=mode_var, value="text2video").pack(side=tk.LEFT)
    ttk.Radiobutton(mode_row, text="Image → Video", variable=mode_var, value="image2video").pack(side=tk.LEFT, padx=(12, 0))

    def unload_models():
        if messagebox.askokcancel(
            "Unload Models",
            "Drop the resident pipeline and hand back ~18GB of RAM?\n\n"
            "The next run reloads everything from disk -- one slow generation."
        ):
            free_resident_models()

    btn_free = ttk.Button(mode_row, text="🧹 Unload Models", command=unload_models)
    btn_free.pack(side=tk.RIGHT)
    tooltip(btn_free,
            "Drop the resident pipeline and hand back ~18GB of RAM.\n"
            "The next run reloads from disk (one slow generation).\n\n"
            "Usually unnecessary -- VRAM is freed automatically after every "
            "run. Use this when you need the RAM for something else.")

    image_row = ttk.Frame(mode_frame)
    image_row.pack(fill=tk.X, pady=(6, 0))
    lbl_image = ttk.Label(image_row, textvariable=image_path_var, foreground="#666666")

    def browse_image():
        path = filedialog.askopenfilename(
            title="Select conditioning image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
        )
        if path:
            image_path_var.set(path)

    btn_browse = ttk.Button(image_row, text="📁 Choose Image...", command=browse_image)

    def on_mode_switch(*_args):
        if mode_var.get() == "image2video":
            btn_browse.pack(side=tk.LEFT)
            lbl_image.pack(side=tk.LEFT, padx=(8, 0))
        else:
            btn_browse.pack_forget()
            lbl_image.pack_forget()

    mode_var.trace_add("write", on_mode_switch)
    on_mode_switch()

    pp_header = ttk.Frame(main_frame)
    pp_header.pack(fill=tk.X)
    ttk.Label(pp_header, text="Positive Prompt:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)

    # 7 rows: an enhanced prompt is ~850 chars, which is about 8 lines at this width.
    text_prompt = tk.Text(main_frame, height=7, wrap=tk.WORD, font=("Arial", 10))

    btn_clear_pp = ttk.Button(pp_header, text="✕ Clear",
                              command=lambda: text_prompt.delete("1.0", tk.END))
    btn_clear_pp.pack(side=tk.RIGHT)
    tooltip(btn_clear_pp, "Empty the positive prompt box.")

    text_prompt.pack(fill=tk.X, pady=(4, 12))
    text_prompt.insert(tk.END, config['prompt'])

    enhance_row = ttk.Frame(main_frame)
    enhance_row.pack(fill=tk.X, pady=(0, 12))

    ttk.Label(
        enhance_row,
        text="Rewrite the prompt in LTX-2.5's trained caption style:",
        foreground="#666666",
    ).pack(side=tk.LEFT)

    def enhance_now():
        """Rewrite the prompt box in place so it can be reviewed/edited before
        generating. Runs in the same throwaway subprocess the generate path uses."""
        p = text_prompt.get("1.0", tk.END).strip()
        if not p:
            messagebox.showerror("Error", "Positive prompt cannot be empty.")
            return
        if not os.path.isdir(ENHANCER_PATH):
            messagebox.showerror("Error", f"Prompt enhancer not found at {ENHANCER_PATH}.")
            return
        img = image_path_var.get().strip() if mode_var.get() == "image2video" else ""

        try:
            max_words = int(enh_words_var.get())
        except (ValueError, tk.TclError):
            max_words = 0
        max_words = max_words or None      # 0 / blank = no limit, stock behaviour

        btn_enhance.config(state="disabled", text="✨ Enhancing...")

        def work():
            out_path = os.path.join(EMBED_CACHE_DIR, "enhanced_prompt.txt")
            try:
                os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
                if os.path.exists(out_path):
                    os.remove(out_path)
                print("\n--- Enhancing prompt ---")
                run_subprocess_logged("enhance_in_subprocess", (p, img, out_path, max_words))
                if not os.path.exists(out_path):
                    raise RuntimeError("Enhancement failed -- see the output above.")
                with open(out_path) as f:
                    enhanced = f.read().strip()
                # Printed here, in the parent, so it reaches the GUI log. The
                # child's stdout is forwarded too, but the prompt is the one
                # thing worth showing verbatim next to the box it lands in.
                print(f"  -> Enhanced prompt ({len(enhanced)} chars):\n{enhanced}\n")
            except Exception as exc:
                msg = str(exc)   # exc is unbound once this block exits
                root.after(0, lambda m=msg: messagebox.showerror("Enhance failed", m))
                enhanced = None

            def done():
                if enhanced:
                    text_prompt.delete("1.0", tk.END)
                    text_prompt.insert(tk.END, enhanced)
                btn_enhance.config(state="normal", text="✨ Enhance Now")
            root.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    btn_enhance = ttk.Button(enhance_row, text="✨ Enhance Now", command=enhance_now)
    btn_enhance.pack(side=tk.RIGHT)

    # Length cap for the enhancer. 0 = stock behaviour (it writes until done,
    # typically 120-160 words).
    enh_words_var = tk.StringVar(value=str(config.get("enhance_max_words", 0)))
    ttk.Spinbox(enhance_row, from_=0, to=200, increment=10, width=4,
                textvariable=enh_words_var).pack(side=tk.RIGHT, padx=(6, 6))
    lbl_enh_words = ttk.Label(enhance_row, text="max words:", foreground="#666666")
    lbl_enh_words.pack(side=tk.RIGHT)
    tooltip(lbl_enh_words,
            "Cap the enhanced prompt's length. 0 = no limit (stock LTX-2.5\n"
            "behaviour, usually 120-160 words).\n\n"
            "The verbosity is deliberate -- LTX-2.5 was trained on exhaustive\n"
            "captions, so trimming may cost some adherence. The limit asks the\n"
            "model to drop background and ambience first, never the subject,\n"
            "its action, or the camera.")
    tooltip(btn_enhance,
            "Rewrite the prompt above into the long, detailed caption style\n"
            "LTX-2.5 was trained on, using Gemma-4.\n\n"
            "The result replaces the box so you can read and edit it before\n"
            "generating. Takes ~1 min; runs in a throwaway subprocess, so it\n"
            "costs no VRAM during the actual render.")


    np_header = ttk.Frame(main_frame)
    np_header.pack(fill=tk.X)
    # Whether this box does anything depends on CFG quality mode, so the label
    # tracks it rather than asserting one state. Updated by update_np_label().
    np_label_var = tk.StringVar()
    ttk.Label(
        np_header,
        textvariable=np_label_var,
        font=("Arial", 10, "bold"),
    ).pack(side=tk.LEFT)
    
    text_np = tk.Text(main_frame, height=5, wrap=tk.WORD, font=("Arial", 10))
    
    def reset_np():
        text_np.delete("1.0", tk.END)
        text_np.insert(tk.END, "Loading default prompt...")
        root.update() 
        try:
            from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
            real_default = DEFAULT_NEGATIVE_PROMPT
        except ImportError:
            real_default = "worst quality, inconsistent, deformed, blurry, watermark"
        text_np.delete("1.0", tk.END)
        text_np.insert(tk.END, real_default)
        
    btn_reset_np = ttk.Button(np_header, text="↺ Reset Default", command=reset_np)
    btn_reset_np.pack(side=tk.RIGHT)
    tooltip(btn_reset_np, "Load the stock LTX-2.5 negative prompt into the box.\n\n"
                          "Only has an effect with CFG quality mode on -- the\n"
                          "distilled schedule is guidance-free and never evaluates\n"
                          "the negative branch.")

    btn_clear_np = ttk.Button(np_header, text="✕ Clear",
                              command=lambda: text_np.delete("1.0", tk.END))
    btn_clear_np.pack(side=tk.RIGHT, padx=(0, 4))
    tooltip(btn_clear_np, "Empty the negative prompt box.")

    text_np.pack(fill=tk.X, pady=(4, 12))
    text_np.insert(tk.END, config.get('negative_prompt', ""))
        
    settings_frame = ttk.Frame(main_frame)
    settings_frame.pack(fill=tk.X, pady=(0, 12))
    
    res_frame = ttk.LabelFrame(settings_frame, text=" Resolution & Quality ", padding="8")
    res_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
    
    res_var = tk.StringVar(value=f"{config['width']}x{config['height']}")
    res_combo = ttk.Combobox(res_frame, textvariable=res_var, state="readonly", width=26)
    # Portrait entries are the landscape ones transposed. Attention cost is
    # latent_frames * (H/32) * (W/32), which is symmetric, so a portrait preset
    # costs exactly the same VRAM as its landscape twin -- the token warning and
    # the README's frame-count guidance carry over unchanged.
    res_combo['values'] = (
        "1280x704 (Landscape, High)",
        "1024x576 (Landscape, Medium)",
        "960x544 (Landscape, 2x = 1920x1088)",
        "768x512 (Landscape, Low)",
        "704x1280 (Portrait, High)",
        "576x1024 (Portrait, Medium)",
        "544x960 (Portrait, 2x = 1088x1920)",
        "512x768 (Portrait, Low)",
        "Custom",
    )
    res_combo.pack(pady=(0, 4))
    tooltip(res_combo,
            "Portrait presets are the landscape ones transposed, so they cost\n"
            "identical VRAM -- attention scales with (H/32)x(W/32), which is\n"
            "symmetric. Frame-count limits apply the same either way.\n\n"
            "Custom lets you type any size; both must be multiples of 32.")
    
    custom_frame = ttk.Frame(res_frame)
    custom_frame.pack()
    ttk.Label(custom_frame, text="W:").pack(side=tk.LEFT)
    entry_w = ttk.Entry(custom_frame, width=5)
    entry_w.pack(side=tk.LEFT, padx=(2, 8))
    entry_w.insert(0, str(config['width']))
    ttk.Label(custom_frame, text="H:").pack(side=tk.LEFT)
    entry_h = ttk.Entry(custom_frame, width=5)
    entry_h.pack(side=tk.LEFT, padx=(2, 0))
    entry_h.insert(0, str(config['height']))

    # Always starts OFF, deliberately not restored from the config -- same
    # reasoning as cfg_var below: 2-stage upscale roughly doubles VRAM
    # pressure and once locked up the whole machine (see commit 740bc65), so
    # it should be a decision made for this run, not something a previous
    # session silently leaves armed.
    upscale_var = tk.BooleanVar(value=False)
    chk_upscale = ttk.Checkbutton(
        res_frame,
        text="2-stage: 2x latent upscale + refine",
        variable=upscale_var,
        command=lambda: update_output_label(),
    )
    chk_upscale.pack(pady=(5, 0))

    # The dropdown lists *base* resolutions; whether a run is 1- or 2-stage is
    # decided by the checkbox above, not by the preset. Spell out the actual
    # output so the two can't be confused.
    lbl_output = ttk.Label(res_frame, foreground="#0a7", font=("Arial", 9, "bold"))
    lbl_output.pack(pady=(2, 0))

    def update_output_label(*_a):
        try:
            w, h = snap_dimension(int(entry_w.get())), snap_dimension(int(entry_h.get()))
        except ValueError:
            lbl_output.config(text="Output: —")
            return
        if upscale_var.get():
            lbl_output.config(text=f"Output: {w*2}x{h*2}  (2-stage from {w}x{h})")
        else:
            lbl_output.config(text=f"Output: {w}x{h}  (single-stage)")

    tooltip(chk_upscale,
            "Generate at the resolution above, then 2x latent upsample and\n"
            "run a short refinement pass -- faster than rendering at full\n"
            "size directly.\n\nCosts VRAM: stage 2 runs 4x the latent tokens.\n"
            "A measured 145-frame 2-stage run reserved 12.04GB, so watch the\n"
            "token warning before pushing resolution or length.")

    # --- CFG quality mode ---
    # Always starts OFF, deliberately not restored from the config. CFG costs
    # ~7-8x the compute, so it should be a decision you make for this run rather
    # than something a previous session silently leaves armed. (The CLI still
    # honours cfg_mode from the config -- there you are stating intent per run.)
    cfg_var = tk.BooleanVar(value=False)

    def update_np_label():
        """The negative prompt is only encoded when CFG mode is on; say which."""
        if cfg_var.get():
            np_label_var.set("Negative Prompt (ACTIVE: CFG quality mode is on):")
        else:
            np_label_var.set("Negative Prompt (INACTIVE: CFG quality mode is off):")

    def on_cfg_toggle():
        if not cfg_var.get():
            update_np_label()
            return
        est_steps = int(config.get("cfg_steps", 30))
        if not messagebox.askokcancel(
            "CFG quality mode",
            "Classifier-free guidance improves how closely the video follows "
            "your prompt -- especially secondary details the distilled schedule "
            "tends to drop.\n\n"
            "IT IS MUCH SLOWER AND USES MUCH MORE VRAM:\n\n"
            f"  • {est_steps} steps instead of 8, and each step runs the\n"
            "    transformer TWICE (prompt + negative). Roughly 7-8x the\n"
            "    compute of a normal run.\n"
            "  • Activation VRAM roughly doubles. On a 16GB card this is\n"
            "    realistically single-stage only -- combining it with 2-stage\n"
            "    upscaling at any real length is likely to run out of memory.\n"
            "  • Your negative prompt becomes live (it is ignored otherwise).\n\n"
            "Recommended: leave 2-stage upscale OFF, start at a modest "
            "resolution and frame count, and increase only once you know how "
            "your GPU copes.\n\nEnable it?",
        ):
            cfg_var.set(False)
        update_np_label()

    chk_cfg = ttk.Checkbutton(
        res_frame,
        text="CFG quality mode (slow, VRAM-hungry)",
        variable=cfg_var,
        command=on_cfg_toggle,
    )
    chk_cfg.pack(pady=(5, 0))
    update_np_label()   # reflect the restored cfg_mode setting at startup

    # --- Spatio-Temporal Guidance ---
    # Unlike CFG this uses no negative prompt: it perturbs a transformer block
    # and steers away from the degraded result, which targets anatomy and
    # object coherence rather than prompt adherence. Also always starts off.
    stg_var = tk.BooleanVar(value=False)

    def on_stg_toggle():
        # From the live spinbox, not the saved config -- the dialog used to
        # quote whatever was last saved, which goes stale the moment someone
        # adjusts the spinbox before ticking the box.
        try:
            eng_scale = float(stg_scale_var.get())
        except (ValueError, tk.TclError):
            eng_scale = 1.0
        if not stg_var.get():
            return
        if not messagebox.askokcancel(
            "Spatio-Temporal Guidance",
            "STG targets structural problems -- duplicated limbs, objects that "
            "float free of the scene -- by running a second pass with one "
            "transformer block perturbed and steering away from that result.\n\n"
            "It uses NO negative prompt. It steers on structure, not text.\n\n"
            "COST: one extra pass per step, so roughly 2x the time and VRAM. "
            "That is about 4x cheaper than CFG quality mode, because it keeps "
            "the 8-step distilled schedule instead of needing 30 steps.\n\n"
            f"Tried on this setup and it visibly improved anatomy and object "
            f"coherence at the default strength of 1.0 -- that was one "
            f"comparison, not a controlled measurement, so judge it on your own "
            f"prompts.\n\n"
            f"If the result looks over-sharpened or contrasty rather than "
            f"better formed, lower the strength beside the checkbox "
            f"(currently {eng_scale}).\n\n"
            "Enable it?",
        ):
            stg_var.set(False)

    stg_row = ttk.Frame(res_frame)
    stg_row.pack(pady=(5, 0))
    chk_stg = ttk.Checkbutton(
        stg_row,
        text="STG: fix anatomy / floating objects",
        variable=stg_var,
        command=on_stg_toggle,
    )
    chk_stg.pack(side=tk.LEFT)

    # Exposed in the GUI, unlike cfg_scale, because STG is untested against the
    # distilled schedule and lowering this is the documented first fix. Editing
    # it in the config instead would mean a restart -- and a restart drops the
    # resident pipeline, so each attempt would cost an 18GB reload.
    stg_scale_var = tk.StringVar(value=str(config.get("stg_scale", 1.0)))
    ttk.Spinbox(stg_row, from_=0.0, to=3.0, increment=0.25, width=5,
                textvariable=stg_scale_var, format="%.2f").pack(side=tk.LEFT, padx=(6, 0))
    tooltip(chk_stg,
            "Spatio-Temporal Guidance. Perturbs one transformer block and\n"
            "steers away from the degraded prediction -- aimed at duplicated\n"
            "limbs and objects not attached to the scene.\n\n"
            "Uses NO negative prompt; it steers on structure, not text.\n\n"
            "~2x time and VRAM (one extra pass per step), but keeps the 8-step\n"
            "distilled schedule -- roughly 4x cheaper than CFG quality mode.\n\n"
            "Tried here and it visibly helped at strength 1.0. The box beside\n"
            "it is the strength; drop to 0.5 if output looks over-sharpened or\n"
            "contrasty rather than better formed.")
    tooltip(chk_cfg,
            "Runs the transformer twice per step and pushes away from the\n"
            "negative prompt -- the standard way to force prompt adherence.\n\n"
            "~7-8x the compute (30 steps vs 8, doubled per step) and roughly\n"
            "double the activation VRAM. Applied to stage 1 only; the stage-2\n"
            "refinement stays guidance-free to limit the cost.\n\n"
            "Intended for a bigger card. On 16GB, single-stage only.\n"
            "Steps and strength: cfg_steps / cfg_scale in ltx2_config.json.")


    def on_res_select(event):
        val = res_combo.get()
        entry_w.config(state="normal"); entry_h.config(state="normal")
        if val != "Custom":
            w, h = val.split(" ")[0].split("x")
            entry_w.delete(0, tk.END); entry_w.insert(0, w)
            entry_h.delete(0, tk.END); entry_h.insert(0, h)
            entry_w.config(state="disabled"); entry_h.config(state="disabled")
        update_output_label()

    res_combo.bind("<<ComboboxSelected>>", on_res_select)
    # Custom W/H are typed, so follow keystrokes too.
    entry_w.bind("<KeyRelease>", update_output_label, add="+")
    entry_h.bind("<KeyRelease>", update_output_label, add="+")
    if f"{config['width']}x{config['height']}" not in [v.split(" ")[0] for v in res_combo['values'][:-1]]:
        res_combo.set("Custom")
    on_res_select(None)
    
    time_frame = ttk.LabelFrame(settings_frame, text=" Timing & Length ", padding="8")
    time_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    fps_sub = ttk.Frame(time_frame)
    fps_sub.pack(anchor=tk.W)
    ttk.Label(fps_sub, text="FPS:").pack(side=tk.LEFT)
    entry_fps = ttk.Entry(fps_sub, width=6)
    entry_fps.pack(side=tk.LEFT, padx=4)
    entry_fps.insert(0, str(config['fps']))
    
    length_type = tk.StringVar(value="frames")
    entry_len = ttk.Entry(time_frame, width=8)
    base_frames = float(config['frames'])
    
    def get_safe_fps():
        try: return max(1.0, float(entry_fps.get()))
        except (ValueError, tk.TclError): return 24.0

    def on_mode_change():
        nonlocal base_frames
        fps = get_safe_fps()
        if length_type.get() == "seconds":
            try: base_frames = float(entry_len.get())
            except (ValueError, tk.TclError): pass
            entry_len.delete(0, tk.END); entry_len.insert(0, f"{base_frames/fps:.2f}".rstrip('0').rstrip('.'))
        else:
            try: base_frames = round(float(entry_len.get()) * fps)
            except (ValueError, tk.TclError): pass
            entry_len.delete(0, tk.END); entry_len.insert(0, str(int(base_frames)))

    def on_fps_typing(event=None):
        if length_type.get() == "seconds":
            fps = get_safe_fps()
            entry_len.delete(0, tk.END); entry_len.insert(0, f"{base_frames/fps:.2f}".rstrip('0').rstrip('.'))
            
    def on_len_typing(event=None):
        nonlocal base_frames
        fps = get_safe_fps()
        try:
            val = float(entry_len.get())
            base_frames = val if length_type.get() == "frames" else val * fps
        except (ValueError, tk.TclError): pass

    mode_sub = ttk.Frame(time_frame)
    mode_sub.pack(anchor=tk.W, pady=4)
    ttk.Radiobutton(mode_sub, text="Frames", variable=length_type, value="frames", command=on_mode_change).pack(side=tk.LEFT)
    ttk.Radiobutton(mode_sub, text="Seconds", variable=length_type, value="seconds", command=on_mode_change).pack(side=tk.LEFT, padx=(5,0))
    entry_len.pack(anchor=tk.W, padx=2)
    entry_len.insert(0, str(config['frames']))
    entry_fps.bind("<KeyRelease>", on_fps_typing)
    entry_len.bind("<KeyRelease>", on_len_typing)

    # Auto Duration: model picks the length, capped for VRAM safety. The cap
    # depends on resolution/upscale/guidance mode, so it's recomputed from the
    # live form values rather than a single fixed number for one card.
    auto_dur_var = tk.BooleanVar(value=config.get("auto_duration", False))

    def current_auto_cap():
        try:
            w, h = int(entry_w.get()), int(entry_h.get())
            fps = float(entry_fps.get())
        except (ValueError, tk.TclError):
            return config.get("auto_max_seconds", 5.0)
        return auto_duration_cap_s(w, h, upscale_var.get(), cfg_var.get(),
                                   stg_var.get(), fps,
                                   config.get("cfg_modality_scale", 1.0))

    def refresh_auto_cap_label(_event=None):
        chk_auto.config(text=f"Auto Duration (max {current_auto_cap():.0f}s)")

    def on_auto_toggle():
        on_auto = auto_dur_var.get()
        entry_len.config(state="disabled" if on_auto else "normal")
        if on_auto:
            refresh_auto_cap_label()
            lbl_auto_max.pack(side=tk.LEFT)
            entry_auto_max.pack(side=tk.LEFT, padx=(2, 0))
        else:
            lbl_auto_max.pack_forget()
            entry_auto_max.pack_forget()

    chk_auto = ttk.Checkbutton(
        time_frame,
        text="Auto Duration",
        variable=auto_dur_var,
        command=on_auto_toggle,
    )
    chk_auto.pack(anchor=tk.W, pady=(6, 0))
    entry_w.bind("<KeyRelease>", refresh_auto_cap_label, add="+")
    entry_h.bind("<KeyRelease>", refresh_auto_cap_label, add="+")
    entry_fps.bind("<KeyRelease>", refresh_auto_cap_label, add="+")
    chk_upscale.config(command=lambda: (update_output_label(), refresh_auto_cap_label()))
    chk_cfg.config(command=lambda: (on_cfg_toggle(), refresh_auto_cap_label()))
    chk_stg.config(command=lambda: (on_stg_toggle(), refresh_auto_cap_label()))

    auto_row = ttk.Frame(time_frame)
    auto_row.pack(anchor=tk.W, pady=(2, 0))
    lbl_auto_max = ttk.Label(auto_row, text="max s:")
    entry_auto_max = ttk.Entry(auto_row, width=4)
    entry_auto_max.insert(0, str(config.get("auto_max_seconds", 5.0)))
    refresh_auto_cap_label()
    on_auto_toggle()
    
    seed_frame = ttk.Frame(main_frame)
    seed_frame.pack(fill=tk.X, pady=(0, 12))
    ttk.Label(seed_frame, text="Seed ('r' for Random):", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
    entry_seed = ttk.Entry(seed_frame, width=15)
    entry_seed.pack(side=tk.LEFT, padx=10)
    tooltip(entry_seed,
            "A number for a reproducible render, or 'r' for a random one.\n"
            "The seed used is written into the output filename.")
    entry_seed.insert(0, str(config['seed']))

    progress_var = tk.IntVar(value=0)
    progress_bar = ttk.Progressbar(main_frame, variable=progress_var, maximum=8, mode='determinate')
    progress_bar.pack(fill=tk.X, pady=(0, 8))
    
    # `state="disabled"` keeps the log read-only, but it also stops the widget
    # taking focus -- so Ctrl+C never reaches it. takefocus + a click-to-focus
    # binding restore copying without making the log editable.
    log_text = scrolledtext.ScrolledText(main_frame, height=10, state="disabled", bg="#1e1e1e",
                                         fg="#00ff00", font=("Consolas", 9), takefocus=True)
    log_text.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

    def copy_log(_event=None):
        try:
            sel = log_text.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            sel = log_text.get("1.0", tk.END)   # nothing selected -> copy it all
        sel = sel.strip()
        if sel:
            root.clipboard_clear()
            root.clipboard_append(sel)
        return "break"

    def select_all_log(_event=None):
        log_text.tag_add(tk.SEL, "1.0", tk.END)
        return "break"

    log_text.bind("<Button-1>", lambda e: log_text.focus_set())
    log_text.bind("<Control-c>", copy_log)
    log_text.bind("<Control-C>", copy_log)
    log_text.bind("<Control-a>", select_all_log)
    log_text.bind("<Control-A>", select_all_log)

    log_menu = tk.Menu(log_text, tearoff=0)
    log_menu.add_command(label="Copy  (Ctrl+C)", command=copy_log)
    log_menu.add_command(label="Select All  (Ctrl+A)", command=select_all_log)
    log_text.bind("<Button-3>", lambda e: log_menu.tk_popup(e.x_root, e.y_root))


    sys.stdout = TextRedirector(log_text)
    sys.stderr = TextRedirector(log_text)

    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill=tk.X)
    
    def start_generation():
        eng.cancel_flag = False 
        
        try:
            p = text_prompt.get("1.0", tk.END).strip()
            if not p:
                messagebox.showerror("Error", "Positive prompt cannot be empty.")
                return
                
            if mode_var.get() == "image2video":
                img = image_path_var.get().strip()
                if not img or not os.path.exists(img):
                    messagebox.showerror("Error", "Image-to-Video mode needs a valid image. Click 'Choose Image...'.")
                    return

            # Store exactly what's in the box, empty included. Substituting the
            # library default here used to get saved straight back to the config,
            # so clearing the box never stuck -- it refilled on the next launch.
            # Nothing is lost: the distilled schedule is guidance-free, so the
            # negative branch is never evaluated. Use "Reset Default" to load
            # the stock text if you ever want to see or edit it.
            np_val = text_np.get("1.0", tk.END).strip()

            # The VAE compresses 32:1 spatially, so width/height must be
            # multiples of 32; anything else is snapped. Say so rather than
            # silently rewriting what was typed -- "I asked for 810 and got
            # 800" is otherwise invisible until you inspect the output file.
            w_raw, h_raw = int(entry_w.get()), int(entry_h.get())
            w_adj, h_adj = snap_dimension(w_raw), snap_dimension(h_raw)
            if (w_adj, h_adj) != (w_raw, h_raw):
                print(f"  [!] Resolution {w_raw}x{h_raw} -> {w_adj}x{h_adj} "
                      f"(must be multiples of {SPATIAL_COMPRESSION}, min {MIN_DIMENSION}).")
                entry_w.delete(0, tk.END); entry_w.insert(0, str(w_adj))
                entry_h.delete(0, tk.END); entry_h.insert(0, str(h_adj))

            fps = float(entry_fps.get())
            val = float(entry_len.get())
            target_frames = int(val * fps) if length_type.get() == "seconds" else int(val)
            # Same idea temporally: the VAE is 8:1, hence the 8k+1 frame rule.
            aligned_frames = align_frames(target_frames)
            if aligned_frames != target_frames:
                print(f"  [!] Frames {target_frames} -> {aligned_frames} (8k+1 rule).")
            
            s_val = entry_seed.get().strip().lower()
            active_seed = random.randint(0, 2**32 - 1) if s_val == 'r' else int(s_val)

            auto_cap = auto_duration_cap_s(w_adj, h_adj, upscale_var.get(), cfg_var.get(),
                                          stg_var.get(), fps,
                                          config.get("cfg_modality_scale", 1.0))
            try:
                auto_max_val = min(float(entry_auto_max.get()), auto_cap)
            except ValueError:
                auto_max_val = auto_cap

            # VRAM sanity check. Transformer sequence length is
            # latent_frames * (H/32) * (W/32), and the threshold now scales with
            # whatever card is present -- see token_warn_threshold().
            scale = 2 if upscale_var.get() else 1  # for the dialog text below
            check_frames = int(auto_max_val * fps) if auto_dur_var.get() else aligned_frames
            tokens = latent_tokens(w_adj, h_adj, check_frames, upscale_var.get())
            threshold = token_warn_threshold(config)
            # Each extra guidance pass (CFG, STG, modality) is close to
            # another full forward call, so its activation cost scales the
            # same way -- multiply the token count by the pass count rather
            # than a hardcoded x2. This used to only ever double for CFG, so
            # STG alone (2 passes) was compared as if it were the 1-pass
            # baseline, and CFG+STG (3 passes) was under-counted as 2.
            # cfg_modality_scale has no GUI control, so read it from the last
            # loaded/saved config -- same as everywhere else it's used.
            passes = guidance_pass_count(cfg_var.get(), stg_var.get(),
                                         config.get("cfg_modality_scale", 1.0))
            eff_tokens = tokens * passes
            if eff_tokens > threshold:
                _, _, vram_total = hw_monitor.get_gpu_stats()
                est = VRAM_BASE_GB + VRAM_GB_PER_TOKEN * eff_tokens
                card = f"{vram_total:.1f}GB card" if vram_total else "this card"
                cfg_note = (f"\n\n{passes} transformer passes per step (CFG"
                            f"{'+STG' if stg_var.get() else ''}"
                            f"{'+modality' if config.get('cfg_modality_scale', 1.0) > 1.0 else ''}"
                            f"), roughly {passes}x the compute of a plain run."
                            ) if cfg_var.get() else (
                            "\n\n2 transformer passes per step (STG), roughly "
                            "2x the compute of a plain run.") if stg_var.get() else ""
                if not messagebox.askokcancel(
                    "Large sequence",
                    f"Final stage would run {tokens:,} latent tokens "
                    f"({w_adj*scale}x{h_adj*scale}, {check_frames} frames"
                    f"{' worst-case under Auto Duration' if auto_dur_var.get() else ''}).\n\n"
                    f"Estimated peak VRAM: ~{est:.1f}GB on a {card}.\n"
                    f"Warning threshold for this GPU: {threshold:,} tokens.{cfg_note}\n\n"
                    "Beyond this you risk exhausting VRAM or tripping the GPU "
                    "driver's timeout watchdog.\n\n"
                    "Continue anyway?",
                ):
                    return


            config.update({
                'prompt': p, 'negative_prompt': np_val,
                'width': w_adj, 'height': h_adj, 'fps': fps, 'frames': aligned_frames, 'seed': s_val,
                'active_seed': active_seed,
                'upscale': upscale_var.get(),
                'mode': mode_var.get(),
                'image_path': image_path_var.get().strip(),
                'auto_duration': auto_dur_var.get(),
                'cfg_mode': cfg_var.get(),
                'stg_mode': stg_var.get(),
                'stg_scale': float(stg_scale_var.get() or 1.0),
                'enhance_max_words': max(0, int(enh_words_var.get() or 0)),
                'auto_max_seconds': auto_max_val,
            })
            save_config(config)
            
            btn_generate.config(state="disabled")
            btn_cancel.config(state="normal")
            progress_var.set(0)

            # Unload Models and Enhance Now both touch the GPU/pipeline and
            # neither checks whether a generation is in flight. Unload would
            # free_resident_models() out from under the worker's live
            # references and pinned buffers mid-transfer; Enhance Now would
            # launch the ~10GB enhancer onto a GPU already busy denoising,
            # risking an OOM on a 16GB card. Disable both for the duration.
            btn_free.config(state="disabled")
            btn_enhance.config(state="disabled")

            def run_and_reenable(*worker_args):
                try:
                    generation_worker(*worker_args)
                finally:
                    root.after(0, lambda: (btn_free.config(state="normal"),
                                           btn_enhance.config(state="normal")))

            thread = threading.Thread(
                target=run_and_reenable,
                args=(config, root, progress_var, progress_bar, btn_generate, btn_cancel),
            )
            thread.daemon = True
            thread.start()
            
        except ValueError:
            messagebox.showerror("Error", "Please ensure numbers are valid.")
            
    def cancel_generation():
        eng.cancel_flag = True
        btn_cancel.config(state="disabled")
        print("\n[!] Cancelling... waiting for current step to yield.")

    # Utility controls (Debug) on the left; primary actions (Cancel, Generate)
    # on the right, Generate flush against the right edge -- the "submit
    # button bottom-right" convention users expect. Unload Models lives next
    # to Generation Mode instead, away from this row entirely.

    def on_debug_toggle():
        eng.debug_flag = debug_var.get()
        # diffusers is muted to `error` in the worker; follow the checkbox so its
        # own warnings (offload, attention backend, tiling) show up too.
        set_diffusers_verbosity(eng.debug_flag)
        print(f"[dbg] debug output {'ON' if eng.debug_flag else 'OFF'}")

    debug_var = tk.BooleanVar(value=False)
    chk_debug = ttk.Checkbutton(btn_frame, text="🐞 Debug", variable=debug_var,
                                command=on_debug_toggle)
    chk_debug.pack(side=tk.LEFT, padx=(0, 8))

    btn_generate = ttk.Button(btn_frame, text="🚀 Generate Video", command=start_generation)
    btn_generate.pack(side=tk.RIGHT, fill=tk.X, expand=True, ipady=8, padx=(4, 0))
    tooltip(btn_generate,
            "Render with the settings above. Progress streams into the log.\n\n"
            "First run of a session reloads ~18GB from disk; later runs reuse\n"
            "the resident pipeline. A warning appears first if the sequence is\n"
            "large enough to risk exhausting VRAM.")

    btn_cancel = ttk.Button(btn_frame, text="🛑 Cancel", command=cancel_generation, state="disabled")
    btn_cancel.pack(side=tk.RIGHT, fill=tk.X, expand=True, ipady=8, padx=(4, 4))
    tooltip(btn_cancel,
            "Stop cleanly at the end of the current diffusion step.\n"
            "Models stay resident, so the next run starts immediately.")
    tooltip(chk_debug,
            "Per-step timing, latent geometry, token count and a VRAM/RAM\n"
            "reading on every line, plus diffusers' own offload and tiling\n"
            "warnings.\n\nSafe to toggle mid-run: the next step picks it up,\n"
            "so you never restart and reload 18GB to diagnose something.")

    root.update_idletasks()
    # Floor the window at whatever the packed layout actually needs, measured
    # rather than hardcoded -- font metrics and CPU core count (the telemetry
    # grid wraps at 8 per row) both change the requirement per machine.
    root.minsize(max(640, root.winfo_reqwidth()), min(root.winfo_reqheight(),
                                                      root.winfo_screenheight()))
    x = (root.winfo_screenwidth() // 2) - (WIN_W // 2)
    y = (root.winfo_screenheight() // 2) - (WIN_H // 2)
    root.geometry(f"+{x}+{y}")
    
    def on_close():
        """Leave immediately, without running Python's finalisers.

        Closing the window used to return from mainloop() and let the
        interpreter finalise. With a resident pipeline, pinned host buffers and
        live HIP streams still alive, it unwinds in an order the ROCm runtime
        does not survive -- observed as:

            segfault at 7f1789f16760 ip 00007f1789f16760 error 15

        ip == fault address: a call through a function pointer into a library
        that had already been unloaded. `os._exit()` skips the unwind entirely,
        which is the fix.

        Note what this deliberately does NOT do: free the models, synchronize,
        or empty any cache. The kernel reclaims all of it -- host, pinned and
        VRAM -- when the process dies, and the amdgpu driver tears down the
        context when the fd closes. Doing it by hand first only delays the
        window disappearing, which is what made closing feel slow.
        """
        if btn_cancel["state"] == "normal":     # a generation is in flight
            if not messagebox.askokcancel(
                    "Quit", "A generation is still running.\n\n"
                            "Quitting now abandons it. Continue?"):
                return
        # os._exit skips atexit and buffer flushing, so flush the real streams
        # (not the log widget's redirect) or trailing output is lost.
        for stream in (sys.__stdout__, sys.__stderr__):
            try:
                stream.flush()
            except Exception:
                pass
        # Also skipped with atexit: multiprocessing's own cleanup, which reaps
        # child processes and unlinks the semaphores they registered. Without
        # it the resource_tracker complains on the way out:
        #     UserWarning: resource_tracker: There appear to be 1 leaked
        #     semaphore objects to clean up at shutdown
        # This is only mp bookkeeping -- no model or GPU teardown -- so it stays
        # cheap and closing stays instant.
        try:
            import multiprocessing.util
            multiprocessing.util._exit_function()
        except Exception:
            pass
        os._exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)

    print("Welcome to LTX-2.5 Control Panel.")
    print("System ready. Modify settings above and click Generate Video.")

    root.mainloop()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
