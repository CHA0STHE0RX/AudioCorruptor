import os
import tempfile
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import scipy.ndimage
import soundfile as sf
import moviepy.editor as mp
from pedalboard import Pedalboard, Reverb, Chorus, PitchShift, Distortion, LowpassFilter, HighpassFilter, Bitcrush
#initial
class AudioCorruptor:
    def __init__(self, root):
        self.root = root
        self.root.title("AudioCorruptor")
        self.root.geometry("650x550")
        self.filepath = None
        self.setup_ui()
        self.apply_theme()

    def setup_ui(self):
        # - Theme selector -
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill="x", pady=5, padx=10)
        tk.Label(top_frame, text="Theme:").pack(side="left")
        
        self.theme_var = tk.StringVar(value="Dark")
        self.theme_menu = ttk.Combobox(top_frame, textvariable=self.theme_var, values=["Dark", "Light", "Purple"], state="readonly", width=10)
        self.theme_menu.pack(side="left", padx=5)
        self.theme_menu.bind("<<ComboboxSelected>>", self.apply_theme)

        # - File selection -
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=10)
        
        self.btn_select = tk.Button(file_frame, text="Browse MP4...", command=self.browse_file)
        self.btn_select.pack()
        self.lbl_file = tk.Label(file_frame, text="No file loaded.")
        self.lbl_file.pack()
        
        # - Sliders -
        slider_frame = tk.Frame(self.root)
        slider_frame.pack(pady=10, fill="both", expand=True)
        left_col = tk.Frame(slider_frame)
        left_col.pack(side="left", fill="both", expand=True, padx=10)
        right_col = tk.Frame(slider_frame)
        right_col.pack(side="right", fill="both", expand=True, padx=10)

        self.params = {}
        def make_knob(parent, name, label, min_v, max_v, res, default):
            var = tk.DoubleVar(value=default)
            s = tk.Scale(parent, from_=min_v, to=max_v, resolution=res, variable=var, orient="horizontal", length=250, label=label)
            s.pack(pady=2)
            self.params[name] = var
            return s
        # make_knob(column, "dictionary_name", "label", min_val, max_val, step_size, default_val)
        # Left column
        make_knob(left_col, "skip", "Stutter / Skip Chance", 0.0, 0.1, 0.01, 0.03)
        make_knob(left_col, "crossfade", "Crossfade", 0.01, 1.0, 0.01, 0.2)
        make_knob(left_col, "pitch_f", "Foreground Pitch (Semitones)", -5.0, 0.0, 0.1, -1.0)
        make_knob(left_col, "pitch_t", "Other Pitch (Semitones)", -12.0, -2.0, 0.1, -6.0)
        make_knob(left_col, "muffle", "Muffle Highs (Hz)", 200, 5000, 50, 800)
        # Right column
        make_knob(right_col, "wobble", "Wobble (Hz)", 0.1, 5.0, 0.1, 2.0)
        make_knob(right_col, "bitcrush", "Bitcrush (Lower=Worse)", 2, 16, 1, 6)
        make_knob(right_col, "reverb", "Reverb Size", 0.1, 1.0, 0.05, 1.0)
        make_knob(right_col, "static", "Background Static", 0.0, 0.05, 0.001, 0.015)
        make_knob(right_col, "distortion", "Harshness", 0.0, 30.0, 1.0, 0.5)

        # - Bottom status -
        bot_frame = tk.Frame(self.root)
        bot_frame.pack(pady=10)
        self.btn_run = tk.Button(bot_frame, text="Process Video", command=self.run_process, height=2, width=20)
        self.btn_run.pack()
        self.lbl_status = tk.Label(bot_frame, text="Ready.")
        self.lbl_status.pack(pady=5)

    def apply_theme(self, event=None):
        theme = self.theme_var.get()
        if theme == "Dark":
            colors = {"bg": "#1c1c1c", "fg": "#cccccc", "btn": "#323131", "trough": "#4A4848"}
        elif theme == "Light":
            colors = {"bg": "#f5f5f5", "fg": "#111111", "btn": "#c4baba", "trough": "#cccccc"}
        else: #Purple
            colors = {"bg": "#24152e", "fg": "#dc9ffa", "btn": "#56306e", "trough": "#45265d"}

        self.root.configure(bg=colors["bg"])
        self._style_children(self.root, colors)

    def _style_children(self, widget, colors):
        #Applies colors. Prevent -fg errors on Frames
        for child in widget.winfo_children():
            wtype = child.winfo_class()
            
            if wtype == "Frame":
                child.configure(bg=colors["bg"])
            elif wtype == "Label":
                child.configure(bg=colors["bg"], fg=colors["fg"])
            elif wtype == "Button":
                child.configure(bg=colors["btn"], fg=colors["fg"])
            elif wtype == "Scale":
                child.configure(bg=colors["bg"], fg=colors["fg"], troughcolor=colors["trough"], highlightthickness=0)
            self._style_children(child, colors)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 Video", "*.mp4")])
        if path:
            self.filepath = path
            self.lbl_file.config(text=os.path.basename(path))

    def run_process(self):
        if not self.filepath:
            return messagebox.showwarning("Wait", "Select an MP4 file first.")
        
        save_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
        if not save_path: 
            return
            
        self.btn_run.config(state="disabled")
        # Run audio/video processing in background
        threading.Thread(target=self.run, args=(self.filepath, save_path), daemon=True).start()

    def run(self, in_file, out_file):
        temp_dir = tempfile.gettempdir()
        raw_wav = os.path.join(temp_dir, "raw_temp.wav")
        fx_wav = os.path.join(temp_dir, "fx_temp.wav")
        vid_clip = new_audio_clip = None
        
        try:
            self.update_status("Extracting audio from MP4 ...")
            vid_clip = mp.VideoFileClip(in_file)
            if vid_clip.audio is None: 
                raise ValueError("No audio track found in this video.")
            vid_clip.audio.write_audiofile(raw_wav, logger=None)
            
            self.update_status("Applying effects ...")
            audio, sr = sf.read(raw_wav)
            # Pedalboard expects (channels, samples). Soundfile gives (samples, channels)
            audio = audio.T if len(audio.shape) > 1 else np.expand_dims(audio, 0)
            # Stuttering/looping
            audio_skip = self.apply_skips(audio, sr, self.params["skip"].get())
            #Symetric boards (hopefully)
            board_A = Pedalboard([
                PitchShift(semitones=self.params["pitch_f"].get()),
                Chorus(rate_hz=0.5, depth=0.3, centre_delay_ms=5.0),
                HighpassFilter(cutoff_frequency_hz=200),
                LowpassFilter(cutoff_frequency_hz=self.params["muffle"].get()),
                Distortion(drive_db=self.params["distortion"].get())
            ])
            
            board_B = Pedalboard([
                PitchShift(semitones=self.params["pitch_t"].get()),
                Chorus(rate_hz=self.params["wobble"].get(), depth=0.8, centre_delay_ms=15.0),
                Bitcrush(bit_depth=int(self.params["bitcrush"].get())),
                LowpassFilter(cutoff_frequency_hz=800),
                Reverb(room_size=self.params["reverb"].get(), damping=0.1, wet_level=1.0, dry_level=0.1)
            ])
            
            layer_A = board_A(audio_skip, sr)
            layer_B = board_B(audio_skip, sr)
            # Match lengths
            min_samples = min(layer_A.shape[1], layer_B.shape[1])
            layer_A, layer_B = layer_A[:, :min_samples], layer_B[:, :min_samples]
            # Smoothed random wave to crossfade
            fade = self.generate_fade(min_samples, sr, self.params["crossfade"].get())
            fade_mask = np.vstack((fade, fade)) # Make it stereo
            
            final_audio = (layer_A * (1.0 - fade_mask)) + (layer_B * fade_mask)
            # Global static layer
            static_lvl = self.params["static"].get()
            if static_lvl > 0:
                final_audio += np.random.normal(0, static_lvl, final_audio.shape)
                
            # Normalize
            max_val = np.max(np.abs(final_audio))
            if max_val > 1.0: final_audio /= max_val
                
            sf.write(fx_wav, final_audio.T, sr)
            # Audio back into video
            self.update_status("Audio to video ...")
            new_audio_clip = mp.AudioFileClip(fx_wav)
            
            # Because the skipping duplicates chunks, the audio is now longer than the video
            # Cut it back down to video length
            new_audio_clip = new_audio_clip.set_duration(vid_clip.duration)
            vid_clip.set_audio(new_audio_clip).write_videofile(out_file, audio_codec="aac")
            
            self.update_status("Done!")
            messagebox.showinfo("Success", f"Saved to:\n{out_file}")
            
        except Exception as e:
            traceback.print_exc()
            self.update_status("Error occurred.")
            messagebox.showerror("Error", str(e))
        finally:
            # Forcefully closing them
            if vid_clip: vid_clip.close()
            if new_audio_clip: new_audio_clip.close()
            for f in (raw_wav, fx_wav):
                if os.path.exists(f): 
                    try: os.remove(f)
                    except: pass
            self.root.after(0, lambda: self.btn_run.config(state="normal"))
    def update_status(self, msg):
        self.root.after(0, lambda: self.lbl_status.config(text=msg))
    def apply_skips(self, audio, sr, skip_prob):
        #Repeat random 1-second chunk 2-4 times.
        out_chunks = []
        samples = audio.shape[1]
        i = 0
        chunk_size = sr * 2 # Evaluate a potential skip every ~ 2 seconds
        while i < samples:
            current_chunk = min(chunk_size, samples - i)
            # Skips more likely to happen towards the end
            prob = skip_prob + ((i / samples) * 0.1)
            
            if np.random.rand() < prob and current_chunk == chunk_size:
                loop_len = int(sr * np.random.uniform(0.6, 1.8))
                phrase = audio[:, i:i+loop_len]
                
                # Loop the phrase 2 to 4 times
                for _ in range(np.random.randint(2, 5)):
                    pop = np.random.normal(0, 0.5, phrase.shape) * np.exp(-np.linspace(0, 50, phrase.shape[1]))
                    out_chunks.append(phrase + (pop * 0.2))
                i += loop_len 
            else:
                out_chunks.append(audio[:, i:i+current_chunk])
                i += current_chunk
        return np.concatenate(out_chunks, axis=1)

    def generate_fade(self, total_samples, sr, speed):
        #  0.0 - 1.0 used to crossfade
        # Create random noise then apply gaussian smoothing
        noise = np.random.rand(int(total_samples / 1000))
        wave = scipy.ndimage.gaussian_filter1d(noise, sigma=sr * speed)
        wave = (wave - np.min(wave)) / np.ptp(wave)
        drift = np.interp(np.linspace(0, 1, total_samples), np.linspace(0, 1, len(wave)), wave)
        # Decay gets worse towards the end
        return np.clip(drift + np.linspace(0, 0.7, total_samples), 0.0, 1.0)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioCorruptor(root)
    root.mainloop()
