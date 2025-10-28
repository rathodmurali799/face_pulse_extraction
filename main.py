import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
from collections import deque
import pandas as pd
from scipy.signal import butter, lfilter
import time
import mediapipe as mp
import matplotlib.pyplot as plt

# --- Global Variables ---
cap = cv2.VideoCapture(0)
running = False
video_cap = None
video_filename = None
video_start_time = None
video_duration = None  # in seconds
frame_counter = 0

cg_signal1, cg_signal2 = [], []
MAX_SIGNAL_LENGTH = 600  # keep ~20 seconds of data at 30 FPS

x1_hist, y1_hist, x2_hist, y2_hist = deque(maxlen=50), deque(maxlen=50), deque(maxlen=50), deque(maxlen=50)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,  # detect weaker faces
    min_tracking_confidence=0.5
)


root = tk.Tk()
root.title("Face Pulse Extraction")
root.geometry("1200x700+50+20")# 780 x 580
root.resizable(True, True)

# --- Functions ---
def bgr_to_ycgco(bgr_img):
    bgr = bgr_img.astype(np.float32) / 255.0
    B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    Y  = 0.25 * R + 0.5 * G + 0.25 * B
    Cg = -0.25 * R + 0.5 * G - 0.25 * B
    Co = 0.5 * R - 0.5 * B
    return Y, Cg, Co

def upload_file():
    global video_filename
    filename = filedialog.askopenfilename(
        title="Select a Video",
        filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")]
    )
    if filename:
        video_filename = filename

def start():
    global running, video_cap, cap, video_filename, video_start_time, video_duration
    video_start_time = time.time()
    running = True
    if video_filename:
        video_cap = cv2.VideoCapture(video_filename)
        if not video_cap.isOpened():
            print("Error: Cannot open video file")
            video_cap = None
            return
        try:
            fps = video_cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            video_duration = frame_count / (fps if fps > 0 else 30.0)
        except Exception:
            video_duration = None
    else:
        video_duration = None
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)

    # clear old signals
    cg_signal1.clear()
    cg_signal2.clear()
    x1_hist.clear()
    y1_hist.clear()
    x2_hist.clear()
    y2_hist.clear()

    update_frame()

def pause():
    global running
    running = False

def resume():
    global running
    running = True
    update_frame()

def quit_app():
    global running
    running = False
    if cap.isOpened():
        cap.release()
    if video_cap is not None:
        video_cap.release()
    root.destroy()

def webcam():
    global running, cap, video_cap, video_filename, video_start_time
    video_start_time = time.time()
    running = True
    
    if video_cap is not None:
        video_cap.release()
        video_cap = None
    video_filename = None
    cap = cv2.VideoCapture(0)
    update_frame()

# --- Signal Processing ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.7, highcut=3.0, fs=30, order=3):
    if len(data) < order * 3:
        return np.array(data)
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

def smooth_signal(signal, window=8):
    if len(signal) < window:
        return np.array(signal)
    smoothed = np.convolve(signal, np.ones(window)/window, mode='valid')
    pad_len = len(signal) - len(smoothed)
    if pad_len > 0:
        smoothed = np.pad(smoothed, (pad_len,0), 'edge')
    return smoothed

def smooth_position(hist, alpha=0.2):
    if len(hist) == 0:
        return 0
    smoothed = hist[0]
    for v in list(hist)[1:]:
        smoothed = alpha*v + (1-alpha)*smoothed
    return int(smoothed)

# --- Draw Axes ---
def draw_axes():
    wave_canvas.delete("all")
    root.update_idletasks()
    W = wave_canvas.winfo_width()
    H = wave_canvas.winfo_height()

    global margin_left, margin_right, margin_bottom, margin_top, plot_w, plot_h, y_center
    margin_left, margin_right, margin_bottom, margin_top = 80, 20, 40, 40
    plot_w = max(0, W - margin_left - margin_right)
    plot_h = max(0, H - margin_top - margin_bottom)
    y_center = margin_top + plot_h / 2

    wave_canvas.create_line(margin_left, margin_top, margin_left, margin_top + plot_h, fill="white", width=2)
    wave_canvas.create_line(margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h, fill="white", width=2)

    y_values = [-0.1025, -0.1000, -0.0975, -0.0950, -0.0925, -0.0900, -0.0875, -0.0850, -0.0825]
    for y_val in y_values:
        y_norm = (y_val - np.min(y_values)) / (np.ptp(y_values) + 1e-6)
        y_canvas = margin_top + (1 - y_norm) * plot_h
        wave_canvas.create_line(margin_left - 5, y_canvas, margin_left + 5, y_canvas, fill="white")
        wave_canvas.create_text(margin_left - 10, y_canvas, text=f"{y_val:.4f}", fill="white", anchor="e", font=("Arial", 10))

    x_ticks = list(range(0, 801, 100))
    max_tick = max(x_ticks)
    for xt in x_ticks:
        x_canvas = margin_left + (xt / max_tick) * plot_w
        wave_canvas.create_line(x_canvas, H - margin_bottom - 5, x_canvas, H - margin_bottom + 5, fill="white")
        wave_canvas.create_text(x_canvas, margin_top + plot_h + 12, text=str(xt), fill="white", anchor="n")

    wave_canvas.create_text(W/2, 15, text="Pulse Signal", fill="white", font=("Arial", 14, "bold"))
    wave_canvas.create_text(W/2, H-10, text="Time", fill="white", font=("Arial", 11))
    wave_canvas.create_text(15, H/2, text="Signal (Cg)", fill="white", font=("Arial", 11), angle=90)

def update_frame():
    global running, video_cap, cap, cg_signal1, cg_signal2, frame_counter, video_start_time

    if not running:
        return

    frame_counter += 1

    frame = None
    ret = False
    if video_cap is not None:
        ret, frame = video_cap.read()
    elif cap.isOpened():
        ret, frame = cap.read()

    if ret and frame is not None:
        h = frame.shape[0]     
        y_start = 0
        y_end = int(h * 0.7)   
        frame = frame[y_start:y_end, :]

    if not ret or frame is None:
        running = False
        print("Video finished or camera stopped.")
        return

    elapsed_time = time.time() - (video_start_time or time.time())

    if video_duration is not None and elapsed_time >= video_duration:
        running = False
        print("Video finished.")
        if video_cap is not None:
            video_cap.release()
        return

    frame = cv2.resize(frame, (400, 320), interpolation=cv2.INTER_CUBIC)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    lx, ly, rx, ry = None, None, None, None

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        left_indices = [234, 93, 115, 348]
        right_indices = [454, 323, 360, 453]

        for idx in left_indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        for idx in right_indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        lx = int(np.mean([landmarks[i].x for i in left_indices]) * w)
        ly = int(np.mean([landmarks[i].y for i in left_indices]) * h)
        rx = int(np.mean([landmarks[i].x for i in right_indices]) * w)
        ry = int(np.mean([landmarks[i].y for i in right_indices]) * h)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w_, h_ = faces[0]
            lx, ly = x + w_ // 4, y + h_ // 3
            rx, ry = x + 3 * w_ // 4, y + h_ // 3
            cv2.rectangle(frame, (x, y), (x + w_, y + h_), (255, 0, 0), 2)
        else:
            lx, ly, rx, ry = 100, 100, 200, 100
            
    x1_hist.append(lx)
    y1_hist.append(ly)
    x2_hist.append(rx)
    y2_hist.append(ry)
    lx_smooth = smooth_position(x1_hist, alpha=0.03)
    ly_smooth = smooth_position(y1_hist, alpha=0.03)
    rx_smooth = smooth_position(x2_hist, alpha=0.03)
    ry_smooth = smooth_position(y2_hist, alpha=0.03)

    roi_size = 50
    h_f, w_f, _ = frame.shape
    x1a = max(0, lx_smooth - roi_size // 2)
    y1a = max(0, ly_smooth - roi_size // 2)
    x1b = min(w_f, lx_smooth + roi_size // 2)
    y1b = min(h_f, ly_smooth + roi_size // 2)

    x2a = max(0, rx_smooth - roi_size // 2)
    y2a = max(0, ry_smooth - roi_size // 2)
    x2b = min(w_f, rx_smooth + roi_size // 2)
    y2b = min(h_f, ry_smooth + roi_size // 2)

    roi1 = frame[y1a:y1b, x1a:x1b]
    roi2 = frame[y2a:y2b, x2a:x2b]

    if roi1.size > 0 and roi2.size > 0:
        _, cg1, _ = bgr_to_ycgco(roi1)
        _, cg2, _ = bgr_to_ycgco(roi2)
        cg_signal1.append(np.mean(cg1))
        cg_signal2.append(np.mean(cg2))
        if len(cg_signal1) > MAX_SIGNAL_LENGTH:
            cg_signal1 = cg_signal1[-MAX_SIGNAL_LENGTH:]
            cg_signal2 = cg_signal2[-MAX_SIGNAL_LENGTH:]

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    webcam_label.imgtk = imgtk
    webcam_label.configure(image=imgtk)

    redraw_every = 1
    if len(cg_signal1) > 10 and (frame_counter % redraw_every == 0):
        raw1 = np.array(cg_signal1[-200:])
        raw2 = np.array(cg_signal2[-200:])
        fs = 30
        if video_cap is not None:
            fps = video_cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                fs = fps
                
        sig1 = bandpass_filter(raw1, 0.7, 3.0, fs=fs, order=3)
        sig2 = bandpass_filter(raw2, 0.7, 3.0, fs=fs, order=3)
        sig1 = smooth_signal(sig1, window=3)
        sig2 = smooth_signal(sig2, window=3)
        sig1 = sig1 - np.mean(sig1)
        sig2 = sig2 - np.mean(sig2)
        
        if np.std(sig1) > 1e-6:
            sig1 = (sig1 - np.mean(sig1)) / np.std(sig1)
        if np.std(sig2) > 1e-6:
            sig2 = (sig2 - np.mean(sig2)) / np.std(sig2)
        
        wave_canvas.delete("wave")

        W = wave_canvas.winfo_width()
        H = wave_canvas.winfo_height()
        margin_left, margin_right, margin_top, margin_bottom = 80, 20, 40, 40
        plot_w = W - margin_left - margin_right
        plot_h = H - margin_top - margin_bottom
        y_center = margin_top + plot_h / 2
        scale = 30

        N = len(sig1)
        if N > 1:
            for i in range(1, N):
                x1 = margin_left + (i - 1) / (N - 1) * plot_w
                x2 = margin_left + i / (N - 1) * plot_w

                y1a = y_center - sig1[i - 1] * scale
                y2a = y_center - sig1[i] * scale
                wave_canvas.create_line(x1, y1a, x2, y2a, fill="blue", width=2, tags="wave")

                y1b = y_center - sig2[i - 1] * scale + 10
                y2b = y_center - sig2[i] * scale + 10
                wave_canvas.create_line(x1, y1b, x2, y2b, fill="red", width=2, tags="wave")

        elapsed_time = time.time() - (video_start_time or time.time())
        wave_canvas.create_text(W - 60, 25, text=f"{elapsed_time:.1f}s", fill="yellow", font=("Arial", 12), tags="wave")
    
    delay = int(1000 / 30)  # 30 FPS
    if running:
        webcam_label.after(delay, update_frame)


# --- Save & Analyze Function ---
def save_and_analyze():
    import os

    if len(cg_signal1) == 0 or len(cg_signal2) == 0:
        print("No signals to save yet.")
        return

    # Save CSV file
    df = pd.DataFrame({'roi1_cg': cg_signal1, 'roi2_cg': cg_signal2})
    csv_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"Cg signals saved to '{csv_path}'")

    # Convert to numpy arrays
    cg_1 = np.array(df['roi1_cg'])
    cg_2 = np.array(df['roi2_cg'])
    duration = max(1, len(cg_1)/30)
    fs = len(cg_1)/duration

    # --- Plot raw signals ---
    t_vals = np.arange(len(cg_1))
    plt.figure(figsize=(10,4))
    plt.plot(t_vals, cg_1, label='Cg_1', color='blue')
    plt.plot(t_vals, cg_2, label='Cg_2', color='red')
    plt.xlabel("Frame")
    plt.ylabel("Cg")
    plt.title("Raw Signals")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ✅ Save raw signal plot as image
    raw_plot_path = os.path.splitext(csv_path)[0] + "_raw_signals.png"
    plt.savefig(raw_plot_path)
    print(f"Raw signals graph saved as '{raw_plot_path}'")
    plt.show()

    # --- Extract pulse waves ---
    def extract_pulse_wave(signal, fs, f_min=0.67, f_max=3.34):
        N = len(signal)
        freqs = np.fft.fftfreq(N, 1/fs)
        X_f = np.fft.fft(signal)
        X_f_zero = np.zeros_like(X_f, dtype=complex)
        mask = (np.abs(freqs) >= f_min) & (np.abs(freqs) <= f_max)
        X_f_zero[mask] = X_f[mask]
        pulse_wave = np.fft.ifft(X_f_zero).real
        return pulse_wave

    pulse_wave_1 = extract_pulse_wave(cg_1, fs)
    pulse_wave_2 = extract_pulse_wave(cg_2, fs)

    # --- Plot filtered pulse waves ---
    t_vals_1 = np.arange(len(pulse_wave_1))
    plt.figure(figsize=(10,4))
    plt.plot(t_vals_1, pulse_wave_1, label='Cg_1', color='blue')
    plt.plot(t_vals_1, pulse_wave_2, label='Cg_2', color='red')
    plt.xlabel("Frame")
    plt.ylabel("Cg (Filtered)")
    plt.title("Pulse Waves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ✅ Save pulse wave plot as image
    pulse_plot_path = os.path.splitext(csv_path)[0] + "_pulse_waves.png"
    plt.savefig(pulse_plot_path)
    print(f"Pulse wave graph saved as '{pulse_plot_path}'")
    plt.show()

    # --- Frequency analysis ---
    def compute_fmax_peak_and_phase(signal, fs):
        n = len(signal)
        freqs = np.fft.fftfreq(n, 1/fs)
        fft_vals = np.fft.fft(signal)
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        fft_vals = fft_vals[pos_mask]
        idx_peak = np.argmax(np.abs(fft_vals))
        f_max_peak = freqs[idx_peak]
        phase = np.angle(fft_vals[idx_peak])
        return f_max_peak, phase

    f1, phase1 = compute_fmax_peak_and_phase(pulse_wave_1, fs)
    f2, phase2 = compute_fmax_peak_and_phase(pulse_wave_2, fs)

    # Compute phase difference
    phase_diff = abs(phase1 - phase2)
    if phase_diff > np.pi:
        theta_d = 2*np.pi - phase_diff
    else:
        theta_d = phase_diff

    print(f"f_max_peak1 = {f1:.3f} Hz, f_max_peak2 = {f2:.3f} Hz")
    print(f"Phase difference θ_d = {theta_d:.3f} rad")

    # Compute Pulse Transit Time (PTT)
    if theta_d > np.pi:
        PTT = (2*np.pi - theta_d) / (2*np.pi) * (1 / f1)
    else:
        PTT = theta_d / (2*np.pi) * (1 / f1)

    print(f"Pulse Transit Time (PTT) = {PTT:.3f} seconds")


# --- GUI Setup ---
Menu = tk.Frame(root, bg="#2c3e50", width=150)
Menu.pack(side=tk.LEFT, fill=tk.Y)
tk.Label(Menu, text="Menu", bg="#2c3e50", fg="white", font=("Arial", 14, "bold")).pack(pady=20)
tk.Button(Menu, text="Upload", width=15, command=upload_file).pack(pady=10)
tk.Button(Menu, text="Start", width=15, command=start).pack(pady=10)
tk.Button(Menu, text="Pause", width=15, command=pause).pack(pady=10)
tk.Button(Menu, text="Resume", width=15, command=resume).pack(pady=10)
tk.Button(Menu, text="Webcam", width=15, command=webcam).pack(pady=10)
tk.Button(Menu, text="Save & Analyze", width=15, command=save_and_analyze).pack(pady=10)
tk.Button(Menu, text="Quit", width=15, command=quit_app).pack(pady=10)

# Video display
webcam_label = tk.Label(root, bg="white", width=400, height=320)
webcam_label.pack(anchor="nw", padx=20, pady=(5,0))

# Waveform canvas
plot_frame = tk.Frame(root, bg="black", width=0, height=0)
plot_frame.place(x=0, rely=1, relwidth=1, height=450, anchor="sw")
wave_canvas = tk.Canvas(plot_frame, bg="black")
wave_canvas.place(x=0, y=0, relwidth=1.0, relheight=1.0)
wave_canvas.bind("<Configure>", lambda e: draw_axes())

root.update()
root.mainloop()
