"""
Dog Head Orientation Labeling - Simple UI
==========================================

A simple graphical interface to label dog head orientation from DeepLabCut data files.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from pathlib import Path
import threading


class HeadOrientationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dog Head Orientation Labeler")
        self.root.geometry("700x600")
        self.root.resizable(True, True)
        
        # Set up project directories
        self.project_dir = Path(__file__).parent
        self.data_dir = self.project_dir / "data"
        self.output_dir = self.project_dir / "output"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.video_duration = tk.DoubleVar(value=9.0)
        self.likelihood_threshold = tk.DoubleVar(value=0.3)
        self.straight_threshold = tk.DoubleVar(value=0.12)
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # Title
        title_label = ttk.Label(main_frame, text="Dog Head Orientation Labeler", 
                                font=('Helvetica', 16, 'bold'))
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1
        
        # Input file selection
        ttk.Label(main_frame, text="Input Excel File:").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(main_frame, textvariable=self.input_file, width=50).grid(row=row, column=1, sticky="ew", padx=5)
        ttk.Button(main_frame, text="Browse...", command=self.browse_input).grid(row=row, column=2)
        row += 1
        
        # Output file selection
        ttk.Label(main_frame, text="Output CSV File:").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(main_frame, textvariable=self.output_file, width=50).grid(row=row, column=1, sticky="ew", padx=5)
        ttk.Button(main_frame, text="Browse...", command=self.browse_output).grid(row=row, column=2)
        row += 1
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky="ew", pady=15)
        row += 1
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        params_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        params_frame.columnconfigure(1, weight=1)
        row += 1
        
        # Video duration
        ttk.Label(params_frame, text="Video Duration (seconds):").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(params_frame, textvariable=self.video_duration, width=10).grid(row=0, column=1, sticky="w", padx=5)
        
        # Likelihood threshold
        ttk.Label(params_frame, text="Likelihood Threshold:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(params_frame, textvariable=self.likelihood_threshold, width=10).grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(params_frame, text="(0-1, frames below this are labeled ELSEWHERE)").grid(row=1, column=2, sticky="w")
        
        # Straight threshold
        ttk.Label(params_frame, text="Straight Threshold:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(params_frame, textvariable=self.straight_threshold, width=10).grid(row=2, column=1, sticky="w", padx=5)
        ttk.Label(params_frame, text="(offset ratio, determines LEFT/RIGHT vs STRAIGHT)").grid(row=2, column=2, sticky="w")
        
        # Process button
        self.process_btn = ttk.Button(main_frame, text="Process File", command=self.process_file)
        self.process_btn.grid(row=row, column=0, columnspan=3, pady=20)
        row += 1
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)
        row += 1
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(row, weight=1)
        row += 1
        
        # Results text with scrollbar
        self.results_text = tk.Text(results_frame, height=15, width=80, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="gray")
        self.status_label.grid(row=row, column=0, columnspan=3, sticky="w")
        
    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input Excel File",
            initialdir=str(self.data_dir),
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            # Auto-generate output filename in the output folder
            input_path = Path(filename)
            output_path = self.output_dir / (input_path.stem + "_labels.csv")
            self.output_file.set(str(output_path))
            
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save Output CSV File",
            initialdir=str(self.output_dir),
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.output_file.set(filename)
            
    def process_file(self):
        # Validate inputs
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input file")
            return
        if not self.output_file.get():
            messagebox.showerror("Error", "Please specify an output file")
            return
            
        # Run processing in a separate thread to keep UI responsive
        self.process_btn.config(state='disabled')
        self.progress.start()
        self.status_label.config(text="Processing...", foreground="blue")
        self.results_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self.run_processing)
        thread.start()
        
    def run_processing(self):
        try:
            results = self.label_head_orientation(
                self.input_file.get(),
                self.output_file.get(),
                self.video_duration.get(),
                self.likelihood_threshold.get(),
                self.straight_threshold.get()
            )
            self.root.after(0, lambda: self.show_results(results))
        except Exception as e:
            self.root.after(0, lambda: self.show_error(str(e)))
            
    def show_results(self, results):
        self.progress.stop()
        self.process_btn.config(state='normal')
        self.status_label.config(text="Processing complete!", foreground="green")
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "=" * 60 + "\n")
        self.results_text.insert(tk.END, "HEAD ORIENTATION RESULTS\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        self.results_text.insert(tk.END, results.to_string(index=False))
        self.results_text.insert(tk.END, "\n\n" + "=" * 60 + "\n")
        self.results_text.insert(tk.END, f"Results saved to:\n{self.output_file.get()}\n")
        
        messagebox.showinfo("Success", f"Processing complete!\nResults saved to:\n{self.output_file.get()}")
        
    def show_error(self, error_msg):
        self.progress.stop()
        self.process_btn.config(state='normal')
        self.status_label.config(text="Error occurred", foreground="red")
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"ERROR: {error_msg}")
        
        messagebox.showerror("Error", f"Processing failed:\n{error_msg}")
        
    def label_head_orientation(self, input_file, output_file, video_duration, 
                                likelihood_threshold, straight_threshold):
        """Main processing function - adapted from label_head_orientation.py"""
        
        # Load data
        df = pd.read_excel(input_file, header=None, skiprows=2)
        
        # Rename columns
        df.columns = [
            'frame', 
            'nose_tip_x', 'nose_tip_y', 'nose_tip_likelihood',
            'nose_right_x', 'nose_right_y', 'nose_right_likelihood',
            'nose_bottom_x', 'nose_bottom_y', 'nose_bottom_likelihood',
            'nose_left_x', 'nose_left_y', 'nose_left_likelihood'
        ]
        
        # Calculate FPS
        fps = len(df) / video_duration
        
        # Calculate metrics
        df['nose_midpoint_x'] = (df['nose_right_x'] + df['nose_left_x']) / 2
        df['nose_width'] = df['nose_right_x'] - df['nose_left_x']
        df['tip_offset'] = df['nose_tip_x'] - df['nose_midpoint_x']
        df['tip_offset_ratio'] = np.where(
            np.abs(df['nose_width']) > 5,
            df['tip_offset'] / df['nose_width'],
            0
        )
        df['avg_likelihood'] = (
            df['nose_tip_likelihood'] + df['nose_right_likelihood'] + 
            df['nose_left_likelihood'] + df['nose_bottom_likelihood']
        ) / 4
        
        # Classify each frame
        def classify(row):
            if row['avg_likelihood'] < likelihood_threshold:
                return "ELSEWHERE"
            if row['nose_width'] < 5 or row['nose_width'] > 200:
                return "ELSEWHERE"
            
            tip_ratio = row['tip_offset_ratio']
            if abs(tip_ratio) <= straight_threshold:
                return "STRAIGHT"
            elif tip_ratio < -straight_threshold:
                return "LEFT"
            elif tip_ratio > straight_threshold:
                return "RIGHT"
            else:
                return "STRAIGHT"
        
        df['orientation'] = df.apply(classify, axis=1)
        
        # Aggregate by second
        df['second'] = (df['frame'] / fps).astype(int)
        
        results = []
        for second, group in df.groupby('second'):
            orientation_counts = group['orientation'].value_counts()
            dominant_orientation = orientation_counts.index[0]
            confidence = orientation_counts.iloc[0] / len(group)
            
            results.append({
                'second': int(second),
                'orientation': dominant_orientation,
                'confidence': round(confidence, 3),
                'avg_likelihood': round(group['avg_likelihood'].mean(), 3),
                'avg_tip_offset_ratio': round(group['tip_offset_ratio'].mean(), 3),
                'frame_count': len(group)
            })
        
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(output_file, index=False)
        
        # Also save frame-level data
        frame_output = output_file.replace('.csv', '_frame_level.csv')
        frame_results = df[['frame', 'nose_tip_x', 'nose_tip_y', 'tip_offset_ratio', 
                           'avg_likelihood', 'orientation']].copy()
        frame_results.to_csv(frame_output, index=False)
        
        return results_df


def main():
    root = tk.Tk()
    app = HeadOrientationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
