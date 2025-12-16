# Manual Labeling Tool - Distribution Guide

## Quick Start

1. **Copy the executable** `ManualLabelingTool.exe` from the `dist` folder to any location on your computer
2. **Create folders** next to the exe (optional - they will be created automatically):
   - `data/` - for input files (videos and Excel files)
   - `output/` - for saved label files
3. **Run** `ManualLabelingTool.exe`

## Usage

1. Click **"📂 Load Video"** to select a video file
2. Click **"📊 Load Excel"** to load the corresponding data file
3. Select the **interval** (1.0s, 0.5s, or 0.2s)
4. Click **"▶ Start Labeling"** to begin

## Folder Structure

When you run the tool, it will automatically create `data/` and `output/` folders in the same directory as the executable.

```
YourFolder/
├── ManualLabelingTool.exe
├── data/           (auto-created for input files)
└── output/         (auto-created for saved labels)
```

## System Requirements

- Windows 10 or later
- No Python installation required
- No additional dependencies needed

## Troubleshooting

- **Slow startup**: The first launch may take 10-30 seconds as files are extracted
- **Antivirus warning**: Some antivirus software may flag the exe - this is a false positive
- **Missing folders**: Run the exe once and it will create the necessary folders

## Building from Source

If you need to rebuild the executable:

```bash
pip install pyinstaller
pyinstaller ManualLabelingTool.spec
```

The new executable will be in the `dist/` folder.
