# Dog Head Orientation Labeling Tool 🐕

A Python tool for labeling and analyzing dog head orientation from DeepLabCut tracking data.

## Quick Start

### 1. Clone and download the tool

```bash
git clone https://github.com/sqrmjdi/dog_head_orientation_tool.git

```
### 2. Change directory to source code
```bash

cd dog_head_orientation_tool
```

### 3. Installation (Windows OS)
Create virtual environment
```bash
python -m venv .venv
```

Activate virtual environment (Windows)
```bash
.venv\Scripts\activate
```

Install dependencies
note: make sure you're in the right directory with requirements.txt there !!!
```bash
pip install -r requirements.txt
```

### 4. Put data folder in the codebase

Create a `head_orientation/data/` folder and add your:

- Video files (`.mp4`, `.avi`, `.mov`)
- DeepLabCut Excel files (`.xlsx`)

### 5. Run the manual labeling tool

```bash
python head_orientation/manual_labeling_ui.py
```

## Features

- **Automatic Orientation Detection** - Classifies head orientation as LEFT, RIGHT, STRAIGHT, or ELSEWHERE
- **Manual Labeling UI** - Review and correct auto-detected labels with video preview
- **Head Tilt Angle Calculation** - Visualizes the angle of head tilt
- **Configurable Frame Intervals** - Label at 1s, 0.5s, or 0.2s intervals
- **Nose Landmark Visualization** - See the nose tracking points used for detection



## Orientation Detection Logic

The tool uses nose landmark positions from DeepLabCut:

| Orientation         | Condition                                      |
| ------------------- | ---------------------------------------------- |
| **LEFT**      | `nose.right_Y < nose.left_Y` (±2px margin)  |
| **RIGHT**     | `nose.left_Y < nose.right_Y` (±2px margin)  |
| **STRAIGHT**  | `nose.right_Y ≈ nose.left_Y` (within ±2px) |
| **ELSEWHERE** | Low detection confidence (<0.3)                |

## Requirements

- Python 3.8+
- pandas, openpyxl, opencv-python, pillow
- tkinter (included with Python)
