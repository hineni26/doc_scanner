# Document Scanner App

A computer vision application that scans documents from images and converts them into clean, readable PDF documents.

Built using **OpenCV** and **Streamlit**.

---

## Features

- Automatic document edge detection  
- Perspective correction (flattening)  
- Shadow and noise removal  
- Black & white enhancement for readability  
- Supports multiple image uploads  
- Export all pages into a single PDF  

---

## Tech Stack

- **Python**
- **OpenCV** — image processing
- **NumPy** — numerical operations
- **Streamlit** — web interface
- **Pillow** — PDF generation

---

## How It Works

The app follows this pipeline:

1. **Image Upload**
   - User uploads one or more images

2. **Document Detection**
   - Detects edges using contour detection
   - Identifies the largest 4-sided shape (document)

3. **Perspective Transform**
   - Warps the image to get a top-down scanned view

4. **Preprocessing**
   - Converts to grayscale  
   - Removes shadows  
   - Applies denoising  

5. **Thresholding**
   - Converts image to high-contrast black & white  

6. **PDF Generation**
   - Combines all processed images into a single PDF  

---

## Project Structure

```
doc_scanner/
├── app.py              # Streamlit UI
├── src/
│   ├── scanner.py     # Core image processing
│   ├── utils.py       # Helper functions
├── data/
│   ├── input/         # Sample input images
│   ├── output/        # Generated images
├── requirements.txt
├── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/hineni26/doc_scanner.git
cd doc_scanner
```

---

### 2. Create a virtual environment

```bash
conda create --prefix ./venv python=3.10
conda activate ./venv
```

(You can also use `venv` if preferred)

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Run the App

```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

---

## Usage

1. Upload one or more document images  
2. The app will automatically:
   - Detect the document  
   - Enhance and scan it  
3. Click **Download PDF** to save the result  

---

## Sample Data

- Place test images in:
  ```
  data/input/
  ```
- Output PDFs will be saved in:
  ```
  data/output/
  ```

---

## Limitations

- Works best with clear, high-contrast images  
- May fail on blurry or cluttered backgrounds  
- Requires visible document boundaries  

---

## Future Improvements

- Live camera scanning  
- Automatic cropping UI  
- OCR (text extraction)  
- Mobile-friendly interface  

---

## Motivation

This project demonstrates practical use of computer vision techniques such as contour detection and perspective transformation to solve a real-world problem.

---

## Demo (Add Screenshots)

_Add screenshots or GIFs here to showcase:_
- Input image  
- Edge detection  
- Final scanned output  
- App interface  

---

## License

GPL License

---

## Author

**Ahan Mondal**