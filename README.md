# Document Scanner App

A simple computer vision project that scans documents from images and converts them into clean, readable PDF files.

Built using OpenCV and Streamlit.

---

## Features

* Detects document edges automatically
* Corrects perspective (flattens the page)
* Removes shadows and background noise
* Enhances readability (black & white scan)
* Supports multiple images
* Exports scanned pages as a single PDF

---

## Tech Stack

* Python
* OpenCV → image processing
* NumPy → numerical operations
* Streamlit → web app interface
* Pillow → PDF generation

---

## How It Works

The app follows this pipeline:

1. **Image Upload**

   * User uploads one or more images.

2. **Document Detection**

   * Uses contour detection to find the document edges.
   * Filters shapes to detect a 4-sided boundary.

3. **Perspective Transform**

   * Converts tilted document into a flat top-down view.

4. **Preprocessing**

   * Converts to grayscale
   * Removes shadows using background normalization
   * Applies denoising

5. **Thresholding**

   * Converts image into clean black & white for readability

6. **PDF Generation**

   * All processed images are combined into a single PDF

---

## Project Structure

```
doc_scanner/
├── app.py
├── src/
│   ├── scanner.py
│   ├── utils.py
├── data/
│   ├── input/
│   ├── output/
├── requirements.txt
├── README.md
```

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/YOUR_USERNAME/doc_scanner.git
cd doc_scanner
```

---

### 2. Create environment (conda recommended)

```
conda create --prefix ./venv python=3.10
conda activate ./venv
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## Run the App

```
streamlit run app.py
```

Then open the link shown in terminal.

---

## Usage

1. Upload one or more images
2. The app will automatically:

   * detect the document
   * scan and enhance it
3. Click **Download PDF** to get the final output

---

## Key Concepts Used

### 1. Edge Detection

Detects boundaries of objects in the image.

### 2. Contour Detection

Finds shapes and identifies the document region.

### 3. Perspective Transform

Maps the document into a flat rectangular view.

### 4. Image Enhancement

* Noise removal
* Shadow removal
* Contrast improvement

### 5. Thresholding

Converts image into high-contrast black & white.

---

## Future Improvements

* Live camera scanning
* Automatic cropping
* OCR (text extraction)
* Cloud deployment

---

## Author

Ahan Mondal
