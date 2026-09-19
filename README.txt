PanelForge Geometry + OCR v1.6.2
=============================

Purpose
-------
Fast webtoon panel extraction with optional OCR-aware text preservation.
Geometry detects the panel first. A second visual pass then splits suspiciously tall
merged regions on safe scene seams. OCR detects nearby text zones, protects them
during those splits, and can expand the final crop so bubbles/captions are not cut off.

Launch
------
1. Double-click run.bat.
2. On first launch, the local .venv and dependencies are installed automatically.
3. Choose either a .CBZ file or an image folder.
4. Keep "Inclure les textes proches / OCR actif" enabled if you want text-aware crops.
5. Click "Découper les panels".

OCR model
---------
- Uses the PP-OCRv4 mobile TEXT DETECTOR only; it does not waste time recognizing words.
- No PaddlePaddle / ONNX Runtime dependency is required: inference runs through OpenCV DNN.
- The ~4.8 MB model is downloaded automatically on the first OCR run and cached in models/.
- If the machine is offline, disable OCR and the original geometry-only mode still works.

Output
------
For a folder:
  <source folder>\PanelCrops_Geometry\

For a CBZ:
  <CBZ parent>\<CBZ name> output\

Generated files:
  panel_0001.png
  panel_0002.png
  ...
  panels.csv
  report.json
  _debug\analysis_overlay.jpg   (if debug preview is enabled)

Debug overlay (OCR mode)
------------------------
- RED: original geometry panel
- ORANGE: OCR-detected text zones
- GREEN: final crop after OCR expansion
- Label Pxx Tn: final panel id + number of text zones assigned

How it works
------------
- Source images are reduced to a small analysis width (default 192 px) for geometry.
- Reduced images are vertically joined into one continuous analysis strip.
- The geometric detector uses occupancy, contrast, saturation, edges,
  connected components, gutters and adaptive valleys/gradients.
- OCR is run per source image at a separate resolution (default 768 px wide), in
  overlapping vertical tiles so long webtoon pages do not explode RAM usage.
- Each text zone is attached to at most one nearby panel using overlap, horizontal
  alignment and vertical distance.
- A panel can receive multiple text zones. OCR does not invent artwork panels, but its
  text zones now protect bubbles/captions while the oversized-panel splitter works.
- Oversized merged panels are split only on safe visual seams; there is no blind fixed-height slicing.
- A panel crossing two source image chunks remains one logical panel.
- Final crops are reconstructed from the ORIGINAL full-resolution images.
- Original HD pages are never concatenated into a giant HD canvas.

Recommended settings
--------------------
Panel geometry: 192 px = fastest/default; 256 px for difficult layouts.
OCR: 768 px = recommended; 512 px faster; 1024 px for small or difficult lettering.
