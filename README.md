# MSML 640 – Urban Growth Detection from Satellite Imagery

This repository contains the implementation of my MSML-640 Computer Vision project:  
**Urban Growth Detection from Satellite Imagery using Deep Learning and Change Detection Techniques.**

The goal of this project is to detect **urban expansion over time** using multi-temporal satellite imagery (Google Earth Engine), building segmentation (U-Net), and temporal mask differencing to identify **newly constructed areas**.

---

## Project Overview

Urban growth is a strong indicator of economic development, but manually analyzing satellite imagery is slow and inconsistent.  
This project builds an automated pipeline to:

- Download & preprocess **before/after** satellite images (Google Earth Engine)
- Perform **building segmentation** using U-Net
- Detect changes over time
- Generate **growth maps, overlays, and heatmaps**
- Quantify urban expansion with interpretable statistics

---

## Repository Structure

> **Note:** This structure is subject to change as the project evolves.