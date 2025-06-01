# Emotion Recognition App

This project is a multi-modal emotion recognition system using **Facial Expressions**, **Voice**, and **Text** inputs. It uses Deep Learning and traditional ML models to analyze human emotions in real-time or through uploaded files.

## Features
- Facial emotion detection using DeepFace
- Audio emotion classification (RAVDESS dataset)
- Text emotion analysis
- Majority voting system for final emotion
- Streamlit interface

## How to Run
1. Clone the repo
2. Install requirements:
      ```bash
   pip install -r requirements.txt
3. Run the app:
   streamlit run app.py

Deployment
This app can be deployed on:

Streamlit Cloud

Hugging Face Spaces

GitHub Pages (for static previews only)

License
MIT License

© 2025 Umaa Maheshwary SV


Save this as `README.md`.

---

## ✅ STEP 2: Add a License (MIT Recommended)

Create a file called `LICENSE` (no extension), and paste this MIT license text:

MIT License

Copyright (c) 2025 Umaa Maheshwary SV

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

...

(standard MIT license continues)


Let me know if you'd prefer **Apache 2.0** instead.

---

## ✅ STEP 3: Push the New Files to GitHub

Make sure you're in your `emotion-recognition-app` folder:

```bash
git add README.md LICENSE
git commit -m "Add README and MIT license"
git push origin main

✅ STEP 4: Deploy the App
Option A: Streamlit Cloud (Recommended)
Go to https://streamlit.io/cloud

Sign in with your GitHub

Click "New app"

Select your repo: emotion-recognition-app

Set main file path: app.py

Click Deploy

Done! 🎉

Option B: Hugging Face Spaces
Create a Hugging Face account

Go to https://huggingface.co/spaces

Click "Create New Space"

Choose Gradio or Streamlit as the SDK

Connect your GitHub repo or upload the files directly

Add requirements.txt and app.py

Option C: GitHub Pages (not suitable for Streamlit apps)
Only works for static HTML/JS sites. Not suitable here.