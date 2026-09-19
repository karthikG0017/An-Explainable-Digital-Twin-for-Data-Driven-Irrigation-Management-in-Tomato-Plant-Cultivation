# An-Explainable-Digital-Twin-for-Data-Driven-Irrigation-Management-in-Tomato-Plant-Cultivation
DEVELOPMENT PRIORITY ORDER — build in this sequence, not frontend-first:
1. Data ingestion script (ESP32 → Firebase, or simulated CSV data for testing 
   before hardware is ready)
2. Label-engineering script (raw sensor data → irrigation_needed label, using 
   the 70% threshold rule above)
3. Sensor model training script (XGBoost, with Isolation Forest anomaly filtering 
   as a preprocessing step)
4. Image model training/fine-tuning script (EfficientNetB0 on PlantVillage)
5. SHAP integration (sensor model) and LIME integration (image model)
6. Fusion logic (combining both model outputs)
7. Flask backend API (serves predictions, connects to Firebase)
8. Frontend dashboard (React) — ONLY after everything above works and is tested
