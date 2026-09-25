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

Item 7 — React frontend dashboard. Build sub-pieces one at a time, not the whole dashboard at once:
7a. Live status view (current sensor readings + prediction, for both plants)
7b. Recommendation display with SHAP explanation (sensor) and Grad-CAM heatmap 
    (image) shown clearly, matched to the exact API field names above
7c. Historical trend chart (Chart.js) — moisture/temp/humidity over time
7d. What-if simulator (manual input sliders re-running prediction)
7e. Treatment recommendation panel (using the treatment{} object)
Do NOT build all of these in one response — I will tell you which sub-item to build next after reviewing each one.