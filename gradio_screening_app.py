import gradio as gr
import io
import os
import json
import logging
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from openai import OpenAI
from datetime import datetime
from pydub import AudioSegment
from new_ import transcribe_audio, extract_medical_review_data, save_medical_review_data

def process_audio(audio, audio_type):
    output_folder = "recorded_audio"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if audio_type == "microphone":
        file_path = os.path.join(output_folder, f"medical_review_{timestamp}.wav")
        wavfile.write(file_path, audio[0], audio[1])
    else:  
        file_path = os.path.join(output_folder, f"medical_review_{timestamp}.wav")
        if isinstance(audio, str):
            AudioSegment.from_file(audio).export(file_path, format="wav")
        else:
            with open(file_path, "wb") as f:
                f.write(audio)
    
    transcript = transcribe_audio(file_path)
    
    medical_review_data = extract_medical_review_data(transcript)
    
    if medical_review_data:
        json_file = save_medical_review_data(medical_review_data)
        return medical_review_data, transcript
    else:
        return None, transcript

def process_audio_and_update_form(audio, audio_type):
    medical_review_data, transcript = process_audio(audio, audio_type)
    
    if medical_review_data is None:
        return ("", None, None, None, None, None, None, None, 
                "Never", "Never", "", "", "", "", "",
                "Failed to extract medical review data. Please try again.", 
                transcript)
    
    return (
        medical_review_data.get("diagnosis", ""),
        medical_review_data.get("biometrics", {}).get("height"),
        medical_review_data.get("biometrics", {}).get("weight"),
        medical_review_data.get("biometrics", {}).get("bmi"),
        medical_review_data.get("biometrics", {}).get("waistCircumference"),
        medical_review_data.get("bgAndHtn", {}).get("bloodGlucose"),
        medical_review_data.get("bgAndHtn", {}).get("systolicBP"),
        medical_review_data.get("bgAndHtn", {}).get("diastolicBP"),
        medical_review_data.get("lifestyle", {}).get("smokingStatus", "Never"),
        medical_review_data.get("lifestyle", {}).get("alcoholStatus", "Never"),
        medical_review_data.get("lifestyle", {}).get("dietNutrition", ""),
        medical_review_data.get("lifestyle", {}).get("physicalActivity", ""),
        medical_review_data.get("examination", {}).get("chiefComplaints", ""),
        medical_review_data.get("examination", {}).get("physicalExamination", ""),
        medical_review_data.get("physicianNotes", ""),
        json.dumps(medical_review_data, indent=2),
        transcript
    )

with gr.Blocks(title="Medical Review System") as iface:
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record Audio")
        with gr.Column():
            file_input = gr.File(label="Upload Audio File", file_types=["audio"])
    
    with gr.Row():
        transcript_output = gr.Textbox(label="Transcript", lines=3)
        json_output = gr.JSON(label="Extracted Data")
    
    with gr.Tabs():
        with gr.Tab("Diagnosis"):
            diagnosis = gr.Textbox(label="Diagnosis")

        with gr.Tab("Biometrics"):
            height = gr.Number(label="Height (cm)")
            weight = gr.Number(label="Weight (kg)")
            bmi = gr.Number(label="BMI")
            waist_circumference = gr.Number(label="Waist Circumference (cm)")

        with gr.Tab("BG and HTN"):
            blood_glucose = gr.Number(label="Blood Glucose (mg/dL)")
            systolic_bp = gr.Number(label="Systolic BP (mmHg)")
            diastolic_bp = gr.Number(label="Diastolic BP (mmHg)")

        with gr.Tab("Lifestyle"):
            smoking_status = gr.Radio(label="Smoking Status", choices=["Never", "Former", "Current"])
            alcohol_status = gr.Radio(label="Alcohol Status", choices=["Never", "Former", "Current"])
            diet_nutrition = gr.Textbox(label="Diet and Nutrition")
            physical_activity = gr.Textbox(label="Physical Activity")

        with gr.Tab("Examination"):
            chief_complaints = gr.Textbox(label="Chief Complaints", lines=3)
            physical_examination = gr.Textbox(label="Physical Examination", lines=3)

        with gr.Tab("Physician Notes"):
            physician_notes = gr.Textbox(label="Physician Notes", lines=5)

    audio_input.change(
        fn=lambda x: process_audio_and_update_form(x, "microphone"),
        inputs=[audio_input],
        outputs=[
            diagnosis, height, weight, bmi, waist_circumference,
            blood_glucose, systolic_bp, diastolic_bp,
            smoking_status, alcohol_status, diet_nutrition, physical_activity,
            chief_complaints, physical_examination, physician_notes,
            json_output, transcript_output
        ]
    )

    file_input.change(
        fn=lambda x: process_audio_and_update_form(x.name if x else None, "file"),
        inputs=[file_input],
        outputs=[
            diagnosis, height, weight, bmi, waist_circumference,
            blood_glucose, systolic_bp, diastolic_bp,
            smoking_status, alcohol_status, diet_nutrition, physical_activity,
            chief_complaints, physical_examination, physician_notes,
            json_output, transcript_output
        ]
    )

iface.launch(share=True)
