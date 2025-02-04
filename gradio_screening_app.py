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
        empty_data = {
            "diagnosis": "",
            "biometrics": {"height": None, "weight": None, "bmi": None, "waistCircumference": None},
            "bgAndHtn": {"bloodGlucose": None, "systolicBP": None, "diastolicBP": None},
            "lifestyle": {"smokingStatus": "Never", "alcoholStatus": "Never", "dietNutrition": "", "physicalActivity": ""},
            "examination": {"chiefComplaints": [], "physicalExamination": []},
            "phq9": {
                "interest_pleasure": None,
                "feeling_down": None,
                "sleep_problems": None,
                "tiredness": None,
                "appetite": None,
                "self_esteem": None,
                "concentration": None,
                "movement": None,
                "suicidal_thoughts": None
            },
            "physicianNotes": ""
        }
        return (
            "", None, None, None, None, None, None, None,
            "Never", "Never", "", "", [], [], "",
            None, None, None, None, None, None, None, None, None,
            "Failed to extract medical review data. Please try again.",
            json.dumps(empty_data, indent=2),
            transcript
        )
    
    physical_exam = medical_review_data.get("examination", {}).get("physicalExamination", [])
    chief_complaints = medical_review_data.get("examination", {}).get("chiefComplaints", [])
    
    phq9 = medical_review_data.get("phq9", {})
    
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
        chief_complaints,
        physical_exam,
        phq9.get("interest_pleasure"),
        phq9.get("feeling_down"),
        phq9.get("sleep_problems"),
        phq9.get("tiredness"),
        phq9.get("appetite"),
        phq9.get("self_esteem"),
        phq9.get("concentration"),
        phq9.get("movement"),
        phq9.get("suicidal_thoughts"),
        medical_review_data.get("physicianNotes", ""),
        json.dumps(medical_review_data, indent=2),
        transcript
    )

def save_edited_data(diagnosis, height, weight, bmi, waist_circumference,
                    blood_glucose, systolic_bp, diastolic_bp,
                    smoking_status, alcohol_status, diet_nutrition, physical_activity,
                    chief_complaints, physical_examination, phq1, phq2, phq3, 
                    phq4, phq5, phq6, phq7, phq8, phq9, physician_notes):
    medical_review_data = {
        "diagnosis": diagnosis,
        "biometrics": {
            "height": height,
            "weight": weight,
            "bmi": bmi,
            "waistCircumference": waist_circumference
        },
        "bgAndHtn": {
            "bloodGlucose": blood_glucose,
            "systolicBP": systolic_bp,
            "diastolicBP": diastolic_bp
        },
        "lifestyle": {
            "smokingStatus": smoking_status,
            "alcoholStatus": alcohol_status,
            "dietNutrition": diet_nutrition,
            "physicalActivity": physical_activity
        },
        "examination": {
            "chiefComplaints": chief_complaints,
            "physicalExamination": physical_examination
        },
        "phq9": {
            "interest_pleasure": phq1,
            "feeling_down": phq2,
            "sleep_problems": phq3,
            "tiredness": phq4,
            "appetite": phq5,
            "self_esteem": phq6,
            "concentration": phq7,
            "movement": phq8,
            "suicidal_thoughts": phq9
        },
        "physicianNotes": physician_notes
    }
    json_file = save_medical_review_data(medical_review_data)
    return f"Data successfully saved to {json_file}", medical_review_data

with gr.Blocks(title="Medical Review System") as iface:
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record Audio")
        with gr.Column():
            file_input = gr.File(label="Upload Audio File", file_types=["audio"])
    
    with gr.Row():
        transcript_output = gr.Textbox(label="Transcript", lines=3)
        json_output = gr.JSON(label="Extracted Data")
    
    save_button = gr.Button("Save Edited Data")
    save_status = gr.Textbox(label="Save Status", interactive=False)
    
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
            chief_complaints_choices = [
                "Focal weakness",
                "Shortness of breath on activity",
                "Loss of consciousness",
                "Palpitations (heart racing)",
                "Foot complaints",
                "Recurrent dizziness",
                "Fainting",
                "Blurring of vision",
                "Leg swelling",
                "Other"
            ]
            chief_complaints = gr.CheckboxGroup(
                label="Chief Complaints",
                choices=chief_complaints_choices
            )
            
            physical_examination_choices = [
                "Eye Exam",
                "Foot Exam",
                "Neurological Exam",
                "Mental Health",
                "Pallor",
                "Foetal Heartbeat",
                "Abdominal Pelvic",
                "Abdominal Exam",
                "Lie & Presentation",
                "Other"
            ]
            physical_examination = gr.CheckboxGroup(
                label="Physical Examinations",
                choices=physical_examination_choices
            )
        
        with gr.Tab("PHQ9 Assessment"):
            phq_options = ["Not at all", "Several days", "More than half the days", "Nearly every day"]
            
            phq1 = gr.Radio(phq_options, label="1. Little interest or pleasure in doing things?")
            phq2 = gr.Radio(phq_options, label="2. Feeling down, depressed or hopeless?")
            phq3 = gr.Radio(phq_options, label="3. Trouble falling/staying asleep or sleeping too much?")
            phq4 = gr.Radio(phq_options, label="4. Feeling tired or having little energy?")
            phq5 = gr.Radio(phq_options, label="5. Poor appetite or overeating?")
            phq6 = gr.Radio(phq_options, label="6. Feeling bad about yourself?")
            phq7 = gr.Radio(phq_options, label="7. Trouble concentrating on things?")
            phq8 = gr.Radio(phq_options, label="8. Moving/speaking slowly or restlessness?")
            phq9 = gr.Radio(phq_options, label="9. Thoughts of being better off dead?")


        with gr.Tab("Physician Notes"):
            physician_notes = gr.Textbox(label="Physician Notes", lines=5)

    audio_input.change(
        fn=lambda x: process_audio_and_update_form(x, "microphone"),
        inputs=[audio_input],
        outputs=[
            diagnosis, height, weight, bmi, waist_circumference,
            blood_glucose, systolic_bp, diastolic_bp,
            smoking_status, alcohol_status, diet_nutrition, physical_activity,
            chief_complaints, physical_examination, 
            phq1, phq2, phq3, phq4, phq5, phq6, phq7, phq8, phq9,
            physician_notes,
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
            chief_complaints, physical_examination, 
            phq1, phq2, phq3, phq4, phq5, phq6, phq7, phq8, phq9,
            physician_notes,
            json_output, transcript_output
        ]
    )

    save_button.click(
        fn=save_edited_data,
        inputs=[
            diagnosis, height, weight, bmi, waist_circumference,
            blood_glucose, systolic_bp, diastolic_bp,
            smoking_status, alcohol_status, diet_nutrition, physical_activity,
            chief_complaints, physical_examination,  
            phq1, phq2, phq3, phq4, phq5, phq6, phq7, phq8, phq9,
            physician_notes
        ],
        outputs=[save_status, json_output]
    )

iface.launch(share=True)