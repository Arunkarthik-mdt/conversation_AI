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
from new_ import transcribe_audio, extract_screening_data, save_screening_data

def process_audio(audio, audio_type):
    output_folder = "recorded_audio"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if audio_type == "microphone":
        file_path = os.path.join(output_folder, f"screening_{timestamp}.wav")
        wavfile.write(file_path, audio[0], audio[1])
    else:  
        file_path = os.path.join(output_folder, f"screening_{timestamp}.wav")
        audio_segment = AudioSegment.from_mp3(audio)
        audio_segment.export(file_path, format="wav")
    
    transcript = transcribe_audio(file_path)
    
    # Extract and structure data
    screening_data = extract_screening_data(transcript)
    
    if screening_data:
        json_file = save_screening_data(screening_data)
        return json.dumps(screening_data, indent=2), transcript
    else:
        return "Failed to extract screening data. Please try again.", transcript

def process_audio_and_update_form(audio, audio_type):
    json_data, transcript = process_audio(audio, audio_type)
    
    if isinstance(json_data, str):
        print(f"json_data: {json_data}")
        json_data = json.loads(json_data)
    
    # Updating the components
    return {
        # Bio Data Tab
        first_name: json_data["bioData"]["firstName"],
        middle_name: json_data["bioData"]["middleName"] or "",
        last_name: json_data["bioData"]["lastName"],
        mobile_number: json_data["bioData"]["mobileNumber"] or "+254",
        mobile_category: json_data["bioData"]["mobileNumberCategory"] or "Personal",
        landmark: json_data["bioData"]["landmark"] or "",
        national_id: json_data["bioData"]["nationalId"] or "",
        
        # Biometrics Tab
        gender: json_data["biometrics"]["gender"] or "Male",
        date_of_birth: json_data["biometrics"]["dateOfBirth"] or "",
        age: json_data["biometrics"]["age"] or None,
        height: json_data["biometrics"]["height"] or None,
        weight: json_data["biometrics"]["weight"] or None,
        bmi: json_data["biometrics"]["bmi"] or "-",
        is_smoker: json_data["biometrics"]["isRegularSmoker"] or "No",
        
        # Blood Pressure Tab
        has_hypertension: json_data["bloodPressure"]["hasHypertensionHistory"] or "No",
        
        # Keep the original JSON output
        json_output: json_data,
        transcript_output: transcript
    }

with gr.Blocks(title="Patient Screening System") as iface:
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record Audio")
        with gr.Column():
            file_input = gr.File(label="Upload MP3 File", file_types=[".mp3"])
    
    with gr.Row():
        transcript_output = gr.Textbox(label="Transcript", lines=3)
        json_output = gr.JSON(label="Extracted Data")
    
    with gr.Tabs():
        with gr.Tab("Bio Data"):
            first_name = gr.Textbox(label="First Name*")
            middle_name = gr.Textbox(label="Middle Name")
            last_name = gr.Textbox(label="Last Name*")
            mobile_number = gr.Textbox(label="Mobile Number*", value="+254")
            mobile_category = gr.Dropdown(
                label="Mobile Number Category*",
                choices=["Personal", "Work", "Other"],
                value="Personal"
            )
            landmark = gr.Textbox(label="Landmark*")
            national_id = gr.Textbox(label="National ID*")

        with gr.Tab("Biometrics"):
            gender = gr.Radio(
                label="Gender*",
                choices=["Male", "Female", "Non-Binary"],
                value="Male"
            )
            with gr.Row():
                date_of_birth = gr.Textbox(label="Date of Birth")
                age = gr.Number(label="Age (in years)*")
            height = gr.Number(label="Height (in cm)")
            weight = gr.Number(label="Weight (in kg)")
            bmi = gr.Textbox(label="BMI", value="-")
            is_smoker = gr.Radio(
                label="Are you a regular smoker?*",
                choices=["Yes", "No"],
                value="No"
            )

        with gr.Tab("Blood Pressure"):
            has_hypertension = gr.Radio(
                label="Have you been diagnosed with High Blood Pressure?*",
                choices=["Yes", "No"],
                value="No"
            )

    # Connect the audio input to update all form fields
    audio_input.change(
        fn=lambda x: process_audio_and_update_form(x, "microphone"),
        inputs=[audio_input],
        outputs=[
            first_name, middle_name, last_name, mobile_number,
            mobile_category, landmark, national_id,
            gender, date_of_birth, age, height, weight,
            bmi, is_smoker, has_hypertension,
            json_output, transcript_output
        ]
    )

    # Connect the file input to update all form fields
    file_input.change(
        fn=lambda x: process_audio_and_update_form(x.name, "file"),
        inputs=[file_input],
        outputs=[
            first_name, middle_name, last_name, mobile_number,
            mobile_category, landmark, national_id,
            gender, date_of_birth, age, height, weight,
            bmi, is_smoker, has_hypertension,
            json_output, transcript_output
        ]
    )

iface.launch()
