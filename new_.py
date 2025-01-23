import io
import os
import json
import logging
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from openai import OpenAI
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MEDICAL_REVIEW_SCHEMA = {
    "diagnosis": "",
    "biometrics": {
        "height": None,
        "weight": None,
        "bmi": None,
        "waistCircumference": None
    },
    "bgAndHtn": {
        "bloodGlucose": None,
        "systolicBP": None,
        "diastolicBP": None
    },
    "lifestyle": {
        "smokingStatus": "",
        "alcoholStatus": "",
        "dietNutrition": "",
        "physicalActivity": ""
    },
    "examination": {
        "chiefComplaints": "",
        "physicalExamination": ""
    },
    "physicianNotes": ""
}

def get_openai_client():
    api_key = "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"
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    return OpenAI(api_key=api_key)

def record_audio(output_folder="recorded_audio", duration=30, fs=44100):
    logger.info(f"Starting {duration} second recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()
    logger.info("Recording complete")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_folder, f"screening_{timestamp}.wav")
    wavfile.write(file_path, fs, (recording * 32767).astype(np.int16))
    logger.info(f"Audio saved to {file_path}")
    return file_path

def transcribe_audio(file_path):
    client = get_openai_client()
    try:
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        logger.info("Transcription successful")
        return transcription.text
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return ""

def extract_medical_review_data(transcript):
    client = get_openai_client()
    
    system_prompt = """
    You are a medical review assistant. Extract patient information from the transcript and create a JSON object that matches exactly this structure:
    {
        "diagnosis": "string",
        "biometrics": {
            "height": "number (in cm)",
            "weight": "number (in kg)",
            "bmi": "number (calculated if height and weight present)",
            "waistCircumference": "number (in cm)"
        },
        "bgAndHtn": {
            "bloodGlucose": "number (in mg/dL)",
            "systolicBP": "number (in mmHg)",
            "diastolicBP": "number (in mmHg)"
        },
        "lifestyle": {
            "smokingStatus": "string (must be one of: Never, Former, Current)",
            "alcoholStatus": "string (must be one of: Never, Former, Current)",
            "dietNutrition": "string",
            "physicalActivity": "string"
        },
        "examination": {
            "chiefComplaints": "string",
            "physicalExamination": "string"
        },
        "physicianNotes": "string"
    }

    Important rules:
    1. Only extract information explicitly mentioned in the transcript
    2. Set fields to null if not mentioned
    3. Calculate BMI if both height and weight are provided using: weight (kg) / (height (m))^2
    4. Validate all fields against their required types and options
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract medical review data from this transcript and format it according to the structure shown above: {transcript}"}
            ],
            temperature=0
        )
        
        structured_data = json.loads(response.choices[0].message.content)
        
        if "biometrics" in structured_data:
            biometrics = structured_data["biometrics"]
            if biometrics.get("height") and biometrics.get("weight"):
                height_m = float(biometrics["height"]) / 100
                weight_kg = float(biometrics["weight"])
                biometrics["bmi"] = round(weight_kg / (height_m * height_m), 2)
        
        structured_data["reviewDate"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        structured_data["reviewId"] = f"REV_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return structured_data
    except Exception as e:
        logger.error(f"Error extracting data: {str(e)}")
        return None
    
def save_medical_review_data(data, output_folder="medical_review_data"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_folder, f"medical_review_{timestamp}.json")
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Medical review data saved to {file_path}")
    return file_path

def main():
    try:
        print("Starting medical review process...")
        
        audio_file = record_audio(duration=30)
        
        transcript = transcribe_audio(audio_file)
        print(f"\nTranscript of recording:\n{transcript}\n")
        
        medical_review_data = extract_medical_review_data(transcript)
        
        if medical_review_data:
            json_file = save_medical_review_data(medical_review_data)
            print("\nMedical review data saved successfully!")
            print(f"JSON file location: {json_file}")
            
            print("\nExtracted Medical Review Information:")
            print(json.dumps(medical_review_data, indent=2))
        else:
            print("\nFailed to extract medical review data. Please try again.")
            
    except Exception as e:
        logger.error(f"Error in medical review process: {str(e)}")
        print("\nAn error occurred during the medical review process. Please try again.")

if __name__ == "__main__":
    main()
