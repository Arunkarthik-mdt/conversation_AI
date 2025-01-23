# Medical Review System Setup

## Prerequisites

# Install required Python packages from requirements.txt
!pip install -r requirements.txt

# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Unix or MacOS
source venv/bin/activate

# On Windows
venv\Scripts\activate

# Install the required packages from requirements.txt
pip install -r requirements.txt

# Set up OpenAI API key as an environment variable
export OPENAI_API_KEY="your_api_key_here"

## Run the Gradio App

# Run the Gradio application
python gradio_screening_app.py
