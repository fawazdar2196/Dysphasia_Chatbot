from flask import Flask, render_template, request, session, redirect, url_for
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
import google.generativeai as genai
import os
import base64
import logging
import json
from datetime import datetime
from time import time
from pydub import AudioSegment

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey123"  # For session management

# Custom Jinja2 filter for timestamp
def timestamp_filter(value):
    return str(int(time() * 1000))

app.jinja_env.filters['timestamp'] = timestamp_filter

# Set Google API key for Gemini
genai.configure(api_key="")

# Log credentials path
logging.debug("GOOGLE_APPLICATION_CREDENTIALS: %s", os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

# Configure Google Text-to-Speech client
try:
    tts_client = texttospeech.TextToSpeechClient()
except Exception as e:
    logging.error("Failed to initialize Text-to-Speech client: %s", e)
    raise

# Configure Google Speech-to-Text client
try:
    stt_client = speech.SpeechClient()
except Exception as e:
    logging.error("Failed to initialize Speech-to-Text client: %s", e)
    raise

# Configure Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Simulated user database
users = {"client1": "password123"}

# Home route (login page)
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username in users and users[username] == password:
            session["username"] = username
            session["conversation"] = []  # Initialize conversation history
            logging.info(f"User {username} logged in")
            return redirect(url_for("chat"))
        else:
            logging.warning("Invalid login attempt for username: %s", username)
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html", error=None)

# Chat route
@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "username" not in session:
        logging.warning("Unauthorized access to /chat")
        return redirect(url_for("login"))
    
    response_text = ""
    audio_file = None
    mcq_options = []
    error_message = None

    if request.method == "POST":
        transcript = ""
        if request.form.get("mcq_option"):
            transcript = request.form["mcq_option"]
            logging.info("Patient selected MCQ: %s", transcript)
            session["conversation"].append({
                "role": "patient",
                "content": transcript,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            response_text = transcript  # Use MCQ selection as response
            audio_file = text_to_speech(transcript)
            mcq_options = []  # Clear MCQs after selection
            session["mcq_options"] = mcq_options
            session.modified = True
        elif request.form.get("audio_data"):
            try:
                audio_data = request.form["audio_data"].split(",")[1]
                audio_bytes = base64.b64decode(audio_data)
                audio_path = "static/temp_audio.webm"
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)
                file_size = os.path.getsize(audio_path)
                logging.info("Audio saved to %s, size: %d bytes", audio_path, file_size)
                if file_size < 1000:
                    raise ValueError("Audio file is too small, likely empty or invalid")
                
                # Normalize audio
                normalized_path = "static/temp_audio_normalized.webm"
                try:
                    audio = AudioSegment.from_file(audio_path, format="webm")
                    max_dbfs = audio.max_dBFS
                    logging.info("Audio max_dBFS: %f", max_dbfs)
                    audio = audio + 15  # Boost volume by 15dB
                    audio = audio.low_pass_filter(2000).high_pass_filter(150)  # Noise reduction
                    audio.export(normalized_path, format="webm", codec="libopus")
                    logging.info("Normalized audio saved to %s, size: %d bytes", normalized_path, os.path.getsize(normalized_path))
                    transcript = speech_to_text(normalized_path)
                except Exception as e:
                    logging.warning("Audio normalization failed: %s", e)
                    logging.info("Falling back to original audio")
                    transcript = speech_to_text(audio_path)  # Fallback to original
                    error_message = "Audio processing issue, using original audio."
                
                logging.info("Speech-to-Text transcript: %s", transcript)
                # Keep audio files for debugging; remove manually
                if not transcript:
                    error_message = error_message or "No speech detected. Please speak louder or check microphone."
                else:
                    session["conversation"].append({
                        "role": "doctor",
                        "content": transcript,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    mcq_options = generate_mcq_options(transcript)
                    session["mcq_options"] = mcq_options
                    session.modified = True
                    response_text = "Please select an option."
            except Exception as e:
                logging.error("Audio processing error: %s", e)
                transcript = ""
                error_message = f"Failed to process audio: {str(e)}. Please try again."
        else:
            transcript = request.form.get("text_input", "").strip()
            logging.info("Text input received: %s", transcript)
            if transcript:
                session["conversation"].append({
                    "role": "doctor",
                    "content": transcript,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                mcq_options = generate_mcq_options(transcript)
                session["mcq_options"] = mcq_options
                session.modified = True
                response_text = "Please select an option."
        
        if not transcript and not mcq_options:
            response_text = error_message or "Sorry, I didn't catch that. Please try again or type something."
            mcq_options = ["Try again", "Type a message", "Ask something else", "Wait a moment", "Need help?", "Speak louder", "NONE of the ABOVE"]
    
    return render_template(
        "chat.html",
        response=response_text,
        audio=audio_file,
        mcq_options=mcq_options,
        conversation=session.get("conversation", []),
        error=error_message
    )

# Logout route
@app.route("/logout")
def logout():
    session.pop("username", None)
    session.pop("conversation", None)
    session.pop("mcq_options", None)
    logging.info("User logged out")
    return redirect(url_for("login"))

# Speech-to-Text function
def speech_to_text(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()
        
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code="en-US",
            model="latest_short",
            enable_automatic_punctuation=True,
            use_enhanced=True,
            speech_contexts=[speech.SpeechContext(phrases=[
                "how are you feeling", "are you in pain", "what symptoms",
                "need help", "hard to talk", "doctor", "patient"
            ])]
        )
        
        response = stt_client.recognize(config=config, audio=audio)
        transcript = ""
        for result in response.results:
            if result.alternatives:
                transcript += result.alternatives[0].transcript
        logging.debug("Speech-to-Text response: %s", response)
        return transcript.strip()
    except Exception as e:
        logging.error("Speech-to-Text error: %s", e)
        return ""

# Text-to-Speech function
def text_to_speech(text):
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-D",
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.8,
        )
        
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        
        audio_path = "static/output.mp3"
        with open(audio_path, "wb") as out:
            out.write(response.audio_content)
        logging.info("Audio response saved to %s", audio_path)
        return audio_path
    except Exception as e:
        logging.error("Text-to-Speech error: %s", e)
        return None

# Generate MCQ options for doctor's question
def generate_mcq_options(question):
    try:
        prompt = (
            "You are an AI assistant helping a patient with expressive dysphasia communicate. "
            "Given a question asked by a doctor, generate exactly 7 contextually relevant multiple-choice options "
            "for the patient to select as a response. Options should be simple, short, and ordered from most likely to least likely. "
            "The last option must be 'NONE of the ABOVE'. "
            "Return the options as a JSON array of strings.\n\n"
            f"Question: {question}\n\n"
            "Output: ```json\n[]\n```"
        )
        response = model.generate_content(prompt)
        options = json.loads(response.text.strip().replace("```json\n", "").replace("\n```", ""))
        if len(options) == 7 and options[-1].upper() == "NONE OF THE ABOVE":
            return options
        else:
            # Fallback options
            return ["Yes", "No", "Maybe", "Not sure", "I need help", "Can you repeat?", "NONE of the ABOVE"]
    except Exception as e:
        logging.error("MCQ generation error: %s", e)
        return ["Yes", "No", "Maybe", "Not sure", "I need help", "Can you repeat?", "NONE of the ABOVE"]

# Run the app
if __name__ == "__main__":
    app.run(debug=True)