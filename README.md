# X-Plane AI ATC

A highly realistic, AI-powered Air Traffic Control system for X-Plane, designed utilizing real-time simulator data and cutting-edge Large Language Models (LLMs) to create a lifelike offline aviation experience.

## Overview
This script dynamically observes the state of your X-Plane aircraft and acts as a fully intelligent Air Traffic Controller (ATC). Whether you are sitting at the gate requesting an IFR clearance, or holding short of the runway preparing for departure, you simply tune your active COM1 radio to the correct real-world frequency for your airport, hold your Push-to-Talk button, and speak naturally into your microphone.

## Features
* **Push-to-Talk (PTT):** Hold the numpad `+` key to speak to ATC, featuring realistic white-noise radio click `squelch` audio injection.
* **Instant Radio Cutoff:** Just like real life, tuning away from a frequency while ATC is speaking will instantly cut their signal mid-sentence.
* **Frequency-Based Roles:** Tuning your COM1 radio automatically switches the AI's role (Ground, Tower, Clearance Delivery, UNICOM) and utilizes different Text-To-Speech (TTS) voices depending on the band.
* **Strict Aviation Logic:** Powered by Llama-3.3-70B, the ATC strictly enforces FAA/ICAO rules. It will deny clearances outside its jurisdiction (e.g., asking for pushback on 122.800 UNICOM).
* **State Awareness:** Reads X-Plane datarefs to know your exact coordinates, altitude, tail number, and whether you are airborne or parked.
* **Location Awareness:** Uses OpenStreetMap reverse geocoding to determine your exact real-world geographical airport location.
* **Live Weather Integration:** Dynamically pulls wind speed, direction, and altimeter settings directly from the simulator to provide accurate condition reports and QNH.

## Requirements
* Python 3.9+
* `xpc` (X-Plane Connect plugin installed in your simulator)
* `sd` / `sounddevice`, `scipy.io.wavfile`
* `edge_tts`, `pygame`
* `groq` API key (for fast Whisper transcription and Llama-3.3-70B responses)
* `keyboard`, `numpy`, `requests`