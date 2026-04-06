import asyncio
import xpc
from groq import Groq
import edge_tts
import pygame
import os
import sounddevice as sd
from scipy.io.wavfile import write
import keyboard
import numpy as np
import requests
import time

# --- CONFIGURATION ---
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)
FILENAME = "mic_input.wav"

async def record_audio_ptt(fs=44100):
    print("\n[Hold '+' to talk to ATC, or tune to 128.000 for ATIS...]")
    
    # Wait until the key is pressed OR frequency is ATIS
    while not keyboard.is_pressed('+'):
        # Check if we tuned to our dedicated ATIS frequency
        try:
            with xpc.XPlaneConnect() as xp:
                try:
                    com1_hz_val = xp.getDREF("sim/cockpit/radios/com1_freq_hz")[0]
                    com1_mhz = com1_hz_val / 100.0
                except:
                    com1_hz_val = xp.getDREF("sim/cockpit2/radios/actuators/com1_frequency_hz_833")[0]
                    com1_mhz = com1_hz_val / 1000.0
                    
                if abs(com1_mhz - 128.000) < 0.01:
                    return "ATIS"
        except:
            pass
            
        await asyncio.sleep(0.1)

    print("--- 🎙️ RECORDING (Release '+' to stop) ---")
    
    recording = []
    
    def callback(indata, frames, time, status):
        recording.append(indata.copy())
        
    # Start audio stream
    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        while keyboard.is_pressed('+'):
            await asyncio.sleep(0.05) # Small sleep to prevent high CPU usage
            
    print("--- ☑️ RECORDING FINISHED ---")
    
    if recording:
        audio_data = np.concatenate(recording, axis=0)
        write(FILENAME, fs, audio_data)
        return "PTT"
    return None

def generate_squelch():
    """Generates a realistic radio mic click/squelch sound using numpy."""
    if not os.path.exists("squelch.wav"):
        fs = 44100
        duration = 0.15 # 150 milliseconds of static
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        envelope = np.exp(-15 * t)
        noise = np.random.normal(0, 0.5, len(t)) * envelope
        audio = np.int16(noise / np.max(np.abs(noise)) * 32767)
        write("squelch.wav", fs, audio)

async def run_atc_loop():
    pygame.mixer.init()
    generate_squelch()
    squelch_sound = pygame.mixer.Sound("squelch.wav")
    chat_history = []
    
    while True:
        # 1. RECORD YOUR VOICE (Push-to-Talk) or wait for ATIS
        action = await record_audio_ptt()
        
        # Pull X-Plane data EARLY so the console prints match the real-time radio tuning!
        # This keeps our internal state instantly synchronized with the simulator.
        on_ground = False
        lat, lon, alt_ft = 0.0, 0.0, 0
        com1_mhz = 122.8
        wind_dir, wind_spd_kts, altim_inhg, qnh_mb = 0, 0, 29.92, 1013
        tail = "UNKNOWN"
        heading, airspeed = 0.0, 0.0
        
        try:
            with xpc.XPlaneConnect() as xp:
                # 1. Positional Data
                try:
                    posi = xp.getPOSI(0)
                    lat, lon = posi[0], posi[1]
                    alt_ft = int(posi[2] * 3.28084)
                except:
                    pass
                
                # 2. Radar Data (Heading & Speed)
                try:
                    heading_drefs = xp.getDREF("sim/flightmodel/position/mag_psi")
                    heading = heading_drefs[0] if heading_drefs else 0.0
                    
                    airspeed_drefs = xp.getDREF("sim/flightmodel/position/indicated_airspeed")
                    airspeed = airspeed_drefs[0] if airspeed_drefs else 0.0
                except:
                    pass
                
                # 3. Ground Status
                try:
                    agl_meters = xp.getDREF("sim/flightmodel/position/y_agl")[0]
                    on_ground = agl_meters < 15.0
                except:
                    on_ground = True
                    
                # 3. Weather Data (These often change names in X-Plane 12)
                try:
                    wind_dir_drefs = xp.getDREF("sim/weather/wind_direction_degs")
                    wind_dir = wind_dir_drefs[0] if wind_dir_drefs else 0
                    
                    wind_spd_drefs = xp.getDREF("sim/weather/wind_speed_kt")
                    wind_spd_kts = wind_spd_drefs[0] if wind_spd_drefs else 0
                    
                    altim_drefs = xp.getDREF("sim/weather/barometer_sealevel_inhg")
                    altim_inhg = altim_drefs[0] if altim_drefs else 29.92
                    qnh_mb = altim_inhg * 33.8639
                except:
                    pass

                # 4. Tail Number
                try:
                    tail_bytes = xp.getDREF("sim/aircraft/view/acf_tailnum")
                    if tail_bytes:
                        tail = "".join([chr(int(b)) for b in tail_bytes if b > 0]).strip()
                        if not tail: tail = "UNKNOWN"
                except:
                    pass

                # 5. Get COM1 Frequency
                try:
                    com1_hz_val = xp.getDREF("sim/cockpit/radios/com1_freq_hz")[0]
                    com1_mhz = com1_hz_val / 100.0  # 12190.0 / 100.0 = 121.90
                except:
                    try:
                        com1_hz_val = xp.getDREF("sim/cockpit2/radios/actuators/com1_frequency_hz_833")[0]
                        com1_mhz = com1_hz_val / 1000.0
                    except:
                        pass
                        
        except Exception as e:
            print(f"[X-Plane Connection Error] {e}")

        # 2. TRANSCRIBE (Voice to Text via Groq Whisper) or bypass for ATIS
        user_text = ""
        if action == "PTT":
            print("Transcribing...")
            with open(FILENAME, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(FILENAME, file.read()),
                    model="whisper-large-v3",
                    response_format="text",
                    language="en",  # FORCE ENGLISH TO PREVENT HALLUCINATIONS
                )
                
            user_text = transcription
            if not user_text.strip():
                print("Didn't hear anything. Try again.")
                continue

            print(f"You said: {user_text}")

        # Format location context for the AI
        location_name = "Local Area"
        if lat != 0.0 and lon != 0.0:
            try:
                headers = {'User-Agent': 'XPlane-AI-ATC/1.0'}
                # Extremely fast HTTP call to get the town/city name from coordinates
                res = requests.get(f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json", headers=headers, timeout=2.0)
                if res.status_code == 200:
                    data = res.json()
                    address = data.get("address", {})
                    # Grab city, or town, or fallback
                    location_name = address.get("city", address.get("town", address.get("county", "Local Area")))
                    print(f"[📍 Detected Location: {location_name}]")
            except Exception as e:
                pass # Location resolution failed, default to 'Local Area'

        # Format location context for the AI
        location_status = "ON THE GROUND" if on_ground else "AIRBORNE"
        
        weather_context = f"Wind {wind_dir:03.0f} at {wind_spd_kts:.0f} knots. Altimeter {altim_inhg:.2f} (QNH {qnh_mb:.0f})."

        # Strictly enforce ATC role to stop the AI from guessing
        atc_role = "APPROACH/CENTER" # Default for anything high frequency (126.0+)
        if 118.0 <= com1_mhz <= 121.5:
            atc_role = "TOWER"
        elif 121.6 <= com1_mhz <= 121.95: # Sola Ground is 121.90
            atc_role = "GROUND"
        elif 122.0 <= com1_mhz <= 123.05:
            atc_role = "UNICOM" # UNICOM (untowered)
        elif 123.05 < com1_mhz <= 126.0: # Sola Clearance is 121.925 or 125+ in standard airspace
            atc_role = "CLEARANCE DELIVERY"

        # 4. GET BRAIN RESPONSE (Llama 3.3 70B)
        if action == "ATIS":
            print(f"📻 Generating ATIS for {location_name}...")
            atis_number = int((time.time() // 3600) % 26)  # Changes every hour
            atis_letter = chr(65 + atis_number)
            system_prompt = (
                f"You are the automated ATIS broadcaster at {location_name}. "
                f"Current weather: Wind {wind_dir:03.0f} at {wind_spd_kts:.0f} knots. Altimeter {altim_inhg:.2f}. "
                f"Generate a short, realistic FAA ATIS broadcast. "
                f"Start with '{location_name} Airport, information {atis_letter}...'. "
                f"End with '...advise on initial contact, you have information {atis_letter}.' "
                f"Do not include any pleasantries or conversational filler. Output the standard script only."
            )
            messages = [{"role": "system", "content": system_prompt}]
        else:
            print(f"ATC Thinking (Role: {atc_role} on {com1_mhz:.3f} MHz)...")
            system_prompt = (
                f"You are a professional Air Traffic Controller acting at {location_name}. "
                f"The pilot's tail number is {tail} (If 'UNKNOWN', use their stated callsign). Aircraft state: {alt_ft:.0f}ft MSL, Heading {heading:03.0f}, Speed {airspeed:.0f} kts, currently {location_status}. "
                f"\n--- RADIO FREQUENCY LOGIC --- \n"
                f"The pilot is transmitting on {com1_mhz:.3f} MHz. I am explicitly assigning you the role of {atc_role}. "
                f"CRITICAL ROLE AUTHORITIES:\n"
                f"- If {atc_role} is UNICOM: You are NOT ATC. You CANNOT give IFR clearances, pushback, taxi, takeoff, or landing clearances. Tell the pilot they are on UNICOM and must tune to the correct ATC frequency for {location_name}.\n"
                f"- If {atc_role} is GROUND: You can give IFR, pushback, and taxi clearances. You CANNOT give takeoff or landing clearances.\n"
                f"- If {atc_role} is TOWER: You can give takeoff and landing clearances. You CANNOT give IFR or pushback clearances.\n"
                f"- If {atc_role} is APPROACH/CENTER: Provide radar vectoring (assign headings and altitudes) and approach clearances. Use the pilot's current heading ({heading:03.0f}) and airspeed ({airspeed:.0f} kts) to formulate instructions.\n"
                f"If the pilot asks for a clearance outside your {atc_role} authority, STRICTLY DENY IT and tell them to contact the proper frequency.\n"
                f"\n--- WEATHER & STATE ---\n"
                f"CURRENT WEATHER: {weather_context} "
                f"\n--- RULES ---\n"
                f"CRITICAL: DO NOT anticipate the pilot's next request. If the pilot reads something back correctly, ONLY acknowledge it ('Roger', 'Readback correct'). DO NOT give the next clearance. "
                f"1. IFR CLEARANCE (Clearance/Ground ONLY): Provide clearance limit, route, initial altitude, real-world departure frequency, squawk code, AND expected departure runway. "
                f"2. READBACKS: Acknowledge briefly and naturally (e.g., 'Roger'). VARY YOUR RESPONSES. Do not repeatedly say 'readback correct'. Do not add any new instructions. "
                f"3. PUSHBACK/START (Ground ONLY): Approve push/start and advise which way to face. "
                f"4. TAXI (Ground ONLY): Assign a single runway, realistic taxiways, give Altimeter/QNH, and state 'hold short of runway [X]'. "
                f"5. TAKEOFF/LANDING (Tower ONLY): Provide wind conditions and state 'cleared for takeoff/land runway [X]'. "
                f"6. VECTORING (Approach/Center ONLY): Instruct headings ('fly heading [XYZ]'), altitudes ('climb/descend and maintain [X] thousand'), and clear for approaches. "
                f"Keep responses strictly in professional FAA/ICAO phraseology."
            )

            # Build Memory Conversation
            chat_history.append({"role": "user", "content": user_text})
            if len(chat_history) > 20: # Keep enough context for the whole ground flow (10 back-and-forths)
                chat_history.pop(0)
                
            messages = [{"role": "system", "content": system_prompt}] + chat_history

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        response_text = completion.choices[0].message.content
        print(f"ATC: {response_text}")

        # Dynamically ensure consistent voices for specific frequencies
        available_voices = [
            "en-US-GuyNeural", "en-GB-RyanNeural", "en-US-AriaNeural", 
            "en-US-ChristopherNeural", "en-GB-SoniaNeural", "en-US-EricNeural"
        ]
        if action == "ATIS":
            tts_voice = "en-US-JennyNeural"  # Professional female automated voice for ATIS
        else:
            # Hash the frequency so 121.9 always sounds like the same person, but 118.3 is someone else!
            voice_index = int(com1_mhz * 1000) % len(available_voices)
            tts_voice = available_voices[voice_index]
        
        # Save response to history ONLY if it was an ATC interaction
        if action == "PTT":
            chat_history.append({"role": "assistant", "content": response_text})

        # 5. SPEAK (Edge-TTS)
        communicate = edge_tts.Communicate(response_text, tts_voice)
        await communicate.save("response.mp3")

        # 🎙️ PLAY RADIO CLICK ON
        squelch_sound.play()
        await asyncio.sleep(0.15) # Wait for the click to finish

        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        
        # Monitor frequency during playback to allow instant cut-off
        try:
            with xpc.XPlaneConnect() as xp_monitor:
                while pygame.mixer.music.get_busy():
                    current_mhz = com1_mhz  # Default to what we had
                    # Poll frequency
                    try:
                        hz_val = xp_monitor.getDREF("sim/cockpit/radios/com1_freq_hz")[0]
                        current_mhz = hz_val / 100.0
                    except:
                        try:
                            hz_val = xp_monitor.getDREF("sim/cockpit2/radios/actuators/com1_frequency_hz_833")[0]
                            current_mhz = hz_val / 1000.0
                        except:
                            pass
                    
                    if abs(current_mhz - com1_mhz) > 0.01:
                        print("[Frequency changed. Radio transmission cut off.]")
                        pygame.mixer.music.stop()
                        break
                        
                    await asyncio.sleep(0.1)
        except Exception:
            # Fallback if connection fails during playback
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)

        # 🎙️ PLAY RADIO CLICK OFF
        squelch_sound.play()
        await asyncio.sleep(0.2) # Wait for the unkey click to finish

        # ADD THESE TWO LINES TO UNLOCK THE FILE
        pygame.mixer.music.unload() 
        pygame.mixer.music.stop()
        
if __name__ == "__main__":
    asyncio.run(run_atc_loop())