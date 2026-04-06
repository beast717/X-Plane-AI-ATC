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
import math

# --- CONFIGURATION ---
GROQ_API_KEY = "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"
SIMBRIEF_USERNAME = "meddho11" # Change this to your Simbrief username to automatically load cleared routes!
client = Groq(api_key=GROQ_API_KEY)
FILENAME = "mic_input.wav"

async def record_audio_ptt(squelch=None, fs=44100):
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

    if squelch:
        squelch.play()
        
    print("--- 🎙️ RECORDING (Release '+' to stop) ---")
    
    recording = []
    
    def callback(indata, frames, time, status):
        recording.append(indata.copy())
        
    # Start audio stream
    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        while keyboard.is_pressed('+'):
            await asyncio.sleep(0.05) # Small sleep to prevent high CPU usage

    if squelch:
        squelch.play()
            
    print("--- ☑️ RECORDING FINISHED ---")
    
    if recording:
        audio_data = np.concatenate(recording, axis=0)
        write(FILENAME, fs, audio_data)
        return "PTT"
    return None

def generate_squelch():
    """Generates a realistic radio mic click/squelch sound using numpy."""
    fs = 44100
    duration = 0.12 # 120 milliseconds of squelch
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # 1. Background static noise
    noise = np.random.normal(0, 0.8, len(t))
    noise_env = np.ones_like(t)
    noise_env[:int(fs * 0.01)] = np.linspace(0, 1, int(fs * 0.01)) # Fast attack
    noise_env[-int(fs * 0.03):] = np.linspace(1, 0, int(fs * 0.03)) # Fast decay
    
    # 2. Synthesize mechanical mic clicks at the beginning and end
    click1_env = np.exp(-500 * t)
    click1 = np.sin(2 * np.pi * 2000 * t) * click1_env
    
    click2_env = np.exp(-400 * t[::-1])
    click2 = np.sin(2 * np.pi * 1200 * t) * click2_env
    
    # Mix together: clicks are louder, noise is in the background
    signal = (noise * noise_env * 0.4) + (click1 * 0.8) + (click2 * 0.5)
    
    # Normalize and lower volume slightly so it doesn't blow out eardrums
    audio = np.int16(signal / np.max(np.abs(signal)) * 24000)
    write("radio_click.wav", fs, audio)

def generate_heavy_static():
    """Generates pure radio static when out of range."""
    if not os.path.exists("heavy_static.wav"):
        fs = 44100
        duration = 1.5 # 1.5 seconds of loud static
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        noise = np.random.normal(0, 0.3, len(t))
        audio = np.int16(noise * 32767)
        write("heavy_static.wav", fs, audio)

def haversine_nm(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in Nautical Miles."""
    R = 3440.065 # Earth radius in NM
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate initial bearing from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    return (bearing + 360) % 360

async def run_atc_loop():
    pygame.mixer.init()
    generate_squelch()
    generate_heavy_static()
    squelch_sound = pygame.mixer.Sound("radio_click.wav")
    heavy_static_sound = pygame.mixer.Sound("heavy_static.wav")
    chat_history = []
    
    # 🌍 Cache for Runways & Airport Info to prevent spamming the APIs
    location_cache = {}
    runway_cache = {}
    taxiway_cache = {}
    metar_cache = {}
    
    # 📻 Radio Line of Sight Tracker
    active_station = {"mhz": 0.0, "lat": 0.0, "lon": 0.0}

    def get_real_metar(lat, lon):
        # Round digits to cache the grid securely
        cache_key = f"{round(lat, 1)},{round(lon, 1)}"
        if cache_key in metar_cache and (time.time() - metar_cache[cache_key]['time'] < 1800):
            return metar_cache[cache_key]['metar']
            
        print("🌤️ Fetching real-world METAR for current location...")
        try:
            bbox = f"{lat-0.5},{lon-0.5},{lat+0.5},{lon+0.5}"
            url = f"https://aviationweather.gov/api/data/metar?format=json&bbox={bbox}"
            r = requests.get(url, timeout=3.0)
            data = r.json()
            if data and len(data) > 0:
                metar_str = data[0].get('rawOb', 'Unknown')
                metar_cache[cache_key] = {'metar': metar_str, 'time': time.time()}
                return metar_str
        except Exception:
            pass
        return None

    def get_runways_from_overpass(lat, lon, radius=5000):
        # Round digits to cache the grid instead of firing for every meter moved
        cache_key = f"{round(lat, 2)},{round(lon, 2)}"
        if cache_key in runway_cache:
            return runway_cache[cache_key]

        print("📡 Querying real-world aviation databases for active runways...")
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        way["aeroway"="runway"](around:{radius},{lat},{lon});
        out tags;
        """
        try:
            r = requests.get(overpass_url, params={'data': query}, timeout=3.0)
            data = r.json()
            runways = []
            for element in data.get('elements', []):
                tags = element.get('tags', {})
                ref = tags.get('ref')
                if ref and len(ref) <= 7: # Make sure it's actually a runway like "11/29" or "09"
                    runways.append(ref)
            
            # Remove duplicates and clean up
            runways = list(set([r.replace(" ", "") for r in runways]))
            runway_cache[cache_key] = runways
            return runways
        except Exception:
            return []

    def get_taxiways_from_overpass(lat, lon, radius=5000):
        cache_key = f"{round(lat, 2)},{round(lon, 2)}"
        if cache_key in taxiway_cache:
            return taxiway_cache[cache_key]

        print("🚖 Querying real-world aviation databases for active taxiways...")
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        way["aeroway"="taxiway"](around:{radius},{lat},{lon});
        out tags;
        """
        try:
            r = requests.get(overpass_url, params={'data': query}, timeout=3.0)
            data = r.json()
            taxiways = []
            for element in data.get('elements', []):
                tags = element.get('tags', {})
                ref = tags.get('ref')
                if ref and len(ref) <= 3: # usually taxiways are 1-2 letters like "A", "M6"
                    taxiways.append(ref.upper())
            
            taxiways = list(set(taxiways))
            taxiway_cache[cache_key] = taxiways
            return taxiways
        except Exception:
            return []

    simbrief_cache = None
    last_simbrief_fetch = 0
    def get_simbrief_flight_plan():
        nonlocal simbrief_cache, last_simbrief_fetch
        if not SIMBRIEF_USERNAME or SIMBRIEF_USERNAME == "YOUR_SIMBRIEF_USERNAME":
            return None
            
        # Only ping the Simbrief API once every 5 minutes to prevent ratelimiting
        if simbrief_cache and (time.time() - last_simbrief_fetch < 300):
            return simbrief_cache
            
        try:
            print(f"📡 Fetching latest SimBrief flight plan for {SIMBRIEF_USERNAME}...")
            url = f"https://www.simbrief.com/api/xml.fetcher.php?username={SIMBRIEF_USERNAME}&json=1"
            headers = {'User-Agent': 'XPlane-AI-ATC/1.0'}
            res = requests.get(url, headers=headers, timeout=5.0)
            
            if res.status_code == 200:
                data = res.json()
                origin = data.get("origin", {}).get("icao_code", "UNKNOWN")
                dest = data.get("destination", {}).get("icao_code", "UNKNOWN")
                dest_lat = float(data.get("destination", {}).get("pos_lat", "0"))
                dest_lon = float(data.get("destination", {}).get("pos_long", "0"))
                
                route = data.get("general", {}).get("route", "Direct")
                alt = data.get("general", {}).get("initial_altitude", "UNKNOWN")
                
                simbrief_cache = {
                    "origin": origin,
                    "destination": dest,
                    "dest_lat": dest_lat,
                    "dest_lon": dest_lon,
                    "route": route,
                    "altitude": alt
                }
                last_simbrief_fetch = time.time()
                print(f"[✅ SimBrief Plan Loaded: {origin} to {dest} at FL{alt}]")
                return simbrief_cache
        except Exception as e:
            # print(f"⚠️ SimBrief error: {e}")
            pass
        return None

    while True:
        # 1. RECORD YOUR VOICE (Push-to-Talk) or wait for ATIS
        action = await record_audio_ptt(squelch_sound)
        
        # Pull X-Plane data EARLY so the console prints match the real-time radio tuning!
        # This keeps our internal state instantly synchronized with the simulator.
        on_ground = False
        vsi_fpm = 0.0
        lat, lon, alt_ft = 0.0, 0.0, 0
        com1_mhz = 122.8
        wind_dir, wind_spd_kts, altim_inhg, qnh_mb = 0, 0, 29.92, 1013
        tail = "UNKNOWN"
        heading, airspeed = 0.0, 0.0
        squawk_code = 1200
        nav1_freq = 0.0
        nav1_hdef = 0.0
        
        try:
            with xpc.XPlaneConnect() as xp:
                # 1. Positional Data
                try:
                    posi = xp.getPOSI(0)
                    lat, lon = posi[0], posi[1]
                    alt_ft = int(posi[2] * 3.28084)
                    
                    vsi_raw = xp.getDREF("sim/flightmodel/position/vh_ind_fpm")
                    if vsi_raw:
                        vsi_fpm = vsi_raw[0]
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
                    
                # 4. Transponder (Squawk Code)
                try:
                    squawk_raw = xp.getDREF("sim/cockpit/radios/transponder_code")
                    if squawk_raw:
                        squawk_code = int(squawk_raw[0])
                except:
                    pass

                # 4.5 ILS / NAV1 Tracking
                try:
                    nav1_raw = xp.getDREF("sim/cockpit/radios/nav1_freq_hz")
                    if nav1_raw:
                        nav1_freq = nav1_raw[0] / 100.0
                    nav1_def = xp.getDREF("sim/cockpit2/radios/indicators/nav1_hdef_dots_pilot")
                    if nav1_def:
                        nav1_hdef = nav1_def[0]
                except:
                    pass
                    
                # 5. Weather Data (These often change names in X-Plane 12)
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
        active_runways = []
        active_taxiways = []
        if lat != 0.0 and lon != 0.0:
            active_runways = get_runways_from_overpass(lat, lon)
            active_taxiways = get_taxiways_from_overpass(lat, lon)
            
            # Use Nominatim for the City Name
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
        
        real_metar = get_real_metar(lat, lon)
        if real_metar:
            weather_context = f"METAR: {real_metar}"
        else:
            weather_context = f"Wind {wind_dir:03.0f} at {wind_spd_kts:.0f} knots. Altimeter {altim_inhg:.2f} (QNH {qnh_mb:.0f})."

        # ILS / NAV1 Tracking
        nav_info = ""
        if nav1_freq >= 108.0 and not on_ground:
            nav_info = f"NAV1 Tuned: {nav1_freq:.2f}."
            if nav1_hdef > 1.5:
                nav_info += " (PILOT IS DRIFTING FAR LEFT OF THE LOCALIZER)"
            elif nav1_hdef < -1.5:
                nav_info += " (PILOT IS DRIFTING FAR RIGHT OF THE LOCALIZER)"
            elif abs(nav1_hdef) <= 1.5:
                nav_info += " (PILOT IS ESTABLISHED ON LOCALIZER)"

        # Automatically look up pilot's filed flight plan
        simbrief_data = get_simbrief_flight_plan()
        flight_plan_context = "Pilot has NO IFR flight plan filed. Ask intentions."
        if simbrief_data:
            flight_plan_context = f"Pilot's filed IFR plan: Destination {simbrief_data['destination']}, Route: {simbrief_data['route']}, Initial FL: {simbrief_data['altitude']}."
            
            # Destination logic calculations
            if simbrief_data.get('dest_lat') and simbrief_data.get('dest_lon') and not on_ground:
                dest_dist = haversine_nm(lat, lon, simbrief_data['dest_lat'], simbrief_data['dest_lon'])
                dest_brg = calculate_bearing(lat, lon, simbrief_data['dest_lat'], simbrief_data['dest_lon'])
                flight_plan_context += f" | Distance to dest: {dest_dist:.1f} NM. Bearing to dest: {dest_brg:03.0f} degrees."

        # Strictly enforce ATC role to stop the AI from guessing
        atc_role = "APPROACH/CENTER" # Default for anything high frequency
        if 118.0 <= com1_mhz <= 121.5:
            atc_role = "TOWER"
        elif 121.6 <= com1_mhz <= 121.95: # Sola Ground is 121.90
            atc_role = "GROUND"
        elif 122.0 <= com1_mhz <= 123.05:
            atc_role = "UNICOM" # UNICOM (untowered)
        elif 123.05 < com1_mhz < 124.0: # Clearance Delivery
            atc_role = "CLEARANCE DELIVERY"

        # VHF RADIO PHYSICS (Line of Sight & Range Checks)
        # Lock onto the station's coordinates when we first tune the frequency
        if abs(com1_mhz - active_station["mhz"]) > 0.01:
            active_station["mhz"] = com1_mhz
            active_station["lat"] = lat
            active_station["lon"] = lon

        # Calculate distance to locked station location
        station_distance_nm = haversine_nm(lat, lon, active_station["lat"], active_station["lon"])
        
        # Max LOS distance rough formula: 1.23 * sqrt(altitude in feet) + 10 NM base
        max_los_nm = 1.23 * math.sqrt(max(alt_ft, 0)) + 10.0
        
        # Restrict heavily by ATC role (Ground shouldn't be heard from 100 NM away even at FL350)
        if atc_role in ["GROUND", "CLEARANCE DELIVERY"]:
            max_range = min(max_los_nm, 15.0) # Ground radios are weak
        elif atc_role == "TOWER":
            max_range = min(max_los_nm, 50.0) # Tower reaches a bit further
        else:
            max_range = max_los_nm # Center has huge antennas
            
        if station_distance_nm > max_range:
            print(f"📻 [OUT OF RANGE] {station_distance_nm:.1f} NM away. Max range: {max_range:.1f} NM.")
            heavy_static_sound.play()
            await asyncio.sleep(1.5)
            continue # ABORT THE LLM REQUEST. The controller never heard you!

        # 4. GET BRAIN RESPONSE (Llama 3.3 70B)
        if action == "ATIS":
            print(f"📻 Generating ATIS for {location_name}...")
            atis_number = int((time.time() // 3600) % 26)  # Changes every hour
            atis_letter = chr(65 + atis_number)
            system_prompt = (
                f"You are the automated ATIS broadcaster at {location_name}. "
                f"Current weather: {weather_context}. "
                f"Generate a short, realistic FAA ATIS broadcast. "
                f"Start with '{location_name} Airport, information {atis_letter}...'. "
                f"End with '...advise on initial contact, you have information {atis_letter}.' "
                f"CRITICAL: Spell out the ATIS letter using the NATO phonetic alphabet (e.g., if it is A, write out 'Alpha'). "
                f"Do not include any pleasantries or conversational filler. Output the standard script only."
            )
            messages = [{"role": "system", "content": system_prompt}]
        else:
            print(f"ATC Thinking (Role: {atc_role} on {com1_mhz:.3f} MHz)...")
            system_prompt = (
                f"You are a professional Air Traffic Controller acting at {location_name}. "
                f"The pilot's tail number is {tail} (If 'UNKNOWN', use their stated callsign). "
                f"Aircraft state: {alt_ft:.0f}ft MSL, Heading {heading:03.0f}, Speed {airspeed:.0f} kts, VSI {vsi_fpm:.0f} fpm, currently {location_status}, Squawk {squawk_code}. {nav_info}\n"
                f"Real reported runways at this physical location: {', '.join(active_runways) if active_runways else 'Unknown (assign a generic runway like 27)'}. "
                f"Real reported taxiways at this physical location: {', '.join(active_taxiways) if active_taxiways else 'Unknown (invent realistic taxiways like A, B, C)'}. "
                f"\n--- RADIO FREQUENCY LOGIC & HANDOFFS --- \n"
                f"The pilot is transmitting on {com1_mhz:.3f} MHz. I am explicitly assigning you the role of {atc_role}. "
                f"Available Frequencies for {location_name}: Clearance (121.92), Ground (121.9), Tower (118.5), Approach/Departure (124.5), Center (126.0). "
                f"CRITICAL ROLE AUTHORITIES & HANDOFF RULES:\n"
                f"- If {atc_role} is UNICOM: You are NOT ATC. Tell the pilot they are on UNICOM and must tune to the correct ATC frequency for {location_name}.\n"
                f"- If {atc_role} is GROUND: You can give IFR, pushback, and taxi clearances. When the pilot is ready for departure at the runway, explicitly tell them 'Contact Tower on 118.5'.\n"
                f"- If {atc_role} is TOWER: You can give takeoff and landing clearances. When the pilot is airborne and climbing out, explicitly tell them 'Contact Departure on 124.5'. Upon landing and exiting the runway, tell them 'Contact Ground on 121.9'.\n"
                f"- If {atc_role} is APPROACH/CENTER: Provide radar vectoring and approach clearances. Use the pilot's current heading ({heading:03.0f}) and airspeed ({airspeed:.0f} kts). \n"
                f"If the pilot asks for a clearance outside your {atc_role} authority, STRICTLY DENY IT.\n"
                f"\n--- FLIGHT PLAN & WEATHER ---\n"
                f"CURRENT WEATHER: {weather_context} "
                f"\nSIMBRIEF FLIGHT PLAN: {flight_plan_context}\n"
                f"\n--- RULES ---\n"
                f"CRITICAL: 1. ONE INSTRUCTION AT A TIME. Respond only to what the pilot is explicitly requesting right now. If they ask for pushback, ONLY give pushback, DO NOT give taxi. NEVER ask the pilot if they are ready for the next step. NEVER give taxi instructions until the pilot explicitly says 'ready for taxi'.\n"
                f"CRITICAL: 2. ACKNOWLEDGING READBACKS: If the pilot is reading back a clearance, you MUST NOT give any further instructions automatically. DO NOT add handoffs or taxi clearances to an acknowledgement. NEVER use 'Roger' to acknowledge a clearance readback. Use strict ICAO/FAA phraseology (e.g., '[Callsign], readback correct', or just state their '[Callsign]').\n"
                f"CRITICAL: 3. DO NOT MICROMANAGE. Maintain a professional, detached ATC tone. Avoid redundant phrasing like 'Clearance for [Callsign]'. Just say '[Callsign], cleared to...'. Simply provide the clearance or vector. Just say 'Taxi via [route] hold short runway [X]'.\n"
                f"CRITICAL: 4. HANDOFFS MUST BE TIMED CORRECTLY AND NEVER COMBINED WITH CLEARANCES:\n"
                f"  - Ground to Tower: DO NOT handoff during taxi clearance. ONLY instruct 'Contact Tower on 118.5' when the pilot explicitly reports holding short of the runway.\n"
                f"  - Tower to Departure: DO NOT handoff during takeoff clearance. ONLY instruct 'Contact Departure on 124.5' when the aircraft is explicitly reported AIRBORNE or location_status is AIRBORNE.\n"
                f"CRITICAL: 5. DO NOT CONTRADICT SIMBRIEF. If {atc_role} is APPROACH/CENTER and aircraft is climbing, clear them up to their cruise altitude of {simbrief_data['altitude'] if simbrief_data else 'unknown'}.\n"
                f"CRITICAL: 6. RUNWAY CONSISTENCY. Maintain a strictly consistent active runway state. Review the chat history! The runway you assign for taxi, holding short, and takeoff MUST identically match the 'expect runway' you assigned in the initial IFR clearance.\n"
                f"CRITICAL: 7. PHONETIC ALPHABET. ALWAYS use the NATO phonetic alphabet for single letters (e.g., say 'Taxiway Alpha', not 'Taxiway A'; 'Information Bravo', not 'Information B').\n"
                f"8. SQUAWK CODES: If squawk is 1200 instruct 'squawk [4-digit code]'.\n"
                f"9. IFR CLEARANCE (Clearance/Ground ONLY): Assign a lower INITIAL altitude (e.g., 5000 or 7000), NOT cruise. Provide clearance limit, route (DO NOT read the entire route string; read only the first waypoint/SID, then state 'then as filed'), initial altitude, departure frequency (which is 124.5 for Departure, NEVER 121.92), squawk code, and expected runway. Do NOT give pushback or taxi instructions here.\n"
                f"10. TRANSCRIPTION: Ignore minor speech transcription artifacts ('10-20', hyphens, '121.94', weird callsign glitches like '81777'). Use the pilot's stated callsign from the transcription.\n"
                f"11. WEATHER CONSIDERATIONS: Review the METAR. You MUST assign runways that prioritize headwinds over tailwinds. If wind speed is > 8kts, strictly avoid assigning crossing or tailwind runways if a direct headwind runway is available.\n"
                f"12. MANAGING DESCENTS (Approach/Center): Monitor the flight plan's Distance to Destination. If distance < 120 NM and altitude > 15000, proactively instruct 'descend and maintain [lower FL/Altitude]'.\n"
                f"13. SMART APPROACH VECTORING (Approach/Center): If distance to destination < 30 NM, use the pilot's Bearing to Dest to vector them into the localizer. (e.g. 'fly heading [intercept heading], cleared ILS approach runway [X]'). Do not give generic vectors forever.\n"
                f"14. ALTITUDE PHRASEOLOGY: For altitudes 18,000 feet and above, you MUST use 'Flight Level' (e.g., 'climb and maintain Flight Level 290'). For altitudes below 18,000 feet, use thousands and hundreds (e.g., 'climb and maintain one two thousand').\n"
                f"15. SPEED CONTROL (Approach/Center): Issue speed restrictions when necessary for spacing (e.g., 'reduce speed to 250 knots', 'maintain 160 knots to the marker').\n"
                f"16. TRANSITION ALTIMETER: When clearing an aircraft to descend below 18,000 feet for the first time, you MUST provide the local altimeter setting (e.g., 'Sola altimeter {altim_inhg:.2f}').\n"
                f"17. MISSED APPROACH / GO-AROUND: If distance to destination is < 10 NM, altitude < 3000, and VSI is strongly positive (> 600 fpm), the aircraft is executing a missed approach. Proactively instruct 'observed going around, fly runway heading, climb and maintain [X]...'.\n"
                f"18. RADAR CONTACT & SQUAWK: On initial contact with Departure/Center, if squawk does not match what you previously assigned, say 'squawk [code]'. If correct, say 'Radar contact [altitude]'.\n"
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

        # START BACKGROUND STATIC
        static_channel = heavy_static_sound.play(loops=-1)
        if static_channel:
            static_channel.set_volume(0.04)

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

        # STOP BACKGROUND STATIC
        if static_channel:
            static_channel.stop()

        # 🎙️ PLAY RADIO CLICK OFF
        squelch_sound.play()
        await asyncio.sleep(0.2) # Wait for the unkey click to finish

        # ADD THESE TWO LINES TO UNLOCK THE FILE
        pygame.mixer.music.unload() 
        pygame.mixer.music.stop()
        
if __name__ == "__main__":
    asyncio.run(run_atc_loop())