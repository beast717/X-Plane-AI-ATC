import asyncio
import xpc
from groq import Groq
import edge_tts
import pygame
import os
from dotenv import load_dotenv
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import butter, sosfilt
import keyboard
import numpy as np
import requests
import time
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import re

load_dotenv()

# --- CONFIGURATION ---
# It is recommended to use environment variables instead of hardcoding credentials:
# export GROQ_API_KEY="your_key_here"
# export SIMBRIEF_USERNAME="your_username_here"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
SIMBRIEF_USERNAME = os.environ.get("SIMBRIEF_USERNAME", "")  # Leave blank to disable SimBrief
client = Groq(api_key=GROQ_API_KEY)
FILENAME = "mic_input.wav"

# --- AUDIO RADIO EFFECT SETTINGS ---
RADIO_LOWPASS_HZ = 3000   # VHF voice top cutoff
RADIO_HIGHPASS_HZ = 300   # VHF voice bottom cutoff
RADIO_NOISE_LEVEL = 0.008 # Subtle noise under voice (0 to disable)

# --- FLIGHT STATE ENUMS ---
class FlightPhase(Enum):
    """Explicit flight phase tracking for phase-aware prompting."""
    PREFLIGHT = "preflight"          # Before startup
    STARTUP = "startup"              # Engine start request
    PUSHBACK = "pushback"            # Pushback clearance given
    TAXI = "taxi"                    # Actively taxiing
    HOLD_SHORT = "hold_short"        # Holding short of runway
    TAKEOFF = "takeoff"              # Takeoff clearance given / airborne
    CLIMB = "climb"                  # Climbing to cruise
    CRUISE = "cruise"                # Cruise phase
    DESCENT = "descent"              # Descending
    APPROACH = "approach"            # On approach
    FINAL = "final"                  # On final approach
    GO_AROUND = "go_around"          # Missed approach / go-around
    LANDED = "landed"                # Touched down
    ROLLOUT = "rollout"              # Landing rollout
    TAXI_IN = "taxi_in"              # Taxiing to parking

# --- FLIGHT STATE TRACKER ---
@dataclass
class FlightStateTracker:
    """
    Persistent state machine that survives frequency changes and pilot interactions.
    Tracks assigned values to prevent hallucination and ensure runway/squawk consistency.
    """
    phase: FlightPhase = FlightPhase.PREFLIGHT
    
    # Assigned clearance values (set once, never re-issued without pilot request)
    assigned_runway: Optional[str] = None          # e.g., "27R"
    assigned_altitude: Optional[str] = None        # e.g., "FL250" or "3000"
    assigned_heading: Optional[float] = None       # magnetic heading in degrees
    assigned_speed: Optional[float] = None         # in knots
    assigned_squawk: Optional[int] = None          # 4-digit code, e.g., 5546
    
    # Tracking what instructions have been given in this phase
    instructions_given_this_phase: list = field(default_factory=list)
    
    def update_phase(self, new_phase: FlightPhase):
        """Transition to a new flight phase and clear phase-specific history."""
        if self.phase != new_phase:
            print(f"[✈️ Flight Phase Transition: {self.phase.name} → {new_phase.name}]")
            self.phase = new_phase
            self.instructions_given_this_phase.clear()
    
    def extract_and_commit_llm_assignment(self, llm_response: str):
        """
        Parse LLM response and extract/commit any newly assigned values.
        E.g., if LLM says "squawk 5546", extract 5546 and update state.
        """
        # Extract squawk code (4 digits)
        squawk_match = re.search(r'\bsquawk\s+(\d{4})\b', llm_response, re.IGNORECASE)
        if squawk_match:
            self.assigned_squawk = int(squawk_match.group(1))
            
        # Extract runway (e.g., "27", "27R", "27L", "27C") - be careful not to catch "runway heading" or similar
        runway_match = re.search(r'\b(?:runway|rwy)\s+([0-3]\d[LRC]?)\b', llm_response, re.IGNORECASE)
        if runway_match:
            self.assigned_runway = runway_match.group(1).upper()
            
        # Extract altitude clearance
        alt_match = re.search(r'\b(?:climb to|descend to|maintain|Flight Level)\s+([FL]*[\d,]+)\b', 
                            llm_response, re.IGNORECASE)
        if alt_match:
            self.assigned_altitude = alt_match.group(1).replace(",", "")

        # Infer phase updates from ATC response keywords
        text = llm_response.lower()
        if "cleared to" in text and "route" in text and "altitude" in text:
            self.update_phase(FlightPhase.PREFLIGHT)
        elif "pushback and start approved" in text or "pushback approved" in text:
            self.update_phase(FlightPhase.PUSHBACK)
        elif "taxi to holding point" in text or "taxi via" in text:
            self.update_phase(FlightPhase.TAXI)
        elif "cleared for takeoff" in text:
            self.update_phase(FlightPhase.TAKEOFF)
        elif "cleared to land" in text:
            self.update_phase(FlightPhase.FINAL)


def apply_radio_effect(audio_path: str, out_path: str, fs: int = 44100):
    """
    Post-processes a WAV file to sound like a VHF radio transmission:
    - Bandpass filter (300–3000 Hz)
    - Subtle background noise under voice
    - Light normalization
    """
    try:
        from scipy.io.wavfile import read as wav_read
        sample_rate, data = wav_read(audio_path)

        # Convert to float32 for processing
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        else:
            audio = data.astype(np.float32)

        if audio.ndim > 1:
            audio = audio[:, 0]  # Use mono

        # Bandpass filter: 300 Hz – 3000 Hz
        sos_hp = butter(4, RADIO_HIGHPASS_HZ / (sample_rate / 2), btype='high', output='sos')
        sos_lp = butter(4, RADIO_LOWPASS_HZ / (sample_rate / 2), btype='low', output='sos')
        audio = sosfilt(sos_hp, audio)
        audio = sosfilt(sos_lp, audio)

        # Add subtle static noise under the voice
        if RADIO_NOISE_LEVEL > 0:
            noise = np.random.normal(0, RADIO_NOISE_LEVEL, len(audio))
            audio = audio + noise

        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.92

        write(out_path, sample_rate, (audio * 32767).astype(np.int16))
    except Exception as e:
        print(f"[Radio effect failed, using clean audio: {e}]")
        import shutil
        shutil.copy(audio_path, out_path)


async def record_audio_ptt(squelch=None, fs=44100):
    print("\n[Hold '+' to talk to ATC, or tune to 128.000 for ATIS...]")

    # Wait until PTT key is pressed OR frequency is ATIS
    while not keyboard.is_pressed('+'):
        try:
            with xpc.XPlaneConnect() as xp:
                try:
                    com1_hz_val = xp.getDREF("sim/cockpit/radios/com1_freq_hz")[0]
                    com1_mhz = com1_hz_val / 100.0
                except Exception:
                    com1_hz_val = xp.getDREF("sim/cockpit2/radios/actuators/com1_frequency_hz_833")[0]
                    com1_mhz = com1_hz_val / 1000.0

                if abs(com1_mhz - 128.000) < 0.01:
                    return "ATIS"
        except Exception:
            pass

        await asyncio.sleep(0.1)

    if squelch:
        squelch.play()

    print("--- 🎙️ RECORDING (Release '+' to stop) ---")

    recording = []

    def callback(indata, frames, time, status):
        recording.append(indata.copy())

    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        while keyboard.is_pressed('+'):
            await asyncio.sleep(0.05)

    if squelch:
        squelch.play()

    print("--- ☑️ RECORDING FINISHED ---")

    if recording:
        audio_data = np.concatenate(recording, axis=0)

        # FIX: Reject recordings that are too short (accidental key tap)
        duration_secs = len(audio_data) / fs
        if duration_secs < 0.3:
            print("[Recording too short, ignoring]")
            return None

        write(FILENAME, fs, audio_data)
        return "PTT"
    return None


def generate_squelch():
    """Generates a realistic radio mic click/squelch sound using numpy.
    Only regenerates if the file doesn't already exist."""
    if os.path.exists("radio_click.wav"):
        return

    fs = 44100
    duration = 0.12
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    noise = np.random.normal(0, 0.8, len(t))
    noise_env = np.ones_like(t)
    noise_env[:int(fs * 0.01)] = np.linspace(0, 1, int(fs * 0.01))
    noise_env[-int(fs * 0.03):] = np.linspace(1, 0, int(fs * 0.03))

    click1_env = np.exp(-500 * t)
    click1 = np.sin(2 * np.pi * 2000 * t) * click1_env

    click2_env = np.exp(-400 * t[::-1])
    click2 = np.sin(2 * np.pi * 1200 * t) * click2_env

    signal = (noise * noise_env * 0.4) + (click1 * 0.8) + (click2 * 0.5)
    audio = np.int16(signal / np.max(np.abs(signal)) * 24000)
    write("radio_click.wav", fs, audio)


def generate_heavy_static():
    """Generates pure radio static when out of range."""
    if not os.path.exists("heavy_static.wav"):
        fs = 44100
        duration = 1.5
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        noise = np.random.normal(0, 0.3, len(t))
        audio = np.int16(noise * 32767)
        write("heavy_static.wav", fs, audio)


def haversine_nm(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in Nautical Miles."""
    R = 3440.065
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
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360

def get_best_runway(runways, wind_dir):
    """Calculates the best runway based on headwind."""
    if not runways:
        return None
        
    best_runway = runways[0]
    smallest_diff = 180  # Max possible difference in degrees

    for rw in runways:
        # Extract the number part of the runway (e.g., "16L" -> 16)
        import re
        match = re.match(r"(\d+)", rw)
        if match:
            rw_hdg = int(match.group(1)) * 10
            
            # Calculate the angular difference between wind and runway heading
            diff = abs((wind_dir - rw_hdg + 180) % 360 - 180)
            
            if diff < smallest_diff:
                smallest_diff = diff
                best_runway = rw
                
    return best_runway


async def run_atc_loop():
    pygame.mixer.init()
    generate_squelch()
    generate_heavy_static()
    squelch_sound = pygame.mixer.Sound("radio_click.wav")
    heavy_static_sound = pygame.mixer.Sound("heavy_static.wav")

    chat_history = []
    flight_tracker = FlightStateTracker()
    last_com1_mhz = -1.0  # Track frequency changes to reset context

    # 🌍 Caches to prevent spamming external APIs
    location_cache = {}
    runway_cache = {}
    taxiway_cache = {}
    metar_cache = {}

    # 📻 Radio Line of Sight Tracker
    # NOTE: lat/lon here should represent the AIRPORT/STATION position, not the aircraft.
    # We approximate by locking the nearest airport centroid on first tune.
    # The station is updated when frequency changes AND aircraft is near the ground
    # (likely at an airport), otherwise it keeps the last known tower position.
    active_station = {"mhz": 0.0, "lat": 0.0, "lon": 0.0}

    # --- Persistent X-Plane connection ---
    # We reuse a single connection object. If it drops, we reconnect on next iteration.
    xp_conn = None

    def get_xp():
        """Returns a fresh XPlaneConnect context. Called per data-fetch block."""
        return xpc.XPlaneConnect()

    def get_real_metar(lat, lon):
        cache_key = f"{round(lat, 1)},{round(lon, 1)}"
        if cache_key in metar_cache and (time.time() - metar_cache[cache_key]['time'] < 1800):
            return metar_cache[cache_key]['metar']

        print("🌤️ Fetching real-world METAR for current location...")
        try:
            bbox = f"{lat-0.5},{lon-0.1},{lat+0.1},{lon+0.1}"
            url = f"https://aviationweather.gov/api/data/metar?format=json&bbox={bbox}"
            r = requests.get(url, timeout=6.0)
            data = r.json()
            if data and len(data) > 0:
                metar_str = data[0].get('rawOb', 'Unknown')
                metar_cache[cache_key] = {'metar': metar_str, 'time': time.time()}
                return metar_str
        except Exception as e:
            print(f"[METAR fetch error: {e}]")
        return None

    def query_overpass_with_retries(query, timeout=5.0):
        # Reduced endpoints and timeouts to fail fast gracefully without hanging the simulation
        endpoints = [
            "https://overpass.kumi.systems/api/interpreter",
            "https://lz4.overpass-api.de/api/interpreter",
            "https://overpass-api.de/api/interpreter"
        ]
        headers = {'User-Agent': 'atc_live/1.0'}
        
        for url in endpoints:
            try:
                r = requests.get(url, params={'data': query}, headers=headers, timeout=timeout)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                continue
        return None

    def get_runways_from_overpass(lat, lon, radius=5000):
        cache_key = f"{round(lat, 2)},{round(lon, 2)}"
        if cache_key in runway_cache:
            return runway_cache[cache_key]

        print("📡 Querying real-world aviation databases for active runways...")
        query = f"""
        [out:json][timeout:2];
        way["aeroway"="runway"](around:{radius},{lat},{lon});
        out tags;
        """
        data = query_overpass_with_retries(query, timeout=5.0)
        
        runways = []
        if data:
            for element in data.get('elements', []):
                tags = element.get('tags', {})
                ref = tags.get('ref')
                if ref and len(ref) <= 7 and not ref.upper().endswith('H'):
                    runways.append(ref)
            runways = list(set([rw.replace(" ", "") for rw in runways]))
        
        runway_cache[cache_key] = runways
        return runways

    def get_taxiways_from_overpass(lat, lon, radius=5000):
        cache_key = f"{round(lat, 2)},{round(lon, 2)}"
        if cache_key in taxiway_cache:
            return taxiway_cache[cache_key]

        print("🚖 Querying real-world aviation databases for active taxiways...")
        query = f"""
        [out:json][timeout:2];
        way["aeroway"="taxiway"](around:{radius},{lat},{lon});
        out tags;
        """
        data = query_overpass_with_retries(query, timeout=5.0)
        
        taxiways = []
        if data:
            for element in data.get('elements', []):
                tags = element.get('tags', {})
                ref = tags.get('ref')
                if ref and len(ref) <= 3:
                    taxiways.append(ref.upper())
            taxiways = list(set(taxiways))
        
        taxiway_cache[cache_key] = taxiways
        return taxiways

    def get_location_name(lat, lon):
        """Reverse geocode lat/lon to a city/airport name. Cached to respect Nominatim ToS."""
        cache_key = f"{round(lat, 1)},{round(lon, 1)}"
        if cache_key in location_cache:
            return location_cache[cache_key]

        try:
            headers = {'User-Agent': 'XPlane-AI-ATC/1.0'}
            res = requests.get(
                f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json",
                headers=headers, timeout=2.0
            )
            if res.status_code == 200:
                address = res.json().get("address", {})
                name = address.get("city", address.get("town", address.get("county", "Local Area")))
                location_cache[cache_key] = name
                print(f"[📍 Detected Location: {name}]")
                return name
        except Exception as e:
            print(f"[Location fetch error: {e}]")
        location_cache[cache_key] = "Local Area"
        return "Local Area"

    simbrief_cache = None
    last_simbrief_fetch = 0

    def get_simbrief_flight_plan():
        nonlocal simbrief_cache, last_simbrief_fetch
        if not SIMBRIEF_USERNAME:
            return None

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

                route_full = data.get("general", {}).get("route", "Direct")
                route_first_wp = route_full.split()[0] if route_full else "Direct"

                # FIX: Variable was named 'alt' but referenced as 'alt_raw' — renamed correctly
                alt_raw = data.get("general", {}).get("initial_altitude", "UNKNOWN")
                alt = str(int(alt_raw) // 100) if (
                    isinstance(alt_raw, str) and alt_raw.isdigit() and int(alt_raw) >= 18000
                ) else alt_raw

                simbrief_cache = {
                    "origin": origin,
                    "destination": dest,
                    "dest_lat": dest_lat,
                    "dest_lon": dest_lon,
                    "route": route_first_wp,
                    "altitude": alt
                }
                last_simbrief_fetch = time.time()
                print(f"[✅ SimBrief Plan Loaded: {origin} to {dest} at FL{alt}]")
                return simbrief_cache
        except Exception as e:
            print(f"[SimBrief error: {e}]")
        return None

    while True:
        # 1. RECORD VOICE (Push-to-Talk) or wait for ATIS
        action = await record_audio_ptt(squelch_sound)

        if action is None:
            continue  # Short recording or cancelled

        # --- Pull X-Plane data ---
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
            with get_xp() as xp:
                # 1. Positional Data
                try:
                    posi = xp.getPOSI(0)
                    lat, lon = posi[0], posi[1]
                    alt_ft = int(posi[2] * 3.28084)

                    vsi_raw = xp.getDREF("sim/flightmodel/position/vh_ind_fpm")
                    if vsi_raw:
                        vsi_fpm = vsi_raw[0]
                except Exception as e:
                    print(f"[Positional data error: {e}]")

                # 2. Radar Data
                try:
                    heading_drefs = xp.getDREF("sim/flightmodel/position/mag_psi")
                    heading = heading_drefs[0] if heading_drefs else 0.0

                    airspeed_drefs = xp.getDREF("sim/flightmodel/position/indicated_airspeed")
                    airspeed = airspeed_drefs[0] if airspeed_drefs else 0.0
                except Exception as e:
                    print(f"[Radar data error: {e}]")

                # 3. Ground Status
                try:
                    agl_meters = xp.getDREF("sim/flightmodel/position/y_agl")[0]
                    on_ground = agl_meters < 15.0
                except Exception as e:
                    print(f"[Ground status error: {e}]")
                    on_ground = True

                # 4. Transponder
                try:
                    squawk_raw = xp.getDREF("sim/cockpit/radios/transponder_code")
                    if squawk_raw:
                        squawk_code = int(squawk_raw[0])
                except Exception as e:
                    print(f"[Transponder error: {e}]")

                # 4.5 ILS / NAV1 Tracking
                try:
                    nav1_raw = xp.getDREF("sim/cockpit/radios/nav1_freq_hz")
                    if nav1_raw:
                        nav1_freq = nav1_raw[0] / 100.0
                    nav1_def = xp.getDREF("sim/cockpit2/radios/indicators/nav1_hdef_dots_pilot")
                    if nav1_def:
                        nav1_hdef = nav1_def[0]
                except Exception as e:
                    print(f"[NAV1 error: {e}]")

                # 5. Weather Data
                try:
                    wind_dir_drefs = xp.getDREF("sim/weather/wind_direction_degs")
                    wind_dir = wind_dir_drefs[0] if wind_dir_drefs else 0

                    wind_spd_drefs = xp.getDREF("sim/weather/wind_speed_kt")
                    wind_spd_kts = wind_spd_drefs[0] if wind_spd_drefs else 0

                    altim_drefs = xp.getDREF("sim/weather/barometer_sealevel_inhg")
                    altim_inhg = altim_drefs[0] if altim_drefs else 29.92
                    qnh_mb = altim_inhg * 33.8639
                except Exception as e:
                    print(f"[Weather data error: {e}]")

                # 6. Tail Number
                try:
                    tail_bytes = xp.getDREF("sim/aircraft/view/acf_tailnum")
                    if tail_bytes:
                        tail = "".join([chr(int(b)) for b in tail_bytes if b > 0]).strip()
                        if not tail:
                            tail = "UNKNOWN"
                except Exception as e:
                    print(f"[Tail number error: {e}]")

                # 7. COM1 Frequency
                try:
                    com1_hz_val = xp.getDREF("sim/cockpit/radios/com1_freq_hz")[0]
                    com1_mhz = com1_hz_val / 100.0
                except Exception:
                    try:
                        com1_hz_val = xp.getDREF("sim/cockpit2/radios/actuators/com1_frequency_hz_833")[0]
                        com1_mhz = com1_hz_val / 1000.0
                    except Exception as e:
                        print(f"[COM1 frequency error: {e}]")

        except Exception as e:
            print(f"[X-Plane Connection Error] {e}")

        # 2. TRANSCRIBE or bypass for ATIS
        user_text = ""
        if action == "PTT":
            print("Transcribing...")
            try:
                with open(FILENAME, "rb") as file:
                    transcription = client.audio.transcriptions.create(
                        file=(FILENAME, file.read()),
                        model="whisper-large-v3",
                        response_format="text",
                        language="en",
                    )
                user_text = transcription
            except Exception as e:
                print(f"[Transcription error: {e}]")
                continue

            if not user_text.strip():
                print("Didn't hear anything. Try again.")
                continue

            print(f"You said: {user_text}")

        # FIX: Reset chat history when the pilot changes frequency (new controller = fresh context)
        if abs(com1_mhz - last_com1_mhz) > 0.01 and last_com1_mhz >= 0:
            print(f"[📻 Frequency changed {last_com1_mhz:.3f} → {com1_mhz:.3f}. Resetting ATC context.]")
            chat_history = []
        last_com1_mhz = com1_mhz

        # Location context (cached)
        location_name = "Local Area"
        active_runways = []
        best_active_runway = None
        active_taxiways = []
        if lat != 0.0 and lon != 0.0:
            active_runways = get_runways_from_overpass(lat, lon)
            
            # Determine the single best runway based on wind direction
            best_active_runway = get_best_runway(active_runways, wind_dir)
            
            active_taxiways = get_taxiways_from_overpass(lat, lon)
            location_name = get_location_name(lat, lon)

        location_status = "ON THE GROUND" if on_ground else "AIRBORNE"

        # FAA vs ICAO region detection
        is_faa_region = (-170 < lon < -50) and (15 < lat < 75)
        altimeter_phrase = f"altimeter {altim_inhg:.2f}" if is_faa_region else f"QNH {qnh_mb:.0f}"
        atis_type = "FAA" if is_faa_region else "ICAO"

        real_metar = get_real_metar(lat, lon)
        weather_context = f"METAR: {real_metar}" if real_metar else (
            f"Wind {wind_dir:03.0f} at {wind_spd_kts:.0f} knots. {altimeter_phrase}."
        )

        # ILS / NAV1 Tracking
        nav_info = ""
        if nav1_freq >= 108.0 and not on_ground:
            nav_info = f"NAV1 Tuned: {nav1_freq:.2f}."
            if nav1_hdef > 1.5:
                nav_info += " (PILOT IS DRIFTING FAR LEFT OF THE LOCALIZER)"
            elif nav1_hdef < -1.5:
                nav_info += " (PILOT IS DRIFTING FAR RIGHT OF THE LOCALIZER)"
            else:
                nav_info += " (PILOT IS ESTABLISHED ON LOCALIZER)"

        # SimBrief flight plan
        simbrief_data = get_simbrief_flight_plan()
        flight_plan_context = "Pilot has NO IFR flight plan filed. Ask intentions."
        if simbrief_data:
            flight_plan_context = (
                f"Pilot's filed IFR plan: Destination {simbrief_data['destination']}, "
                f"Route: {simbrief_data['route']}, Initial FL: {simbrief_data['altitude']}."
            )
            if simbrief_data.get('dest_lat') and simbrief_data.get('dest_lon') and not on_ground:
                dest_dist = haversine_nm(lat, lon, simbrief_data['dest_lat'], simbrief_data['dest_lon'])
                dest_brg = calculate_bearing(lat, lon, simbrief_data['dest_lat'], simbrief_data['dest_lon'])
                flight_plan_context += f" | Distance to dest: {dest_dist:.1f} NM. Bearing to dest: {dest_brg:03.0f} degrees."

        # ATC Role from frequency
        atc_role = "APPROACH/CENTER"
        if 118.0 <= com1_mhz <= 121.5:
            atc_role = "TOWER"
        elif 121.6 <= com1_mhz <= 121.95:
            atc_role = "GROUND"
        elif 122.0 <= com1_mhz <= 123.05:
            atc_role = "UNICOM"
        elif 123.05 < com1_mhz < 124.0:
            atc_role = "CLEARANCE DELIVERY"

        # VHF Radio Line-of-Sight Range Check
        # FIX: Lock station to aircraft position only when on the ground tuning a local freq
        # (best approximation of actual tower/antenna position without a full airport database)
        if abs(com1_mhz - active_station["mhz"]) > 0.01:
            active_station["mhz"] = com1_mhz
            # Only re-anchor station when on the ground (near the actual transmitter)
            if on_ground:
                active_station["lat"] = lat
                active_station["lon"] = lon

        station_distance_nm = haversine_nm(lat, lon, active_station["lat"], active_station["lon"])
        max_los_nm = 1.23 * math.sqrt(max(alt_ft, 0)) + 10.0

        if atc_role in ["GROUND", "CLEARANCE DELIVERY"]:
            max_range = min(max_los_nm, 15.0)
        elif atc_role == "TOWER":
            max_range = min(max_los_nm, 50.0)
        else:
            max_range = max_los_nm

        if station_distance_nm > max_range:
            print(f"📻 [OUT OF RANGE] {station_distance_nm:.1f} NM away. Max range: {max_range:.1f} NM.")
            heavy_static_sound.play()
            await asyncio.sleep(1.5)
            continue

        # 4. GET AI RESPONSE
        # FIX: Use faster model for simple readbacks; full 70B for clearances
        # FIX: Always use the smarter model for realism as per the plan
        model_choice = "llama-3.3-70b-versatile"

        if action == "ATIS":
            print(f"📻 Generating ATIS for {location_name}...")
            atis_number = int((time.time() // 3600) % 26)
            atis_letter = chr(65 + atis_number)
            system_prompt = (
                f"You are the automated ATIS broadcaster at {location_name}. "
                f"Current weather: {weather_context}. "
                f"Active runways in use: {', '.join(active_runways) if active_runways else 'not determined'}. "
                f"Generate a short, realistic {atis_type} standard ATIS broadcast. "
                f"Start with '{location_name} Airport, information {atis_letter}...'. "
                f"End with '...advise on initial contact, you have information {atis_letter}.' "
                f"CRITICAL RULES: "
                f"1. Spell out the ATIS letter using the NATO phonetic alphabet. "
                f"2. You MUST translate raw METAR code into spoken English (e.g., read '15024G36KT' as 'wind one five zero at two four knots, gusting three six'). DO NOT read raw METAR strings like '092150Z'. "
                f"3. Ensure you use the correct regional pressure setting ({altimeter_phrase}). "
                f"4. Do not include any pleasantries."
            )
            messages = [{"role": "system", "content": system_prompt}]
        else:
            print(f"ATC Thinking (Role: {atc_role} on {com1_mhz:.3f} MHz, model: {model_choice})...")
            # --- PROACTIVE PHASE UPDATES BASED ON USER REQUEST ---
            user_text_lower = user_text.lower()
            if flight_tracker.phase == FlightPhase.PREFLIGHT and ("push" in user_text_lower or "start" in user_text_lower):
                flight_tracker.update_phase(FlightPhase.PUSHBACK)
            elif flight_tracker.phase == FlightPhase.PUSHBACK and ("taxi" in user_text_lower):
                flight_tracker.update_phase(FlightPhase.TAXI)
            elif flight_tracker.phase == FlightPhase.TAXI and ("short" in user_text_lower or "takeoff" in user_text_lower):
                flight_tracker.update_phase(FlightPhase.HOLD_SHORT)

            # --- FAILSAFE PHASE UPDATES BASED ON TELEMETRY ---
            if not on_ground and airspeed > 60 and flight_tracker.phase in [FlightPhase.PREFLIGHT, FlightPhase.PUSHBACK, FlightPhase.TAXI, FlightPhase.HOLD_SHORT]:
                print("\n[✈️ Failsafe Transition: Aircraft is airborne, forcing CLIMB phase]")
                flight_tracker.update_phase(FlightPhase.CLIMB)
            
            # --- DYNAMIC PHASE INSTRUCTIONS ---
            phase_instructions = ""
            if flight_tracker.phase == FlightPhase.PREFLIGHT:
                phase_instructions = "Pilot is at the gate. Issue IFR/VFR clearance ONLY. Do NOT approve pushback or taxi yet. " \
                    "For IFR clearances, you MUST follow standard structure: Clearance limit, Route (SID), Altitude, Departure Frequency, and Squawk. " \
                    "If the pilot reads back the clearance correctly, YOU MUST confirm it by saying 'readback correct'. " \
                    "CRITICAL: You MUST explicitly state the departure runway (e.g., 'Departure runway 18'). " \
                    "CRITICAL: Generate a random 4-digit squawk code between 2000 and 7777. DO NOT use 1200 or 7000." \
                    "CRITICAL: Do NOT list the clearance as 'Clearance limit: X, Route: Y'. Speak it naturally like a real controller. Example: 'Scandinavian 123, cleared to Oslo airport via the UPLE1G departure, runway 18. Climb and maintain 5000 feet, departure frequency 118.5, squawk 4321.'"
            elif flight_tracker.phase == FlightPhase.PUSHBACK:
                phase_instructions = "Pilot is in the pushback phase. If they request pushback, provide startup and pushback instructions (include a logical pushback direction like 'face North'). Do NOT add conversational padding like 'call me for taxi'. If they are just reading back your pushback clearance, simply confirm it briefly (e.g., 'Roger' or 'Callsign'). DO NOT re-approve pushback if already approved. Do NOT say 'readback correct' for pushback."
            elif flight_tracker.phase == FlightPhase.TAXI:
                phase_instructions = "Pilot is taxiing. Provide a logical taxi routing to the assigned runway holding point using multiple taxiways if necessary (e.g., 'Taxi via Alpha, Bravo, hold short runway 18'). Do NOT clear for takeoff. If the pilot states they are holding short, instruct them to contact Tower."
            elif flight_tracker.phase == FlightPhase.HOLD_SHORT:
                phase_instructions = "Pilot is holding short. If they are talking to Ground, instruct them to contact Tower. If they are talking to Tower, issue a takeoff clearance OR 'line up and wait'. A takeoff clearance MUST include the current wind (e.g., 'Wind 270 at 10, Runway 18, cleared for takeoff'). Do NOT re-issue the squawk code or altitude. Do NOT issue 'line up and wait' and 'cleared for takeoff' in the same transmission."
            elif flight_tracker.phase == FlightPhase.TAKEOFF or flight_tracker.phase == FlightPhase.CLIMB:
                phase_instructions = "Pilot is airborne. Hand off to Departure, or provide climb clearances and radar vectors."
            elif flight_tracker.phase == FlightPhase.CRUISE:
                phase_instructions = "Pilot is enroute. Provide traffic advisories, cruise altitude changes, or handoffs to next Center."
            elif flight_tracker.phase == FlightPhase.DESCENT or flight_tracker.phase == FlightPhase.APPROACH:
                phase_instructions = "Pilot is arriving. Provide descent clearances, approach vectors (PTA format), and hand off to Tower when established."
            elif flight_tracker.phase == FlightPhase.FINAL:
                phase_instructions = "Pilot is on final approach. Issue landing clearance and winds."
            elif flight_tracker.phase == FlightPhase.ROLLOUT or flight_tracker.phase == FlightPhase.TAXI_IN:
                phase_instructions = "Pilot has landed. Instruct them to exit the runway and contact Ground."

            system_prompt = (
                f"You are a professional Air Traffic Controller acting at {location_name}. "
                f"The pilot's tail number is {tail} BUT if the pilot states a callsign, use it strictly. "
                f"\n--- AIRCRAFT STATE ---\n"
                f"{alt_ft:.0f}ft MSL, Heading {heading:03.0f}, Speed {airspeed:.0f} kts, VSI {vsi_fpm:.0f} fpm. {nav_info}\n"
                f"Currently: {location_status}\n"
                f"\n--- ISSUED CLEARANCES & HISTORY (DO NOT RE-ISSUE IF ALREADY GIVEN) ---\n"
                f"Assigned Runway: {flight_tracker.assigned_runway or 'None'}\n"
                f"Assigned Altitude: {flight_tracker.assigned_altitude or 'None'}\n"
                f"Assigned Squawk: {flight_tracker.assigned_squawk or '1200'}\n"
                f"Current Flight Phase: {flight_tracker.phase.name}\n"
                f"Your Role: {atc_role} on {com1_mhz:.3f} MHz.\n"
                f"\n--- REAL WORLD DATA ---\n"
                f"Weather: {weather_context}\n"
                f"Active Departure/Arrival Runway: {best_active_runway if best_active_runway else '27'}\n"
                f"Reported Taxiways: {', '.join(active_taxiways) if active_taxiways else 'Invent realistic taxiways like A, B'}\n"
                f"SimBrief Plan: {flight_plan_context}\n"
                f"\n--- DYNAMIC ATC RULES ---\n"
                f"1. YOU ARE {atc_role}. Only give clearances appropriate for your role. Deny others.\n"
                f"2. CURRENT PHASE DIRECTIVE: {phase_instructions}\n"
                f"3. PREVIOUS ASSIGNMENTS: If a runway/squawk/altitude is listed in 'ISSUED CLEARANCES' above, YOU MUST STRICTLY USE IT. Do NOT change runways once assigned. NEVER repeat the squawk code or initial altitude after the PREFLIGHT phase unless the pilot explicitly asks for it.\n"
                f"4. ONE STEP AT A TIME: NEVER combine pushback and taxi clearances in the same transmission. If pilot requests pushback, ONLY approve pushback. NEVER add conversational padding like 'call for taxi when ready'.\n"
                f"5. READBACKS: If the pilot correctly reads back an instruction, acknowledge it and DO NOT re-issue it. In the PREFLIGHT phase for IFR clearance readbacks, you MUST say 'Readback correct'. For ALL other phases, you MUST use extreme brevity and acknowledge with just 'Roger' or their callsign (do NOT say 'readback correct'). Accept partial readbacks as long as the core meaning is correct.\n"
                f"6. HANDOFFS: Only issue handoffs when the aircraft is transitioning phases or leaving your airspace. Never bounce the pilot between frequencies (e.g., Center to Departure and back). Instruct them to contact Departure only when Airborne, Center when enroute, Tower when near the runway surface.\n"
                f"7. IFR CLEARANCES: Use CRAFT format. DO NOT re-issue if the pilot is past PREFLIGHT phase.\n"
                f"8. PHONETICS: Use NATO alphabet (e.g. Taxiway Alpha).\n"
                f"9. PROFESSIONALISM: Extreme brevity is required. Do not use conversational padding like 'call me back for...', 'when ready...', or 'have a safe flight'. If an instruction changes, simply issue the new clearance directly (e.g., 'Climb and maintain flight level 330').\n"
                f"Keep responses strictly in professional FAA/ICAO phraseology."
            )

            chat_history.append({"role": "user", "content": user_text})

            # FIX: Trim history correctly — keep last 20 messages, not just pop one
            while len(chat_history) > 20:
                chat_history.pop(0)

            messages = [{"role": "system", "content": system_prompt}] + chat_history

        # FIX: Add timeout to LLM call to prevent sim freezing on slow API responses
        try:
            completion = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.chat.completions.create(
                        model=model_choice,
                        messages=messages,
                        max_tokens=300,
                    )
                ),
                timeout=10.0
            )
            response_text = completion.choices[0].message.content
        except asyncio.TimeoutError:
            print("[⚠️ ATC response timed out. Stand by...]")
            heavy_static_sound.play()
            await asyncio.sleep(1.0)
            continue
        except Exception as e:
            print(f"[LLM error: {e}]")
            continue

        print(f"ATC: {response_text}")

        # Voice selection: consistent per-frequency, role-weighted gender
        # FIX: Skew voice gender by ATC role to better match real-world norms
        male_voices = ["en-US-GuyNeural", "en-GB-RyanNeural", "en-US-ChristopherNeural", "en-US-EricNeural"]
        female_voices = ["en-US-AriaNeural", "en-GB-SoniaNeural"]

        if action == "ATIS":
            tts_voice = "en-US-JennyNeural"
        elif atc_role in ["APPROACH/CENTER", "TOWER"]:
            # Radar/Tower: mostly male
            all_voices = male_voices + female_voices[:1]
            tts_voice = all_voices[int(com1_mhz * 1000) % len(all_voices)]
        else:
            # Ground/Clearance: mixed
            all_voices = male_voices + female_voices
            tts_voice = all_voices[int(com1_mhz * 1000) % len(all_voices)]

        # Save response to history ONLY if it was an ATC interaction
        if action == "PTT":
            chat_history.append({"role": "assistant", "content": response_text})
            flight_tracker.extract_and_commit_llm_assignment(response_text)

        # 5. SPEAK (Edge-TTS) + Radio Effect
        communicate = edge_tts.Communicate(response_text, tts_voice)
        await communicate.save("response_raw.mp3")

        # Convert to WAV and apply radio bandpass + noise effect
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_mp3("response_raw.mp3")
            seg.export("response_raw.wav", format="wav")
            apply_radio_effect("response_raw.wav", "response_processed.wav")
            playback_file = "response_processed.wav"
        except Exception as e:
            print(f"[pydub not available or conversion failed, using raw mp3: {e}]")
            playback_file = "response_raw.mp3"

        # 🎙️ RADIO CLICK ON
        squelch_sound.play()
        await asyncio.sleep(0.15)

        # START BACKGROUND STATIC
        static_channel = heavy_static_sound.play(loops=-1)
        if static_channel:
            static_channel.set_volume(0.04)

        pygame.mixer.music.load(playback_file)
        pygame.mixer.music.play()

        # Monitor frequency during playback to allow instant cut-off on retune
        try:
            with get_xp() as xp_monitor:
                while pygame.mixer.music.get_busy():
                    current_mhz = com1_mhz
                    try:
                        hz_val = xp_monitor.getDREF("sim/cockpit/radios/com1_freq_hz")[0]
                        current_mhz = hz_val / 100.0
                    except Exception:
                        try:
                            hz_val = xp_monitor.getDREF("sim/cockpit2/radios/actuators/com1_frequency_hz_833")[0]
                            current_mhz = hz_val / 1000.0
                        except Exception:
                            pass

                    if abs(current_mhz - com1_mhz) > 0.01:
                        print("[Frequency changed. Radio transmission cut off.]")
                        pygame.mixer.music.stop()
                        break

                    await asyncio.sleep(0.1)
        except Exception:
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)

        # STOP BACKGROUND STATIC
        if static_channel:
            static_channel.stop()

        # 🎙️ RADIO CLICK OFF
        squelch_sound.play()
        await asyncio.sleep(0.2)

        # FIX: unload() already stops music — stop() after is redundant, but harmless; kept for clarity
        pygame.mixer.music.unload()


if __name__ == "__main__":
    asyncio.run(run_atc_loop())