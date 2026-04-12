"""
ATC Live - Ultra-Realistic Air Traffic Control Simulator
Enhanced version with maximum realism improvements
"""
import asyncio
import xpc
from groq import Groq
import edge_tts
import pygame
import os
from dotenv import load_dotenv
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import butter, sosfilt, iirfilter, lfilter
from scipy.fft import fft, ifft
import numpy as np
import requests
import time
import math
import csv
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
import re
import struct
import keyboard
from airport_routing import fetch_airport_geometry, get_taxi_route

load_dotenv()

# --- CONFIGURATION ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
SIMBRIEF_USERNAME = os.environ.get("SIMBRIEF_USERNAME", "")
client = Groq(api_key=GROQ_API_KEY)
FILENAME = "mic_input.wav"

# --- ENHANCED AUDIO CONFIGURATION ---
SAMPLE_RATE = 44100
RADIO_LOWPASS_HZ = 2800      # Realistic VHF channel bandwidth
RADIO_HIGHPASS_HZ = 400      # Voice presence
RADIO_NOISE_LEVEL = 0.006    # Subtle channel noise
AM_MODULATION_DEPTH = 0.85  # Simulate AM transmission

# Pre-emphasis/de-emphasis for more authentic radio sound
PRE_EMPHASIS = 1.5          # High-frequency boost before transmission
DE_EMPHASIS = 0.7           # High-frequency cut after reception

# --- FLIGHT STATE ENUMS ---

NATO_ALPHABET = {
    "A": "Alpha", "B": "Bravo", "C": "Charlie", "D": "Delta",
    "E": "Echo", "F": "Foxtrot", "G": "Golf", "H": "Hotel",
    "I": "India", "J": "Juliett", "K": "Kilo", "L": "Lima",
    "M": "Mike", "N": "November", "O": "Oscar", "P": "Papa",
    "Q": "Quebec", "R": "Romeo", "S": "Sierra", "T": "Tango",
    "U": "Uniform", "V": "Victor", "W": "Whiskey", "X": "X-ray",
    "Y": "Yankee", "Z": "Zulu",
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
    "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Niner"
}

def to_phonetic_string(text: str) -> str:
    """Accurately converts alphanumeric identifiers (like 'A1' or 'J') to NATO phonetic words ('Alpha One', 'Juliett')."""
    if not text:
        return text
    return " ".join(NATO_ALPHABET.get(char.upper(), char) for char in text.replace(" ", ""))

class FlightPhase(Enum):
    """Detailed flight phase tracking for phase-aware prompting."""
    PREFLIGHT = "preflight"
    STARTUP = "startup"
    PUSHBACK = "pushback"
    TAXI = "taxi"
    HOLD_SHORT = "hold_short"
    TAKEOFF = "takeoff"
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    APPROACH = "approach"
    FINAL = "final"
    GO_AROUND = "go_around"
    LANDED = "landed"
    ROLLOUT = "rollout"
    TAXI_IN = "taxi_in"

class AircraftCategory(Enum):
    """Wake turbulence categories for realistic separation."""
    SUPER = "Super (A380, AN-225)"
    HEAVY = "Heavy (B747, B777, A350)"
    LARGE = "Large (B737, A320, E190)"
    MEDIUM = "Medium (CJ1, Phenom 100)"
    SMALL = "Small (C172, PA-28)"

class EmergencyStatus(Enum):
    """Emergency priority levels."""
    NONE = "none"
    URGENT = "urgent"          # 7600 - Radio failure
    INTERCEPT = "intercept"    # 7500 - Hijack
    EMERGENCY = "emergency"    # 7700 - General emergency

# --- ENHANCED FLIGHT STATE TRACKER ---
@dataclass
class FlightStateTracker:
    """
    Enhanced state machine with detailed clearance tracking.
    Prevents hallucination and ensures runway/squawk consistency.
    """
    phase: FlightPhase = FlightPhase.PREFLIGHT
    emergency: EmergencyStatus = EmergencyStatus.NONE

    # Assigned clearances (set once, never changed without pilot request)
    assigned_runway: Optional[str] = None
    assigned_sid: Optional[str] = None          # Standard Instrument Departure
    assigned_star: Optional[str] = None          # Standard Terminal Arrival
    assigned_altitude: Optional[str] = None
    assigned_climb_violation_alt: Optional[int] = None
    assigned_heading: Optional[float] = None
    assigned_speed: Optional[float] = None
    assigned_squawk: Optional[int] = None
    assigned_transponder_mode: str = "C"        # C mode (Mode S)

    # IFR Route details
    cleared_route: Optional[str] = None
    clearance_limit: Optional[str] = None

    # Taxi clearance
    taxi_instructions: List[str] = field(default_factory=list)

    # Track what has been read back correctly
    readback_verified: Dict[str, bool] = field(default_factory=dict)

    # Instructions this phase
    instructions_given_this_phase: List[str] = field(default_factory=list)

    # Aircraft category (for wake turbulence)
    aircraft_category: AircraftCategory = AircraftCategory.LARGE

    # Previous controller for handoff tracking
    previous_controller: Optional[str] = None

    def update_phase(self, new_phase: FlightPhase):
        """Transition to a new flight phase."""
        if self.phase != new_phase:
            print(f"[✈️ Flight Phase Transition: {self.phase.name} → {new_phase.name}]")
            self.phase = new_phase
            self.instructions_given_this_phase.clear()
            self.readback_verified.clear()

# --- DYNAMIC FREQUENCY MANAGER ---
class DynamicFrequencyManager:
    """Fetches real-world airport frequencies for accurate ATC role assignments."""
    def __init__(self):
        self.airports_file = "airports.csv"
        self.frequencies_file = "airport-frequencies.csv"
        self.airports = []
        self.frequencies = {}
        self._metar_cache = {}         # NEW: Cache METAR data
        self._atis_cache = {}           # NEW: Cache ATIS data

    def load_data(self):
        print("🌍 Loading global airport frequency database...")
        if not os.path.exists(self.airports_file):
            print("   -> Downloading airports.csv...")
            r = requests.get("https://ourairports.com/data/airports.csv", timeout=10.0)
            with open(self.airports_file, "wb") as f:
                f.write(r.content)
        if not os.path.exists(self.frequencies_file):
            print("   -> Downloading airport-frequencies.csv...")
            r = requests.get("https://ourairports.com/data/airport-frequencies.csv", timeout=10.0)
            with open(self.frequencies_file, "wb") as f:
                f.write(r.content)

        try:
            with open(self.airports_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['type'] in ['large_airport', 'medium_airport', 'small_airport']:
                        try:
                            self.airports.append({
                                'ident': row['ident'],
                                'name': row['name'],
                                'lat': float(row['latitude_deg']),
                                'lon': float(row['longitude_deg']),
                                'elevation_ft': int(row.get('elevation_ft', 0) or 0)
                            })
                        except ValueError:
                            pass

            with open(self.frequencies_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ident = row['airport_ident']
                    if ident not in self.frequencies:
                        self.frequencies[ident] = {}

                    try:
                        mhz = round(float(row['frequency_mhz']), 3)
                        role_str = row['type'].upper()
                        mapped_role = self._map_frequency_role(role_str)
                        self.frequencies[ident][mhz] = mapped_role
                    except ValueError:
                        pass
            print(f"✅ Loaded frequencies for {len(self.frequencies)} airports worldwide.")
        except Exception as e:
            print(f"[Frequency DB Error: {e}] - Falling back to hardcoded bands.")

    def _map_frequency_role(self, role_str: str) -> str:
        """Map frequency type to ATC role."""
        if role_str in ['TWR', 'CTAF', 'UNIC', 'UNICOM']:
            return "TOWER"
        elif role_str in ['GND', 'RAMP', 'SMC']:
            return "GROUND"
        elif role_str in ['DEL', 'CLD', 'CD']:
            return "CLEARANCE DELIVERY"
        elif role_str in ['ATIS', 'AWOS', 'ASOS', 'AFIS']:
            return "ATIS"
        elif role_str in ['APP', 'DEP', 'CTR', 'Radar']:
            return "APPROACH/CENTER"
        return "APPROACH/CENTER"

    def get_atc_role(self, lat, lon, target_mhz, in_air=False):
        closest_ident = None
        min_dist = float('inf')

        for ap in self.airports:
            if abs(ap['lat'] - lat) < 1.0 and abs(ap['lon'] - lon) < 1.0:
                d = haversine_nm(lat, lon, ap['lat'], ap['lon'])
                if d < min_dist:
                    min_dist = d
                    closest_ident = ap['ident']

        if closest_ident and closest_ident in self.frequencies:
            freqs = self.frequencies[closest_ident]
            for mhz, role in freqs.items():
                if abs(mhz - target_mhz) <= 0.006:
                    return role

        # Fallback to frequency bands
        role = "APPROACH/CENTER"
        if 121.6 <= target_mhz <= 121.95:
            role = "GROUND"
        elif 122.0 <= target_mhz <= 123.05:
            role = "UNICOM"
        elif 123.05 < target_mhz < 124.0:
            role = "CLEARANCE DELIVERY"
        elif 118.0 <= target_mhz <= 121.5:
            role = "APPROACH/CENTER" if in_air else "TOWER"
        return role

    def get_nearest_airport(self, lat, lon):
        closest_ap = None
        min_dist = float('inf')

        for ap in self.airports:
            if abs(ap['lat'] - lat) < 1.0 and abs(ap['lon'] - lon) < 1.0:
                d = haversine_nm(lat, lon, ap['lat'], ap['lon'])
                if d < min_dist:
                    min_dist = d
                    closest_ap = ap
        return closest_ap

    def get_nearest_frequencies(self, lat, lon):
        """Returns a formatted string of real frequencies for the nearest airport."""
        closest_ident = None
        min_dist = float('inf')

        for ap in self.airports:
            if abs(ap['lat'] - lat) < 1.0 and abs(ap['lon'] - lon) < 1.0:
                d = haversine_nm(lat, lon, ap['lat'], ap['lon'])
                if d < min_dist:
                    min_dist = d
                    closest_ident = ap['ident']

        freq_context = ""
        if closest_ident and closest_ident in self.frequencies:
            freqs = self.frequencies[closest_ident]
            # Reverse map: {Role: Frequency}
            role_to_freq = {}
            for mhz, role in freqs.items():
                if role not in role_to_freq: # Keep first matching per role
                    role_to_freq[role] = mhz
            
            # Create a readable context string for the AI prompt
            freq_context = ", ".join(f"{role.capitalize()}: {mhz:.3f} MHz" for role, mhz in role_to_freq.items())
            
        return freq_context if freq_context else "Frequencies: Ground 121.9, Tower 118.1, Approach 121.5"

    def extract_and_commit_llm_assignment(self, llm_response: str):
        """Parse LLM response and extract/commit any newly assigned values."""
        # Extract squawk code
        squawk_match = re.search(r'\bsquawk\s+(\d{4})\b', llm_response, re.IGNORECASE)
        if squawk_match:
            return int(squawk_match.group(1))
        return None

# --- ENHANCED AUDIO PROCESSING ---
def apply_pre_emphasis(audio: np.ndarray, fs: int, coefficient: float = 0.95) -> np.ndarray:
    """Apply pre-emphasis filter for transmission clarity."""
    b = [1.0, -coefficient]
    a = [1.0]
    return lfilter(b, a, audio)

def apply_de_emphasis(audio: np.ndarray, fs: int, coefficient: float = 0.95) -> np.ndarray:
    """Apply de-emphasis filter for receiver authenticity."""
    b = [1.0]
    a = [1.0, -coefficient]
    return lfilter(b, a, audio)

def apply_am_modulation(audio: np.ndarray, depth: float = 0.85) -> np.ndarray:
    """Simulate AM modulation with carrier."""
    carrier = 1.0 - depth  # DC offset acts as carrier
    return carrier + audio * depth

def apply_radio_effect(audio_path: str, out_path: str, fs: int = SAMPLE_RATE,
                      signal_strength: float = 1.0, distance_nm: float = 0.0):
    """
    Enhanced radio effect chain for maximum realism:
    1. Pre-emphasis (transmission)
    2. Bandpass filter (VHF channel)
    3. AM modulation simulation
    4. Signal strength attenuation
    5. Atmospheric fading based on distance
    6. Channel noise
    7. De-emphasis (reception)
    8. Compression/limiting
    """
    try:
        from scipy.io.wavfile import read as wav_read
        sample_rate, data = wav_read(audio_path)

        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        else:
            audio = data.astype(np.float32)

        if audio.ndim > 1:
            audio = audio[:, 0]

        # 1. Pre-emphasis for transmission realism
        audio = apply_pre_emphasis(audio, sample_rate)

        # 2. Bandpass filter (280Hz - 2800Hz for authentic VHF)
        sos_hp = butter(4, RADIO_HIGHPASS_HZ / (sample_rate / 2), btype='high', output='sos')
        sos_lp = butter(4, RADIO_LOWPASS_HZ / (sample_rate / 2), btype='low', output='sos')
        audio = sosfilt(sos_hp, audio)
        audio = sosfilt(sos_lp, audio)

        # 3. AM modulation simulation
        audio = apply_am_modulation(audio, AM_MODULATION_DEPTH)

        # 4. Signal strength attenuation + distance-based fading
        # Real VHF signal weakens with distance and has multipath fading
        base_attenuation = 0.15 + (distance_nm / 200.0) * 0.3  # Progressive attenuation
        signal_attenuation = max(0.05, 1.0 - base_attenuation) * signal_strength
        audio = audio * signal_attenuation

        # 5. Atmospheric noise modeling (pink noise characteristics)
        if RADIO_NOISE_LEVEL > 0:
            # Generate noise with spectrum more realistic for VHF
            noise = np.random.normal(0, RADIO_NOISE_LEVEL * 0.7, len(audio))
            # Apply light low-pass to pinken the noise
            sos_n = butter(2, 1500 / (sample_rate / 2), btype='low', output='sos')
            noise = sosfilt(sos_n, noise)
            audio = audio + noise

        # 6. Light intermodulation distortion for realism
        if np.max(np.abs(audio)) > 0.5:
            # Add subtle clipping character
            audio = np.tanh(audio * 1.2) / 1.2

        # 7. De-emphasis for receiver
        audio = apply_de_emphasis(audio, sample_rate)
        
        # Remove DC offset caused by AM carrier and de-emphasis integration
        audio = audio - np.mean(audio)

        # 8. Compression for consistent loudness
        # Soft knee compressor simulation
        threshold = 0.6
        ratio = 4.0
        attack_coef = 0.95
        release_coef = 0.98

        envelope = np.abs(audio)
        smoothed_env = np.zeros_like(envelope)
        for i in range(len(envelope)):
            if i == 0:
                smoothed_env[i] = envelope[i]
            else:
                if envelope[i] > smoothed_env[i-1]:
                    smoothed_env[i] = attack_coef * smoothed_env[i-1] + (1 - attack_coef) * envelope[i]
                else:
                    smoothed_env[i] = release_coef * smoothed_env[i-1] + (1 - release_coef) * envelope[i]

        # Apply compression
        gain_reduction = np.where(smoothed_env > threshold,
                                   1.0 / ratio + (smoothed_env - threshold) / (smoothed_env * ratio),
                                   1.0)
        audio = audio * gain_reduction

        # Final normalization
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.88

        write(out_path, sample_rate, (audio * 32767).astype(np.int16))
    except Exception as e:
        print(f"[Radio effect failed, using clean audio: {e}]")
        import shutil
        shutil.copy(audio_path, out_path)


def generate_realistic_squelch():
    """Generate a more realistic radio squelch click sequence."""
    if os.path.exists("radio_click.wav"):
        return

    fs = SAMPLE_RATE
    duration = 0.15
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Initial noise burst
    noise = np.random.normal(0, 0.6, len(t))

    # Attack envelope
    attack_env = np.exp(-200 * t)

    # Main click component
    click = np.sin(2 * np.pi * 1800 * t) * np.exp(-400 * t)
    click2 = np.sin(2 * np.pi * 2400 * t) * np.exp(-500 * t[::-1]) * 0.3

    # Tail rumble
    rumble = np.sin(2 * np.pi * 400 * t) * np.exp(-800 * t) * 0.2

    signal = (noise[:len(t)] * attack_env * 0.3) + (click + click2 + rumble) * 0.7
    audio = np.int16(signal / np.max(np.abs(signal)) * 22000)
    write("radio_click.wav", fs, audio)


def generate_heavy_static():
    """Generate more realistic channel noise/static."""
    if not os.path.exists("heavy_static.wav"):
        fs = SAMPLE_RATE
        duration = 1.5

        # Create noise with realistic VHF characteristics
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        noise = np.random.normal(0, 0.25, len(t))

        # Add some periodic components for "alive" static
        for freq in [1200, 2400, 3600]:
            noise += np.sin(2 * np.pi * freq * t) * 0.05 * np.random.random()

        audio = np.int16(noise * 28000)
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
    """Calculates the best runway based on headwind component."""
    if not runways:
        return None

    best_runway = runways[0]
    smallest_diff = 180

    for rw in runways:
        match = re.match(r"(\d+)", rw)
        if match:
            rw_hdg = int(match.group(1)) * 10
            # Calculate crosswind component
            crosswind = abs((wind_dir - rw_hdg + 180) % 360 - 180)

            if crosswind < smallest_diff:
                smallest_diff = crosswind
                best_runway = rw

    return best_runway


def get_aircraft_category(icao_type: str) -> AircraftCategory:
    """Determine wake turbulence category based on ICAO type."""
    # Super heavy
    if icao_type.upper() in ['A388', 'AN225']:
        return AircraftCategory.SUPER
    # Heavy
    heavy_types = ['B744', 'B772', 'B773', 'B788', 'A332', 'A333', 'A335', 'A338', 'A339']
    if icao_type.upper() in heavy_types:
        return AircraftCategory.HEAVY
    # Large
    large_types = ['B738', 'B739', 'B37X', 'A320', 'A319', 'A321', 'A20N', 'A21N', 'E190', 'E195']
    if icao_type.upper() in large_types:
        return AircraftCategory.LARGE
    return AircraftCategory.MEDIUM


def calculate_min_separation(own_category: AircraftCategory, traffic_category: AircraftCategory,
                            distance_nm: float, altitude_diff_ft: int) -> Tuple[bool, str]:
    """Calculate if minimum separation is maintained."""
    # Simplified wake turbulence separation minima
    wake_minima = {
        (AircraftCategory.SUPER, AircraftCategory.HEAVY): 8.0,
        (AircraftCategory.SUPER, AircraftCategory.LARGE): 10.0,
        (AircraftCategory.SUPER, AircraftCategory.MEDIUM): 12.0,
        (AircraftCategory.HEAVY, AircraftCategory.LARGE): 5.0,
        (AircraftCategory.HEAVY, AircraftCategory.MEDIUM): 6.0,
        (AircraftCategory.LARGE, AircraftCategory.MEDIUM): 3.0,
    }

    min_distance = wake_minima.get((own_category, traffic_category), 3.0)

    # Altitude-based separation (1000ft minimum for opposite direction)
    alt_separation_ok = altitude_diff_ft >= 1000 if altitude_diff_ft < 1000 else True

    # Radar separation (3NM horizontal or 1000ft vertical)
    radar_ok = distance_nm >= 3.0 or altitude_diff_ft >= 1000

    separation_ok = distance_nm >= min_distance and alt_separation_ok

    if not separation_ok:
        if distance_nm < 3.0:
            return False, f"Traffic {distance_nm:.1f} miles, maintain separation"
        return False, f"Wake turbulence separation required"
    return True, "Clear"


async def record_audio_ptt(squelch=None, fs=SAMPLE_RATE):
    """Record audio with PTT, waits for ATIS check."""
    print("\n[Hold '+' to talk to ATC, or tune to 121.5 for GUARD/emergency...]")

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
                if abs(com1_mhz - 121.5) < 0.01:
                    return "GUARD"
        except Exception:
            pass

        await asyncio.sleep(0.1)

    if squelch:
        squelch.play()

    print("--- 🎙️ RECORDING ---")

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
        duration_secs = len(audio_data) / fs
        if duration_secs < 0.3:
            print("[Recording too short, ignoring]")
            return None

        write(FILENAME, fs, audio_data)
        return "PTT"
    return None


# --- ENHANCED AI PROMPTS FOR REALISM ---
def generate_atis_system_prompt(location_name: str, weather_context: str,
                                 active_runways: List[str], altimeter_phrase: str,
                                 atis_type: str, atis_letter: str, notams: str = "") -> str:
    """Generate highly realistic ATIS broadcast."""
    return f"""You are the automated ATIS (Automatic Terminal Information Service) at {location_name}.

CRITICAL RULES:
1. Use EXACT format: "{location_name} Airport, information {atis_letter}."
2. Translate ALL weather codes into spoken English - NEVER read raw METAR codes
   - Wind: "15010KT" → "Wind one five zero at one zero knots"
   - "15010G20KT" → "Wind one five zero at one zero knots gusting two zero"
   - Visibility: "9999" → "Visibility one zero kilometers" or "Cavor one zero"
   - "4500" → "Visibility four thousand five hundred meters"
3. Use the correct pressure format: {altimeter_phrase}
4. Active runways: {', '.join(active_runways) if active_runways else 'not available'}
5. Include NOTAMs if provided: {notams}
6. NEVER use pleasantries or filler phrases
7. End exactly with: 'Advise on initial contact you have information {atis_letter}.'
Example: 'JFK Airport information Bravo. Wind two seven zero at one five. Altimeter two niner niner two. ILS runway one niner left in use. Advise on initial contact you have information Bravo.'
"""


def generate_atc_system_prompt(
    location_name: str,
    flight_tracker: FlightStateTracker,
    atc_role: str,
    com1_mhz: float,
    alt_ft: int,
    heading: float,
    airspeed: float,
    vsi_fpm: float,
    on_ground: bool,
    weather_context: str,
    best_runway: str,
    taxiways: List[str],
    simbrief_data: dict,
    nav_info: str,
    squawk_code: int,
    tail: str,
    is_faa_region: bool,
    altimeter_phrase: str,
    emergency: EmergencyStatus,
    station_distance_nm: float,
    local_freqs: str,
    intercept_heading: Optional[int]
) -> str:
    """Generate highly realistic ATC system prompt with proper phraseology."""

    # Phase-specific instructions
    phase_directives = {
        FlightPhase.PREFLIGHT: f"""Pilot is PREFLIGHT at the gate. Issue clearance ONLY.
CRITICAL RULES:
- For IFR clearances, you MUST use CRAFT format: Clearance limit, Route, Altitude, Frequency, Transponder
- MUST mention the assigned departure runway (Active runway: {best_runway}).
- IMPORTANT: Use ACTUAL local frequencies for Departure handoffs. Local frequencies are: {local_freqs}
- ALTITUDE ASSIGNMENT: Assign a realistic INITIAL altitude for a SID (e.g. 4000-8000 ft or FL060) and tell pilot to expect their filed Cruise FL 10 minutes after departure. Do NOT clear them to Cruise FL right away.
- Example: "N12345, cleared to JFK via direct, departure runway {best_runway}, climb via SID to five thousand, expect flight level three five zero ten minutes after departure, departure frequency [USE REAL FREQ], squawk four three two one"
- Generate a RANDOM 4-digit Mode-S transponder code between 2000-6277 (never 1200 or 7000)
- If pilot reads back correctly, say ONLY "Readback correct" and their callsign. Do NOT add instructions to contact ground or request pushback.
- NEVER approve taxi, takeoff, or pushback during this phase
- Acknowledge SimBrief flight plan if available""",

        FlightPhase.PUSHBACK: """Pilot has requested pushback/startup OR is reading back your clearance.
CRITICAL RULES:
- If pilot requests pushback: Issue startup approval with pushback direction (e.g., "Face North"). Example: "Cessna foxtrot, pushback approved, face north".
- If pilot is ONLY reading back your pushback approval: DO NOT ISSUE TAXI INSTRUCTIONS.
- DO NOT issue taxi clearance until pilot explicitly reports ready to taxi!
- NEVER re-approve if already approved""",

        FlightPhase.TAXI: """Pilot is taxiing.
CRITICAL RULES:
- Issue taxi clearance to holding point of assigned runway using the PROVIDED EXACT TAXI ROUTE
- Strictly adhere to the order of the provided taxiways
- Include runway holding point and any restrictions
- Example if route is Alpha, Bravo: "Taxi to holding point runway two seven via Alpha, Bravo"
- If requesting to cross active runway, require position and hold
- If pilot states holding short, instruct to contact Tower""",

        FlightPhase.HOLD_SHORT: f"""Pilot is holding short of runway.
CRITICAL RULES:
- If acting as GROUND: Instruct "Contact Tower" and provide REAL local tower frequency from: {local_freqs}. If the pilot is reading back the handoff, respond ONLY with "Good day". UNDER NO CIRCUMSTANCES issue a takeoff clearance.
- If acting as TOWER: issue takeoff clearance OR "Line up and wait"
- Takeoff clearance MUST include wind: "Wind one eight zero at one zero, runway two seven, cleared for takeoff"
- NEVER issue "line up and wait" and "cleared for takeoff" in same transmission
- If landing traffic exists, issue "continue, traffic in sight" OR "hold short"
- IF THE PILOT IS READING BACK YOUR TAKEOFF CLEARANCE: You MUST respond ONLY with their callsign or "Roger". Do NOT issue a handoff to departure, and do NOT give any further instructions. Wait for them to become airborne.
""",

        FlightPhase.TAKEOFF: f"""Pilot is in takeoff roll OR just airborne.
CRITICAL RULES:
- If still rolling: "airspeed, call sign" (you may still cancel)
- If airborne: hand off to Departure, provide frequency
- Use REAL local frequencies for handoff: {local_freqs}
- If climbing out of your airspace: "contact Departure [REAL FREQ]"
- NEVER clear for approach during this phase""",

        FlightPhase.CLIMB: f"""Pilot is climbing.
CRITICAL RULES:
- Assign altitude consistent with filed cruise altitude or current traffic
- Use proper phraseology: "Climb and maintain flight level [number]"
- IF pilot was on an initial SID altitude, clear them higher toward their filed Cruise Altitude
- Provide heading vectors if needed for traffic or airspace
- Handoff instructions MUST use real frequencies from: {{local_freqs}}
- FL numbers are spoken as "flight level [three-two-zero]" for 320
- When leaving your airspace: provide next controller frequency""",

        FlightPhase.CRUISE: """Pilot is enroute at cruise altitude.
CRITICAL RULES:
- Monitor flight progress
- Provide traffic advisories when required (5NM/1000ft separation)
- Example traffic advisory: "Traffic, ten o'clock, forty miles, Boeing 737, climbing to flight level three five zero"
- Issue frequency changes when appropriate
- NEVER issue approach clearances during cruise""",

        FlightPhase.DESCENT: """Pilot is descending.
CRITICAL RULES:
- Assign STAR (Standard Terminal Arrival) if applicable
- Issue cross descent if pilot descends early
- Provide approach clearance when: 50NM from destination or assigned fix
- Example: "Descend to flight level two four zero, expect approach runway two seven"
- Transfer to Tower frequency when appropriate""",

        FlightPhase.APPROACH: f"""Pilot is on approach.
CRITICAL RULES:
- Issue approach clearance: "Cleared [type] approach runway {best_runway if best_runway else 'two seven'}"
- Provide vectors if needed (PMS/DMS format)
- MUST USE THIS EXACT VECTOR FOR INTERCEPT: "Fly heading {f'{intercept_heading:03d}' if intercept_heading else '...'}, intercept localizer"
- Monitor glideslope intercept
- When established: "continue, report outer marker" OR "continue, tower advise no traffic"
- Transfer to Tower when runway is assured""",

        FlightPhase.FINAL: """Pilot is on final approach.
CRITICAL RULES:
- Issue landing clearance: "Wind [direction] at [speed], runway [number], cleared to land"
- If wind updated: "wind check [direction] at [speed], runway [number], cleared to land"
- If go-around required: "go around, I say again, go around, [reason]"
- Do NOT issue takeoff clearances when aircraft on final""",

        FlightPhase.LANDED: """Aircraft has touched down.
CRITICAL RULES:
- Assign runway exit taxiway
- Example: "turn left at Charlie, contact Ground [frequency]"
- Do NOT issue taxi instructions yet""",

        FlightPhase.TAXI_IN: """Pilot is taxiing to gate/parking.
CRITICAL RULES:
- Provide taxi clearance to stand
- Example: "Taxi to stand Alpha one via Bravo, contact Ground"
- Include apron entry instructions if applicable"""
    }

    phase_instruction = phase_directives.get(flight_tracker.phase, "Monitor and assist as needed.")

    # Emergency handling
    emergency_instruction = ""
    if emergency == EmergencyStatus.EMERGENCY:
        emergency_instruction = """
!!! EMERGENCY TRAFFIC - PRIORITY OVER ALL OTHER TRAFFIC !!!
Respond with extreme urgency:
- "EMERGENCY TRAFFIC ROGER, I SAY AGAIN [clearance/instruction]"
- Provide immediate vectors to nearest airport if requested
- Alert emergency services
- Clear all conflicting traffic"""
    elif emergency == EmergencyStatus.URGENT:
        emergency_instruction = """
!!! RADIO FAILURE / URGENT TRAFFIC !!!
If squawk 7600: Assume pilot has lost radio capability
- Provide traffic advisories
- Issue instructions assuming no readback
- Monitor for pilot flashing landing lights or making standard turns"""
    elif emergency == EmergencyStatus.INTERCEPT:
        emergency_instruction = """
!!! INTERCEPT SITUATION - SQUAWK 7500 !!!
Contact supervisor immediately
Follow intercept protocols"""

    # Build SimBrief data block
    simbrief_data_block = ""
    if simbrief_data:
        simbrief_data_block = f"""
=== FLIGHT PLAN ===
Destination: {simbrief_data['destination']}
Route: {simbrief_data['route']}
Cruise Altitude: {simbrief_data['altitude']}"""

    return f"""You are a professional, certified Air Traffic Controller acting as {atc_role} at {location_name}.

=== AIRCRAFT IDENTIFICATION ===
Tail number: {tail}
Squawk: {squawk_code}
Station distance: {station_distance_nm:.1f} NM
{emergency_instruction}
=== AIRCRAFT STATE ===
Altitude: {alt_ft} feet MSL
Heading: {heading:03.0f} degrees magnetic
Airspeed: {airspeed:.0f} knots
Vertical speed: {vsi_fpm:+.0f} feet per minute
Position: {'ON THE GROUND' if on_ground else 'AIRBORNE'}
{nav_info}

=== CURRENT FLIGHT PHASE ===
{flight_tracker.phase.name}

=== ISSUED CLEARANCES (DO NOT REPEAT UNLESS PILOT ASKS) ===
Assigned Runway: {flight_tracker.assigned_runway or 'Not yet assigned'}
Assigned SID: {flight_tracker.assigned_sid or 'Not assigned'}
Assigned STAR: {flight_tracker.assigned_star or 'Not assigned'}
Assigned Altitude: {flight_tracker.assigned_altitude or 'Not assigned'}
Assigned Squawk: {flight_tracker.assigned_squawk or squawk_code}
Taxi Route: {', '.join(flight_tracker.taxi_instructions) if flight_tracker.taxi_instructions else 'Not assigned'}

=== REAL-WORLD CONTEXT ===
Weather: {weather_context}
Active Runway: {best_runway if best_runway else 'Runway 27'}
Calculated Taxi Route: {', '.join(taxiways) if (flight_tracker.phase in [FlightPhase.TAXI, FlightPhase.TAXI_IN, FlightPhase.LANDED] and taxiways) else 'DO NOT ISSUE TAXI YET'}
Altimeter: {altimeter_phrase}
{simbrief_data_block}

=== CRITICAL OPERATIONAL RULES ===
1. ROLE BOUNDARY: You are {atc_role}. Approve ONLY clearances within your authority.
   - GROUND: Taxi, pushback, runway entry
   - TOWER: Takeoff, landing, runway operations
   - CLEARANCE DELIVERY: IFR/VFR clearances only
   - APPROACH: Vectors, approach clearance, altitude assignments

2. PHASE DIRECTIVE:
{phase_instruction}

3. SEPARATION RESPONSIBILITY:
   - You are responsible for separation until handoff is acknowledged
   - Issue "contact [next controller]" - pilot confirms with "changing"
   - Do NOT issue conflicting clearances after handoff

4. READBACK RULES:
   - For PREFLIGHT IFR clearances: MUST say ONLY "Readback correct" completely omitting any handoff or pushback/taxi instructions. Let the pilot initiate the next phase.
   - For ALL OTHER PHASES (Pushback, Taxi, Takeoff, etc.): DO NOT say "Roger", "Readback correct", or acknowledge the readback in any way if it is correct. Be realistically concise. (A simple "Good Day" during handoffs is acceptable).
   - If readback is incorrect: interrupt immediately with the correction.

5. PHRASEOLOGY STANDARDS:
   - Use NATO phonetic alphabet for letters (Alpha, Bravo, Charlie...)
   - Numbers: 3→"tree", 9→"niner", 0→"zero", 7→"seven"
   - Flight levels: 320 → "flight level three two zero"
   - Runway numbers: 27 → "two seven", 01 → "zero one"
   - Altimeter: "altimeter two niner niner two" (FAA) or "QNH one zero one three" (ICAO)
   - Compass directions: 090→"zero niner zero"

6. TIMING & CONCISENESS:
   - Keep transmissions under 30 seconds
   - Group information efficiently
   - Do NOT use: "when ready", "call me back", "have a nice flight"
   - One clearance per transmission (except in emergencies)

7. TRAFFIC ADVISORIES:
   - Provide when traffic within 10NM and 2000ft vertically
   - Format: "[Direction] [Distance] miles, [Aircraft type or position], [Altitude or direction]"

8. WORKLOAD MANAGEMENT:
   - If frequency is busy, use "standby"
   - If no contact after multiple calls, attempt contact at 2-minute intervals

Respond using ONLY professional ICAO/FAA phraseology. Be concise but complete."""


async def run_atc_loop():
    pygame.mixer.init()
    generate_realistic_squelch()
    generate_heavy_static()
    squelch_sound = pygame.mixer.Sound("radio_click.wav")
    heavy_static_sound = pygame.mixer.Sound("heavy_static.wav")

    chat_history = []
    flight_tracker = FlightStateTracker()
    last_com1_mhz = -1.0
    frequency_station_data = {}  # Track station info per frequency

    freq_manager = DynamicFrequencyManager()
    freq_manager.load_data()

    # Caches
    location_cache = {}
    runway_cache = {}
    taxiway_cache = {}
    metar_cache = {}

    # SimBrief cache
    simbrief_cache = None
    last_simbrief_fetch = 0

    def get_xp():
        """Returns a fresh XPlaneConnect context."""
        return xpc.XPlaneConnect()

    def get_real_metar(lat, lon):
        cache_key = f"{round(lat, 1)},{round(lon, 1)}"
        if cache_key in metar_cache and (time.time() - metar_cache[cache_key]['time'] < 1800):
            return metar_cache[cache_key]['metar']

        print("🌤️ Fetching real-world METAR...")
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

    def get_location_name(lat, lon):
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

    def get_simbrief_flight_plan():
        nonlocal simbrief_cache, last_simbrief_fetch
        if not SIMBRIEF_USERNAME:
            return None

        if simbrief_cache and (time.time() - last_simbrief_fetch < 300):
            return simbrief_cache

        try:
            print(f"📡 Fetching latest SimBrief flight plan...")
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
                print(f"[✅ SimBrief Plan: {origin} to {dest} at FL{alt}]")
                return simbrief_cache
        except Exception as e:
            print(f"[SimBrief error: {e}]")
        return None

    def check_emergency_status(squawk_code: int) -> EmergencyStatus:
        """Check for emergency squawk codes."""
        if squawk_code == 7700:
            return EmergencyStatus.EMERGENCY
        elif squawk_code == 7600:
            return EmergencyStatus.URGENT
        elif squawk_code == 7500:
            return EmergencyStatus.INTERCEPT
        return EmergencyStatus.NONE

    # Voice selection weights by role
    male_voices = ["en-US-GuyNeural", "en-GB-RyanNeural", "en-US-ChristopherNeural",
                   "en-US-EricNeural", "en-GB-ConnorNeural", "en-US-BrianNeural"]
    female_voices = ["en-US-AriaNeural", "en-GB-SoniaNeural", "en-US-JennyNeural"]

    while True:
        action = await record_audio_ptt(squelch_sound)

        if action is None:
            continue

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
        nav1_vdef = 0.0
        aircraft_icao = "B738"

        try:
            with get_xp() as xp:
                # Positional Data
                try:
                    posi = xp.getPOSI(0)
                    lat, lon = posi[0], posi[1]
                    alt_ft = int(posi[2] * 3.28084)

                    vsi_raw = xp.getDREF("sim/flightmodel/position/vh_ind_fpm")
                    if vsi_raw:
                        vsi_fpm = vsi_raw[0]
                except Exception as e:
                    print(f"[Positional data error: {e}]")

                # Radar Data
                try:
                    heading_drefs = xp.getDREF("sim/flightmodel/position/mag_psi")
                    heading = heading_drefs[0] if heading_drefs else 0.0

                    airspeed_drefs = xp.getDREF("sim/flightmodel/position/indicated_airspeed")
                    airspeed = airspeed_drefs[0] if airspeed_drefs else 0.0
                except Exception as e:
                    print(f"[Radar data error: {e}]")

                # Ground Status
                try:
                    agl_meters = xp.getDREF("sim/flightmodel/position/y_agl")[0]
                    on_ground = agl_meters < 15.0
                except Exception as e:
                    print(f"[Ground status error: {e}]")
                    on_ground = True

                # Transponder
                try:
                    squawk_raw = xp.getDREF("sim/cockpit/radios/transponder_code")
                    if squawk_raw:
                        squawk_code = int(squawk_raw[0])
                except Exception as e:
                    print(f"[Transponder error: {e}]")

                # ILS / NAV1
                try:
                    nav1_raw = xp.getDREF("sim/cockpit/radios/nav1_freq_hz")
                    if nav1_raw:
                        nav1_freq = nav1_raw[0] / 100.0
                    nav1_hdef = xp.getDREF("sim/cockpit2/radios/indicators/nav1_hdef_dots_pilot")[0]
                    nav1_vdef = xp.getDREF("sim/cockpit2/radios/indicators/nav1_vdef_dots_pilot")[0]
                except Exception as e:
                    print(f"[NAV1 error: {e}]")

                # Weather Data
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

                # Tail Number and Aircraft Type
                try:
                    tail_bytes = xp.getDREF("sim/aircraft/view/acf_tailnum")
                    if tail_bytes:
                        tail = "".join([chr(int(b)) for b in tail_bytes if b > 0]).strip()
                        if not tail:
                            tail = "UNKNOWN"

                    # Get aircraft ICAO type
                    acf_type = xp.getDREF("sim/aircraft/acf_icao_code")
                    if acf_type:
                        aircraft_icao = "".join([chr(int(b)) for b in acf_type if b > 0]).strip()
                except Exception as e:
                    print(f"[Aircraft data error: {e}]")

                # COM1 Frequency
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

        # Emergency status check
        emergency = check_emergency_status(squawk_code)

        # Aircraft category
        flight_tracker.aircraft_category = get_aircraft_category(aircraft_icao)

        # Transcription
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

        # Frequency change detection
        if abs(com1_mhz - last_com1_mhz) > 0.01 and last_com1_mhz >= 0:
            print(f"[📻 Frequency changed {last_com1_mhz:.3f} → {com1_mhz:.3f}. Resetting context.]")
            chat_history = []
        last_com1_mhz = com1_mhz

        # Location context
        location_name = "Local Area"
        active_runways = []
        best_active_runway = None
        active_taxiways = []
        intercept_heading = None
        
        if lat != 0.0 and lon != 0.0:
            active_runways = get_runways_from_overpass(lat, lon)
            best_active_runway = get_best_runway(active_runways, wind_dir)
            active_taxiways = [to_phonetic_string(twy) for twy in get_taxi_route(lat, lon, best_active_runway)]
            location_name = get_location_name(lat, lon)

            # --- PRECISION VECTORING CALCULATION ---
            if best_active_runway:
                match = re.match(r"(\d+)", best_active_runway)
                if match:
                    rw_hdg = int(match.group(1)) * 10
                    nearest_ap = freq_manager.get_nearest_airport(lat, lon)
                    if nearest_ap:
                        brg = calculate_bearing(lat, lon, nearest_ap['lat'], nearest_ap['lon'])
                        hdg1 = (rw_hdg - 30) % 360
                        hdg2 = (rw_hdg + 30) % 360
                        if hdg1 == 0: hdg1 = 360
                        if hdg2 == 0: hdg2 = 360
                        diff1 = abs((hdg1 - brg + 180) % 360 - 180)
                        diff2 = abs((hdg2 - brg + 180) % 360 - 180)
                        intercept_heading = int(hdg1 if diff1 < diff2 else hdg2)

        location_status = "ON THE GROUND" if on_ground else "AIRBORNE"

        # Region detection
        is_faa_region = (-170 < lon < -50) and (15 < lat < 75)
        altimeter_phrase = f"altimeter {altim_inhg:.2f}" if is_faa_region else f"QNH {qnh_mb:.0f}"
        atis_type = "FAA" if is_faa_region else "ICAO"

        real_metar = get_real_metar(lat, lon)
        weather_context = f"METAR: {real_metar}" if real_metar else (
            f"Wind {wind_dir:03.0f} at {wind_spd_kts:.0f} knots. {altimeter_phrase}."
        )

        # ILS / NAV1
        nav_info = ""
        if nav1_freq >= 108.0 and not on_ground:
            nav_info = f"NAV1 Tuned: {nav1_freq:.2f}."
            if nav1_hdef > 1.5:
                nav_info += " (PILOT IS DRIFTING FAR LEFT)"
            elif nav1_hdef < -1.5:
                nav_info += " (PILOT IS DRIFTING FAR RIGHT)"
            if nav1_vdef > 1.5:
                nav_info += " (PILOT IS FAR BELOW GLIDESLOPE)"
            elif nav1_vdef < -1.5:
                nav_info += " (PILOT IS FAR ABOVE GLIDESLOPE)"
            if abs(nav1_hdef) < 0.5 and abs(nav1_vdef) < 0.5:
                nav_info += " (PILOT IS ESTABLISHED)"

        # SimBrief flight plan
        simbrief_data = get_simbrief_flight_plan()
        flight_plan_context = "No IFR flight plan filed. Ask intentions."
        if simbrief_data:
            flight_plan_context = (
                f"IFR plan filed: {simbrief_data['origin']} to {simbrief_data['destination']}, "
                f"Route: {simbrief_data['route']}, Cruise FL: {simbrief_data['altitude']}."
            )

        # ATC Role
        in_air = not on_ground and flight_tracker.phase in [
            FlightPhase.CLIMB, FlightPhase.CRUISE, FlightPhase.DESCENT,
            FlightPhase.APPROACH, FlightPhase.TAKEOFF
        ]
        atc_role = freq_manager.get_atc_role(lat, lon, com1_mhz, in_air=in_air)

        # Failsafe phase updates
        if flight_tracker.phase == FlightPhase.TAKEOFF and not on_ground and alt_ft > 500:
            flight_tracker.update_phase(FlightPhase.CLIMB)
            atc_role = freq_manager.get_atc_role(lat, lon, com1_mhz, in_air=True)

        if not on_ground and airspeed > 60 and flight_tracker.phase in [
            FlightPhase.PREFLIGHT, FlightPhase.PUSHBACK, FlightPhase.TAXI,
            FlightPhase.HOLD_SHORT, FlightPhase.TAKEOFF
        ]:
            print("\n[✈️ Failsafe: Aircraft airborne, transitioning to CLIMB]")
            flight_tracker.update_phase(FlightPhase.CLIMB)

        # VHF Radio Line-of-Sight
        cache_key = f"{com1_mhz:.3f}"
        if cache_key not in frequency_station_data:
            frequency_station_data[cache_key] = {"lat": lat, "lon": lon}
        elif on_ground:
            frequency_station_data[cache_key] = {"lat": lat, "lon": lon}

        station_data = frequency_station_data.get(cache_key, {"lat": lat, "lon": lon})
        station_distance_nm = haversine_nm(lat, lon, station_data["lat"], station_data["lon"])
        max_los_nm = 1.23 * math.sqrt(max(alt_ft, 0)) + 10.0

        if atc_role in ["GROUND", "CLEARANCE DELIVERY"]:
            max_range = min(max_los_nm, 15.0)
        elif atc_role == "TOWER":
            max_range = min(max_los_nm, 50.0)
        else:
            max_range = max_los_nm

        if station_distance_nm > max_range:
            print(f"📻 [OUT OF RANGE] {station_distance_nm:.1f} NM (max {max_range:.1f})")
            heavy_static_sound.play()
            await asyncio.sleep(1.5)
            continue

        # Proactive phase updates
        user_text_lower = user_text.lower()
        if flight_tracker.phase == FlightPhase.PREFLIGHT and any(x in user_text_lower for x in ["request push", "ready to push", "ready for start", "request start"]):
            flight_tracker.update_phase(FlightPhase.PUSHBACK)
        elif flight_tracker.phase == FlightPhase.PUSHBACK and any(x in user_text_lower for x in ["request taxi", "ready to taxi", "ready for taxi"]):
            flight_tracker.update_phase(FlightPhase.TAXI)
        elif flight_tracker.phase == FlightPhase.TAXI and any(x in user_text_lower for x in ["holding short", "approaching holding", "approaching runway"]):
            flight_tracker.update_phase(FlightPhase.HOLD_SHORT)

        # GET AI RESPONSE
        model_choice = "llama-3.3-70b-versatile"

        if action == "ATIS":
            print(f"📻 Generating ATIS for {location_name}...")
            atis_number = int((time.time() // 3600) % 26)
            atis_letter = chr(65 + atis_number)
            
            phonetic_letter = to_phonetic_string(atis_letter)
            
            system_prompt = generate_atis_system_prompt(
                location_name, weather_context, active_runways,
                altimeter_phrase, atis_type, phonetic_letter
            )
            messages = [{"role": "system", "content": system_prompt}]
        elif action == "GUARD":
            # You are monitoring guard frequency 121.5...
            print("📻 GUARD frequency (121.5) - Emergency monitoring...")
            system_prompt = """You are monitoring guard frequency 121.5.
If this is an emergency call, provide immediate assistance with standard phraseology.
If general call, respond briefly and direct to appropriate frequency."""
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
        else:
            local_freqs = freq_manager.get_nearest_frequencies(lat, lon)

            print(f"ATC Thinking ({atc_role} on {com1_mhz:.3f} MHz)...")
            system_prompt = generate_atc_system_prompt(
                location_name, flight_tracker, atc_role, com1_mhz,
                alt_ft, heading, airspeed, vsi_fpm, on_ground,
                weather_context, best_active_runway, active_taxiways,
                simbrief_data, nav_info, squawk_code, tail,
                is_faa_region, altimeter_phrase, emergency, station_distance_nm,
                local_freqs, intercept_heading
            )
            chat_history.append({"role": "user", "content": user_text})

            while len(chat_history) > 20:
                chat_history.pop(0)

            messages = [{"role": "system", "content": system_prompt}] + chat_history

        # LLM call with timeout
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

        # Voice selection - role-based with deterministic selection per frequency
        if action == "ATIS":
            tts_voice = "en-US-JennyNeural"  # Female for ATIS
        elif atc_role in ["APPROACH/CENTER", "TOWER"]:
            all_voices = male_voices + female_voices[:1]
            tts_voice = all_voices[int(com1_mhz * 1000) % len(all_voices)]
        else:
            all_voices = male_voices + female_voices
            tts_voice = all_voices[int(com1_mhz * 1000) % len(all_voices)]

        # Save to history
        if action == "PTT":
            chat_history.append({"role": "assistant", "content": response_text})
            new_squawk = freq_manager.extract_and_commit_llm_assignment(response_text)
            if new_squawk:
                flight_tracker.assigned_squawk = new_squawk

        # TTS + Radio Effect
        if not response_text or not response_text.strip():
            print("[⚠️ ATC response was empty, skipping audio generation.]")
            continue

        communicate = edge_tts.Communicate(response_text, tts_voice)
        
        try:
            await communicate.save("response_raw.mp3")
        except Exception as e:
            print(f"[⚠️ TTS Generation Error: {e}]")
            continue

        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_mp3("response_raw.mp3")
            seg.export("response_raw.wav", format="wav")

            # Calculate signal strength based on distance
            signal_strength = max(0.3, 1.0 - (station_distance_nm / max_range) * 0.7)
            apply_radio_effect("response_raw.wav", "response_processed.wav",
                            signal_strength=signal_strength, distance_nm=station_distance_nm)
            playback_file = "response_processed.wav"
        except Exception as e:
            print(f"[Audio processing error, using raw: {e}]")
            playback_file = "response_raw.mp3"

        # Playback with radio effects
        squelch_sound.play()
        await asyncio.sleep(0.12)

        static_channel = heavy_static_sound.play(loops=-1)
        if static_channel:
            static_channel.set_volume(0.035)

        pygame.mixer.music.load(playback_file)
        pygame.mixer.music.play()

        # Monitor for frequency change during playback
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
                        print("[Frequency changed. Transmission cut off.]")
                        pygame.mixer.music.stop()
                        break
                    await asyncio.sleep(0.1)
        except Exception:
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)

        if static_channel:
            static_channel.stop()

        squelch_sound.play()
        await asyncio.sleep(0.15)
        pygame.mixer.music.unload()


if __name__ == "__main__":
    asyncio.run(run_atc_loop())
