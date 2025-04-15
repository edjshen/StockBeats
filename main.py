import streamlit as st
import yfinance as yf
import pretty_midi
import numpy as np
import pandas as pd
import io
from scipy.io import wavfile
import time
import tempfile
import os
from midi2audio import FluidSynth as MidiSynth

# Set page config
st.set_page_config(
    page_title="Financial MIDI Melody Generator",
    page_icon="🎵",
    layout="wide"
)

# Define musical keys and modes
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MODES = {
    "Major": [0, 2, 4, 5, 7, 9, 11],
    "Minor": [0, 2, 3, 5, 7, 8, 10],
    "Dorian": [0, 2, 3, 5, 7, 9, 10],
    "Phrygian": [0, 1, 3, 5, 7, 8, 10],
    "Lydian": [0, 2, 4, 6, 7, 9, 11],
    "Mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "Locrian": [0, 1, 3, 5, 6, 8, 10]
}

# Instrument options (General MIDI)
INSTRUMENTS = {
    "Piano": 0,
    "Electric Piano": 4,
    "Acoustic Guitar": 24,
    "Electric Guitar": 27,
    "Violin": 40,
    "Cello": 42,
    "Trumpet": 56,
    "Saxophone": 65,
    "Flute": 73,
    "Synth Lead": 80
}

# Soundfont path - specified explicitly
SOUNDFONT_PATH = r"C:\ProgramData\soundfonts\default.sf2"

@st.cache_data(ttl=3600)
def get_financial_data(ticker_symbol, period="5y", interval="1mo"):
    """Fetch financial data for a given ticker and calculate monthly returns."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist_data = ticker.history(period=period, interval=interval)
        
        if hist_data.empty:
            return None
        
        # Calculate monthly returns
        hist_data['Monthly_Return'] = hist_data['Close'].pct_change() * 100
        
        # Get company info
        try:
            info = ticker.info
            hist_data.attrs['company_name'] = info.get('shortName', ticker_symbol)
            hist_data.attrs['currency'] = info.get('currency', 'USD')
        except:
            hist_data.attrs['company_name'] = ticker_symbol
            hist_data.attrs['currency'] = 'USD'
        
        return hist_data
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {e}")
        return None

def create_midi_from_returns(returns, key, mode, instrument=0, base_octave=4, note_duration=0.5, tempo=120):
    """Create a MIDI file from financial returns data based on key and mode."""
    # Create a PrettyMIDI object
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    
    # Create an Instrument instance
    instrument_obj = pretty_midi.Instrument(program=instrument)
    
    # Get key and mode
    key_offset = KEYS.index(key)
    mode_intervals = MODES[mode]
    
    # Map returns to notes in the scale
    for i, ret in enumerate(returns):
        if pd.isna(ret):
            continue
            
        # Map return to a note in the scale (positive returns: higher notes)
        normalized_return = min(max(ret, -20), 20)  # Clamp between -20% and 20%
        scale_index = int((normalized_return + 20) / 40 * len(mode_intervals))
        scale_index = min(max(scale_index, 0), len(mode_intervals) - 1)
        
        # Get the note number
        note_offset = mode_intervals[scale_index]
        note_number = key_offset + note_offset + (base_octave * 12)
        
        # Adjust octave for very high/low returns
        octave_adjust = 0
        if ret > 10:
            octave_adjust = min(int(ret / 10), 2)  # Up to 2 octaves higher for big positive returns
        elif ret < -10:
            octave_adjust = max(int(ret / 10), -2)  # Up to 2 octaves lower for big negative returns
        
        note_number += (octave_adjust * 12)
        note_number = min(max(note_number, 0), 127)  # Ensure within MIDI range
        
        # Create a Note instance
        note_start = i * 60 / tempo * (note_duration * 4)
        note = pretty_midi.Note(
            velocity=max(60, min(100 + int(ret), 127)),  # Velocity based on return magnitude
            pitch=note_number,
            start=note_start,
            end=note_start + (60 / tempo * (note_duration * 4))
        )
        
        # Add the note to our instrument
        instrument_obj.notes.append(note)
    
    # Add the instrument to the PrettyMIDI object
    midi_data.instruments.append(instrument_obj)
    
    # Get MIDI as bytes for download
    midi_bytes = io.BytesIO()
    midi_data.write(midi_bytes)
    midi_bytes.seek(0)
    
    return midi_data, midi_bytes.getvalue()

def convert_midi_to_audio(midi_data):
    """Convert MIDI data to audio for playback using direct FluidSynth call."""
    try:
        # Get MIDI as bytes for conversion
        midi_bytes = io.BytesIO()
        midi_data.write(midi_bytes)
        midi_bytes.seek(0)
        
        # Create a temporary MIDI file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_midi:
            temp_midi.write(midi_bytes.getvalue())
            temp_midi_path = temp_midi.name
        
        # Create a temporary WAV file path
        temp_wav_path = temp_midi_path.replace('.mid', '.wav')
        
        # Construct the FluidSynth command directly
        cmd = [
            'fluidsynth', 
            '-ni',  # Non-interactive mode
            '-g', '1.0',  # Gain
            '-r', '44100',  # Sample rate
            '-F', temp_wav_path,  # Output file
            SOUNDFONT_PATH,  # Soundfont path
            temp_midi_path  # Input MIDI file
        ]
        
        # Execute the command
        import subprocess
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Read the WAV file into memory
        virtual_file = io.BytesIO()
        with open(temp_wav_path, 'rb') as wav_file:
            virtual_file.write(wav_file.read())
        virtual_file.seek(0)
        
        # Clean up temporary files
        os.remove(temp_midi_path)
        os.remove(temp_wav_path)
        
        return virtual_file
    except Exception as e:
        st.error(f"Error converting MIDI to audio: {e}")
        st.info(f"Audio preview requires FluidSynth to be installed and a valid soundfont file at {SOUNDFONT_PATH}")
        return None



def main():
    st.title("🎵 Financial MIDI Melody Generator")
    st.markdown("Transform stock market performance into musical melodies")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.info("""
        This app converts financial market data into musical melodies.
        
        1. Enter a ticker symbol
        2. Select a musical key and mode
        3. Adjust advanced options if desired
        4. Generate and listen to your melody
        5. Download as a MIDI file
        """)
        
        st.header("How it works")
        st.write("""
        - Monthly price changes are mapped to musical notes
        - Positive returns create higher notes
        - Negative returns create lower notes
        - The magnitude of returns affects the pitch range
        """)
        
        # Display soundfont information
        st.header("Audio Settings")
        st.info(f"Using soundfont: {SOUNDFONT_PATH}")
    
    # Main input area with clean layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ticker_symbol = st.text_input("Enter Ticker Symbol", "AAPL")
    
    with col2:
        key = st.selectbox("Select Key", KEYS)
    
    with col3:
        mode = st.selectbox("Select Mode", list(MODES.keys()))
    
    # Advanced options in expandable section
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            period = st.selectbox("Time Period", 
                           ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], 
                           index=3)
            tempo = st.slider("Tempo (BPM)", 60, 180, 120, 5)
        
        with col2:
            base_octave = st.slider("Base Octave", 2, 6, 4)
            instrument_name = st.selectbox("Instrument", list(INSTRUMENTS.keys()))
            instrument = INSTRUMENTS[instrument_name]
    
    # Generate button with full width
    if st.button("Generate Melody", use_container_width=True):
        if not ticker_symbol:
            st.error("Please enter a ticker symbol")
            return
        
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Fetch data
        status_text.text(f"Fetching data for {ticker_symbol}...")
        data = get_financial_data(ticker_symbol, period=period)
        progress_bar.progress(33)
        
        if data is not None and not data.empty:
            # Get company information
            company_name = data.attrs.get('company_name', ticker_symbol)
            
            # Step 2: Process data
            status_text.text("Processing financial data...")
            monthly_returns = data['Monthly_Return'].dropna().tolist()
            progress_bar.progress(66)
            
            # Display the financial data
            st.subheader(f"{company_name} ({ticker_symbol}) - Monthly Returns")
            
            # Chart and metrics
            chart_col, metrics_col = st.columns([3, 1])
            with chart_col:
                st.line_chart(data['Monthly_Return'].dropna())
            
            with metrics_col:
                st.metric("Average Monthly Return", f"{data['Monthly_Return'].mean():.2f}%")
                st.metric("Maximum Return", f"{data['Monthly_Return'].max():.2f}%")
                st.metric("Minimum Return", f"{data['Monthly_Return'].min():.2f}%")
            
            # Step 3: Generate MIDI
            status_text.text("Creating melody...")
            midi_data, midi_bytes = create_midi_from_returns(
                monthly_returns, key, mode, instrument, base_octave, 0.5, tempo
            )
            
            # Step 4: Convert to audio
            status_text.text("Converting to audio with custom soundfont...")
            virtual_file = convert_midi_to_audio(midi_data)
            progress_bar.progress(100)
            status_text.empty()
            
            # Display playback and download section
            st.subheader(f"Your {key} {mode} Financial Melody")
            
            # Audio player and download button
            audio_col, download_col = st.columns([3, 1])
            with audio_col:
                if virtual_file:
                    st.audio(virtual_file)
                else:
                    st.warning("Audio preview not available. Please download the MIDI file.")
            
            with download_col:
                st.download_button(
                    "Download MIDI File",
                    midi_bytes,
                    file_name=f"{ticker_symbol}_{key}_{mode}.mid",
                    mime="audio/midi",
                    use_container_width=True
                )
            
            # Explanation of the melody
            st.info(f"""
            This melody represents the monthly price performance of {company_name} ({ticker_symbol}).
            - Each note corresponds to a monthly return
            - Higher notes represent positive returns
            - Lower notes represent negative returns
            - Melody is in {key} {mode} at {tempo} BPM using {instrument_name}
            """)
        else:
            progress_bar.empty()
            status_text.empty()
            st.error(f"No data available for '{ticker_symbol}'. Please check the ticker symbol and try again.")

if __name__ == "__main__":
    main()
