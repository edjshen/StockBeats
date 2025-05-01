import streamlit as st
import yfinance as yf
import pretty_midi
import numpy as np
import pandas as pd
import io
import time
import pygame
import base64
import requests
import requests
from alpha_vantage.timeseries import TimeSeries

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
})
yf.set_tz_cache_location(".cache")  # Set cache location to a writable directory

# Set page config
st.set_page_config(
    page_title="Financial MIDI Melody Generator",
    page_icon="🎵",
    layout="wide"
)

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

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

# Define measure options
MEASURE_OPTIONS = [4, 8, 16, 32]

@st.cache_data(ttl=3600)
def getAlphaVantageData(ticker_symbol, period="5y", interval="1mo"):
    """Fetch financial data from Alpha Vantage API."""
    # Get API key from streamlit secrets
    api_key = st.secrets["AlphaVantageAPIkey"]
    
    # Initialize Alpha Vantage TimeSeries
    ts = TimeSeries(key=api_key, output_format='pandas')
    
    # Choose the appropriate function based on the interval
    if interval == "1mo":
        data, meta_data = ts.get_monthly(symbol=ticker_symbol)
    elif interval == "1d":
        data, meta_data = ts.get_daily(symbol=ticker_symbol, outputsize='full')
    else:
        # For other intervals, we will fall back to yfinance
        raise ValueError(f"Interval {interval} not directly supported with Alpha Vantage in this app.")
    
    # Rename columns to match yfinance format
    data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    }, inplace=True)
    
    # Apply period filtering
    if period != "max":
        if period == "5y":
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=5)
        elif period == "3y":
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=3)
        elif period == "1y":
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=1)
        elif period == "3mo":
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=3)
        elif period == "10y":
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=10)
        data = data[data.index >= cutoff_date]
    
    # Calculate monthly returns
    data['Monthly_Return'] = data['Close'].pct_change() * 100
    
    # Calculate volatility
    volatility = data['Monthly_Return'].rolling(window=3).std().fillna(
        data['Monthly_Return'].std()
    )
    data['Volatility'] = volatility
    
    # Set metadata
    data.attrs['company_name'] = ticker_symbol
    data.attrs['currency'] = 'USD'
    
    return data

@st.cache_data(ttl=3600)
def getFinancialData(ticker_symbol, period="5y", interval="1mo", max_retries=1):
    """Fetch financial data for a given ticker, first trying Alpha Vantage, then yfinance as fallback."""
    # First try using Alpha Vantage
    try:
        # Check if Alpha Vantage API key is available
        if "AlphaVantageAPIkey" in st.secrets:
            st.info("Fetching data from Alpha Vantage...")
            data = getAlphaVantageData(ticker_symbol, period, interval)
            if data is not None and not data.empty:
                st.success("Successfully retrieved data from Alpha Vantage")
                return data
        else:
            st.warning("Alpha Vantage API key not found in secrets. Falling back to yfinance.")
    except Exception as e:
        st.warning(f"Alpha Vantage error: {e}. Falling back to yfinance.")
    
    # If Alpha Vantage failed or key not found, fall back to yfinance
    st.info("Fetching data from yfinance...")
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist_data = ticker.history(period=period, interval=interval)
            
            if hist_data.empty:
                return None
            
            # Calculate monthly returns
            hist_data['Monthly_Return'] = hist_data['Close'].pct_change() * 100
            
            # Calculate volatility (standard deviation of returns)
            volatility = hist_data['Monthly_Return'].rolling(window=3).std().fillna(
                hist_data['Monthly_Return'].std()
            )
            hist_data['Volatility'] = volatility
            
            # Get company info
            try:
                info = ticker.info
                hist_data.attrs['company_name'] = info.get('shortName', ticker_symbol)
                hist_data.attrs['currency'] = info.get('currency', 'USD')
            except:
                hist_data.attrs['company_name'] = ticker_symbol
                hist_data.attrs['currency'] = 'USD'
            
            st.success("Successfully retrieved data from yfinance")
            return hist_data
        except Exception as e:
            st.error(f"Error fetching data for {ticker_symbol}: {e}")
            if "Rate limited" in str(e) and attempt < max_retries - 1:
                # Exponential backoff: wait longer between retries
                wait_time = (2 ** attempt) + 1
                print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                raise


def createMidiFromReturns(returns, volatilities, key, mode, instrument=0, base_octave=4, tempo=120, num_notes=None):
    """Create a MIDI file from financial returns data based on key and mode with varying rhythm."""
    # Create a PrettyMIDI object
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    
    # Create an Instrument instance
    instrument_obj = pretty_midi.Instrument(program=instrument)
    
    # Get key and mode
    key_offset = KEYS.index(key)
    mode_intervals = MODES[mode]
    
    # Define note durations
    min_duration = 0.25  # 16th note
    max_duration = 2.0   # half note
    
    # Handle case where volatilities list might be shorter than returns
    if len(volatilities) < len(returns):
        mean_vol = sum(volatilities) / len(volatilities) if volatilities else 0
        volatilities.extend([mean_vol] * (len(returns) - len(volatilities)))
    
    # If num_notes is specified, use only the last num_notes
    if num_notes is not None:
        returns = returns[-num_notes:]
        volatilities = volatilities[-num_notes:]
    
    current_time = 0.0
    
    for i, (ret, vol) in enumerate(zip(returns, volatilities)):
        if pd.isna(ret) or pd.isna(vol):
            continue
            
        normalized_return = min(max(ret, -20), 20)
        scale_index = int((normalized_return + 20) / 40 * len(mode_intervals))
        scale_index = min(max(scale_index, 0), len(mode_intervals) - 1)
        
        note_offset = mode_intervals[scale_index]
        note_number = key_offset + note_offset + (base_octave * 12)
        
        octave_adjust = 0
        if ret > 10:
            octave_adjust = min(int(ret / 10), 2)
        elif ret < -10:
            octave_adjust = max(int(ret / 10), -2)
        
        note_number += (octave_adjust * 12)
        note_number = min(max(note_number, 0), 127)
        
        vol_range = max(volatilities) - min(volatilities) if len(set(volatilities)) > 1 else 1.0
        if vol_range > 0:
            norm_vol = (vol - min(volatilities)) / vol_range
        else:
            norm_vol = 0.5
        
        inv_norm_vol = 1.0 - norm_vol
        note_duration = min_duration + (inv_norm_vol * (max_duration - min_duration))
        duration_seconds = 60 / tempo * note_duration
        
        note = pretty_midi.Note(
            velocity=max(60, min(100 + int(ret), 127)),
            pitch=note_number,
            start=current_time,
            end=current_time + duration_seconds
        )
        
        current_time += duration_seconds
        instrument_obj.notes.append(note)
    
    midi_data.instruments.append(instrument_obj)
    midi_bytes = io.BytesIO()
    midi_data.write(midi_bytes)
    midi_bytes.seek(0)
    
    return midi_data, midi_bytes.getvalue()

def simple_note_synthesis(frequency, duration, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    envelope = np.ones_like(t)
    attack = int(0.05 * sample_rate)
    release = int(0.1 * sample_rate)
    if len(t) > attack:
        envelope[:attack] = np.linspace(0, 1, attack)
    if len(t) > release:
        envelope[-release:] = np.linspace(1, 0, release)
    note = amplitude * np.sin(2 * np.pi * frequency * t) * envelope
    return note

def midi_to_wav(midi_data, sample_rate=44100):
    duration = midi_data.get_end_time() + 1
    audio = np.zeros(int(sample_rate * duration))
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            frequency = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            start_sample = int(note.start * sample_rate)
            end_sample = int(note.end * sample_rate)
            note_duration = note.end - note.start
            note_audio = simple_note_synthesis(
                frequency, 
                note_duration, 
                sample_rate, 
                amplitude=note.velocity / 127.0 * 0.5
            )
            end_idx = min(start_sample + len(note_audio), len(audio))
            audio[start_sample:end_idx] += note_audio[:end_idx-start_sample]
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    audio_int16 = np.int16(audio * 32767)
    audio_stereo = np.column_stack((audio_int16, audio_int16))
    import wave
    import struct
    virtual_file = io.BytesIO()
    with wave.open(virtual_file, 'wb') as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for sample in audio_stereo:
            wav_file.writeframes(struct.pack('<hh', sample[0], sample[1]))
    virtual_file.seek(0)
    return virtual_file

def create_audio_player_html(wav_data):
    base64_wav = base64.b64encode(wav_data.read()).decode('utf-8')
    wav_data.seek(0)
    audio_html = f"""
    <audio controls style="width: 100%;">
        <source src="data:audio/wav;base64,{base64_wav}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html

def main():
    st.title("🎵 Financial MIDI Melody Generator")
    st.markdown("Transform stock market performance into musical melodies")
    
    with st.sidebar:
        st.header("About")
        st.info("""
        This app converts financial market data into musical melodies.
        
        1. Enter a ticker symbol
        2. Select a musical key and mode
        3. Adjust advanced options if desired
        4. Generate your melody
        5. Preview and download as a MIDI file
        """)
        st.header("How it works")
        st.write("""
        - Monthly price changes are mapped to musical notes
        - Positive returns create higher notes
        - Negative returns create lower notes
        - The magnitude of returns affects the pitch range
        """)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker_symbol = st.text_input("Enter Ticker Symbol", "AAPL")
    with col2:
        key = st.selectbox("Select Key", KEYS)
    with col3:
        mode = st.selectbox("Select Mode", list(MODES.keys()))
    
    # User can select MIDI length (number of measures)
    midi_length_measures = st.selectbox("Select MIDI Length (Measures)", MEASURE_OPTIONS, index=0)
    
    optionsList = ["3mo", "1y", "3y", "5y", "10y", "max"]
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox("Lookback Period (Financial Data)", optionsList, index=1)
            tempo = st.slider("Tempo (BPM)", 60, 180, 120, 5)
        with col2:
            base_octave = st.slider("Base Octave", 2, 6, 4)
            instrument_name = st.selectbox("Instrument", list(INSTRUMENTS.keys()))
            instrument = INSTRUMENTS[instrument_name]
    
    if st.button("Generate Melody", use_container_width=True):
        if not ticker_symbol:
            st.error("Please enter a ticker symbol")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data using the selected lookback period
        selected_period = period
        status_text.text(f"Fetching data for {ticker_symbol}...")
        data = getFinancialData(ticker_symbol, period=selected_period)
        progress_bar.progress(40)
        
        if data is not None and not data.empty:
            company_name = data.attrs.get('company_name', ticker_symbol)
            status_text.text("Processing financial data...")
            monthly_returns = data['Monthly_Return'].dropna().tolist()
            volatilities = data['Volatility'].dropna().tolist()
            progress_bar.progress(60)
            
            st.subheader(f"{company_name} ({ticker_symbol}) - Monthly Returns")
            chart_col, metrics_col = st.columns([3, 1])
            with chart_col:
                st.line_chart(data['Monthly_Return'].dropna())
            with metrics_col:
                st.metric("Average Monthly Return", f"{data['Monthly_Return'].mean():.2f}%")
                st.metric("Maximum Return", f"{data['Monthly_Return'].max():.2f}%")
                st.metric("Minimum Return", f"{data['Monthly_Return'].min():.2f}%")
            
            # Each measure is 4 notes (assuming 4/4), so total notes = measures * 4
            num_notes = midi_length_measures * 4
            
            status_text.text("Creating melody...")
            midi_data, midi_bytes = createMidiFromReturns(
                monthly_returns, volatilities, key, mode, instrument, base_octave, tempo, num_notes=num_notes
            )
            progress_bar.progress(80)
            
            status_text.text("Generating audio preview...")
            wav_file = midi_to_wav(midi_data)
            progress_bar.progress(100)
            status_text.empty()
            
            st.subheader(f"Your {key} {mode} Financial Melody")
            preview_tab, download_tab = st.tabs(["Audio Preview", "Download MIDI"])
            with preview_tab:
                if wav_file:
                    audio_player = create_audio_player_html(wav_file)
                    st.markdown(audio_player, unsafe_allow_html=True)
                else:
                    st.warning("Audio preview could not be generated. You can still download the MIDI file.")
            with download_tab:
                st.download_button(
                    "Download MIDI File",
                    midi_bytes,
                    file_name=f"{ticker_symbol}_{key}_{mode}.mid",
                    mime="audio/midi",
                    use_container_width=True
                )
                st.caption("Download the MIDI file to play in your favorite media player or DAW software.")
            st.info(f"""
            This melody represents the monthly price performance of {company_name} ({ticker_symbol}).
            - Each note corresponds to a monthly return
            - Higher notes represent positive returns
            - Lower notes represent negative returns
            - Higher volatility creates faster rhythms (shorter notes)
            - Lower volatility creates slower rhythms (longer notes)
            - Melody is in {key} {mode} at {tempo} BPM using {instrument_name}
            """)
        else:
            progress_bar.empty()
            status_text.empty()
            st.error(f"No data available for '{ticker_symbol}'. Please check the ticker symbol and try again.")

if __name__ == "__main__":
    main()
