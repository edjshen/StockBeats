# pages/4_📊_Portfolio_Builder.py

import streamlit as st
import yfinance as yf
import pretty_midi
import numpy as np
import pandas as pd
import io
import base64
import wave
import struct

# Set page config
st.set_page_config(
    page_title="Portfolio MIDI Melody Generator",
    page_icon="📊",
    layout="wide"
)

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Define constants (keep the same as in the main app)
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
# Define measures options - no longer tied to periods
MEASURES_OPTIONS = ["4 Measures", "8 Measures", "16 Measures", "32 Measures"]

@st.cache_data(ttl=3600)
def getFinancialData(ticker_symbol, period="5y", interval="1mo"):
    """Fetch financial data for a given ticker and calculate monthly returns."""
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
        
        return hist_data
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {e}")
        return None

def createMidiFromReturns(returns, volatilities, key, mode, instrument=0, base_octave=4, tempo=120):
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
    
    # Keep track of current position in time
    current_time = 0.0
    
    # Map returns to notes in the scale
    for i, (ret, vol) in enumerate(zip(returns, volatilities)):
        if pd.isna(ret) or pd.isna(vol):
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
        
        # Map volatility to note duration (higher volatility = shorter notes)
        vol_range = max(volatilities) - min(volatilities) if len(set(volatilities)) > 1 else 1.0
        if vol_range > 0:
            norm_vol = (vol - min(volatilities)) / vol_range
        else:
            norm_vol = 0.5  # Default to middle value if all volatilities are the same
        
        # Invert normalized volatility (higher volatility = shorter notes)
        inv_norm_vol = 1.0 - norm_vol
        
        # Map to duration between min_duration and max_duration
        note_duration = min_duration + (inv_norm_vol * (max_duration - min_duration))
        
        # Duration in seconds
        duration_seconds = 60 / tempo * note_duration
        
        # Create a Note instance
        note = pretty_midi.Note(
            velocity=max(60, min(100 + int(ret), 127)),  # Velocity based on return magnitude
            pitch=note_number,
            start=current_time,
            end=current_time + duration_seconds
        )
        
        # Update current time for next note
        current_time += duration_seconds
        
        # Add the note to our instrument
        instrument_obj.notes.append(note)
    
    # Add the instrument to the PrettyMIDI object
    midi_data.instruments.append(instrument_obj)
    
    # Get MIDI as bytes for download
    midi_bytes = io.BytesIO()
    midi_data.write(midi_bytes)
    midi_bytes.seek(0)
    
    return midi_data, midi_bytes.getvalue()

def simple_note_synthesis(frequency, duration, sample_rate=44100, amplitude=0.5):
    """Generate a simple sine wave for a musical note."""
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
    """Convert MIDI data to WAV audio using pure Python synthesis."""
    # Initialize an empty audio array
    duration = midi_data.get_end_time() + 1  # Add a bit of padding
    audio = np.zeros(int(sample_rate * duration))
    
    # For each instrument and note, synthesize and add to the audio
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Get the frequency for this note
            frequency = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            
            # Calculate start and end samples
            start_sample = int(note.start * sample_rate)
            end_sample = int(note.end * sample_rate)
            note_duration = note.end - note.start
            
            # Generate the note audio
            note_audio = simple_note_synthesis(
                frequency, 
                note_duration, 
                sample_rate, 
                amplitude=note.velocity / 127.0 * 0.5
            )
            
            # Add the note to the main audio
            end_idx = min(start_sample + len(note_audio), len(audio))
            audio[start_sample:end_idx] += note_audio[:end_idx-start_sample]
    
    # Normalize to prevent clipping
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    
    # Convert to 16-bit PCM
    audio_int16 = np.int16(audio * 32767)
    
    # Create a stereo array
    audio_stereo = np.column_stack((audio_int16, audio_int16))
    
    # Create a WAV file
    virtual_file = io.BytesIO()
    with wave.open(virtual_file, 'wb') as wav_file:
        wav_file.setnchannels(2)  # Stereo
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Pack data as binary
        for sample in audio_stereo:
            wav_file.writeframes(struct.pack('<hh', sample[0], sample[1]))
    
    virtual_file.seek(0)
    return virtual_file

def create_audio_player_html(wav_data):
    """Create an HTML audio player for WAV data."""
    base64_wav = base64.b64encode(wav_data.read()).decode('utf-8')
    wav_data.seek(0)  # Reset the file pointer
    
    # Create HTML audio element
    audio_html = f"""
    <audio controls style="width: 100%;">
        <source src="data:audio/wav;base64,{base64_wav}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html

def calculate_portfolio_returns(tickers, weights, period):
    """Calculate weighted returns for a portfolio of stocks"""
    all_data = {}
    # Fetch data for all tickers
    for ticker in tickers:
        data = getFinancialData(ticker, period=period)
        if data is not None:
            all_data[ticker] = data
    
    if not all_data:
        return None, None
    
    # Get a common date range for all stocks
    returns_df = pd.DataFrame()
    volatility_df = pd.DataFrame()
    
    for ticker, data in all_data.items():
        returns_df[ticker] = data['Monthly_Return']
        volatility_df[ticker] = data['Volatility']
    
    # Drop dates where any stock has missing data
    returns_df = returns_df.dropna()
    volatility_df = volatility_df.dropna()
    
    if returns_df.empty:
        return None, None
    
    # Calculate weighted returns and volatility
    weighted_returns = pd.Series(0.0, index=returns_df.index)
    weighted_volatility = pd.Series(0.0, index=volatility_df.index)
    
    for i, ticker in enumerate(tickers):
        if ticker in returns_df.columns:
            weighted_returns += returns_df[ticker] * weights[i]
            weighted_volatility += volatility_df[ticker] * weights[i]
    
    return weighted_returns.tolist(), weighted_volatility.tolist()

def main():
    st.title("📊 Portfolio MIDI Melody Generator")
    st.markdown("Transform your investment portfolio performance into musical melodies")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.info("""
        This page allows you to create melodies from a portfolio of securities.
        
        1. Add tickers to your portfolio
        2. Set the weight for each security
        3. Select musical parameters
        4. Generate your portfolio melody
        5. Preview and download as a MIDI file
        """)
        
        st.header("How it works")
        st.write("""
        - Monthly portfolio returns are mapped to musical notes
        - Positive returns create higher notes
        - Negative returns create lower notes
        - The magnitude of returns affects the pitch range
        - Higher volatility creates faster rhythms
        """)
    
    # Portfolio composition section
    st.subheader("Portfolio Composition")
    
    # Initialize portfolio in session state if not already
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = [{"ticker": "AAPL", "weight": 50}, {"ticker": "MSFT", "weight": 50}]
    
    # Functions to manage portfolio
    def add_stock():
        st.session_state.portfolio.append({"ticker": "", "weight": 0})
    
    def remove_stock(index):
        st.session_state.portfolio.pop(index)
    
    # Portfolio input table headers
    cols = st.columns([3, 2, 1])
    with cols[0]:
        st.write("**Ticker Symbol**")
    with cols[1]:
        st.write("**Weight (%)**")
    
    # Display portfolio inputs
    for i, stock in enumerate(st.session_state.portfolio):
        cols = st.columns([3, 2, 1])
        with cols[0]:
            st.session_state.portfolio[i]["ticker"] = st.text_input(
                "Ticker", value=stock["ticker"], key=f"ticker_{i}", 
                label_visibility="collapsed"
            )
        with cols[1]:
            st.session_state.portfolio[i]["weight"] = st.number_input(
                "Weight", min_value=0.0, max_value=100.0, value=float(stock["weight"]), 
                key=f"weight_{i}", label_visibility="collapsed"
            )
        with cols[2]:
            if len(st.session_state.portfolio) > 1:  # Ensure at least one stock remains
                st.button("🗑️", key=f"remove_{i}", on_click=remove_stock, args=(i,))
    
    # Add stock button
    st.button("➕ Add Security", on_click=add_stock)
    
    # Check if weights sum to 100%
    total_weight = sum(stock["weight"] for stock in st.session_state.portfolio)
    if abs(total_weight - 100) > 0.01:
        st.warning(f"Total portfolio weight is {total_weight}%. Adjust to total 100%.")
    
    # Melody options section - MODIFIED to separate MIDI length from lookback period
    st.subheader("Melody Options")
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    
    with col1:
        measures = st.selectbox("Select MIDI Length", MEASURES_OPTIONS, index=0)
    
    with col2:
        lookback_period = st.selectbox("Select Lookback Period", ["3mo", "1y", "3y", "5y", "10y", "max"], index=1)
    
    with col3:
        key = st.selectbox("Select Key", KEYS)
    
    with col4:
        mode = st.selectbox("Select Mode", list(MODES.keys()))
    
    # Advanced options in expandable section
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            tempo = st.slider("Tempo (BPM)", 60, 180, 120, 5)
        
        with col2:
            base_octave = st.slider("Base Octave", 2, 6, 4)
            instrument_name = st.selectbox("Instrument", list(INSTRUMENTS.keys()))
            instrument = INSTRUMENTS[instrument_name]
    
    # Generate button with full width
    if st.button("Generate Portfolio Melody", use_container_width=True):
        # Validate portfolio
        if abs(total_weight - 100) > 1:
            st.error("Portfolio weights must sum to 100%.")
            return
        
        # Extract ticker symbols and weights
        tickers = [stock["ticker"] for stock in st.session_state.portfolio if stock["ticker"]]
        weights = [stock["weight"] / 100.0 for stock in st.session_state.portfolio if stock["ticker"]]
        
        if not tickers:
            st.error("Please add at least one valid ticker to your portfolio.")
            return
        
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Fetch data for each ticker using the selected lookback period
        status_text.text(f"Fetching data for portfolio...")
        progress_bar.progress(20)
        
        # Calculate portfolio returns
        weighted_returns, weighted_volatility = calculate_portfolio_returns(tickers, weights, lookback_period)
        progress_bar.progress(40)
        
        if weighted_returns is None or weighted_volatility is None:
            progress_bar.empty()
            status_text.empty()
            st.error("Could not retrieve data for all tickers in the portfolio.")
            return
        
        # Step 2: Process portfolio data
        status_text.text("Processing portfolio returns...")
        progress_bar.progress(60)
        
        # Display the portfolio performance
        st.subheader("Portfolio Performance")
        
        # Convert to pandas series for charting
        returns_series = pd.Series(weighted_returns)
        
        # Chart and metrics
        chart_col, metrics_col = st.columns([3, 1])
        with chart_col:
            st.line_chart(returns_series)
        
        with metrics_col:
            st.metric("Average Monthly Return", f"{returns_series.mean():.2f}%")
            st.metric("Maximum Return", f"{returns_series.max():.2f}%")
            st.metric("Minimum Return", f"{returns_series.min():.2f}%")
        
        # Step 3: Generate MIDI
        status_text.text("Creating melody...")
        midi_data, midi_bytes = createMidiFromReturns(
            weighted_returns, weighted_volatility, key, mode, instrument, base_octave, tempo
        )
        progress_bar.progress(80)
        
        # Step 4: Create audio preview
        status_text.text("Generating audio preview...")
        wav_file = midi_to_wav(midi_data)
        progress_bar.progress(100)
        status_text.empty()
        
        # Display melody section
        st.subheader(f"Your {key} {mode} Portfolio Melody")
        
        # Preview and Download in tabs
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
                file_name=f"Portfolio_{key}_{mode}.mid",
                mime="audio/midi",
                use_container_width=True
            )
            st.caption("Download the MIDI file to play in your favorite media player or DAW software.")
        
        # Explanation of the melody
        st.info(f"""
        This melody represents the monthly performance of your investment portfolio.
        - Each note corresponds to a monthly return
        - Higher notes represent positive returns
        - Lower notes represent negative returns
        - Higher volatility creates faster rhythms (shorter notes)
        - Lower volatility creates slower rhythms (longer notes)
        - Melody is in {key} {mode} at {tempo} BPM using {instrument_name}
        """)

if __name__ == "__main__":
    main()
