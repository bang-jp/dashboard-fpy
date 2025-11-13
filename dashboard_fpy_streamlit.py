# dashboard_fpy_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os, re, io, tempfile, configparser, time
from typing import Optional

# -------------------------
# Config / Constants
# -------------------------
CONFIG_FILE = "config.ini"
BLACKLIST = [
    "_Dummy_ENG_", "_ENG_", "_Dummy_PROD_", "_DUMMY_LASER_TEST_ENG_",
    "_DM_DUMMY_PROD_", "_DM_DUMMY_ENG_", "_DUMMY_LASER_ENG_"
]

# ported MODEL_MAP/tester_model_map (trimmed to common entries — keep yours)
MODEL_MAP = {
    "10104905_CORAL_FT": "End Tester Dual Motor",
    "10104905_CORAL_MOTORCAL": "Calibration Dual Motor",
    "10104911_CORAL_FT": "Color Sensor",
    "10104902_CORAL_MOTORCAL": "Calibration Single Motor",
    "10104902_CORAL_FT": "End Tester Single Motor",
    "10103651-ACSTC-": "Acoustic",
    "LASERMARK-FT01": "Laser Marking",
    "LASERMARK-FT02": "Laser Marking",
    "10103651_JOURNEY_END_TESTER": "Journey",
    "DistanceSensor-FT02": "Distance Sensor",
    "DistanceSensor-FT01": "Distance Sensor",
}

tester_model_map = {
    "10104905_CORAL_FT01": "End Tester Dual Motor",
    "10104905_CORAL_MOTORCAL01": "Calibration Dual Motor TS1",
    "10104905_CORAL_MOTORCAL02": "Calibration Dual Motor TS2",
    "10104911_CORAL_FT01": "Color Sensor",
    "10104902_CORAL_FT01": "End Tester Single Motor",
    "10104902_CORAL_MOTORCAL01": "Calibration Single Motor",
    # add other keys if needed...
}

CAVITY_REGEX = re.compile(r"(CA\d+)", re.IGNORECASE)
JOURNEY_REGEX = re.compile(r"(Journey_End_TesterTS\d+-\d+)", re.IGNORECASE)
ACOUSTIC_REGEX = re.compile(r"(Journey_Acoustic_TesterTS\d+)", re.IGNORECASE)
LASER_MARKING_REGEX = re.compile(r"(Journey_Laser_MarkingTS\d+)", re.IGNORECASE)
DISTANCE_REGEX = re.compile(r"(10037316\(STM(?:8|32)\)_DISTANCE SENSOR_FT\d+)", re.IGNORECASE)

POSSIBLE_CAVITY_COLS = [
    "TesterCavity", "Tester", "TesterName", "Fixture", "FixtureName",
    "Station", "StationName", "Cavity", "Location", "Machine", "Line"
]

# -------------------------
# Config helpers
# -------------------------
def load_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    settings = {}
    if 'SETTINGS' in config:
        folder_string = config['SETTINGS'].get('DefaultFolders', '')
        settings['admin_password'] = config['SETTINGS'].get('AdminPassword', 'john')
        folders = [f.strip() for f in folder_string.split(';') if f.strip() and os.path.isdir(f.strip())]
        settings['default_folders'] = folders
    else:
        settings['default_folders'] = []
        settings['admin_password'] = 'john'
    return settings

def save_config(key, value):
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    if 'SETTINGS' not in config:
        config['SETTINGS'] = {}
    if key == 'DefaultFolders' and isinstance(value, list):
        config['SETTINGS'][key] = ";".join(value)
    else:
        config['SETTINGS'][key] = value
    try:
        with open(CONFIG_FILE, 'w') as cf:
            config.write(cf)
    except Exception as e:
        st.error(f"Error saving config: {e}")

# -------------------------
# Utility functions (ported)
# -------------------------
def is_blacklisted(filename: str) -> bool:
    fn = filename.upper()
    return any(b.upper() in fn for b in BLACKLIST)

def extract_model_from_filename(filename: str) -> str:
    fn = filename.upper()
    for key, model in tester_model_map.items():
        if key.upper() in fn:
            return model
    for key, model in MODEL_MAP.items():
        if key.upper() in fn:
            return model
    return "Unknown"

def get_model_from_testername(tester_name: str) -> str:
    if not tester_name or pd.isna(tester_name): return "Unknown"
    tn = str(tester_name).upper()
    for key, model in MODEL_MAP.items():
        if key.upper() in tn:
            return model
    return "Unknown"

def try_parse_datetime_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    formats = [
        "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"
    ]
    parsed = pd.to_datetime(s, format=formats[0], errors="coerce")
    remaining_mask = parsed.isna()
    if remaining_mask.any():
        for fmt in formats[1:]:
            parsed.loc[remaining_mask] = pd.to_datetime(s[remaining_mask], format=fmt, errors="coerce")
            remaining_mask = parsed.isna()
            if not remaining_mask.any():
                break
    if remaining_mask.any():
        parsed.loc[remaining_mask] = pd.to_datetime(s[remaining_mask], errors="coerce", dayfirst=True)
    return parsed

def extract_date_from_filename(filename: str):
    m = re.search(r'_(\d{2})(\d{2})(\d{4})_', filename)
    if m:
        try:
            return datetime.strptime(m.group(0).strip('_'), "%d%m%Y")
        except ValueError:
            pass
    m2 = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if m2:
        try:
            return datetime.strptime(m2.group(0), "%Y%m%d")
        except ValueError:
            pass
    return pd.NaT

def infer_tester_cavity(df: pd.DataFrame, filepath: str) -> pd.Series:
    m_journey = JOURNEY_REGEX.search(os.path.basename(filepath))
    if m_journey:
        return pd.Series([m_journey.group(1).replace("_", " ").upper()] * len(df), index=df.index)
    m_acoustic = ACOUSTIC_REGEX.search(os.path.basename(filepath))
    if m_acoustic:
        return pd.Series([m_acoustic.group(1).replace("_", " ").upper()] * len(df), index=df.index)
    m_laser = LASER_MARKING_REGEX.search(os.path.basename(filepath))
    if m_laser:
        return pd.Series([m_laser.group(1).replace("_", " ").upper()] * len(df), index=df.index)
    m_distance = DISTANCE_REGEX.search(os.path.basename(filepath))
    if m_distance:
        return pd.Series([m_distance.group(1).replace("_", " ").upper()] * len(df), index=df.index)
    for col in POSSIBLE_CAVITY_COLS:
        if col in df.columns:
            s = df[col].astype(str)
            found = s.str.extract(CAVITY_REGEX, expand=False)
            if found.notna().any():
                return found.str.upper().fillna("Unknown")
    m_cavity = CAVITY_REGEX.search(os.path.basename(filepath))
    if m_cavity:
        return pd.Series([m_cavity.group(1).upper()] * len(df), index=df.index)
    return pd.Series(["Unknown"] * len(df), index=df.index)

def infer_model(df: pd.DataFrame, filepath: str) -> pd.Series:
    if "TesterName" in df.columns:
        inferred = df["TesterName"].astype(str).apply(get_model_from_testername)
        inferred = inferred.replace("Unknown", extract_model_from_filename(os.path.basename(filepath)))
        return inferred
    filename_model = extract_model_from_filename(os.path.basename(filepath))
    return pd.Series([filename_model] * len(df), index=df.index)

# -------------------------
# File loading (support filepath & file-like)
# -------------------------
def load_and_clean(filepath: str) -> Optional[pd.DataFrame]:
    """
    Original function expecting a filesystem path.
    Keep it as-is for reading local files.
    """
    try:
        header_index = 0
        with open(filepath, 'r', encoding='latin1', errors='ignore') as f:
            for i, line in enumerate(f):
                if any(col in line for col in ["SerialNumber", "TestStatus", "Result"]):
                    header_index = i
                    break

        df = None
        for enc in ['utf-8', 'latin1', 'ISO-8859-1']:
            for sep in ['\t', ',', ';', ' ']:
                try:
                    df = pd.read_csv(
                        filepath, sep=sep, encoding=enc, engine='python',
                        on_bad_lines='skip', skiprows=header_index, skipinitialspace=True
                    )
                    df.columns = df.columns.str.strip()
                    if any(col in df.columns for col in ["SerialNumber", "TestStatus", "Result"]):
                        if "TestStatus" in df.columns:
                            df.rename(columns={"TestStatus": "Result"}, inplace=True)
                        break
                    else:
                        df = None
                except Exception:
                    continue
            if df is not None:
                break

        if df is None or df.empty:
            return None

        # Normalize / fill columns
        df['SerialNumber'] = df.get('SerialNumber', pd.Series([""] * len(df))).astype(str).str.strip()
        df['Result'] = df.get('Result', pd.Series([""] * len(df))).astype(str).str.strip()
        df['FailedID'] = df.get('FailedID', pd.Series(["N/A"] * len(df))).astype(str).str.strip().fillna("N/A")
        df['FailedValue'] = pd.to_numeric(df.get('FailedValue', pd.Series([0] * len(df))), errors='coerce').fillna(0)
        df["ProStage"] = df.get("ProStage", pd.Series(["Test"] * len(df))).astype(str).str.strip().replace({"": "Test", "nan": "Test", "none": "Test", "na": "Test"})

        if "TestDate" in df.columns and "TestStart" in df.columns:
            dt_str = df["TestDate"].astype(str) + " " + df["TestStart"].astype(str)
            df["TestStartNorm"] = pd.to_datetime(dt_str, errors="coerce", dayfirst=True, infer_datetime_format=True)
        elif "TestStart" in df.columns:
            df["TestStartNorm"] = try_parse_datetime_series(df["TestStart"])
        else:
            df["TestStartNorm"] = pd.NaT

        fallback_fill = extract_date_from_filename(os.path.basename(filepath))
        if pd.isna(fallback_fill):
            try:
                fallback_fill = datetime.fromtimestamp(os.path.getmtime(filepath))
            except Exception:
                fallback_fill = datetime.now()

        df["TestStartNorm"] = df["TestStartNorm"].fillna(fallback_fill)
        df["TesterCavity"] = infer_tester_cavity(df, filepath)
        df["Model"] = infer_model(df, filepath)
        df['SerialNumber'] = df.get('SerialNumber', pd.Series([""] * len(df))).astype(str).str.strip()
        df["_source_filepath"] = filepath
        return df
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
        return None

def load_and_clean_filelike(filelike, filename: str) -> Optional[pd.DataFrame]:
    """Accepts an uploaded file (BytesIO) — saves to temp file then reuses load_and_clean."""
    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext in ('.xlsx', '.xls'):
            # read directly with pandas then reuse normalization logic
            try:
                df = pd.read_excel(filelike, engine='openpyxl' if filename.lower().endswith('xlsx') else None)
            except Exception:
                filelike.seek(0)
                df = pd.read_excel(filelike)
            df.columns = df.columns.str.strip()
            # try to mimic remaining logic quickly:
            if "TestStart" in df.columns and "TestDate" in df.columns:
                dt_str = df["TestDate"].astype(str) + " " + df["TestStart"].astype(str)
                df["TestStartNorm"] = pd.to_datetime(dt_str, errors="coerce", dayfirst=True, infer_datetime_format=True)
            elif "TestStart" in df.columns:
                df["TestStartNorm"] = try_parse_datetime_series(df["TestStart"])
            else:
                df["TestStartNorm"] = pd.NaT
            df['SerialNumber'] = df.get('SerialNumber', pd.Series([""] * len(df))).astype(str).str.strip()
            df['Result'] = df.get('Result', pd.Series([""] * len(df))).astype(str).str.strip()
            df['FailedID'] = df.get('FailedID', pd.Series(["N/A"] * len(df))).astype(str).str.strip().fillna("N/A")
            df['FailedValue'] = pd.to_numeric(df.get('FailedValue', pd.Series([0] * len(df))), errors='coerce').fillna(0)
            df["ProStage"] = df.get("ProStage", pd.Series(["Test"] * len(df))).astype(str).str.strip().replace({"": "Test", "nan": "Test", "none": "Test", "na": "Test"})
            fallback_fill = extract_date_from_filename(filename)
            if pd.isna(fallback_fill):
                fallback_fill = datetime.now()
            df["TestStartNorm"] = df["TestStartNorm"].fillna(fallback_fill)
            df["TesterCavity"] = infer_tester_cavity(df, filename)
            df["Model"] = infer_model(df, filename)
            df["_source_filepath"] = filename
            return df

        # for text/csv: write temp file and call load_and_clean to preserve heuristics
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp.write(filelike.read())
            tmp.flush()
            tmp_path = tmp.name

        df = load_and_clean(tmp_path)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return df
    except Exception as e:
        print(f"Error load_and_clean_filelike: {e}")
        return None

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="FPY Monitoring (Web)", layout="wide")
st.title("📊 FPY Monitoring Dashboard (Web)")

# session state init
if 'df_cache' not in st.session_state:
    st.session_state.df_cache = {}   # key: filename or path -> DataFrame
if 'files' not in st.session_state:
    st.session_state.files = []      # list of file keys
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

# left column: controls
with st.sidebar:
    st.header("Controls")
    upload = st.file_uploader("Upload files (.txt/.csv/.xlsx)", accept_multiple_files=True, type=['txt','csv','xlsx','xls'])
    use_folder_load = st.checkbox("Load from local folder (server)", value=False)
    settings = load_config()
    default_folders = settings.get('default_folders', [])
    st.write("Default Folders (from config.ini):")
    for f in default_folders:
        st.write(f"- {f}")

    col1, col2 = st.columns(2)
    with col1:
        add_folder_btn = st.button("Add Folder to config")
    with col2:
        clear_files_btn = st.button("Clear Data")

    if add_folder_btn:
        new_folder = st.text_input("Enter absolute folder path to add", value="")
        # (we show input only when button clicked, but Streamlit reruns; handle via st.session_state)
        st.session_state._add_folder_prompt = True

    if clear_files_btn:
        st.session_state.df_cache.clear()
        st.session_state.files.clear()
        st.success("Cleared loaded files and cache.")

    # Add/Remove folder from config
    if st.session_state.get('_add_folder_prompt', False):
        new_folder = st.text_input("New folder path:", "")
        if st.button("Save folder to config"):
            if os.path.isdir(new_folder):
                default_folders.append(new_folder)
                save_config('DefaultFolders', default_folders)
                st.success("Folder added to config.ini")
                st.session_state._add_folder_prompt = False
            else:
                st.error("Folder path not valid on server.")

    # File type filters
    st.subheader("File Type Filter")
    ext_txt = st.checkbox(".txt", value=True)
    ext_csv = st.checkbox(".csv", value=False)
    ext_xlsx = st.checkbox(".xlsx/.xls", value=False)
    selected_exts = []
    if ext_txt: selected_exts.append('.txt')
    if ext_csv: selected_exts.append('.csv')
    if ext_xlsx: selected_exts += ['.xlsx', '.xls']

    # Date/time filters
    st.subheader("Date / Time Filter")
    today = datetime.now().date()
    start_date = st.date_input("Start date", value=today)
    end_date = st.date_input("End date", value=today)
    start_time = st.time_input("Start time", value=datetime.strptime("00:00:00","%H:%M:%S").time())
    end_time = st.time_input("End time", value=datetime.strptime("23:59:59","%H:%M:%S").time())

    # Analysis filters
    st.subheader("Other Filters")
    prostage = st.selectbox("ProStage", options=["All","Test","Retest","Rework"], index=0)
    tester_select = st.selectbox("Tester (after load)", options=["All Testers"])
    week_select = st.selectbox("Week", options=["All Weeks"])
    auto_refresh = st.checkbox("Auto Refresh", value=False)
    refresh_interval = st.selectbox("Interval (sec)", options=[5, 30, 60, 300], index=2)

    # Manual control
    st.button("Run Analysis (manual)", key="run_analysis_btn")

# Main area: load files if uploaded
def add_uploaded_files(uploaded_files):
    added = 0
    for f in uploaded_files:
        fname = f.name
        ext = os.path.splitext(fname)[1].lower()
        if selected_exts and ext not in selected_exts: 
            continue
        if is_blacklisted(fname):
            continue
        # load to df using helper
        file_content = f.read()
        filelike = io.BytesIO(file_content)
        df = load_and_clean_filelike(filelike, fname)
        if df is not None:
            key = f"UPLOAD::{fname}::{len(st.session_state.df_cache)}"
            st.session_state.df_cache[key] = df
            st.session_state.files.append(key)
            added += 1
    return added

def load_folder_to_cache(folder_path):
    new = 0
    for root, _, files in os.walk(folder_path):
        for fname in files:
            full = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            if selected_exts and ext not in selected_exts: continue
            if is_blacklisted(fname): continue
            if full in st.session_state.df_cache: continue
            df = load_and_clean(full)
            if df is not None:
                key = f"LOCAL::{full}"
                st.session_state.df_cache[key] = df
                st.session_state.files.append(key)
                new += 1
    return new

# process uploads
if upload:
    added = add_uploaded_files(upload)
    st.info(f"Added {added} uploaded files.")

# process local folder load if user requested
if use_folder_load:
    # show selector of available default folders or allow manual path
    folder_choice = st.selectbox("Choose folder to load from server", options=["Select..."] + default_folders + ["Enter manually..."])
    if folder_choice == "Enter manually...":
        manual_path = st.text_input("Enter absolute folder path:")
        if st.button("Load manual folder"):
            if os.path.isdir(manual_path):
                cnt = load_folder_to_cache(manual_path)
                st.success(f"Loaded {cnt} files from {manual_path}")
            else:
                st.error("Path invalid or not accessible by server.")
    elif folder_choice and folder_choice != "Select...":
        if st.button("Load selected folder"):
            cnt = load_folder_to_cache(folder_choice)
            st.success(f"Loaded {cnt} files from {folder_choice}")

# Auto-refresh logic (non-blocking)
if auto_refresh:
    now = time.time()
    if now - st.session_state.last_refresh > refresh_interval:
        st.session_state.last_refresh = now
        st.experimental_rerun()

# If no data loaded, show hint
if not st.session_state.df_cache:
    st.info("No data loaded. Upload files or load a folder from server (sidebar).")
    st.stop()

# Combine cached dfs (copy to avoid mutation)
all_df = pd.concat([df.copy() for df in st.session_state.df_cache.values()], ignore_index=True)
# normalize names
all_df.columns = all_df.columns.str.strip()
if "TestStatus" in all_df.columns and "Result" not in all_df.columns:
    all_df.rename(columns={"TestStatus": "Result"}, inplace=True)
if "Result" not in all_df.columns:
    st.warning("No 'Result' column detected — many metrics may be unavailable.")

# Try normalize TestStartNorm if missing
if "TestStartNorm" not in all_df.columns:
    if "TestStart" in all_df.columns:
        all_df["TestStartNorm"] = try_parse_datetime_series(all_df["TestStart"])
    else:
        extracted_dt = all_df.get("_source_filepath", "").apply(lambda x: extract_date_from_filename(os.path.basename(str(x))) if isinstance(x, str) else pd.NaT)
        fallback = datetime.now()
        all_df["TestStartNorm"] = extracted_dt.fillna(fallback)

# Filter by date/time
start_dt = datetime.combine(start_date, start_time)
end_dt = datetime.combine(end_date, end_time)
mask_dt = (all_df['TestStartNorm'] >= start_dt) & (all_df['TestStartNorm'] <= end_dt)
df_filtered = all_df[mask_dt].copy()

# Filter by prostage
if prostage != "All":
    df_filtered = df_filtered[df_filtered['ProStage'].astype(str).str.upper() == prostage.upper()]

# Update selectable tester/week lists
unique_testers = sorted(df_filtered['TesterCavity'].dropna().unique().tolist())
tester_options = ["All Testers"] + unique_testers
# rewrite tester_select if changed
tester_select = st.sidebar.selectbox("Tester (filter)", options=tester_options, index=0)

if tester_select and tester_select != "All Testers":
    df_filtered = df_filtered[df_filtered['TesterCavity'] == tester_select]

# weeks
df_filtered['Week'] = df_filtered['TestStartNorm'].dt.isocalendar().week.astype(str)
unique_weeks = sorted(df_filtered['Week'].dropna().unique().tolist())
week_options = ["All Weeks"] + unique_weeks
week_select = st.sidebar.selectbox("Week (filter)", options=week_options, index=0)
if week_select and week_select != "All Weeks":
    df_filtered = df_filtered[df_filtered['Week'] == week_select]

# show summary metrics
total_test = len(df_filtered)
pass_count = (df_filtered['Result'].astype(str).str.upper() == 'PASS').sum() if 'Result' in df_filtered.columns else 0
fail_count = total_test - pass_count
fpy_overall = (pass_count / total_test * 100) if total_test else 0

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Total units", total_test)
col_b.metric("Passed units", pass_count)
col_c.metric("Failed units", fail_count)
col_d.metric("FPY (%)", f"{fpy_overall:.2f}")

# show yield per Model & Tester
st.subheader("Yield per Model & Tester")
if df_filtered.empty:
    st.warning("No data after filters.")
else:
    yield_groups = df_filtered.groupby(['Model', 'TesterCavity'])
    rows = []
    for (model, tester), group in yield_groups:
        cnt = len(group)
        p_cnt = (group['Result'].astype(str).str.upper() == 'PASS').sum()
        f_cnt = cnt - p_cnt
        fpy = (p_cnt / cnt * 100) if cnt else 0
        rows.append((model, tester, cnt, p_cnt, f_cnt, fpy))
    yield_df = pd.DataFrame(rows, columns=["Model","Tester","Total","Pass","Fail","FPY"])
    st.dataframe(yield_df.sort_values(by="Total", ascending=False).reset_index(drop=True))

# Top failures chart
st.subheader("Top Failures (by FailedID)")
fail_rows = df_filtered[df_filtered['Result'].astype(str).str.upper().isin(['FAIL','FAILED','NG'])]
if not fail_rows.empty and 'FailedID' in fail_rows.columns:
    top_fails = fail_rows['FailedID'].value_counts().nlargest(10)
    fig, ax = plt.subplots(figsize=(8,3))
    top_fails.plot(kind='bar', ax=ax)
    ax.set_xlabel("Failed ID")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.write(top_fails.to_frame("Count"))
else:
    st.info("No failure rows or 'FailedID' column missing.")

# Failure detail table with filter
st.subheader("Failure Details")
failed_id_options = ["All Failures"] + (fail_rows['FailedID'].value_counts().index.tolist() if 'FailedID' in fail_rows.columns else [])
selected_failed_id = st.selectbox("Filter by Failed ID", failed_id_options)
display_fail = fail_rows.copy()
if selected_failed_id and selected_failed_id != "All Failures":
    display_fail = display_fail[display_fail['FailedID'] == selected_failed_id]
if not display_fail.empty:
    st.dataframe(display_fail[['SerialNumber','FailedID','FailedValue','Model','TesterCavity','TestStartNorm']].reset_index(drop=True))
else:
    st.info("No failure rows to display for this filter.")

# Raw data toggle
with st.expander("Show raw combined data"):
    st.dataframe(df_filtered.head(1000))

# Footer: actions
st.write("---")
colx, coly = st.columns([1,1])
with colx:
    if st.button("Refresh Now"):
        st.experimental_rerun()
with coly:
    if st.button("Clear Cache & Restart"):
        st.session_state.df_cache.clear()
        st.session_state.files.clear()
        st.experimental_rerun()

st.caption("Converted from Tkinter application. You can extend filters or visualizations as needed.")
