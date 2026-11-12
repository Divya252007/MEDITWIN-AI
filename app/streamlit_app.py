import os
import sys
import io
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
import wfdb


# Resolve project root before importing project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if not (PROJECT_ROOT / "src").is_dir():
    raise FileNotFoundError(
        f"Cannot find the src directory at {PROJECT_ROOT / 'src'}"
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tabular_adapter import TabularAdapter
from src.database import init_database, get_connection

from src.models.ecg_inference import extract_ecg_features

from src.integration.demo_encoders import (
    load_image_encoder,
    load_text_encoder,
    extract_image_embedding,
    extract_text_embedding,
)
# ---------------- DESIGN SYSTEM ----------------
BG = "#05090B"
PANEL = "#101A1D"
ACCENT = "#10D9A0"
TEXT = "#F4F8F7"
MUTED = "#92A6A2"
BORDER = "rgba(146,166,162,0.18)"

st.set_page_config(
    page_title="MediTwin-AI",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<style>
/* Green sidebar */
[data-testid="stSidebar"] {
    background: #124B3A !important;
}

/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #FFFFFF !important;
}

/* Sidebar radio buttons */
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: transparent;
    border-radius: 9px;
    padding: 8px;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #28765A;
}

/* Main background */
.stApp {
    background: #F7FAF8;
}

/* Main headings */
h1, h2, h3 {
    color: #124B3A !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #D5E7DC;
    border-radius: 12px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DEMO DATA ----------------
def demo_patients():
    return pd.DataFrame([
        {
            "Patient ID": "MT-1042",
            "Age": 58,
            "Indicators": "ECG uploaded",
            "AI status": "Review needed",
            "Priority": "High",
            "Updated": "2 min ago",
        },
        {
            "Patient ID": "MT-1038",
            "Age": 67,
            "Indicators": "Patient CSV available",
            "AI status": "Queued",
            "Priority": "Medium",
            "Updated": "11 min ago",
        },
        {
            "Patient ID": "MT-1031",
            "Age": 49,
            "Indicators": "ECG inspection complete",
            "AI status": "Ready for review",
            "Priority": "Low",
            "Updated": "24 min ago",
        },
        {
            "Patient ID": "MT-1024",
            "Age": 72,
            "Indicators": "Image and notes uploaded",
            "AI status": "Input incomplete",
            "Priority": "Medium",
            "Updated": "37 min ago",
        },
    ])


def metric(label, value, note, icon="◈"):
    st.markdown(f"""
    <div class="mt-card">
      <div class="mt-label">{icon} &nbsp; {label}</div>
      <div class="mt-value">{value}</div>
      <div class="mt-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)




def checkpoint_status():
    names = {
        "Image encoder": "image_encoder.pth",
        "ECG encoder": "ecg_encoder.pth",
        "Text encoder": "text_encoder.pth",
        "Tabular model": "tabular_model.pth",
        "Fusion model": "fusion_model.pth",
    }

    folder = PROJECT_ROOT / "models" / "checkpoints"

    return {
        name: (folder / filename).exists()
        for name, filename in names.items()
    }




def read_csv_upload(upload):
    """Read and normalize an uploaded patient CSV."""

    if upload is None:
        return None

    try:
        raw = upload.getvalue()

        if not raw or not raw.strip():
            st.error("The uploaded CSV is empty.")
            return None

        # Automatically detect comma, semicolon or tab separators.
        df = pd.read_csv(
            io.BytesIO(raw),
            sep=None,
            engine="python",
            encoding="utf-8-sig"
        )

        # Remove unnamed columns.
        df = df.loc[
            :,
            ~df.columns.astype(str).str.match(
                r"^\s*Unnamed", case=False
            )
        ]

        # Normalize column names.
        df.columns = (
            df.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.lower()
        )

        # Remove duplicate spaces and convert spaces to underscores.
        df.columns = (
            df.columns
            .str.replace(r"\s+", "_", regex=True)
            .str.replace("-", "_", regex=False)
        )

        if df.empty:
            st.error("The CSV contains no patient records.")
            return None

        if len(df.columns) == 1:
            st.error(
                "Only one column was detected. "
                "Please check the CSV delimiter and header row."
            )
            st.write("Detected column:", list(df.columns))
            return None

        return df

    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return None

# ---------------- MULTIMODAL MODEL HELPERS ----------------

@st.cache_resource
def get_image_model():
    return load_image_encoder()


@st.cache_resource
def get_text_model():
    return load_text_encoder()


@st.cache_resource
def get_tabular_model():
    checkpoint = (
        PROJECT_ROOT
        / "models"
        / "checkpoints"
        / "tabular_model.pth"
    )
    return TabularAdapter(str(checkpoint))


def load_diagnostic_ecg(dat_file, hea_file):
    """Read a matching WFDB .dat and .hea pair."""

    with tempfile.TemporaryDirectory() as temp_dir:
        record_name = "diagnostic_record"

        dat_path = Path(temp_dir) / f"{record_name}.dat"
        hea_path = Path(temp_dir) / f"{record_name}.hea"

        dat_path.write_bytes(dat_file.getvalue())

        header = hea_file.getvalue().decode(
            "utf-8", errors="replace"
        )

        lines = header.splitlines()

        if not lines:
            raise ValueError("ECG header is empty.")

        first_parts = lines[0].split()

        if len(first_parts) < 2:
            raise ValueError("Invalid ECG header.")

        first_parts[0] = record_name
        lines[0] = " ".join(first_parts)

        for i in range(1, len(lines)):
            parts = lines[i].split()

            if parts and parts[0].lower().endswith(
                (".dat", ".mat")
            ):
                parts[0] = f"{record_name}.dat"
                lines[i] = " ".join(parts)

        hea_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        record = wfdb.rdrecord(
            str(Path(temp_dir) / record_name)
        )

        if record.p_signal is None:
            raise ValueError("Could not read ECG signal.")

        return (
            record.p_signal.copy(),
            float(record.fs),
            list(record.sig_name or []),
        )


def preprocess_diagnostic_ecg(signal, sampling_rate):
    """
    Prepare ECG for the experimental encoder.

    Expected input:
    12 leads, 100 Hz, exactly 1000 samples.
    """

    if signal.ndim != 2:
        raise ValueError("ECG must be a 2D signal.")

    if signal.shape[1] != 12:
        raise ValueError(
            f"Expected 12 leads; received {signal.shape[1]}."
        )

    if sampling_rate != 100:
        raise ValueError(
            f"Expected 100 Hz; received {sampling_rate} Hz. "
            "Resampling is not implemented."
        )

    if signal.shape[0] != 1000:
        raise ValueError(
            f"Expected exactly 1000 samples; received "
            f"{signal.shape[0]}. No padding or truncation "
            "will be applied."
        )

    signal = np.asarray(signal, dtype=np.float32)
    signal = np.nan_to_num(
        signal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Normalize each lead independently.
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)

    std = np.where(std < 1e-8, 1.0, std)

    normalized = (signal - mean) / std

    # (1000, 12) -> (1, 12, 1000)
    tensor = torch.from_numpy(
        normalized.T.copy()
    ).unsqueeze(0)

    return tensor
# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown(
        '<div class="mt-brand">MediTwin<span>-AI</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("CLINICAL INTELLIGENCE · RESEARCH DEMO")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Patients",
            "AI Diagnostics",
            "ECG Analysis",
            "Remote Monitoring",
            "Hospital Analytics",
            "Research & Models",
            "Settings",
        ],
    )

    st.divider()
    st.markdown("🟢 **Interface online**")
    st.caption("Research prototype · Demo mode")
    st.caption("No hospital or wearable connection configured.")
    st.info(
    "RESEARCH PROTOTYPE — MediTwin-AI demonstrates multimodal "
    "health-data processing and experimental model components. "
    "Extracted features and experimental outputs are not validated "
    "medical diagnoses or clinical risk predictions. Do not use "
    "this application to make patient-care decisions."
    )

# ---------------- TOP BAR ----------------
left, right = st.columns([4, 1])
with left:
    st.markdown(
        '<div class="mt-muted">MULTIMODAL CARDIOVASCULAR RESEARCH PLATFORM</div>',
        unsafe_allow_html=True,
    )
with right:
    st.caption(datetime.now().strftime("%d %b %Y"))


# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.markdown(f"""
    <div class="mt-hero">
      <div style="color:{ACCENT};font-size:.8rem;font-weight:700;
      letter-spacing:2px">AI-ASSISTED HEALTHCARE</div>
      <h1>Intelligent healthcare.<br>
      <span style="color:{ACCENT}">Earlier insights.</span><br>
      Better decisions.</h1>
      <p class="mt-muted">
      Explore a multimodal research workflow combining medical imaging,
      ECG signals, clinical text, and structured patient data.
      Designed to support research and clinical review.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Platform overview")
    st.caption("Synthetic demonstration metrics — not real hospital statistics.")

    cols = st.columns(5)
    values = [
        ("Registered records", "1,284", "Synthetic demo count"),
        ("Monitoring samples", "86", "Illustrative records"),
        ("Pending analyses", "12", "Demo queue only"),
        ("Priority flags", "04", "Fictional examples"),
        ("Processing time", "—", "Not benchmarked"),
    ]
    for col, item in zip(cols, values):
        with col:
            metric(*item)

    a, b = st.columns([1.4, 1])
    with a:
        st.markdown("### Patient activity")
        dates = pd.date_range(end=pd.Timestamp.today(), periods=14)
        activity = pd.DataFrame({
            "Date": dates,
            "Admissions (demo)": [
                18, 21, 17, 25, 23, 27, 20,
                24, 29, 26, 31, 28, 33, 30
            ],
            "ECG checks (demo)": [
                25, 28, 24, 31, 34, 30, 36,
                33, 39, 42, 40, 45, 47, 44
            ],
        }).set_index("Date")
        st.line_chart(activity)

    with b:
        st.markdown("### Research risk categories")
        st.caption("Illustrative distribution only")
        st.bar_chart(pd.DataFrame({
            "Demo records": [48, 33, 19]
        }, index=["Low", "Medium", "High"]))

    st.markdown("### Patient priority queue")
    st.dataframe(
        demo_patients(),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Patient identifiers and records above are fictional.")


# ---------------- PATIENTS ----------------
elif page == "Patients":
    st.title("Patient workspace")
    st.caption("Patient records stored in the local SQLite database.")

    # Ensure the database table exists
    init_database()

    # Load saved patient records
    with get_connection() as conn:
        records = conn.execute(
            """
            SELECT patient_id, age, sex, created_at
            FROM patients
            ORDER BY id DESC
            """
        ).fetchall()

    # Convert database rows into the existing table format
    patient_data = []
    for record in records:
        patient_data.append({
            "Patient ID": record["patient_id"],
            "Age": record["age"] if record["age"] is not None else "Not entered",
            "Sex": record["sex"] or "Not entered",
            "Priority": "Low",
            "Indicators": "No indicators recorded",
            "AI status": "Not assessed",
            "Created At": record["created_at"],
        })

    import pandas as pd
    df = pd.DataFrame(
        patient_data,
        columns=[
            "Patient ID", "Age", "Sex", "Priority",
            "Indicators", "AI status", "Created At"
        ],
    )

    # Add a patient
    with st.expander("➕ Register a patient"):
        with st.form("register_patient_form"):
            patient_id = st.text_input("Patient ID").strip()
            age = st.number_input("Age", min_value=0, max_value=120, value=25)
            sex = st.selectbox("Sex", ["Not specified", "Female", "Male", "Other"])
            submitted = st.form_submit_button("Save patient")

            if submitted:
                if not patient_id:
                    st.error("Please enter a patient ID.")
                else:
                    try:
                        with get_connection() as conn:
                            conn.execute(
                                """
                                INSERT INTO patients (patient_id, age, sex)
                                VALUES (?, ?, ?)
                                """,
                                (
                                    patient_id,
                                    int(age),
                                    None if sex == "Not specified" else sex,
                                ),
                            )
                        st.success("Patient saved successfully.")
                        st.rerun()
                    except Exception as exc:
                        if "UNIQUE constraint failed" in str(exc):
                            st.error("That patient ID already exists.")
                        else:
                            st.error(f"Could not save patient: {exc}")

    # Search and filter
    search = st.text_input("Search by patient ID or indicator")

    priority = st.multiselect(
        "Filter priority",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )

    filtered = df[df["Priority"].isin(priority)]

    if search:
        mask = filtered.astype(str).apply(
            lambda col: col.str.contains(
                search, case=False, na=False
            )
        ).any(axis=1)
        filtered = filtered[mask]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    # Patient details
    if not filtered.empty:
        selected = st.selectbox(
            "Open patient details",
            filtered["Patient ID"].tolist(),
        )
        row = filtered[
            filtered["Patient ID"] == selected
        ].iloc[0]

        st.markdown("### Patient details")

        c1, c2, c3 = st.columns(3)
        with c1:
            metric("Patient ID", row["Patient ID"], "Saved record")
        with c2:
            metric("Age", str(row["Age"]), "Patient information")
        with c3:
            metric("Workflow status", row["AI status"], "Not a diagnosis")

        st.write("**Sex:**", row["Sex"])
        st.write("**Created:**", row["Created At"])
        st.write("**Indicators:**", row["Indicators"])

        tabs = st.tabs([
            "Overview", "History", "Vitals", "ECG",
            "Imaging", "Clinical notes", "AI assessments", "Reports"
        ])

        with tabs[0]:
            st.write(row["Indicators"])
        with tabs[1]:
            st.info("Medical history has not been recorded.")
        with tabs[2]:
            st.info("No live patient vital-sign stream is connected.")
        with tabs[3]:
            st.info("Use ECG Analysis to inspect a WFDB record.")
        with tabs[4]:
            st.info("No imaging result is stored for this patient.")
        with tabs[5]:
            st.info("No clinical note is stored for this patient.")
        with tabs[6]:
            st.info("No clinically validated prediction is available.")
        with tabs[7]:
            st.info("Report export can be added after validated outputs exist.")

    else:
        st.info("No saved patient records match your search and filters.")

# ---------------- AI DIAGNOSTICS ----------------
elif page == "AI Diagnostics":

    st.title("Multimodal AI Diagnostics")
    st.caption(
        "Experimental feature extraction from four cardiovascular data modalities."
    )

    st.warning(
        "Research prototype only. Extracted embeddings are not "
        "validated medical diagnoses or clinical risk predictions."
    )

    REQUIRED_FEATURES = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalch", "exang",
        "oldpeak", "slope", "ca", "thal"
    ]

    # ---------------- FILE UPLOADS ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Medical image")

        image_upload = st.file_uploader(
            "Upload chest X-ray",
            type=["png", "jpg", "jpeg"],
            key="diagnostic_image"
        )

        st.subheader("2. ECG signal")

        ecg_dat = st.file_uploader(
            "Upload ECG .dat file",
            type=["dat"],
            key="diagnostic_ecg_dat"
        )

        ecg_hea = st.file_uploader(
            "Upload matching ECG .hea file",
            type=["hea"],
            key="diagnostic_ecg_hea"
        )

    with col2:
        st.subheader("3. Clinical text")

        text_upload = st.file_uploader(
            "Upload clinical notes",
            type=["txt"],
            key="diagnostic_text"
        )

        st.subheader("4. Patient tabular data")

        csv_upload = st.file_uploader(
            "Upload patient CSV",
            type=["csv"],
            key="diagnostic_csv"
        )

    # ---------------- CSV PROCESSING ----------------
    patient_df = read_csv_upload(csv_upload)

    selected_patient = None

    if patient_df is not None:

        st.markdown("### Patient data preview")

        st.dataframe(
            patient_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### CSV column validation")

        missing_columns = [
            feature
            for feature in REQUIRED_FEATURES
            if feature not in patient_df.columns
        ]

        if missing_columns:
            st.error(
                "The uploaded CSV is missing these required columns: "
                + ", ".join(missing_columns)
            )

            st.write("Detected CSV columns:")
            st.write(list(patient_df.columns))

            st.info(
                "Required columns: "
                + ", ".join(REQUIRED_FEATURES)
            )

        else:
            st.success("All 13 required patient columns were detected.")

            selected_index = st.selectbox(
                "Select patient row",
                options=list(range(len(patient_df))),
                format_func=lambda i: f"Patient row {i + 1}",
                key="diagnostic_patient_row"
            )

            selected_patient = patient_df.iloc[
                [selected_index]
            ].copy()

            # Pass only the expected features in the expected order.
            selected_patient = selected_patient[
                REQUIRED_FEATURES
            ].copy()

            # Convert feature values to numeric.
            for feature in REQUIRED_FEATURES:
                selected_patient[feature] = pd.to_numeric(
                    selected_patient[feature],
                    errors="coerce"
                )

            if selected_patient.isnull().any().any():
                invalid = selected_patient.columns[
                    selected_patient.isnull().any()
                ].tolist()

                st.error(
                    "Invalid or missing numeric values in: "
                    + ", ".join(invalid)
                )

                selected_patient = None

            else:
                st.write("Selected patient features:")
                st.dataframe(
                    selected_patient,
                    use_container_width=True,
                    hide_index=True
                )

    st.divider()

    # ---------------- EMBEDDING EXTRACTION ----------------
    if st.button(
        "Extract multimodal embeddings",
        use_container_width=True,
        key="extract_multimodal_embeddings"
    ):

        missing = []

        if image_upload is None:
            missing.append("Medical image")

        if ecg_dat is None or ecg_hea is None:
            missing.append("Matching ECG .dat and .hea files")

        if text_upload is None:
            missing.append("Clinical text")

        if selected_patient is None:
            missing.append("Valid patient CSV data")

        if missing:
            st.error(
                "Please provide all required inputs: "
                + ", ".join(missing)
            )

        else:
            try:
                status = checkpoint_status()

                required_models = [
                    "Image encoder",
                    "ECG encoder",
                    "Text encoder",
                    "Tabular model"
                ]

                unavailable = [
                    name for name in required_models
                    if not status.get(name, False)
                ]

                if unavailable:
                    st.error(
                        "Missing model checkpoints: "
                        + ", ".join(unavailable)
                    )

                else:

                    with st.spinner(
                        "Extracting experimental embeddings..."
                    ):

                        # 1. IMAGE EMBEDDING
                        image_model = get_image_model()

                        image_embedding = extract_image_embedding(
                            image_model,
                            image_upload
                        )

                        # 2. ECG EMBEDDING
                        signal, fs, lead_names = load_diagnostic_ecg(
                            ecg_dat,
                            ecg_hea
                        )

                        ecg_tensor = preprocess_diagnostic_ecg(
                            signal,
                            fs
                        )

                        ecg_result = extract_ecg_features(
                            ecg_tensor
                        )

                        ecg_embedding = ecg_result["features"]

                        # 3. CLINICAL TEXT EMBEDDING
                        text_model, tokenizer = get_text_model()

                        clinical_text = (
                            text_upload.getvalue().decode(
                                "utf-8",
                                errors="replace"
                            )
                        )

                        if not clinical_text.strip():
                            raise ValueError(
                                "The clinical text file is empty."
                            )

                        text_embedding = extract_text_embedding(
                            text_model,
                            tokenizer,
                            clinical_text
                        )

                        # 4. TABULAR EMBEDDING
                        tabular_model = get_tabular_model()

                        # Final column check before model inference.
                        if list(selected_patient.columns) != REQUIRED_FEATURES:
                            raise ValueError(
                                "Patient feature columns do not match "
                                "the expected model input."
                            )

                        tabular_embedding = tabular_model(
                            selected_patient
                        )

                    # Store embeddings in session state.
                    embeddings = {
                        "image": image_embedding,
                        "ecg": ecg_embedding,
                        "text": text_embedding,
                        "tabular": tabular_embedding
                    }

                    for name, embedding in embeddings.items():
                        if not torch.is_tensor(embedding):
                            raise TypeError(
                                f"{name} encoder did not return a tensor."
                            )

                        if not torch.isfinite(embedding).all():
                            raise ValueError(
                                f"{name} embedding contains invalid values."
                            )

                    st.session_state[
                        "multimodal_embeddings"
                    ] = {
                        name: embedding.detach().cpu()
                        for name, embedding in embeddings.items()
                    }

                    st.success(
                        "All four experimental embeddings were extracted."
                    )

                    st.markdown("### Embedding dimensions")

                    result_cols = st.columns(4)

                    labels = {
                        "image": "Medical image",
                        "ecg": "ECG signal",
                        "text": "Clinical text",
                        "tabular": "Tabular data"
                    }

                    for col, (name, embedding) in zip(
                        result_cols,
                        embeddings.items()
                    ):
                        with col:
                            st.metric(
                                labels[name],
                                str(tuple(embedding.shape))
                            )

                    st.info(
                        "The four embeddings have been extracted "
                        "independently. This prototype does not provide "
                        "a validated multimodal fusion prediction."
                    )

            except Exception as exc:
                st.error(
                    f"Multimodal extraction failed: {exc}"
                )

                with st.expander("Technical error details"):
                    st.exception(exc)

#---------------- ECG ANALYSIS ----------------#
elif page == "ECG Analysis":
    import matplotlib.pyplot as plt
    import torch
    import wfdb

    st.title("ECG Analysis")
    st.caption(
        "ECG waveform visualization and experimental feature extraction."
    )

    st.warning(
        "Research prototype only. Model output is not a confirmed "
        "diagnosis and must not be used for clinical decisions."
    )

    st.subheader("Upload ECG recording")

    col1, col2 = st.columns(2)

    with col1:
        dat_upload = st.file_uploader(
            "ECG signal (.dat)",
            type=["dat"],
            key="ecg_dat_upload",
        )

    with col2:
        hea_upload = st.file_uploader(
            "Matching ECG header (.hea)",
            type=["hea"],
            key="ecg_hea_upload",
        )

    def load_uploaded_ecg(dat_file, hea_file, temp_dir):
        """Save matching ECG files and read the WFDB record."""
        record_name = "uploaded_record"

        dat_path = os.path.join(
            temp_dir, f"{record_name}.dat"
        )
        hea_path = os.path.join(
            temp_dir, f"{record_name}.hea"
        )

        with open(dat_path, "wb") as f:
            f.write(dat_file.getvalue())

        header_text = hea_file.getvalue().decode(
            "utf-8", errors="replace"
        )
        header_lines = header_text.splitlines()

        if not header_lines:
            raise ValueError("The ECG header is empty.")

        first_parts = header_lines[0].split()
        if len(first_parts) < 2:
            raise ValueError("The ECG header format is invalid.")

        first_parts[0] = record_name
        header_lines[0] = " ".join(first_parts)

        for i in range(1, len(header_lines)):
            parts = header_lines[i].split()

            if parts and (
                parts[0].lower().endswith(".dat")
                or parts[0].lower().endswith(".mat")
            ):
                parts[0] = f"{record_name}.dat"
                header_lines[i] = " ".join(parts)

        with open(hea_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header_lines) + "\n")

        return wfdb.rdrecord(
            os.path.join(temp_dir, record_name)
        )

    if dat_upload is None or hea_upload is None:
        st.info(
            "Upload both the .dat signal and matching .hea header "
            "to inspect an ECG recording."
        )
    else:
        ecg_action = st.radio(
            "Choose an operation",
            ["Inspect waveform", "Extract experimental features"],
            horizontal=True,
            key="ecg_action",
        )

        if st.button(
            "Run ECG operation",
            use_container_width=True,
            key="run_ecg_operation",
        ):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    record = load_uploaded_ecg(
                        dat_upload,
                        hea_upload,
                        temp_dir,
                    )

                    signal = record.p_signal

                    if signal is None or signal.size == 0:
                        st.error("The ECG signal could not be read.")

                    elif ecg_action == "Inspect waveform":
                        st.success("ECG signal loaded.")
                        st.write("Sampling rate:", record.fs, "Hz")
                        st.write("Samples:", signal.shape[0])
                        st.write("Leads:", signal.shape[1])

                        lead_names = (
                            record.sig_name
                            if record.sig_name
                            else [
                                f"Lead {i + 1}"
                                for i in range(signal.shape[1])
                            ]
                        )

                        selected_lead = st.selectbox(
                            "Select ECG lead",
                            options=list(range(signal.shape[1])),
                            format_func=lambda i: lead_names[i],
                            key="ecg_lead_select",
                        )

                        time_axis = (
                            np.arange(signal.shape[0]) / record.fs
                        )

                        fig, ax = plt.subplots(figsize=(12, 4))
                        ax.plot(
                            time_axis,
                            signal[:, selected_lead],
                            linewidth=0.8,
                        )
                        ax.set_title(
                            f"ECG Waveform - {lead_names[selected_lead]}"
                        )
                        ax.set_xlabel("Time (seconds)")
                        ax.set_ylabel("Amplitude")
                        ax.grid(True, alpha=0.3)

                        st.pyplot(fig)
                        plt.close(fig)

                        st.info(
                            "This displays the signal waveform only; "
                            "it does not diagnose a cardiac condition."
                        )

                    else:
                        if signal.shape[1] != 12:
                            st.error(
                                f"Expected 12 ECG leads, but found "
                                f"{signal.shape[1]}."
                            )
                        elif record.fs != 100:
                            st.error(
                                f"This experimental model expects 100 Hz, "
                                f"but the recording is {record.fs} Hz. "
                                "Resampling is not implemented here."
                            )
                        else:
                            ecg_tensor = torch.tensor(
                                signal.T.copy(),
                                dtype=torch.float32,
                            ).unsqueeze(0)

                            with st.spinner(
                                "Extracting experimental ECG features..."
                            ):
                                result = extract_ecg_features(ecg_tensor)

                            features = result["features"]
                            logits = result["logits"]
                            classes = result["classes"]

                            if (
                                not torch.isfinite(features).all()
                                or not torch.isfinite(logits).all()
                            ):
                                st.error(
                                    "The model returned invalid numerical "
                                    "values."
                                )
                            else:
                                st.success(
                                    "Experimental feature extraction completed."
                                )
                                st.write(
                                    "Feature vector shape:",
                                    tuple(features.shape),
                                )
                                st.write(
                                    "Model output shape:",
                                    tuple(logits.shape),
                                )

                                score_df = pd.DataFrame({
                                    "Category": classes,
                                    "Raw logit": (
                                        logits[0]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    ),
                                }).sort_values(
                                    "Raw logit",
                                    ascending=False,
                                ).reset_index(drop=True)

                                st.dataframe(
                                    score_df,
                                    use_container_width=True,
                                    hide_index=True,
                                )

                                st.caption(
                                    "Raw logits are uncalibrated scores, "
                                    "not probabilities. The top-ranked "
                                    "category is not a confirmed diagnosis."
                                )

                                st.warning(
                                    "Experimental research output only. "
                                    "Clinical accuracy and reliability have "
                                    "not been established."
                                )

            except Exception as exc:
                st.error(f"ECG operation failed: {exc}")
# ---------------- REMOTE MONITORING ----------------
elif page == "Remote Monitoring":
    st.title("Remote patient monitoring")
    st.caption(
        "All values below are synthetic demonstration data. "
        "No wearable device is connected."
    )

    cols = st.columns(3)
    with cols[0]:
        metric("Heart rate", "72 BPM", "Synthetic · 10:42 AM", "♥")
    with cols[1]:
        metric("Blood pressure", "118/76 mmHg", "Synthetic · 10:42 AM", "⌁")
    with cols[2]:
        metric("Oxygen saturation", "98%", "Synthetic · 10:42 AM", "◉")

    cols = st.columns(3)
    with cols[0]:
        metric("Respiratory rate", "16 /min", "Synthetic sample", "↗")
    with cols[1]:
        metric("Temperature", "36.8 °C", "Synthetic sample", "♨")
    with cols[2]:
        metric("Device status", "Not connected", "Demo placeholder", "⌁")

    st.markdown("### Vital-sign trend")
    times = pd.date_range(end=pd.Timestamp.now(), periods=24, freq="h")
    vitals = pd.DataFrame({
        "Time": times,
        "Heart rate (BPM)": [
            68,67,66,65,64,65,68,72,76,79,78,75,
            73,72,74,76,73,70,69,68,70,71,72,72
        ],
    }).set_index("Time")
    st.line_chart(vitals)

    st.selectbox(
        "Monitoring range",
        ["1 hour", "24 hours", "7 days", "30 days"],
    )
    if st.button("Refresh demo view"):
        st.toast("Demo view refreshed; no external device was contacted.")

    st.warning(
        "This prototype does not provide emergency detection or live "
        "clinical monitoring."
    )


# ---------------- HOSPITAL ANALYTICS ----------------
elif page == "Hospital Analytics":
    st.title("Hospital analytics")
    st.caption("Charts use synthetic demonstration data.")

    range_choice = st.selectbox(
        "Date range", ["Last 7 days", "Last 14 days", "Last 30 days"]
    )
    days = 7 if range_choice == "Last 7 days" else 14
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)

    activity = pd.DataFrame({
        "Date": dates,
        "Admissions (demo)": [18,21,17,25,23,27,20,24,29,26,31,28,33,30][-days:],
        "ECG checks (demo)": [25,28,24,31,34,30,36,33,39,42,40,45,47,44][-days:],
    }).set_index("Date")

    a, b = st.columns(2)
    with a:
        st.markdown("### Activity over time")
        st.line_chart(activity)
    with b:
        st.markdown("### Analysis workflow status")
        st.bar_chart(pd.DataFrame({
            "Count": [12, 7, 18, 5]
        }, index=["Queued", "In progress", "Ready for review", "Incomplete"]))

    a, b = st.columns(2)
    with a:
        st.markdown("### Research category distribution")
        st.bar_chart(pd.DataFrame({
            "Demo records": [48, 33, 19]
        }, index=["Low", "Medium", "High"]))
    with b:
        st.markdown("### System connectivity")
        st.dataframe(pd.DataFrame({
            "Component": ["API", "Model latency", "Wearable sync", "Database"],
            "Status": ["Not connected", "Not benchmarked", "Not connected", "Demo only"],
        }), use_container_width=True, hide_index=True)


# ---------------- RESEARCH AND MODELS ----------------
elif page == "Research & Models":
    st.title("Research and model status")
    st.info(
        "MediTwin-AI is a research prototype. Its current integration "
        "checks do not establish clinical readiness."
    )

    st.markdown("### Checkpoint files")
    st.dataframe(pd.DataFrame([
        {
            "Component": name,
            "Status": "Found" if exists else "Missing",
        }
        for name, exists in checkpoint_status().items()
    ]), use_container_width=True, hide_index=True)

    st.markdown("### Current limitations")
    st.markdown("""
    - File presence does not prove model quality or validation.
    - The current fusion checkpoint was trained using synthetic data.
    - Modality datasets must be matched appropriately before multimodal
      clinical training.
    - Random or untrained projection layers must not be presented as
      validated clinical features.
    - Do not display accuracy, confidence, or risk predictions without
      documented evaluation on an appropriate test set.
    - Clinical use requires independent validation, privacy controls,
      governance, and clinician oversight.
    """)


# ---------------- SETTINGS ----------------
elif page == "Settings":
    st.title("Settings")
    st.caption(
        "These are local session preferences. Authentication and role-based "
        "access control are not implemented in this prototype."
    )

    with st.form("preferences"):
        display_name = st.text_input(
            "Workspace name", value="Research Workspace"
        )
        chart_range = st.selectbox(
            "Default chart range", ["7 days", "14 days", "30 days"]
        )
        submitted = st.form_submit_button("Save preferences")

        if submitted:
            st.session_state["display_name"] = display_name
            st.session_state["chart_range"] = chart_range
            st.success("Preferences saved for this session.")

    st.markdown("### Privacy")
    st.write(
        "Use synthetic data for demonstrations. Do not upload identifiable "
        "patient information unless your environment and permissions "
        "explicitly authorize it."
    )


# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "MediTwin-AI · Research prototype · Synthetic demo data · "
    "Not for clinical decision-making"
)