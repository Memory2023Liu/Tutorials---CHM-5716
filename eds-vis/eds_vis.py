"""
Tutorial script: plot real and/or simulated EDS spectra using:
- quantification.csv
- characteristic_xray_line_by_element_kev.json
- spectrum.emsa (required for real mode)

Modes:
- "real"
- "simulate"
- "both"
"""

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter


def configure_plot_style():
    """Set global matplotlib style once for the whole script."""
    plt.rcParams["font.sans-serif"] = ["Helvetica"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["ytick.major.width"] = 1.5
    plt.rcParams["xtick.minor.width"] = 1.0
    plt.rcParams["ytick.minor.width"] = 1.0


def read_emsa(file_path):
    """Read EMSA spectrum and return energy axis (eV) and counts."""
    energy = []
    counts = []
    x_per_chan, offset, npoints = None, None, None
    parsing_spectrum = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            clean_line = line.strip().upper()

            if clean_line.startswith("#XPERCHAN"):
                match = re.search(r"[-+]?\d*\.?\d+(?:E[-+]?\d+)?", clean_line)
                if match:
                    x_per_chan = float(match.group(0))

            elif clean_line.startswith("#OFFSET"):
                match = re.search(r"[-+]?\d*\.?\d+(?:E[-+]?\d+)?", clean_line)
                if match:
                    offset = float(match.group(0))

            elif clean_line.startswith("#NPOINTS"):
                match = re.search(r"\d+", clean_line)
                if match:
                    npoints = int(match.group(0))

            elif clean_line.startswith("#SPECTRUM"):
                parsing_spectrum = True

            elif clean_line.startswith("#ENDOFDATA"):
                break

            elif parsing_spectrum:
                try:
                    counts.extend(float(v) for v in re.split(r"[, ]+", line.strip()) if v)
                except (ValueError, TypeError):
                    continue

    if x_per_chan is None or offset is None:
        return [], []

    energy = [offset + i * x_per_chan for i in range(len(counts))]

    if npoints is not None and len(counts) != npoints:
        print(f"Warning: parsed points = {len(counts)}, #NPOINTS = {npoints}")

    return energy, counts


def create_scientific_formatter():
    """Create scientific-notation tick formatter."""
    def format_func(x, pos):
        if x == 0:
            return "0"
        exp = int(np.floor(np.log10(abs(x))))
        mantissa = x / (10 ** exp)
        return f"{mantissa:.1f}×10$^{{{exp}}}$"
    return FuncFormatter(format_func)


def apply_publication_style(ax, title_text="", xlabel_text="Energy (eV)", ylabel_text="Counts", num_ticks=6):
    """Apply publication-style axis formatting."""
    ax.set_xlabel(xlabel_text, weight="bold", fontsize=24)
    ax.set_ylabel(ylabel_text, weight="bold", fontsize=24)
    ax.set_title(title_text, weight="bold", fontsize=24)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.tick_params(axis="both", which="major", color="k", length=7, width=1.5, labelcolor="k")
    ax.tick_params(axis="both", which="minor", color="k", length=4, width=1)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(24)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(num_ticks))
    ax.grid(True, linestyle="--", alpha=0.7)


def format_line_name_for_plot(line_name):
    """Convert Greek letters to matplotlib mathtext."""
    return line_name.replace("α", "$\\alpha$").replace("β", "$\\beta$").replace("γ", "$\\gamma$")


def gaussian(x, mu, amp, sigma):
    """Gaussian peak."""
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def simulate_eds_spectrum(ref_energy_eV, ref_intensity, x_min=0, x_max=8000, npts=4000, fwhm_ev=130):
    """
    Simulate EDS spectrum from reference peak positions and intensities.

    f = a * exp(-(x - b)^2 / (2*sigma^2))
    where:
    - a: peak height
    - b: peak center
    - sigma: peak width
    """
    x_energy = np.linspace(x_min, x_max, npts)
    gaussian_simulated_intensity = np.zeros_like(x_energy)
    sigma = fwhm_ev / 2.355

    for i in range(len(ref_energy_eV)):
        a = ref_intensity[i]
        b = ref_energy_eV[i]
        gaussian_simulated_intensity += a * np.exp(-((x_energy - b) ** 2) / (2 * sigma ** 2))

    return x_energy, gaussian_simulated_intensity


def build_simulation_inputs(df_quant, line_db, preferred_lines, line_weights):
    """Build peak positions and intensities for simulation from quantification.csv."""
    ref_energy_eV = []
    ref_intensity = []

    for _, row in df_quant.iterrows():
        element = str(row["Element symbol"]).strip()
        at_pct = float(row["Atomic concentration percentage"])

        if element not in line_db:
            continue

        for line_name, line_energy_keV in line_db[element].items():
            if line_name not in preferred_lines:
                continue
            ref_energy_eV.append(float(line_energy_keV) * 1000)
            ref_intensity.append(at_pct * line_weights.get(line_name, 0.2))

    return ref_energy_eV, ref_intensity


def annotate_peaks(ax, energy, counts, df_quant, line_db, preferred_lines, show_line_type=True, peak_label_color="#c82423"):
    """Annotate peak positions using quantified elements and line database."""
    present_elements = df_quant["Element symbol"].tolist()
    energy_arr = np.array(energy)
    counts_arr = np.array(counts)
    y_max = counts_arr.max()

    for element in present_elements:
        if element not in line_db:
            continue

        for line_name, line_energy_keV in line_db[element].items():
            if line_name not in preferred_lines:
                continue

            peak_energy_eV = float(line_energy_keV) * 1000
            if not (energy_arr[0] <= peak_energy_eV <= energy_arr[-1]):
                continue

            search_window = (energy_arr > (peak_energy_eV - 75)) & (energy_arr < (peak_energy_eV + 75))
            if not np.any(search_window):
                continue

            counts_in_window = counts_arr[search_window]
            energy_in_window = energy_arr[search_window]
            peak_idx = np.argmax(counts_in_window)
            x_pos = energy_in_window[peak_idx]
            y_pos = counts_in_window[peak_idx]

            label = f"{element} {format_line_name_for_plot(line_name)}" if show_line_type else element
            ax.text(
                x_pos,
                y_pos + y_max * 0.03,
                label,
                ha="center",
                va="bottom",
                fontsize=18,
                color=peak_label_color,
                weight="bold",
            )
            ax.plot(
                [x_pos, x_pos],
                [y_pos, y_pos + y_max * 0.02],
                color=peak_label_color,
                linestyle="-",
                linewidth=1,
            )


def plot_spectrum(energy, counts, df_quant, line_db, preferred_lines, title,
                  spectrum_color="#3480b8", peak_label_color="#c82423",
                  x_range=None, y_range=None, show_line_type=True, save_path=None):
    """Plot one spectrum using shared real/simulated format."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(energy, counts, color=spectrum_color, linewidth=1.5)

    ax.set_xlim(x_range if x_range is not None else (0, max(energy)))
    ax.set_ylim(y_range if y_range is not None else (0, max(counts) * 1.1))

    apply_publication_style(ax, title_text=title, xlabel_text="Energy (eV)", ylabel_text="Counts")
    ax.yaxis.set_major_formatter(create_scientific_formatter())
    ax.xaxis.set_major_formatter(create_scientific_formatter())

    annotate_peaks(
        ax=ax,
        energy=energy,
        counts=counts,
        df_quant=df_quant,
        line_db=line_db,
        preferred_lines=preferred_lines,
        show_line_type=show_line_type,
        peak_label_color=peak_label_color,
    )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        print(f"✅ Spectrum plot saved to: {save_path}")
    plt.show()


def main():
    configure_plot_style()

    mode = "both"  # "real", "simulate", or "both"

    quant_file = "quantification.csv"
    line_json = "characteristic_xray_line_by_element_kev.json"
    spectrum_file = "spectrum.emsa"

    preferred_lines = {"Kα", "Kβ"}
    # preferred_lines = {"Kα", "Kβ", "Lα", "Mα"}

    line_weights = {
        "Kα": 1.00,
        "Kβ": 0.18,
        "Lα": 0.55,
        "Mα": 0.35,
    }

    show_line_type = True
    print("✅ Note that the Duane-Hunt limit should be 20 keV based on the experimental setup, we plot up to 10 keV (10000 eV) to show the relevant peaks clearly.")
    x_range = (0, 10000)
    # x_range = None
    y_range = None

    spectrum_color = "#3480b8"
    peak_label_color = "#c82423"

    simulate_x_min = 0
    simulate_x_max = 10000
    simulate_npts = 4000 # for smooth curve, use 4000 points; for faster but more jagged, use 1000 points
    simulate_fwhm_ev = 130
    simulate_scale_to_real_max = True
    simulate_max_counts = 10000

    df_quant = pd.read_csv(quant_file)
    df_quant.columns = [col.strip() for col in df_quant.columns]

    required_cols = [
        "Element symbol",
        "Atomic concentration percentage",
        "Weight concentration percentage",
    ]
    missing = [c for c in required_cols if c not in df_quant.columns]
    if missing:
        raise ValueError(f"Missing required columns in quantification.csv: {missing}")

    with open(line_json, "r", encoding="utf-8") as f:
        line_db = json.load(f)

    real_max_counts = None

    if mode in {"real", "both"}:
        energy_real, counts_real = read_emsa(spectrum_file)
        if not energy_real or not counts_real:
            raise ValueError("Could not read valid spectrum from spectrum.emsa")

        real_max_counts = max(counts_real)

        plot_spectrum(
            energy=energy_real,
            counts=counts_real,
            df_quant=df_quant,
            line_db=line_db,
            preferred_lines=preferred_lines,
            title="Real EDS Spectrum",
            spectrum_color=spectrum_color,
            peak_label_color=peak_label_color,
            x_range=x_range,
            y_range=y_range,
            show_line_type=show_line_type,
            save_path="real_eds_spectrum.pdf"
        )

    if mode in {"simulate", "both"}:
        ref_energy_eV, ref_intensity = build_simulation_inputs(
            df_quant=df_quant,
            line_db=line_db,
            preferred_lines=preferred_lines,
            line_weights=line_weights,
        )

        energy_sim, counts_sim = simulate_eds_spectrum(
            ref_energy_eV=ref_energy_eV,
            ref_intensity=ref_intensity,
            x_min=simulate_x_min,
            x_max=simulate_x_max,
            npts=simulate_npts,
            fwhm_ev=simulate_fwhm_ev,
        )

        if counts_sim.max() > 0:
            target_max = real_max_counts if (simulate_scale_to_real_max and real_max_counts is not None) else simulate_max_counts
            counts_sim = counts_sim / counts_sim.max() * target_max

        plot_spectrum(
            energy=energy_sim.tolist(),
            counts=counts_sim.tolist(),
            df_quant=df_quant,
            line_db=line_db,
            preferred_lines=preferred_lines,
            title="Simulated EDS Spectrum",
            spectrum_color=spectrum_color,
            peak_label_color=peak_label_color,
            x_range=x_range,
            y_range=y_range,
            show_line_type=show_line_type,
            save_path="simulated_eds_spectrum.pdf"
        )

    if mode not in {"real", "simulate", "both"}:
        raise ValueError("mode must be 'real', 'simulate', or 'both'")


if __name__ == "__main__":
    main()