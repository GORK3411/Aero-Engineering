import os
import folium
import pandas as pd
from traffic.core import Flight
import webbrowser
import argparse


def plot_combined_map(flights_with_colors, save_path):
    first_df = list(flights_with_colors.values())[0][0].data
    center = [first_df["latitude"].mean(), first_df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=7)

    for name, (flight, color) in flights_with_colors.items():
        coords = list(zip(flight.data["latitude"], flight.data["longitude"]))
        folium.PolyLine(coords, color=color, weight=2.5, tooltip=name).add_to(m)

    m.save(save_path)
    webbrowser.open(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", type=str)
    parser.add_argument("selected_flight", type=str)
    args = parser.parse_args()

    step = args.step
    selected_flight = args.selected_flight

    reconstructed_path = "final_reconstructions"
    cleaned_data_folder = "cleaned_data_final"

    df_ads_b_before = pd.read_parquet(rf"{cleaned_data_folder}\{step}\{selected_flight}\adsb_before.parquet")
    df_ads_c        = pd.read_parquet(rf"{cleaned_data_folder}\{step}\{selected_flight}\adsc.parquet")
    df_ads_b_after  = pd.read_parquet(rf"{cleaned_data_folder}\{step}\{selected_flight}\adsb_after.parquet")
    df_final        = pd.read_parquet(rf"{reconstructed_path}\{step}\{selected_flight}\full_reconstruction.parquet")

    flight_ads_b_before = Flight(df_ads_b_before)
    flight_ads_c        = Flight(df_ads_c)
    flight_ads_b_after  = Flight(df_ads_b_after)
    final_flight        = Flight(df_final)

    output_folder = rf"{reconstructed_path}\{step}\{selected_flight}"

    flights_map = {
        "BiGRU":        (final_flight,        "orange"),
        "ADS-B Before": (flight_ads_b_before, "blue"),
        "ADS-C":        (flight_ads_c,        "green"),
        "ADS-B After":  (flight_ads_b_after,  "red"),
    }

    baseline_path = os.path.join(reconstructed_path, step, selected_flight, "baseline_full_reconstruction.parquet")
    if os.path.exists(baseline_path):
        flights_map["Baseline (GC)"] = (Flight(pd.read_parquet(baseline_path)), "purple")

    kalman_path = os.path.join(reconstructed_path, step, selected_flight, "kalman_full_reconstruction.parquet")
    if os.path.exists(kalman_path):
        flights_map["Kalman"] = (Flight(pd.read_parquet(kalman_path)), "brown")

    plot_combined_map(flights_map, save_path=os.path.join(output_folder, "combined_map.html"))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        print(fr"Maybe you did not reconstruct the flight yet? in that case run python .\reconstruction.py step flight_name ")
        print(fr"Example: python .\reconstruction.py step1_raw_2023-08-10_to_2023-09-10 20230816_7380c1_065449_081655")