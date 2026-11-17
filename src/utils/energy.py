
# Spesifikasi BYD Atto 3 Advanced Standard

BATTERY_CAPACITY = 49.92  # kWh
NEDC_RANGE = 410          # km

# Estimasi konsumsi energi
ENERGY_CONSUMPTION = BATTERY_CAPACITY / NEDC_RANGE


def calculate_energy(
    distance_km,
    consumption_kwh_per_km=ENERGY_CONSUMPTION
):
    """
    Menghitung estimasi energi yang dibutuhkan
    berdasarkan jarak perjalanan.
    """
    consumption = (
        ENERGY_CONSUMPTION
        if consumption_kwh_per_km is None
        else consumption_kwh_per_km
    )
    return distance_km * consumption


def calculate_remaining_battery(
    initial_battery_kwh,
    distance_km
):
    """
    Menghitung sisa energi baterai setelah perjalanan.
    """
    energy_used = calculate_energy(distance_km)

    return initial_battery_kwh - energy_used


def calculate_battery_percentage(energy_kwh):
    """
    Mengubah kapasitas energi menjadi persentase baterai.
    """
    return (energy_kwh / BATTERY_CAPACITY) * 100


if __name__ == "__main__":
    distance = 50

    energy = calculate_energy(distance)
    remaining = calculate_remaining_battery(
        BATTERY_CAPACITY,
        distance
    )

    print("SIMULASI KONSUMSI ENERGI BYD ATTO 3")
    print(f"Kapasitas baterai: {BATTERY_CAPACITY} kWh")
    print(f"Konsumsi energi: {ENERGY_CONSUMPTION:.4f} kWh/km")
    print(f"Jarak perjalanan: {distance} km")
    print(f"Energi terpakai: {energy:.2f} kWh")
    print(f"Sisa baterai: {remaining:.2f} kWh")
    print(
        f"Sisa persentase: "
        f"{calculate_battery_percentage(remaining):.2f}%"
    )
