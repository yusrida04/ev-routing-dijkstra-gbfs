from src.utils.energy import (
    BATTERY_CAPACITY,
    ENERGY_CONSUMPTION,
    calculate_energy
)

# Skenario persentase baterai awal
BATTERY_SCENARIOS = [80, 50, 20]


def percentage_to_kwh(percentage):
    """
    Mengubah persentase baterai menjadi energi (kWh).
    """
    if not 0 <= percentage <= 100:
        raise ValueError("Persentase baterai harus 0-100.")

    return BATTERY_CAPACITY * percentage / 100


def kwh_to_percentage(energy_kwh):
    """
    Mengubah energi (kWh) menjadi persentase baterai.
    """
    return (energy_kwh / BATTERY_CAPACITY) * 100


def can_reach_destination(
    current_battery_kwh,
    distance_km
):
    """
    Memeriksa apakah baterai cukup untuk menempuh jarak.
    """
    required_energy = calculate_energy(distance_km)

    return required_energy <= current_battery_kwh


def get_battery_scenarios():
    """
    Menghasilkan kapasitas baterai untuk setiap skenario.
    """
    return {
        percentage: percentage_to_kwh(percentage)
        for percentage in BATTERY_SCENARIOS
    }


if __name__ == "__main__":
    print("SIMULASI SKENARIO BATERAI BYD ATTO 3")
    print(f"Kapasitas baterai: {BATTERY_CAPACITY} kWh")
    print(f"Konsumsi energi: {ENERGY_CONSUMPTION:.4f} kWh/km")

    print("\nKONDISI BATERAI AWAL:")

    for percentage, energy in get_battery_scenarios().items():
        estimated_range = energy / ENERGY_CONSUMPTION

        print(
            f"Baterai {percentage}%: "
            f"{energy:.2f} kWh | "
            f"Estimasi jarak: {estimated_range:.2f} km"
        )

    print("\nPENGUJIAN KECUKUPAN BATERAI:")

    test_distance = 50

    for percentage, energy in get_battery_scenarios().items():
        result = can_reach_destination(
            energy,
            test_distance
        )

        status = "CUKUP" if result else "TIDAK CUKUP"

        print(
            f"Baterai {percentage}% untuk "
            f"perjalanan {test_distance} km: {status}"
        )
