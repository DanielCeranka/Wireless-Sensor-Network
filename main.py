import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SensorNetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Symulacja Sieci Sensorowej")

        self.create_widgets()

    def create_widgets(self):
        # Tworzenie i umieszczanie elementów GUI
        tk.Label(self.root, text="Liczba sensorów:").pack()
        self.num_sensors_entry = tk.Entry(self.root)
        self.num_sensors_entry.pack()

        tk.Label(self.root, text="Liczba punktów:").pack()
        self.num_points_entry = tk.Entry(self.root)
        self.num_points_entry.pack()

        tk.Label(self.root, text="Poziom pokrycia (%):").pack()
        self.coverage_entry = tk.Entry(self.root)
        self.coverage_entry.pack()

        tk.Label(self.root, text="Rozmiar obszaru (np. 300,300):").pack()
        self.area_entry = tk.Entry(self.root)
        self.area_entry.pack()

        tk.Label(self.root, text="Zasięg sensora:").pack()
        self.radius_entry = tk.Entry(self.root)
        self.radius_entry.pack()

        self.start_button = tk.Button(self.root, text="Start", command=self.start_simulation)
        self.start_button.pack()

        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack()

        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().pack()

    def start_simulation(self):
        # Rozpoczęcie symulacji po kliknięciu przycisku "Start"
        try:
            num_sensors = int(self.num_sensors_entry.get())
            num_points = int(self.num_points_entry.get())
            coverage = float(self.coverage_entry.get())
            area_size = tuple(map(int, self.area_entry.get().split(',')))
            coverage_radius = float(self.radius_entry.get())

            if num_sensors <= 0 or num_points <= 0 or coverage <= 0 or coverage > 100:
                raise ValueError("Nieprawidłowe dane wejściowe.")

            sensors, points = self.generate_grid(num_sensors, num_points, area_size)
            best_schedule, best_lifetime, active_sensors_over_time, coverage_over_time = self.genetic_algorithm(sensors, points, coverage, area_size, coverage_radius)
            self.plot_results(sensors, points, best_schedule, active_sensors_over_time[:best_lifetime], coverage_over_time[:best_lifetime], coverage_radius)
            self.result_label.config(text=f"Najlepszy czas życia sieci: {best_lifetime}")

        except ValueError as e:
            messagebox.showerror("Błąd", str(e))
            self.clear_entries()
        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił nieoczekiwany błąd: {str(e)}")
            self.clear_entries()

    def generate_grid(self, num_sensors, num_points, area_size):
        # Generowanie rozmieszczenia sensorów i punktów w obszarze
        sensors = np.random.rand(num_sensors, 2) * area_size
        points = np.random.rand(num_points, 2) * area_size
        return sensors, points

    def genetic_algorithm(self, sensors, points, coverage, area_size, coverage_radius, pop_size=50, generations=100):
        # Algorytm genetyczny do optymalizacji czasu życia sieci sensorowej
        population = [self.random_schedule(sensors.shape[0]) for _ in range(pop_size)]
        best_schedule = None
        best_lifetime = 0
        best_active_sensors_over_time = []
        best_coverage_over_time = []

        for gen in range(generations):
            new_active_sensors_over_time = []
            new_coverage_over_time = []
            fitness_scores = [self.evaluate_lifetime(ind, sensors, points, coverage, area_size, coverage_radius, new_active_sensors_over_time, new_coverage_over_time) for ind in population]
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: -pair[0])]
            population = sorted_population[:pop_size // 2]
            population = list(population)  # Upewnienie się, że populacja jest listą

            while len(population) < pop_size:
                parent1_idx, parent2_idx = np.random.choice(len(population), 2, replace=False)
                parent1, parent2 = population[parent1_idx], population[parent2_idx]
                child = self.crossover(parent1, parent2)
                if np.random.rand() < 0.1:
                    child = self.mutate(child)
                population.append(child)

            best_candidate = population[0]
            candidate_lifetime = self.evaluate_lifetime(best_candidate, sensors, points, coverage, area_size, coverage_radius, new_active_sensors_over_time, new_coverage_over_time)
            print(f"Generacja {gen+1}: Czas życia {candidate_lifetime}")

            if candidate_lifetime > best_lifetime:
                best_lifetime = candidate_lifetime
                best_schedule = best_candidate
                best_active_sensors_over_time = new_active_sensors_over_time[:]
                best_coverage_over_time = new_coverage_over_time[:]

        return best_schedule, best_lifetime, best_active_sensors_over_time, best_coverage_over_time

    def random_schedule(self, num_sensors):
        # Generowanie losowego harmonogramu dla sensorów
        return np.random.permutation(num_sensors)

    def evaluate_lifetime(self, schedule, sensors, points, coverage, area_size, coverage_radius,
                          active_sensors_over_time, coverage_over_time):
        # Ocena czasu życia sieci na podstawie danego harmonogramu
        battery = np.ones(len(sensors))
        covered_points = np.zeros(len(points), dtype=bool)
        lifetime = 0

        required_coverage = coverage / 100.0 * len(points)
        active_last_step = np.zeros(len(sensors), dtype=bool)

        while np.any(battery > 0):
            active_sensors = 0
            for sensor in schedule:
                if battery[sensor] > 0:
                    if not active_last_step[sensor]:  # Jeśli sensor nie był aktywny w poprzednim kroku
                        battery[sensor] -= 0.1  # Zmniejsz baterię tylko jeśli sensor jest aktywowany
                    active_last_step[sensor] = True
                    covered_points |= self.check_coverage(sensors[sensor], points, coverage_radius)
                    active_sensors += 1
                    if np.sum(covered_points) >= required_coverage:
                        active_sensors_over_time.append(active_sensors)
                        coverage_over_time.append(np.sum(covered_points) / len(points) * 100)
                        return lifetime
                    lifetime += 1
            active_sensors_over_time.append(active_sensors)
            coverage_over_time.append(np.sum(covered_points) / len(points) * 100)
        return lifetime

    def check_coverage(self, sensor, points, radius):
        # Sprawdzanie pokrycia punktów przez sensor
        distances = np.linalg.norm(points - sensor, axis=1)
        return distances <= radius

    def crossover(self, parent1, parent2):
        # Krzyżowanie dwóch harmonogramów
        crossover_point = np.random.randint(1, len(parent1) - 1)
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return self.fix_schedule(child)

    def mutate(self, schedule):
        # Mutacja harmonogramu
        i, j = np.random.choice(len(schedule), 2, replace=False)
        schedule[i], schedule[j] = schedule[j], schedule[i]
        return self.fix_schedule(schedule)

    def fix_schedule(self, schedule):
        # Naprawianie harmonogramu, aby zawierał unikalne wartości
        unique, indices = np.unique(schedule, return_index=True)
        full_schedule = np.arange(len(schedule))
        missing = np.setdiff1d(full_schedule, unique)
        fixed_schedule = np.zeros_like(schedule)
        fixed_schedule[indices] = unique
        fixed_schedule[np.setdiff1d(full_schedule, indices)] = missing
        return fixed_schedule

    def plot_results(self, sensors, points, schedule, active_sensors_over_time, coverage_over_time, coverage_radius):
        # Rysowanie wyników symulacji
        self.ax1.clear()
        self.ax2.clear()

        # Rysowanie sensorów i punktów
        self.ax1.scatter(sensors[:, 0], sensors[:, 1], c='blue', label='Sensory')
        self.ax1.scatter(points[:, 0], points[:, 1], c='red', label='Punkty')
        self.ax1.set_title('Rozmieszczenie Sensorów i Punktów')
        self.ax1.legend()

        # Rysowanie kręgów zasięgu
        for sensor in sensors:
            circle = plt.Circle(sensor, coverage_radius, color='blue', fill=False, linestyle='--', alpha=0.5)
            self.ax1.add_artist(circle)

        # Rysowanie aktywnych sensorów i pokrycia w czasie
        self.ax2.plot(range(len(active_sensors_over_time)), active_sensors_over_time, label='Aktywne Sensory')
        self.ax2.plot(range(len(coverage_over_time)), coverage_over_time, label='Pokrycie (%)')
        self.ax2.set_title('Aktywne Sensory i Pokrycie w Czasie')
        self.ax2.set_xlabel('Czas')
        self.ax2.set_ylabel('Liczba / Procent')
        self.ax2.legend()

        self.canvas.draw()

    def clear_entries(self):
        # Czyszczenie pól wejściowych
        self.num_sensors_entry.delete(0, tk.END)
        self.num_points_entry.delete(0, tk.END)
        self.coverage_entry.delete(0, tk.END)
        self.area_entry.delete(0, tk.END)
        self.radius_entry.delete(0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = SensorNetworkApp(root)
    root.mainloop()