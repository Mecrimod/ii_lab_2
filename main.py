import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


class AviationSystem:
    def __init__(self):
        self.height_universe = np.arange(0, 15001, 100)
        self.speed_universe = np.arange(0, 1201, 10)

        #нечеткие переменные
        self.height = ctrl.Antecedent(self.height_universe, 'height')
        self.speed = ctrl.Antecedent(self.speed_universe, 'speed')

        #функции принадлежности
        self.height['low'] = fuzz.trapmf(self.height.universe, [0, 0, 2000, 4000])
        self.height['medium'] = fuzz.trapmf(self.height.universe, [2000, 4000, 6000, 8000])
        self.height['high'] = fuzz.trapmf(self.height.universe, [6000, 8000, 15000, 15000])

        self.speed['slow'] = fuzz.trapmf(self.speed.universe, [0, 0, 300, 500])
        self.speed['normal'] = fuzz.trapmf(self.speed.universe, [300, 500, 700, 900])
        self.speed['fast'] = fuzz.trapmf(self.speed.universe, [700, 900, 1200, 1200])

    def calculate_intersection(self, height_value, speed_value):
        #вычисление степеней принадлежности
        height_low = fuzz.interp_membership(self.height.universe, self.height['low'].mf, height_value)
        height_medium = fuzz.interp_membership(self.height.universe, self.height['medium'].mf, height_value)
        height_high = fuzz.interp_membership(self.height.universe, self.height['high'].mf, height_value)

        speed_slow = fuzz.interp_membership(self.speed.universe, self.speed['slow'].mf, speed_value)
        speed_normal = fuzz.interp_membership(self.speed.universe, self.speed['normal'].mf, speed_value)
        speed_fast = fuzz.interp_membership(self.speed.universe, self.speed['fast'].mf, speed_value)

        #операция пересечения (минимум) для всех комбинаций
        intersection_values = {
            'low_slow': min(height_low, speed_slow),
            'low_normal': min(height_low, speed_normal),
            'low_fast': min(height_low, speed_fast),
            'medium_slow': min(height_medium, speed_slow),
            'medium_normal': min(height_medium, speed_normal),
            'medium_fast': min(height_medium, speed_fast),
            'high_slow': min(height_high, speed_slow),
            'high_normal': min(height_high, speed_normal),
            'high_fast': min(height_high, speed_fast)
        }

        return intersection_values

    def get_intersection_result(self, height_value, speed_value):
        results = self.calculate_intersection(height_value, speed_value)
        max_intersection = max(results.values())
        best_combinations = [comb for comb, value in results.items() if value == max_intersection]
        return best_combinations, results

    def plot(self, height_value, speed_value, best_combination):
        height_term, speed_term = best_combination.split('_')

        #график с двумя подграфиками
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        #Высота
        ax1.plot(self.height_universe, self.height['low'].mf, 'b', linewidth=1.5, label='Low', alpha=0.7)
        ax1.plot(self.height_universe, self.height['medium'].mf, 'g', linewidth=1.5, label='Medium', alpha=0.7)
        ax1.plot(self.height_universe, self.height['high'].mf, 'r', linewidth=1.5, label='High', alpha=0.7)

        #Точка для высоты
        height_membership = fuzz.interp_membership(self.height.universe, self.height[height_term].mf, height_value)
        ax1.plot(height_value, height_membership, 'ro', markersize=10, label=f'Input: {height_value}m')

        ax1.set_title(f'Height Membership\n{height_term.capitalize()} height')
        ax1.set_xlabel('Height (m)')
        ax1.set_ylabel('Membership')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        #Скорость
        ax2.plot(self.speed_universe, self.speed['slow'].mf, 'b', linewidth=1.5, label='Slow', alpha=0.7)
        ax2.plot(self.speed_universe, self.speed['normal'].mf, 'g', linewidth=1.5, label='Normal', alpha=0.7)
        ax2.plot(self.speed_universe, self.speed['fast'].mf, 'r', linewidth=1.5, label='Fast', alpha=0.7)

        # Точка для скорости
        speed_membership = fuzz.interp_membership(self.speed.universe, self.speed[speed_term].mf, speed_value)
        ax2.plot(speed_value, speed_membership, 'ro', markersize=10, label=f'Input: {speed_value}km/h')

        ax2.set_title(f'Speed Membership\n{speed_term.capitalize()} speed')
        ax2.set_xlabel('Speed (km/h)')
        ax2.set_ylabel('Membership')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    aviation_system = AviationSystem()
    try:
        height_val = float(input("Введите высоту полета (м, 0-15000): "))
        speed_val = float(input("Введите скорость полета (км/ч, 0-1200): "))

        best_combinations, all_results = aviation_system.get_intersection_result(height_val,speed_val)

        for combination, value in all_results.items():
            height_term, speed_term = combination.split('_')
            print(f"{height_term} height & {speed_term} speed: {value:.3f}")

        print("Лучшие комбинации:")
        for comb in best_combinations:
            height_term, speed_term = comb.split('_')
            print(f"- {height_term} height & {speed_term} speed")

        if best_combinations:
            aviation_system.plot(height_val, speed_val, best_combinations[0])

    except ValueError:
        print("Ошибка: введите корректные числовые значения.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()