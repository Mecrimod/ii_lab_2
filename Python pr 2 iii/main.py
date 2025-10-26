import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


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

    except ValueError:
        print("Ошибка: введите корректные числовые значения.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()