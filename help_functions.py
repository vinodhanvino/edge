import random

import numpy as np
import math


def uploading_time(length_of_task, upload_rate):
    # length task divide by upload rate
    return length_of_task / upload_rate


def upload_rate(bandwidth, number_of_user, power, gains, N0):
    formula_1 = (bandwidth / number_of_user)
    formula_2 = math.log2(1 + (power * gains))
    result = formula_1 * formula_2 / (formula_1 * N0)
    return round(result)


def create_frequncies(start_range, end_range, Hz):
    start_frequncy = int(start_range * Hz)
    end_frequncy = int(end_range * Hz)
    random_frequncy = random.randint(start_frequncy, end_frequncy)
    return random_frequncy


def get_T_local(no_of_cycle, local_comp_pwr):
    return (no_of_cycle / local_comp_pwr)


def response_time(twait, tcomp):
    #twait + tcomputation
    return twait, tcomp


def tcomp(cpu_power, total_ed_comp_pwr):
    #cpu resource every task divided by total edge server computing power
    return cpu_power / total_ed_comp_pwr


def calculate_distance(source, destination):
    return math.sqrt((destination[0] - source[0])**2 + (destination[1] - source[1])**2)




