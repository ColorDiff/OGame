from typing import Dict

import numpy as np
import matplotlib.pyplot as plt


V = 450_000
p_ress = 0.325
p_occurance = {'M': 0.685,
               'K': 0.25,
               'D': 0.075}
p_normal = 0.89
p_large = 0.1
p_huge = 0.01
v_muls = {'M': 1,
          'K': 1 / 2,
          'D': 1 / 3}
res_name = {'M': 'Metall',
            'K': 'Kristall',
            'D': 'Deuterium'}
res_col = {'M': 'r',
           'K': 'b',
           'D': 'g'}


def exp_res_val(capacity: float, res_mult: float = 1.0) -> Dict[str, float]:
    # Expected values for normal finds
    fsz_normal_src = np.arange(10, 50) * V
    exp_val = {ress: 0.0 for ress in v_muls.keys()}
    for ress, mul in v_muls.items():
        exp_val_normal_ress = (np.clip(res_mult * fsz_normal_src * mul, None, capacity) / len(fsz_normal_src)).sum()
        exp_val[ress] += p_normal * p_occurance[ress] * exp_val_normal_ress

    # Exp val for large finds
    fsz_large_src = np.arange(50, 100) * V
    for ress, mul in v_muls.items():
        exp_val_large_ress = (np.clip(res_mult * fsz_large_src * mul, None, capacity) / len(fsz_large_src)).sum()
        exp_val[ress] += p_large * p_occurance[ress] * exp_val_large_ress

    # Exp val for hughe finds
    fsz_huge_src = np.arange(100, 200) * V
    for ress, mul in v_muls.items():
        exp_val_huge_ress = (np.clip(res_mult * fsz_huge_src * mul, None, capacity) / len(fsz_huge_src)).sum()
        exp_val[ress] += p_huge * p_occurance[ress] * exp_val_huge_ress
    return {res: val * p_ress for res, val in exp_val.items()}


def make_plot(exp_vals, capacities, res_mult, max_hline=False):
    f = plt.figure(dpi=300)
    ax = f.gca()
    f.suptitle('Erwartungswert der Rohstofffunde pro Expedition\nnach Ladekapazität für V=450 000, $s_k$={:.2f}'
               .format(res_mult))
    ticks = []
    for res, v in exp_vals.items():
        if max_hline:
            ax.hlines([max(v) / 1000], 0, max(capacities[res]), colors=['k'], linestyles=['--'])
            ticks.append(max(v) / 1000)
        ax.plot(capacities[res], v / 1000, label=res_name[res], color=res_col[res])
        ax.set_xlabel('Ladekapazität in Mio. [1 000 000]')
        ax.set_ylabel('Erwartungswert Rohstoff in Tausenden [1 000]')
    if max_hline:
        ax.set_yticks(sorted(ax.get_yticks().tolist() + ticks)[1:])
    ax.grid()
    ax.legend()
    return f


def get_cost(n_gt: int, gt_cost: float, pos: int):
    cost_flight = round(n_gt * gt_cost * ((1000000 + 5000 * (16 - pos)) / 35000000) * (1 + 1) ** 2) + 1
    cost_expo = n_gt * round(gt_cost) / 10
    return cost_flight + cost_expo


def get_exp_values_for_scenario(gt_kapa: int, gt_consumption: float, num_gts: int, position: int, res_mult: float):
    deut_cost = get_cost(num_gts, gt_consumption, position)
    exp_val = exp_res_val(gt_kapa * num_gts, res_mult=res_mult)
    exp_val['D'] -= deut_cost
    return exp_val


def to_dse(res, vals=None):
    if vals is None:
        vals = {'M': 1 / 3, 'K': 1 / 2, 'D': 1}
    acc = 0
    for k, v in res.items():
        acc += vals[k] * v
    return acc


def main():
    res_mult = 1.2
    exp_vals = {res: [] for res in v_muls.keys()}
    capacities = {res: [] for res in v_muls.keys()}
    for capacity_mln in range(1, 120):
        for res, val in exp_res_val(capacity_mln * 1_000_000, res_mult).items():
            exp_vals[res].append(val)
            capacities[res].append(capacity_mln)
    print(exp_vals['M'][87:])
    for res in exp_vals.keys():
        exp_vals[res] = np.array(exp_vals[res])
        capacities[res] = np.array(capacities[res])

    f_all = make_plot(exp_vals, capacities, res_mult, max_hline=True)
    f_detail = make_plot({k: v[:10] for k, v in exp_vals.items()},
                         {k: v[:10] for k, v in capacities.items()}, res_mult)
    f_all.show()
    f_detail.show()


def plot_scenario(n_gts, dse_321, kapa_gt, sk, consumption):
    f = plt.figure(dpi=300)
    f.suptitle('Erwartungswert der Rohstofffunde in DSE pro Expo. inkl. Verbrauch'
               '\n GT Kapazität={}, Verbrauch={}, V=450 000, $s_k$={:.2f}, Kurs 3:2:1'.format(kapa_gt, consumption, sk))
    ax = f.gca()
    dse_321 = dse_321 / 1000
    ax.plot(n_gts, dse_321)
    ax.set_xlabel('Anzahl Große Transporter (GTs)')
    ax.set_ylabel('Deuterium Standardeinheiten (DSE) in Tausend [1000]')
    ax.vlines([n_gts[np.argmax(dse_321)]], min(dse_321), max(dse_321), colors=(0, 0, 0, .3), linestyles='-.')
    ax2 = ax.twinx()
    ax2.plot(n_gts, n_gts * kapa_gt / 1_000_000, 'k--')
    ax2.set_ylabel('Transportkapazität in Mio. [1 000 000]')
    ax.grid()
    return f


if __name__ == '__main__':
    main()

    # Szenario 1 No-Kaelsh: GT 40 Kapa, 50 Deutverbrauch, 0% s_k
    # -> Maxfind = 90kk met -> 90kk Kapa vs 80 vs 70 vs 60
    # Szenario 2 Kaelsh-Basic: GT 42k Kapazität, 49 Deutverbrauch (Kaelsh research), 15 % s_k
    # -> Maxfind = 99kk met -> 99kk Kapa vs 90 vs 80 vs 70 vs 60
    # Szenario 3 kaelsh-Medium: GT 45k Kapa, 48 Deutverbrauch (Kaelsh reseach), 30% s_k (Kaelsh reseach)
    # -> Maxfind = 118kk met -> 118kk Kapa vs 100 vs 90 vs 80 vs 70 vs 60
    # Szenario 3 kaelsh-Advanced: GT 50k Kapa, 47 Deutverbrauch (Kaelsh reseach), 45% s_k (Kaelsh reseach)
    # -> Maxfind = 118kk met -> 118kk Kapa vs 100 vs 90 vs 80 vs 70 vs 60
    n_gts = []
    exp_vals = []
    dse = []
    cons = 50
    kapa = 40_000
    sk = 1.50
    for target_kapa_mln in range(50, 121):
        n_gts.append(round(target_kapa_mln * 1_000_000 / kapa) + 1)
        exp_vals.append(get_exp_values_for_scenario(kapa, cons, n_gts[-1], 1, sk))
        dse.append(to_dse(exp_vals[-1]))
        print(target_kapa_mln, n_gts[-1], exp_vals[-1], to_dse(exp_vals[-1]))
    n_gts = np.array(n_gts)
    dse = np.array(dse)
    plot_scenario(n_gts, dse, kapa, sk, cons).show()

