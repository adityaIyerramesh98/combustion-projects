import cantera as ct
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")
    import sys
    sys.exit(1)

one_atm = 101325

def combust(T: float, P: float, species_dict: dict, reactor: str,
            dt: float = 5e-4, till_ignition: bool = True):

    gas = ct.Solution('gri30.yaml')
    gas.TPX = T, P, species_dict

    # Reactor selection
    if reactor == 'constant volume':
        r = ct.Reactor(gas)
    elif reactor == 'constant pressure':
        r = ct.ConstPressureReactor(gas)
    else:
        raise ValueError(f"Invalid reactor type: {reactor}")

    sim = ct.ReactorNet([r])

    states = ct.SolutionArray(gas, extra=['time'])

    simulation_time = 10.0
    ignition_delay = None
    ignited = False

    time = 0.0
    while time < simulation_time:
        time += dt
        sim.advance(time)

        states.append(r.thermo.state, time=time)

        # Use latest temperature safely
        current_T = states.T[-1]

        if not ignited and current_T >= (T + 400):
            ignition_delay = time
            ignited = True

            if till_ignition:
                break

    return gas, states, ignition_delay


# ------------------ PART I-A ------------------
fig1, axs = plt.subplots(2, 1, figsize=(10, 10), layout='tight')

igd_lst = []
P_range = np.linspace(1, 5, 9)

for p in P_range:
    _, states, igd = combust(
        1250, p * one_atm,
        {'CH4': 1, 'O2': 2, 'N2': 7.52},
        'constant volume',
        dt=2e-4
    )

    igd_ms = igd * 1000 if igd else np.nan
    axs[0].plot(states.time * 1e3, states.T, '.-',
                label=f'{p:.2f} atm, I.D.={igd_ms:.3f} ms')

    igd_lst.append(igd_ms)

axs[0].set_xlabel('Time [ms]')
axs[0].set_ylabel('Temperature [K]')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(igd_lst, P_range, '.-')
axs[1].set_xlabel('Ignition Delay [ms]')
axs[1].set_ylabel('Pressure [atm]')
axs[1].grid(True)

fig1.suptitle("Auto-ignition of methane at 1250 K (Pressure variation)")


# ------------------ PART I-B ------------------
fig2, axs = plt.subplots(2, 1, figsize=(10, 10), layout='tight')

igd_lst = []
T_range = np.linspace(950, 1450, 9)

for temp in T_range:
    _, states, igd = combust(
        temp, 5 * ct.one_atm,
        {'CH4': 1, 'O2': 2, 'N2': 7.52},
        'constant pressure'
    )

    igd_ms = igd * 1000 if igd else np.nan
    axs[0].plot(states.time * 1e3, states.T, '.-',
                label=f'{temp:.0f} K, I.D.={igd_ms:.3f} ms')

    igd_lst.append(igd_ms)

axs[0].set_xlabel('Time [ms]')
axs[0].set_ylabel('Temperature [K]')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(igd_lst, T_range, '.-')
axs[1].set_xlabel('Ignition Delay [ms]')
axs[1].set_ylabel('Initial Temperature [K]')
axs[1].grid(True)

fig2.suptitle("Auto-ignition of methane at 5 atm (Temperature variation)")


# ------------------ PART II ------------------
gasA, statesA, igdA = combust(
    1000, 5 * ct.one_atm,
    {'CH4': 1, 'O2': 2, 'N2': 7.52},
    'constant volume',
    till_ignition=False
)

gasB, statesB, igdB = combust(
    500, 5 * ct.one_atm,
    {'CH4': 1, 'O2': 2, 'N2': 7.52},
    'constant volume',
    dt=0.01,
    till_ignition=False
)

fig3, axs = plt.subplots(3, 1, figsize=(10, 10), layout='tight')

species_list = ['H2O', 'O2', 'OH']

for i, sp in enumerate(species_list):
    axs[i].plot(statesA.time, statesA.X[:, gasA.species_index(sp)],
                '.-', label=f'1000 K')
    axs[i].plot(statesB.time, statesB.X[:, gasB.species_index(sp)],
                '.-', label=f'500 K')

    axs[i].set_ylabel(f'[{sp}]')
    axs[i].grid(True)
    axs[i].legend()

fig3.supxlabel('Time [s]')
fig3.supylabel('Mole Fraction')

plt.show()