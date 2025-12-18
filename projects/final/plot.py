#acknowledgement: the code in this file is 99% AI generated

import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List, Union, Tuple
import itertools # Used for cycling through colors and markers

# Define the expected types for a single trajectory's data
TrajectoryData = Union[np.ndarray, List[float]]

def plot_agent_trajectory_with_cost(
    timestamps: TrajectoryData,
    positions: List[TrajectoryData],
    cost_function: Callable[[float], float],
    labels: List[str] | None = None,
    y_margin_percent: float = 0.05,
    figsize: Tuple[float, float] = (9, 6),
    title: str = 'Agent Trajectories and Associated Positional Cost Profile'
):
    """
    Generates a Matplotlib plot showing MULTIPLE agent trajectories over time 
    and the associated cost function C(x) in a marginal plot, with aligned grids.

    Args:
        timestamps: A single array/list of time points, common to all trajectories.
        positions: A LIST of agent position arrays/lists (y-axis of both plots).
        cost_function: A function that takes an array of positions and
                       returns an array of corresponding costs C(x).
        labels: Optional list of strings for the legend, one for each trajectory.
        y_margin_percent: Percentage margin to add to the min/max of the 
                          position axis (Y-axis).
        figsize: The size of the figure (width, height).
        title: The main title for the figure.
    """
    
    # Ensure timestamps is a numpy array for consistent indexing/checking
    time_array = np.asarray(timestamps)
    
    # 0. Input Validation
    num_trajectories = len(positions)
    time_len = len(time_array)
    
    # Check if all position arrays have the same length as the time array
    for i, pos in enumerate(positions):
        if len(pos) != time_len:
            raise ValueError(
                f"Trajectory {i+1} length ({len(pos)}) does not match "
                f"the common timestamps length ({time_len})."
            )

    # Generate default labels if none are provided
    if labels is None:
        labels = [f"Trajectory {i+1}" for i in range(num_trajectories)]
    elif len(labels) != num_trajectories:
        raise ValueError("The number of labels must match the number of trajectories.")

    # 1. Determine Overall Position (Y-axis) Range with Margin
    # Flatten all position data to find the global min/max
    all_positions = np.concatenate([np.asarray(pos) for pos in positions])
    
    pos_min = np.min(all_positions)
    pos_max = np.max(all_positions)
    pos_range = pos_max - pos_min
    
    y_margin = pos_range * y_margin_percent
    
    y_lim_min = pos_min - y_margin
    y_lim_max = pos_max + y_margin

    y_lim_min = -2.1
    y_lim_max = 1.9
    
    # 2. Calculate Cost Data for the Position Range
    positions_space = np.linspace(y_lim_min, y_lim_max, 200)
    costs = np.asarray([cost_function(float(x)) for x in positions_space])
    
    # --- 3. Create the Plot Structure (Mosaic) ---

    fig, axs = plt.subplot_mosaic([['cost_func', 'trajectory']],
                                  figsize=figsize,
                                  width_ratios=(1.5, 5), 
                                  layout='constrained')

    ax_cost_func = axs['cost_func']
    ax_trajectory = axs['trajectory']

    # Set shared Y-axis limits (Position x)
    ax_trajectory.set_ylim(y_lim_min, y_lim_max)
    ax_cost_func.set_ylim(y_lim_min, y_lim_max)
    ax_cost_func.sharey(ax_trajectory)

    # Define a color cycle for the trajectories
    color_cycle = itertools.cycle(plt.cm.Set1.colors) 

    ## A. Plot the Cost Function C(x) (Left Marginal Plot) ##

    ax_cost_func.plot(costs, positions_space, color='darkgreen', linestyle='-', linewidth=2, label='Cost $\\mathcal{C}(x)$')

    ax_cost_func.set_xlabel('Cost ($\\mathcal{C}$)', fontsize=12)
    ax_cost_func.set_ylabel('Position ($x$)', fontsize=14)
    ax_cost_func.tick_params(axis='y', labelleft=True) 

    # Add horizontal grid lines to the cost chart
    ax_cost_func.grid(axis='y', linestyle=':', alpha=0.6) 

    ## B. Plot the Agent Trajectories (Main Plot) ##

    # Iterate through all position arrays and plot them against the single time array
    for i in range(num_trajectories):
        color = next(color_cycle)
        ax_trajectory.plot(
            time_array, # <-- CHANGE: Use the single time_array here
            positions[i], 
            color=color, 
            linewidth=2, 
            label=labels[i]
        )

    ax_trajectory.set_xlabel('Time ($t$)', fontsize=14)
    ax_trajectory.tick_params(axis='y', labelleft=False) # Remove y-labels here
    
    # Add a legend to the trajectory plot
    ax_trajectory.legend(loc='lower right', fontsize=10)
    
    # Grid for the trajectory chart (aligned with the cost chart's grid)
    ax_trajectory.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle(title, fontsize=16)

    plt.show()


if __name__ == '__main__':
    # =================================================================
    # --- Example Usage with Mock Data for Multiple Trajectories ---
    # =================================================================

    # 1. Define the Cost Function (same as before)
    def example_cost_function(x: np.ndarray) -> np.ndarray:
        """A sample cost function with a local minimum around x=3.5."""
        cost = 1.0 + 0.5 * (x - 5)**2
        wobble = 0.5 * np.sin(x * 5)
        min_dip = 2.0 * np.exp(-((x - 3.5)**2) / 0.5)
        
        final_cost = cost + wobble - min_dip
        return np.clip(final_cost, 0.1, 10) 

    # 2. Generate Trajectory Data (Multiple Trajectories)
    TIME_STEPS = 400
    
    # SINGLE time array
    common_time_array = np.arange(TIME_STEPS) 
    
    # Trajectory 1: Starts high, settles near a local minimum of cost (around 3.5)
    position_array_1 = 8.0 - 4.5 * (1 - np.exp(-common_time_array / 120.0)) + 0.5 * np.sin(common_time_array / 10.0)

    # Trajectory 2: Starts low, moves away from the minimum
    position_array_2 = 2.0 + 1.0 * (1 - np.exp(-common_time_array / 80.0)) + 0.5 * np.cos(common_time_array / 15.0)

    # Trajectory 3: Stays relatively high
    position_array_3 = 6.0 + 1.5 * np.sin(common_time_array / 25.0) + 0.1 * common_time_array / 400.0

    # Bundle the data for the function call
    # Note: Only one time array is passed now
    all_positions = [position_array_1, position_array_2, position_array_3]
    all_labels = ['Policy A: Optimal', 'Policy B: Sub-optimal', 'Policy C: Oscillating']

    # 3. Call the Function
    plot_agent_trajectory_with_cost(
        timestamps=common_time_array,
        positions=all_positions, #type: ignore
        cost_function=example_cost_function,
        labels=all_labels,
        y_margin_percent=0.10,
        title='Comparison of Three Agent Trajectories (Common Time)'
    )
