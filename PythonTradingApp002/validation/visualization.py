"""
Rolling Window Visualization Module

This module provides visualization tools for rolling window validation results,
including performance plots, stability analysis, and detailed window metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import logging
import traceback
from scipy import stats

def plot_rolling_window_performance(app_config, diagnostics, plt=plt, np=np):
    """
    Create a visualization of rolling window validation performance
    
    Args:
        app_config: Application configuration
        diagnostics: Dictionary containing diagnostics information including window_metrics
        plt: Matplotlib pyplot instance
        np: NumPy instance
        
    Returns:
        fig: Matplotlib figure with rolling window visualization
    """
    # Create the figure
    fig = plt.figure(figsize=app_config.gui.figure_size)
    
    # Check if window metrics are available
    if 'window_metrics' not in diagnostics or not diagnostics['window_metrics']:
        plt.text(0.5, 0.5, "No rolling window metrics available", 
                 ha='center', va='center', fontsize=12)
        plt.tight_layout()
        return fig
    
    # Extract metrics for plotting
    window_metrics = diagnostics['window_metrics']
    window_indices = [m['window_idx'] for m in window_metrics]
    smape_values = [m['smape'] for m in window_metrics]
    
    # Create the main performance plot
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(window_indices, smape_values, marker='o', linestyle='-', color='blue')
    ax1.set_title('Rolling Window Validation Performance')
    ax1.set_xlabel('Window Index')
    ax1.set_ylabel('SMAPE (%)')
    ax1.grid(True)
    
    # Add horizontal line for average performance
    avg_smape = np.mean(smape_values)
    ax1.axhline(y=avg_smape, color='red', linestyle='--', 
               label=f'Average SMAPE: {avg_smape:.2f}%')
    ax1.legend()
    
    # Create a histogram of performance
    ax2 = plt.subplot(2, 1, 2)
    ax2.hist(smape_values, bins=min(10, len(smape_values)), 
             color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_title('Distribution of SMAPE Values')
    ax2.set_xlabel('SMAPE (%)')
    ax2.set_ylabel('Frequency')
    
    # Add descriptive statistics
    min_smape = np.min(smape_values)
    max_smape = np.max(smape_values)
    median_smape = np.median(smape_values)
    std_smape = np.std(smape_values)
    
    stats_text = f"Min: {min_smape:.2f}%  Max: {max_smape:.2f}%\nMedian: {median_smape:.2f}%  StdDev: {std_smape:.2f}%"
    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig

def create_rolling_window_plot_gui(app_config, diagnostics):
    """
    Create a GUI window for rolling window validation visualization
    
    Args:
        app_config: Application configuration
        diagnostics: Dictionary containing window_metrics and other diagnostics
        
    Returns:
        root: Tkinter window for the rolling window visualization
    """
    root = tk.Toplevel()
    root.title("Rolling Window Validation Performance")
    root.geometry("1000x700")
    
    if 'window_metrics' not in diagnostics or not diagnostics['window_metrics']:
        ttk.Label(root, text="No rolling window metrics available", 
                 font=("Arial", 12)).pack(expand=True)
        return root
        
    # Create a notebook for tabs
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Create frames for each tab
    frames = {
        'performance': ttk.Frame(notebook),
        'metrics': ttk.Frame(notebook),
        'windows': ttk.Frame(notebook),
        'comparison': ttk.Frame(notebook)
    }
    
    for name, frame in frames.items():
        notebook.add(frame, text=name.capitalize())
    
    # Performance Plot tab
    frame_performance = frames['performance']
    try:
        fig = plot_rolling_window_performance(app_config, diagnostics)
        canvas = FigureCanvasTkAgg(fig, master=frame_performance)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, frame_performance)
        toolbar.update()
    except Exception as e:
        logging.error(f"Error plotting performance: {e}")
        logging.error(traceback.format_exc())
        ttk.Label(frame_performance, text=f"Error plotting performance: {e}", 
                 wraplength=500).pack(expand=True)
    
    # Metrics Summary tab
    frame_metrics = frames['metrics']
    try:
        metrics = diagnostics['window_metrics']
        
        # Calculate summary statistics
        smape_values = [m['smape'] for m in metrics]
        avg_smape = np.mean(smape_values)
        min_smape = np.min(smape_values)
        max_smape = np.max(smape_values)
        std_smape = np.std(smape_values)
        median_smape = np.median(smape_values)
        
        # Create a summary frame
        summary_frame = ttk.LabelFrame(frame_metrics, text="Performance Summary")
        summary_frame.pack(fill="x", padx=10, pady=10)
        
        # Get configuration values
        training_config = diagnostics.get('training_config', {})
        window_size = training_config.get('window_size', app_config.rolling_window.window_size)
        step_size = training_config.get('step_size', app_config.rolling_window.step_size)
        refit_freq = training_config.get('refit_frequency', app_config.rolling_window.refit_frequency)
        
        summary_labels = [
            f"Number of Windows: {len(metrics)}",
            f"Average SMAPE: {avg_smape:.2f}%",
            f"Median SMAPE: {median_smape:.2f}%",
            f"Minimum SMAPE: {min_smape:.2f}%",
            f"Maximum SMAPE: {max_smape:.2f}%",
            f"Standard Deviation: {std_smape:.2f}%",
            f"Window Size: {window_size} days",
            f"Step Size: {step_size} days",
            f"Refit Frequency: {refit_freq} days"
        ]
        
        for i, label_text in enumerate(summary_labels):
            ttk.Label(summary_frame, text=label_text, font=("Arial", 10)).grid(
                row=i//2, column=i%2, sticky="w", padx=20, pady=5
            )
        
        # Create a stability analysis frame
        stability_frame = ttk.LabelFrame(frame_metrics, text="Model Stability Analysis")
        stability_frame.pack(fill="x", padx=10, pady=10)
        
        # Create a small figure for stability visualization
        fig_stability = plt.Figure(figsize=(8, 3))
        ax = fig_stability.add_subplot(111)
        
        # Plot the distribution of SMAPE values with a kernel density estimate
        if len(smape_values) > 3:  # Need at least 3 points for KDE
            x = np.linspace(min_smape * 0.8, max_smape * 1.2, 100)
            kde = stats.gaussian_kde(smape_values)
            ax.plot(x, kde(x), 'b-', label='SMAPE Distribution')
            ax.fill_between(x, kde(x), alpha=0.3)
        
        # Add reference lines
        ax.axvline(avg_smape, color='r', linestyle='--', label=f'Mean: {avg_smape:.2f}%')
        ax.axvline(median_smape, color='g', linestyle='-.', label=f'Median: {median_smape:.2f}%')
        
        ax.set_title("SMAPE Distribution Across Windows")
        ax.set_xlabel("SMAPE (%)")
        ax.set_ylabel("Density")
        ax.legend()
        
        # Add the stability plot to the frame
        canvas_stability = FigureCanvasTkAgg(fig_stability, master=stability_frame)
        canvas_stability.draw()
        canvas_stability.get_tk_widget().pack(side=tk.TOP, fill=tk.X, expand=1)
        
        # Add stability description based on coefficient of variation
        cv = std_smape / avg_smape if avg_smape > 0 else 0
        
        if cv < 0.1:
            stability_text = "Excellent stability: Model performance is very consistent across different time periods."
        elif cv < 0.2:
            stability_text = "Good stability: Model shows consistent performance with minimal variation."
        elif cv < 0.3:
            stability_text = "Moderate stability: Some variation in model performance across time periods."
        elif cv < 0.5:
            stability_text = "Variable performance: Model shows significant variation across different time periods."
        else:
            stability_text = "Unstable performance: Model results vary substantially depending on the window period."
        
        ttk.Label(stability_frame, text=stability_text, font=("Arial", 10, "italic"), wraplength=500).pack(
            padx=10, pady=5, anchor="w"
        )
        
        # Add stability metrics
        stability_metrics_frame = ttk.Frame(stability_frame)
        stability_metrics_frame.pack(fill="x", padx=10, pady=5)
        
        stability_metrics = [
            f"Coefficient of Variation: {cv:.2f}",
            f"SMAPE Range: {max_smape - min_smape:.2f}%",
            f"95% Confidence Interval: {avg_smape - 1.96*std_smape:.2f}% - {avg_smape + 1.96*std_smape:.2f}%"
        ]
        
        for i, metric in enumerate(stability_metrics):
            ttk.Label(stability_metrics_frame, text=metric, font=("Arial", 9)).grid(
                row=0, column=i, padx=20, pady=5
            )
        
        # Add improvement recommendations based on stability analysis
        recommendations_frame = ttk.LabelFrame(frame_metrics, text="Recommendations")
        recommendations_frame.pack(fill="x", padx=10, pady=10)
        
        recommendations = []
        
        if cv > 0.3:
            recommendations.append("Consider using a larger training window or more frequent model refitting to improve stability.")
        
        if len(metrics) < 10:
            recommendations.append("Increase the number of validation windows for more reliable performance assessment.")
        
        if step_size > window_size / 2:
            recommendations.append("Reduce step size to create more overlapping windows for smoother evaluation.")
        
        if refit_freq > window_size:
            recommendations.append("Increase refit frequency to adapt to changing market conditions more quickly.")
        
        if avg_smape > 15:
            recommendations.append("Consider adding more features or using ensemble methods to improve forecast accuracy.")
        
        if not recommendations:
            recommendations.append("Current configuration shows good results. No specific improvements needed.")
        
        for i, recommendation in enumerate(recommendations):
            ttk.Label(recommendations_frame, text=f"� {recommendation}", 
                     font=("Arial", 9), wraplength=500).pack(anchor="w", padx=10, pady=2)
    
    except Exception as e:
        logging.error(f"Error creating metrics summary: {e}")
        logging.error(traceback.format_exc())
        ttk.Label(frame_metrics, text=f"Error creating metrics summary: {e}", 
                 wraplength=500).pack(expand=True)
    
    # Windows Detail tab - Create a treeview to display all windows
    frame_windows = frames['windows']
    try:
        # Create column headers
        columns = ("window", "start_idx", "end_idx", "smape")
        tree = ttk.Treeview(frame_windows, columns=columns, show="headings")
        
        # Define column headings
        tree.heading("window", text="Window")
        tree.heading("start_idx", text="Start Index")
        tree.heading("end_idx", text="End Index")
        tree.heading("smape", text="SMAPE (%)")
        
        # Set column widths
        tree.column("window", width=100)
        tree.column("start_idx", width=100)
        tree.column("end_idx", width=100)
        tree.column("smape", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame_windows, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert window data
        for i, metric in enumerate(metrics):
            tree.insert("", tk.END, values=(
                f"Window {i+1}",
                metric['start_idx'],
                metric['end_idx'],
                f"{metric['smape']:.2f}%"
            ))
        
        # Highlight best and worst windows
        if metrics:
            best_index = np.argmin([m['smape'] for m in metrics])
            worst_index = np.argmax([m['smape'] for m in metrics])
            
            # Add an info frame
            info_frame = ttk.Frame(frame_windows)
            info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
            
            ttk.Label(info_frame, text=f"Best Window: Window {best_index+1} (SMAPE: {metrics[best_index]['smape']:.2f}%)",
                     font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=20)
            
            ttk.Label(info_frame, text=f"Worst Window: Window {worst_index+1} (SMAPE: {metrics[worst_index]['smape']:.2f}%)",
                     font=("Arial", 10, "bold")).pack(side=tk.RIGHT, padx=20)
            
        # Add actions to view window details
        def on_window_select(event):
            selected_items = tree.selection()
            if selected_items:
                selected_item = selected_items[0]
                window_idx = int(tree.item(selected_item)['values'][0].split()[1]) - 1
                
                if window_idx < len(metrics):
                    window_data = metrics[window_idx]
                    show_window_details(window_idx, window_data)
        
        tree.bind("<<TreeviewSelect>>", on_window_select)
        
        # Add label with instructions
        ttk.Label(frame_windows, text="Click on a window to see detailed metrics and visualizations", 
                 font=("Arial", 9, "italic")).pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
    except Exception as e:
        logging.error(f"Error creating windows detail: {e}")
        logging.error(traceback.format_exc())
        ttk.Label(frame_windows, text=f"Error creating windows detail: {e}", 
                 wraplength=500).pack(expand=True)
    
    # Comparison tab
    frame_comparison = frames['comparison']
    try:
        # Create a frame for selecting windows to compare
        selection_frame = ttk.Frame(frame_comparison)
        selection_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(selection_frame, text="Select windows to compare:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Create variables to store selected windows
        window1_var = tk.StringVar()
        window2_var = tk.StringVar()
        
        # Create dropdown menus
        window_options = [f"Window {i+1}" for i in range(len(metrics))]
        
        # If we have at least 2 windows, set default selections to best and worst
        if len(metrics) >= 2:
            best_index = np.argmin([m['smape'] for m in metrics])
            worst_index = np.argmax([m['smape'] for m in metrics])
            window1_var.set(window_options[best_index])
            window2_var.set(window_options[worst_index])
        elif len(metrics) == 1:
            window1_var.set(window_options[0])
            window2_var.set(window_options[0])
        
        dropdown1 = ttk.Combobox(selection_frame, textvariable=window1_var, values=window_options, state="readonly")
        dropdown1.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(selection_frame, text="vs").pack(side=tk.LEFT, padx=5)
        
        dropdown2 = ttk.Combobox(selection_frame, textvariable=window2_var, values=window_options, state="readonly")
        dropdown2.pack(side=tk.LEFT, padx=5)
        
        # Create a frame for the comparison visualization
        comparison_viz_frame = ttk.Frame(frame_comparison)
        comparison_viz_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Function to update comparison visualization
        def update_comparison():
            for widget in comparison_viz_frame.winfo_children():
                widget.destroy()
            
            window1_idx = int(window1_var.get().split()[1]) - 1
            window2_idx = int(window2_var.get().split()[1]) - 1
            
            window1_data = metrics[window1_idx]
            window2_data = metrics[window2_idx]
            
            # Create figure for comparison
            fig = plt.Figure(figsize=(8, 6))
            
            # Create prediction vs actual subplot
            ax1 = fig.add_subplot(211)
            
            # Check if we have prediction and actual data
            has_predictions = ('prediction' in window1_data and 'actual' in window1_data and
                              'prediction' in window2_data and 'actual' in window2_data)
            
            if has_predictions:
                # Get prediction and actual data
                pred1 = np.array(window1_data['prediction'])
                actual1 = np.array(window1_data['actual'])
                pred2 = np.array(window2_data['prediction'])
                actual2 = np.array(window2_data['actual'])
                
                # Normalize to percentage change from first point for easier comparison
                def normalize_series(series):
                    if len(series) > 0:
                        return 100 * (series / series[0] - 1)
                    return series
                
                # Plot data
                ax1.plot(normalize_series(actual1), 'k-', label=f'Window {window1_idx+1} Actual')
                ax1.plot(normalize_series(pred1), 'b-', label=f'Window {window1_idx+1} Prediction')
                ax1.plot(normalize_series(actual2), 'k--', label=f'Window {window2_idx+1} Actual')
                ax1.plot(normalize_series(pred2), 'r-', label=f'Window {window2_idx+1} Prediction')
                
                ax1.set_title("Prediction vs Actual (% Change)")
                ax1.set_xlabel("Days")
                ax1.set_ylabel("% Change from Start")
                ax1.legend()
                ax1.grid(True)
            else:
                ax1.text(0.5, 0.5, "Prediction and actual data not available for comparison",
                        ha='center', va='center', transform=ax1.transAxes)
            
            # Create error comparison subplot
            ax2 = fig.add_subplot(212)
            
            # Comparison table of metrics
            metrics_table = [
                ["Metric", f"Window {window1_idx+1}", f"Window {window2_idx+1}", "Difference"],
                ["SMAPE (%)", f"{window1_data['smape']:.2f}", f"{window2_data['smape']:.2f}", 
                 f"{window1_data['smape'] - window2_data['smape']:.2f}"],
                ["Start Index", f"{window1_data['start_idx']}", f"{window2_data['start_idx']}", ""],
                ["End Index", f"{window1_data['end_idx']}", f"{window2_data['end_idx']}", ""]
            ]
            
            # Create table
            ax2.axis('tight')
            ax2.axis('off')
            table = ax2.table(cellText=metrics_table, loc='center', cellLoc='center', edges='horizontal')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Color code the SMAPE difference cell
            smape_diff = window1_data['smape'] - window2_data['smape']
            if smape_diff > 0:
                table[(1, 3)].set_facecolor('#ffcccc')  # Red for worse performance
            elif smape_diff < 0:
                table[(1, 3)].set_facecolor('#ccffcc')  # Green for better performance
            
            fig.tight_layout()
            
            # Display the figure
            canvas = FigureCanvasTkAgg(fig, master=comparison_viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            NavigationToolbar2Tk(canvas, comparison_viz_frame)
        
        # Button to update comparison
        ttk.Button(selection_frame, text="Compare", command=update_comparison).pack(side=tk.LEFT, padx=20)
        
        # Initialize the comparison
        if len(metrics) >= 1:
            update_comparison()
        else:
            ttk.Label(comparison_viz_frame, text="No windows available for comparison").pack(expand=True)
    
    except Exception as e:
        logging.error(f"Error creating comparison tab: {e}")
        logging.error(traceback.format_exc())
        ttk.Label(frame_comparison, text=f"Error creating comparison tab: {e}", 
                 wraplength=500).pack(expand=True)
    
    # Function to show window details in a new window
    def show_window_details(window_idx, window_data):
        detail_window = tk.Toplevel(root)
        detail_window.title(f"Window {window_idx+1} Details")
        detail_window.geometry("800x600")
        
        # Add window data
        ttk.Label(detail_window, text=f"Window {window_idx+1} Details", 
                 font=("Arial", 12, "bold")).pack(padx=10, pady=5)
        
        # Create basic info frame
        info_frame = ttk.LabelFrame(detail_window, text="Window Information")
        info_frame.pack(fill="x", padx=10, pady=5)
        
        info_grid = ttk.Frame(info_frame)
        info_grid.pack(fill="x", padx=10, pady=5)
        
        # Window metadata
        ttk.Label(info_grid, text="Start Index:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(info_grid, text=str(window_data['start_idx'])).grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(info_grid, text="End Index:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        ttk.Label(info_grid, text=str(window_data['end_idx'])).grid(row=0, column=3, sticky="w", padx=5, pady=2)
        
        ttk.Label(info_grid, text="SMAPE:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(info_grid, text=f"{window_data['smape']:.2f}%").grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        # Calculate window rank
        smape_values = [m['smape'] for m in metrics]
        rank = sorted(range(len(smape_values)), key=lambda k: smape_values[k]).index(window_idx) + 1
        
        ttk.Label(info_grid, text="Rank:").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        ttk.Label(info_grid, text=f"{rank} of {len(metrics)}").grid(row=1, column=3, sticky="w", padx=5, pady=2)
        
        # Check if we have prediction and actual data
        has_data = 'prediction' in window_data and 'actual' in window_data
        
        if has_data:
            pred = window_data['prediction']
            actual = window_data['actual']
            
            # Create visualization
            fig = plt.Figure(figsize=(8, 6))
            
            # Prediction vs Actual
            ax1 = fig.add_subplot(211)
            ax1.plot(actual, 'k-', label='Actual')
            ax1.plot(pred, 'b-', label='Prediction')
            ax1.set_title("Prediction vs Actual")
            ax1.set_xlabel("Days")
            ax1.set_ylabel("Scaled Price")
            ax1.legend()
            ax1.grid(True)
            
            # Error over time
            ax2 = fig.add_subplot(212)
            if len(pred) == len(actual):
                error = np.array(pred) - np.array(actual)
                ax2.plot(error, 'r-')
                ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax2.set_title("Prediction Error")
                ax2.set_xlabel("Days")
                ax2.set_ylabel("Error (Prediction - Actual)")
                ax2.grid(True)
            else:
                ax2.text(0.5, 0.5, "Error cannot be calculated (length mismatch)",
                        ha='center', va='center', transform=ax2.transAxes)
            
            fig.tight_layout()
            
            # Display the figure
            canvas = FigureCanvasTkAgg(fig, master=detail_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            NavigationToolbar2Tk(canvas, detail_window)
        else:
            ttk.Label(detail_window, text="Detailed prediction and actual data not available").pack(expand=True)
    
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    return root

def update_plot_gui_for_rolling_window(plot_gui_module):
    """
    Patch the plot_gui module to support rolling window visualization.
    
    Args:
        plot_gui_module: The plot_gui module to patch
    """
    # Import necessary functions
    from types import MethodType
    
    # Save reference to original create_plot_gui_with_data function
    original_create_plot_gui_with_data = plot_gui_module.create_plot_gui_with_data
    
    # Define the patched function
    def patched_create_plot_gui_with_data(app_config, plot_data):
        """
        Updated version of create_plot_gui_with_data that adds support for rolling window results
        
        Args:
            app_config: Application configuration
            plot_data: Plot data tuple from forecast module
        """
        import matplotlib.pyplot as plt
        import tkinter as tk
        from tkinter import ttk, messagebox
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        import logging
        import traceback
        
        logging.debug(f"Received plot_data: {type(plot_data)}, length: {len(plot_data) if plot_data else 'None'}")

        if not isinstance(plot_data, tuple):
            logging.error(f"plot_data is not a tuple. Received type: {type(plot_data)}")
            messagebox.showwarning("Error", "Unexpected data format received. Check logs.")
            return

        if len(plot_data) < 11:
            logging.error(f"Expected plot_data of length 11+, but received {len(plot_data)}")
            messagebox.showwarning("Error", "Incomplete simulation results. Please rerun.")
            return

        # Extract components from plot_data
        (df_train_scaled, date_ori, close_scaler, predictions_mean, predictions_std, results_array, 
         smape_scores, top_indices, df_train_raw, df_test_raw, diagnostics) = plot_data[:11]
        
        # Extract feature_importance if available
        feature_importance = plot_data[11] if len(plot_data) > 11 else None

        logging.debug(f"Data available: df_train_scaled={df_train_scaled.shape if df_train_scaled is not None else 'None'}, "
                      f"predictions_mean={predictions_mean.shape if predictions_mean is not None else 'None'}, "
                      f"diagnostics available={diagnostics is not None}, feature_importance available={feature_importance is not None}")

        plt.close('all')

        root = tk.Toplevel()
        root.title("Stock Prediction Plot Viewer")
        root.geometry("1200x800")

        notebook = ttk.Notebook(root)
        notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # Detect if we have rolling window results
        has_rolling_window = (diagnostics is not None and 
                             'window_metrics' in diagnostics and 
                             len(diagnostics['window_metrics']) > 0)

        # Create frames based on available results
        frames = {
            'forecast': ttk.Frame(notebook),
            'candlestick': ttk.Frame(notebook),
            'diagnostics': ttk.Frame(notebook),
            'feature': ttk.Frame(notebook)
        }

        # Add rolling window frame if needed
        if has_rolling_window:
            frames['rolling_window'] = ttk.Frame(notebook)

        for name, frame in frames.items():
            notebook.add(frame, text=name.capitalize().replace('_', ' '))

        # Create standard visualizations with existing code
        # [Forecast tab, Candlestick tab, Diagnostics tab, Feature tab]
        
        # Create rolling window validation results tab if available
        if has_rolling_window:
            frame_rolling_window = frames['rolling_window']
            try:
                logging.info("Generating Rolling Window Validation results")
                
                # Add a button to open the detailed rolling window view
                btn_frame = ttk.Frame(frame_rolling_window)
                btn_frame.pack(fill="x", padx=10, pady=10)
                
                ttk.Label(
                    btn_frame, 
                    text="Rolling window validation was used to train and evaluate this model.",
                    font=("Arial", 10, "bold")
                ).pack(side="left", padx=10)
                
                def open_rolling_window_detail():
                    create_rolling_window_plot_gui(app_config, diagnostics)
                
                ttk.Button(
                    btn_frame,
                    text="View Detailed Rolling Window Analysis",
                    command=open_rolling_window_detail
                ).pack(side="right", padx=10)
                
                # Show a preview of the rolling window performance
                fig_rolling = plot_rolling_window_performance(app_config, diagnostics)
                canvas_rolling = FigureCanvasTkAgg(fig_rolling, master=frame_rolling_window)
                canvas_rolling.draw()
                canvas_rolling.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                toolbar_rolling = NavigationToolbar2Tk(canvas_rolling, frame_rolling_window)
                toolbar_rolling.update()
                
            except Exception as e:
                logging.error(f"Error displaying Rolling Window results: {e}")
                logging.error(traceback.format_exc())
                ttk.Label(frame_rolling_window, text=f"Error displaying Rolling Window results: {e}").pack(expand=True)

        root.protocol("WM_DELETE_WINDOW", root.destroy)
        
        # Call the original function to complete the standard visualization
        original_create_plot_gui_with_data(app_config, plot_data)
    
    # Replace the original function with our patched version
    plot_gui_module.create_plot_gui_with_data = patched_create_plot_gui_with_data