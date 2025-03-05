import tkinter as tk
from tkinter import ttk, messagebox
from stock_predictor import forecast_module
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import logging
import traceback
import threading

# Import for Plotly
try:
    import plotly.graph_objects as go
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_in_main_thread(func, *args, **kwargs):
    """
    Ensures a function runs in the main thread.
    If called from a non-main thread, creates a temporary Tk instance to run in main thread.
    
    Args:
        func: Function to run
        *args, **kwargs: Arguments to the function
        
    Returns:
        The result of calling func
    """
    if threading.current_thread() is threading.main_thread():
        # Already in main thread, run directly
        return func(*args, **kwargs)
    else:
        # We're in a non-main thread, need to use main thread
        logging.info("Function called from non-main thread, redirecting to main thread")
        temp_root = tk.Tk()
        temp_root.withdraw()  # Hide the window
        
        # Use a mutable object to store the result
        result_container = [None]
        exception_container = [None]
        completed = threading.Event()
        
        def wrapper():
            try:
                result_container[0] = func(*args, **kwargs)
            except Exception as e:
                exception_container[0] = e
                logging.error(f"Error in main thread execution: {e}")
                logging.error(traceback.format_exc())
            finally:
                completed.set()
                temp_root.quit()
        
        temp_root.after(0, wrapper)
        
        # Run the main loop until the function completes
        temp_root.mainloop()
        
        # Check for exceptions
        if exception_container[0] is not None:
            raise exception_container[0]
            
        return result_container[0]

def safe_close_plots():
    """Safely close all matplotlib plots"""
    try:
        if threading.current_thread() is threading.main_thread():
            plt.close('all')
        else:
            logging.warning("Skipping plt.close('all') in non-main thread")
    except Exception as e:
        logging.warning(f"Error closing plots: {e}")

def create_plot_gui_with_data(app_config, plot_data):
    """Creates a GUI window displaying forecast results, candlestick charts, diagnostics, and feature importance."""
    # Ensure we're running in the main thread
    if threading.current_thread() is not threading.main_thread():
        return run_in_main_thread(create_plot_gui_with_data, app_config, plot_data)
    
    logging.debug(f"Received plot_data: {type(plot_data)}, length: {len(plot_data) if plot_data else 'None'}")

    if not isinstance(plot_data, tuple):
        logging.error(f"plot_data is not a tuple. Received type: {type(plot_data)}")
        messagebox.showwarning("Error", "Unexpected data format received. Check logs.")
        return

    if len(plot_data) < 11:
        logging.error(f"Expected plot_data of length at least 11, but received {len(plot_data)}")
        messagebox.showwarning("Error", "Incomplete simulation results. Please rerun.")
        return

    (df_train_scaled, date_ori, close_scaler, predictions_mean, predictions_std, results_array, 
     smape_scores, top_indices, df_train_raw, df_test_raw, diagnostics) = plot_data[:11]
    
    # Extract feature_importance if available
    feature_importance = plot_data[11] if len(plot_data) > 11 else None

    logging.debug(f"Data available: df_train_scaled={df_train_scaled.shape if df_train_scaled is not None else 'None'}, "
                  f"predictions_mean={predictions_mean.shape if predictions_mean is not None else 'None'}, "
                  f"diagnostics available={diagnostics is not None}, feature_importance available={feature_importance is not None}")

    # Safely close existing plots
    safe_close_plots()

    # Create toplevel window instead of Tk() to avoid multiple root windows
    root = tk.Toplevel()
    root.title("Stock Prediction Plot Viewer")
    root.geometry("1200x800")

    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both", padx=10, pady=10)

    frames = {
        'forecast': ttk.Frame(notebook),
        'candlestick': ttk.Frame(notebook),
        'diagnostics': ttk.Frame(notebook),
        'feature': ttk.Frame(notebook)
    }

    for name, frame in frames.items():
        notebook.add(frame, text=name.capitalize())

    # Forecast tab with matplotlib
    frame_forecast = frames['forecast']
    try:
        logging.info("Generating Forecast Results plot")
        fig_forecast = forecast_module.plot_forecast_results(
            app_config, df_train_scaled, date_ori, close_scaler, predictions_mean, predictions_std,
            results_array, smape_scores, top_indices, df_train_raw, df_test_raw
        )
        canvas_forecast = FigureCanvasTkAgg(fig_forecast, master=frame_forecast)
        canvas_forecast.draw()
        canvas_forecast.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar_forecast = NavigationToolbar2Tk(canvas_forecast, frame_forecast)
        toolbar_forecast.update()
    except Exception as e:
        logging.error(f"Error plotting Forecast Results: {e}\n{traceback.format_exc()}")
        ttk.Label(frame_forecast, text=f"Error: {e}").pack(expand=True)

    # Candlestick tab with Plotly (use a browser window or embedded HTML)
    frame_candlestick = frames['candlestick']
    try:
        logging.info("Generating Candlestick plot")
        plotly_figure = forecast_module.plot_forecast_results_candlestick(
            app_config, df_train_scaled, date_ori, close_scaler, predictions_mean, predictions_std,
            results_array, smape_scores, top_indices, df_train_raw, df_test_raw
        )
        
        if PLOTLY_AVAILABLE:
            # Option 1: Create button to open in browser
            html_bytes = plotly_figure.to_html(include_plotlyjs='cdn').encode()
            
            # Check if webview is available
            try:
                import webview
                WEBVIEW_AVAILABLE = True
            except ImportError:
                WEBVIEW_AVAILABLE = False
                
            if WEBVIEW_AVAILABLE:
                def show_plotly_window():
                    try:
                        webview.create_window("Candlestick Chart", html=html_bytes.decode())
                        webview.start()
                    except Exception as e:
                        logging.error(f"Error showing webview: {e}")
                        messagebox.showerror("Error", f"Could not display interactive chart: {e}")
                
                open_button = ttk.Button(frame_candlestick, text="Open Candlestick Chart", command=show_plotly_window)
                open_button.pack(pady=20)
                
                # Display preview message
                preview_label = ttk.Label(frame_candlestick, 
                                        text="Click the button above to open the interactive candlestick chart in a separate window")
                preview_label.pack(pady=10)
            else:
                # Option 2: Save to temp file and open browser
                import tempfile
                import webbrowser
                import os
                
                def open_in_browser():
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
                            f.write(html_bytes.decode())
                            temp_path = f.name
                        webbrowser.open('file://' + os.path.abspath(temp_path))
                    except Exception as e:
                        logging.error(f"Error opening browser: {e}")
                        messagebox.showerror("Error", f"Could not open browser: {e}")
                
                open_button = ttk.Button(frame_candlestick, text="Open Candlestick Chart in Browser", command=open_in_browser)
                open_button.pack(pady=20)
                
                ttk.Label(frame_candlestick, 
                        text="Click the button above to open the interactive candlestick chart in your web browser").pack(pady=10)
        else:
            ttk.Label(frame_candlestick, 
                   text="Plotly not available. Install plotly package for interactive charts.").pack(expand=True)
            
    except Exception as e:
        logging.error(f"Error plotting Candlestick: {e}\n{traceback.format_exc()}")
        ttk.Label(frame_candlestick, text=f"Error: {e}").pack(expand=True)

    # Diagnostics tab with matplotlib
    frame_diagnostics = frames['diagnostics']
    if app_config.plot.show_diagnostics:
        try:
            logging.info("Generating Diagnostics plot")
            if diagnostics:
                fig_diagnostics = forecast_module.plot_diagnostics(app_config, diagnostics)
                canvas_diagnostics = FigureCanvasTkAgg(fig_diagnostics, master=frame_diagnostics)
                canvas_diagnostics.draw()
                canvas_diagnostics.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                toolbar_diagnostics = NavigationToolbar2Tk(canvas_diagnostics, frame_diagnostics)
                toolbar_diagnostics.update()
            else:
                ttk.Label(frame_diagnostics, text="No diagnostics data available").pack(expand=True)
        except Exception as e:
            logging.error(f"Error plotting Diagnostics: {e}\n{traceback.format_exc()}")
            ttk.Label(frame_diagnostics, text=f"Error: {e}").pack(expand=True)
    else:
        ttk.Label(frame_diagnostics, text="Diagnostics disabled").pack(expand=True)

    # Feature importance tab with matplotlib
    frame_feature = frames['feature']
    if app_config.plot.show_diagnostics:
        try:
            logging.info("Generating Feature Importance plot")
            if feature_importance:
                fig_feature = forecast_module.plot_feature_importance(app_config, feature_importance)
                canvas_feature = FigureCanvasTkAgg(fig_feature, master=frame_feature)
                canvas_feature.draw()
                canvas_feature.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                toolbar_feature = NavigationToolbar2Tk(canvas_feature, frame_feature)
                toolbar_feature.update()
            else:
                ttk.Label(frame_feature, text="No feature importance data available").pack(expand=True)
        except Exception as e:
            logging.error(f"Error plotting Feature Importance: {e}\n{traceback.format_exc()}")
            ttk.Label(frame_feature, text=f"Error: {e}").pack(expand=True)
    else:
        ttk.Label(frame_feature, text="Feature Importance disabled").pack(expand=True)

    # Handle window close
    def on_window_close():
        try:
            # Clean up resources
            safe_close_plots()
            root.destroy()
        except Exception as e:
            logging.error(f"Error closing window: {e}")
            # Ensure window gets destroyed even if there's an error
            try:
                root.destroy()
            except:
                pass
            
    root.protocol("WM_DELETE_WINDOW", on_window_close)
    
    return root

def create_parallel_plot_gui(app_config, results):
    """Creates a GUI window to display multiple forecast scenarios with confidence intervals."""
    # Ensure we're running in the main thread
    if threading.current_thread() is not threading.main_thread():
        return run_in_main_thread(create_parallel_plot_gui, app_config, results)
    
    # Safely close existing plots
    safe_close_plots()
    
    root = tk.Toplevel()
    root.title("Parallel Forecast Results")
    root.geometry("1200x800")
    
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both", padx=10, pady=10)
    
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Parallel Scenarios")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()
    
    for i, result in enumerate(results):
        if len(result) < 13:
            logging.warning(f"Result {i} has insufficient data: {len(result)} elements")
            continue
        dates = result[1]  # date_ori
        pred_mean = result[3]  # predictions_mean
        ci = result[12]  # confidence_intervals
        
        ax.plot(dates[-len(pred_mean):], pred_mean, label=f'Scenario {i+1}', alpha=0.7)
        if app_config.plot.plot_confidence_interval and ci:
            ax.fill_between(dates[-len(pred_mean):], ci['lower'], ci['upper'], 
                           alpha=0.2, label=f'Scenario {i+1} CI')
    
    ax.legend()
    ax.set_title("Parallel Forecast Scenarios")
    plt.xticks(rotation=45)
    canvas.draw()
    
    def on_window_close():
        try:
            # Clean up resources
            safe_close_plots()
            root.destroy()
        except Exception as e:
            logging.error(f"Error closing window: {e}")
            # Ensure window gets destroyed even if there's an error
            try:
                root.destroy()
            except:
                pass
            
    root.protocol("WM_DELETE_WINDOW", on_window_close)
    
    return root

def create_model_comparison_gui(app_config, model_results):
    """Creates a GUI window to display comparison of different model architectures."""
    # Ensure we're running in the main thread
    if threading.current_thread() is not threading.main_thread():
        return run_in_main_thread(create_model_comparison_gui, app_config, model_results)
    
    # Safely close existing plots
    safe_close_plots()
    
    root = tk.Toplevel()
    root.title("Model Architecture Comparison")
    root.geometry("1200x800")
    
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Create tabs for different visualization types
    frames = {
        'performance': ttk.Frame(notebook),
        'metrics': ttk.Frame(notebook)
    }
    
    for name, frame in frames.items():
        notebook.add(frame, text=name.capitalize())
    
    # Performance plot
    try:
        frame_perf = frames['performance']
        fig, ax = plt.subplots(figsize=(10, 6))
        canvas = FigureCanvasTkAgg(fig, master=frame_perf)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, frame_perf)
        toolbar.update()
        
        for model_name, result in model_results:
            if len(result) < 9:
                logging.warning(f"Model {model_name} has insufficient data")
                continue
                
            dates = result[1]  # date_ori
            pred_mean = result[3]  # predictions_mean
            
            if pred_mean is not None and len(pred_mean) > 0:
                ax.plot(dates[-len(pred_mean):], pred_mean, label=f'{model_name.upper()}', alpha=0.8)
        
        ax.legend()
        ax.set_title("Model Architecture Comparison")
        plt.xticks(rotation=45)
        canvas.draw()
        
    except Exception as e:
        logging.error(f"Error creating performance plot: {e}\n{traceback.format_exc()}")
        ttk.Label(frames['performance'], text=f"Error creating plot: {e}").pack(expand=True)
    
    # Metrics comparison
    try:
        frame_metrics = frames['metrics']
        
        # Create a treeview for the metrics
        columns = ("model", "smape", "mae", "mse", "training_time")
        tree = ttk.Treeview(frame_metrics, columns=columns, show="headings")
        
        # Define column headings
        tree.heading("model", text="Model")
        tree.heading("smape", text="SMAPE (%)")
        tree.heading("mae", text="MAE")
        tree.heading("mse", text="MSE")
        tree.heading("training_time", text="Training Time (s)")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame_metrics, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack tree and scrollbar
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Insert data
        for model_name, result in model_results:
            if len(result) < 7:
                continue
                
            smape_scores = result[6] if len(result) > 6 else []
            avg_smape = np.mean(smape_scores) if len(smape_scores) > 0 else 0
            
            # Get training time if available
            training_time = "N/A"
            if len(result) > 10 and isinstance(result[10], dict) and 'training_time' in result[10]:
                training_time = f"{result[10]['training_time']:.2f}"
            
            tree.insert("", "end", values=(
                model_name.upper(),
                f"{avg_smape:.2f}",
                "N/A",  # MAE not available
                "N/A",  # MSE not available
                training_time
            ))
            
    except Exception as e:
        logging.error(f"Error creating metrics comparison: {e}\n{traceback.format_exc()}")
        ttk.Label(frames['metrics'], text=f"Error creating metrics comparison: {e}").pack(expand=True)
    
    def on_window_close():
        try:
            # Clean up resources
            safe_close_plots()
            root.destroy()
        except Exception as e:
            logging.error(f"Error closing window: {e}")
            # Ensure window gets destroyed even if there's an error
            try:
                root.destroy()
            except:
                pass
            
    root.protocol("WM_DELETE_WINDOW", on_window_close)
    
    return root

def create_backtest_plot_gui(app_config, backtest_data):
    """Creates a GUI window to display backtesting results."""
    # Ensure we're running in the main thread
    if threading.current_thread() is not threading.main_thread():
        return run_in_main_thread(create_backtest_plot_gui, app_config, backtest_data)
    
    # Safely close existing plots
    safe_close_plots()
    
    root = tk.Toplevel()
    root.title("Backtesting Results")
    root.geometry("1200x800")
    
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both", padx=10, pady=10)
    
    # Create tabs for different visualization types
    frames = {
        'performance': ttk.Frame(notebook),
        'metrics': ttk.Frame(notebook),
        'validation': ttk.Frame(notebook)
    }
    
    for name, frame in frames.items():
        notebook.add(frame, text=name.capitalize())
    
    # Extract data from backtest_data
    try:
        # Performance visualization
        frame_perf = frames['performance']
        fig, ax = plt.subplots(figsize=(10, 6))
        canvas = FigureCanvasTkAgg(fig, master=frame_perf)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, frame_perf)
        toolbar.update()
        
        # Create simple placeholder content
        ax.text(0.5, 0.5, "Backtesting visualization would be shown here", 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        canvas.draw()
        
    except Exception as e:
        logging.error(f"Error creating backtest visualization: {e}\n{traceback.format_exc()}")
        ttk.Label(frames['performance'], text=f"Error creating visualization: {e}").pack(expand=True)
    
    def on_window_close():
        try:
            # Clean up resources
            safe_close_plots()
            root.destroy()
        except Exception as e:
            logging.error(f"Error closing window: {e}")
            # Ensure window gets destroyed even if there's an error
            try:
                root.destroy()
            except:
                pass
            
    root.protocol("WM_DELETE_WINDOW", on_window_close)
    
    return root