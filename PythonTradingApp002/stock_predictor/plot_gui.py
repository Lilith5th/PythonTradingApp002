import tkinter as tk
from tkinter import ttk, messagebox
from stock_predictor import forecast_module
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import logging
import traceback

# Import for Plotly
import plotly.graph_objects as go
from plotly.offline import plot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_plot_gui_with_data(app_config, plot_data):
    """Creates a GUI window displaying forecast results, candlestick charts, diagnostics, and feature importance."""
    logging.debug(f"Received plot_data: {type(plot_data)}, length: {len(plot_data) if plot_data else 'None'}")

    if not isinstance(plot_data, tuple):
        logging.error(f"plot_data is not a tuple. Received type: {type(plot_data)}")
        messagebox.showwarning("Error", "Unexpected data format received. Check logs.")
        return

    if len(plot_data) != 12:
        logging.error(f"Expected plot_data of length 12, but received {len(plot_data)}")
        messagebox.showwarning("Error", "Incomplete simulation results. Please rerun.")
        return

    (df_train_scaled, date_ori, close_scaler, predictions_mean, predictions_std, results_array, 
     smape_scores, top_indices, df_train_raw, df_test_raw, diagnostics, feature_importance) = plot_data

    logging.debug(f"Data available: df_train_scaled={df_train_scaled.shape if df_train_scaled is not None else 'None'}, "
                  f"predictions_mean={predictions_mean.shape if predictions_mean is not None else 'None'}, "
                  f"diagnostics available={diagnostics is not None}, feature_importance available={feature_importance is not None}")

    plt.close('all')

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
        
        # Option 1: Open in browser
        # plot(plotly_figure, filename='candlestick.html', auto_open=True)
        # ttk.Label(frame_candlestick, text="Candlestick chart opened in browser").pack(expand=True)
        
        # Option 2: Embed in Tkinter
        html_bytes = plotly_figure.to_html(include_plotlyjs='cdn').encode()
        
        from tkinter import scrolledtext
        import webview  # You may need to install pywebview: pip install pywebview
        
        def show_plotly_window():
            webview.create_window("Candlestick Chart", html=html_bytes.decode())
            webview.start()
            
        open_button = ttk.Button(frame_candlestick, text="Open Candlestick Chart", command=show_plotly_window)
        open_button.pack(pady=20)
        
        # Display preview message
        preview_label = ttk.Label(frame_candlestick, 
                                text="Click the button above to open the interactive candlestick chart in a separate window")
        preview_label.pack(pady=10)
        
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

    root.protocol("WM_DELETE_WINDOW", root.destroy)

def create_parallel_plot_gui(app_config, results):
    """Creates a GUI window to display multiple forecast scenarios with confidence intervals."""
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
    
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

if __name__ == '__main__':
    app_config = forecast_module.AppConfig()
    create_plot_gui_with_data(app_config, None)