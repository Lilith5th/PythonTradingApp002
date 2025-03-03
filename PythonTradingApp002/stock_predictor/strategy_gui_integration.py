"""
Strategy GUI integration module for the stock prediction application.
Provides visualization and reporting tools for strategy backtesting results.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np
import logging
import threading
import traceback

# Try to import plotly for interactive charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def run_strategy_backtest(app_config, data_handler):
    """
    Run a strategy backtest and display results
    
    Args:
        app_config: Application configuration
        data_handler: DataHandler with historical data
    """
    try:
        # Import the strategy module
        from stock_predictor.strategy_implementation import StrategyBacktester
        
        # Check if we have the necessary data
        if data_handler is None or data_handler.df_train_raw is None or data_handler.df_train_raw.empty:
            messagebox.showerror("Error", "No data available for backtesting. Please load data first.")
            return
        
        # Create a progress window
        progress_window = tk.Toplevel()
        progress_window.title("Running Strategy Backtest")
        progress_window.geometry("300x150")
        progress_window.resizable(False, False)
        
        # Center the window
        progress_window.update_idletasks()
        width = progress_window.winfo_width()
        height = progress_window.winfo_height()
        x = (progress_window.winfo_screenwidth() // 2) - (width // 2)
        y = (progress_window.winfo_screenheight() // 2) - (height // 2)
        progress_window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Add a progress label
        progress_label = ttk.Label(progress_window, text="Running backtest...", font=("Arial", 12))
        progress_label.pack(pady=20)
        
        # Add a progress bar
        progress_bar = ttk.Progressbar(progress_window, mode="indeterminate")
        progress_bar.pack(fill="x", padx=20, pady=10)
        progress_bar.start(10)
        
        # Function to run backtest in a separate thread
        def run_backtest_thread():
            try:
                # Initialize the backtester
                backtester = StrategyBacktester(app_config.strategy, data_handler)
                
                # Run the backtest
                metrics, results = backtester.backtest()
                
                # Close progress window
                progress_window.destroy()
                
                # Display results
                display_strategy_results(app_config, backtester, metrics, results)
                
            except Exception as e:
                logging.error(f"Error in strategy backtest: {e}")
                logging.error(traceback.format_exc())
                progress_window.destroy()
                messagebox.showerror("Backtest Error", f"Error running backtest: {str(e)}")
        
        # Start the backtest thread
        threading.Thread(target=run_backtest_thread, daemon=True).start()
        
    except ImportError as e:
        messagebox.showerror("Module Error", f"Required module not found: {str(e)}")
    except Exception as e:
        logging.error(f"Error in strategy backtest: {e}")
        logging.error(traceback.format_exc())
        messagebox.showerror("Backtest Error", f"Error running backtest: {str(e)}")

def display_strategy_results(app_config, backtester, metrics, results):
    """
    Display strategy backtest results
    
    Args:
        app_config: Application configuration
        backtester: StrategyBacktester instance
        metrics: Dictionary of performance metrics
        results: DataFrame of trading results
    """
    # Create results window
    results_window = tk.Toplevel()
    results_window.title(f"Strategy Results: {app_config.strategy.strategy_type}")
    results_window.geometry("1200x800")
    
    # Create notebook for tabs
    notebook = ttk.Notebook(results_window)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Create frames for tabs
    summary_frame = ttk.Frame(notebook)
    equity_frame = ttk.Frame(notebook)
    trades_frame = ttk.Frame(notebook)
    interactive_frame = ttk.Frame(notebook) if PLOTLY_AVAILABLE else None
    
    # Add tabs to notebook
    notebook.add(summary_frame, text="Summary")
    notebook.add(equity_frame, text="Equity Curve")
    notebook.add(trades_frame, text="Trades")
    if PLOTLY_AVAILABLE:
        notebook.add(interactive_frame, text="Interactive Charts")
    
    # Create summary tab
    create_summary_tab(summary_frame, app_config, backtester, metrics)
    
    # Create equity curve tab
    create_equity_tab(equity_frame, backtester, results)
    
    # Create trades tab
    create_trades_tab(trades_frame, results)
    
    # Create interactive charts tab if plotly is available
    if PLOTLY_AVAILABLE:
        create_interactive_tab(interactive_frame, backtester)

def create_summary_tab(parent, app_config, backtester, metrics):
    """
    Create the summary tab
    
    Args:
        parent: Parent frame
        app_config: Application configuration
        backtester: StrategyBacktester instance
        metrics: Dictionary of performance metrics
    """
    # Left panel - Strategy details and metrics
    left_frame = ttk.LabelFrame(parent, text="Strategy Performance")
    left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    
    # Strategy details
    details_frame = ttk.Frame(left_frame)
    details_frame.pack(fill="x", padx=10, pady=10)
    
    ttk.Label(details_frame, text="Strategy:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(details_frame, text=app_config.strategy.strategy_type).grid(row=0, column=1, sticky="w", padx=5, pady=2)
    
    ttk.Label(details_frame, text="Initial Capital:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(details_frame, text=f"${app_config.strategy.initial_capital:.2f}").grid(row=1, column=1, sticky="w", padx=5, pady=2)
    
    ttk.Label(details_frame, text="Final Equity:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(details_frame, text=f"${backtester.results['equity'].iloc[-1]:.2f}").grid(row=2, column=1, sticky="w", padx=5, pady=2)
    
    # Performance metrics
    metrics_frame = ttk.LabelFrame(left_frame, text="Performance Metrics")
    metrics_frame.pack(fill="x", padx=10, pady=10)
    
    row = 0
    for metric_name, metric_value in [
        ("Total Return", f"{metrics['total_return']:.2f}%"),
        ("Annualized Return", f"{metrics['annualized_return']:.2f}%"),
        ("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"),
        ("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}"),
        ("Maximum Drawdown", f"{metrics['max_drawdown']:.2f}%"),
        ("Win Rate", f"{metrics['win_rate']:.2f}%"),
        ("Profit Factor", f"{metrics['profit_factor']:.2f}"),
        ("Number of Trades", f"{metrics['trades']}"),
        ("Winning Trades", f"{metrics['winning_trades']}"),
        ("Losing Trades", f"{metrics['losing_trades']}")
    ]:
        ttk.Label(metrics_frame, text=metric_name + ":", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(metrics_frame, text=metric_value).grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
    
    # Strategy parameters
    params_frame = ttk.LabelFrame(left_frame, text="Strategy Parameters")
    params_frame.pack(fill="x", padx=10, pady=10)
    
    row = 0
    for param_name, param_value in [
        ("Position Size", f"{app_config.strategy.position_size_pct:.2f}%"),
        ("Take Profit", f"{app_config.strategy.take_profit_pct:.2f}%"),
        ("Stop Loss", f"{app_config.strategy.stop_loss_pct:.2f}%"),
        ("Trailing Stop", f"{app_config.strategy.trailing_stop_pct:.2f}%"),
        ("Max Positions", f"{app_config.strategy.max_positions}"),
        ("Reinvest Profits", "Yes" if app_config.strategy.reinvest_profits else "No")
    ]:
        ttk.Label(params_frame, text=param_name + ":", font=("Arial", 10)).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(params_frame, text=param_value).grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1
    
    # Right panel - Performance visualization
    right_frame = ttk.Frame(parent)
    right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
    
    # Create a basic summary visualization
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Create a pie chart for win/loss ratio
    winning = metrics['winning_trades']
    losing = metrics['losing_trades']
    
    if winning + losing > 0:
        ax.pie(
            [winning, losing],
            labels=['Winning', 'Losing'],
            autopct='%1.1f%%',
            colors=['green', 'red'],
            startangle=90
        )
        ax.set_title('Win/Loss Ratio')
    else:
        ax.text(0.5, 0.5, "No trades available", ha='center', va='center', fontsize=12)
    
    ax.axis('equal')
    
    # Display the figure
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    # Add export button
    def export_results():
        """Export results to CSV and text report"""
        from tkinter import filedialog
        import os
        
        # Ask for directory
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
        
        try:
            # Export results to CSV
            csv_path = os.path.join(export_dir, f"{app_config.strategy.strategy_type}_results.csv")
            backtester.results.to_csv(csv_path)
            
            # Export performance report to text file
            report_path = os.path.join(export_dir, f"{app_config.strategy.strategy_type}_report.txt")
            with open(report_path, 'w') as f:
                f.write(backtester.generate_performance_report())
            
            messagebox.showinfo("Export Successful", f"Results exported to:\n{export_dir}")
            
        except Exception as e:
            logging.error(f"Error exporting results: {e}")
            messagebox.showerror("Export Error", f"Error exporting results: {str(e)}")
    
    # Add export button at the bottom
    export_button = ttk.Button(parent, text="Export Results", command=export_results)
    export_button.pack(side="bottom", pady=10)

def create_equity_tab(parent, backtester, results):
    """
    Create the equity curve tab
    
    Args:
        parent: Parent frame
        backtester: StrategyBacktester instance
        results: DataFrame of trading results
    """
    # Create matplotlib figure
    fig = backtester.plot_performance(plt)
    
    # Display the figure
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    # Add navigation toolbar
    toolbar = NavigationToolbar2Tk(canvas, parent)
    toolbar.update()

def create_trades_tab(parent, results):
    """
    Create the trades tab
    
    Args:
        parent: Parent frame
        results: DataFrame of trading results
    """
    # Extract trades from results
    trades = results[results['trade_type'].isin(['buy', 'sell', 'take_profit', 'stop_loss'])]
    
    # Create a treeview to display trades
    columns = ("date", "type", "price", "shares", "value", "pnl")
    tree = ttk.Treeview(parent, columns=columns, show="headings")
    
    # Define column headings
    tree.heading("date", text="Date")
    tree.heading("type", text="Type")
    tree.heading("price", text="Price")
    tree.heading("shares", text="Shares")
    tree.heading("value", text="Value")
    tree.heading("pnl", text="P&L")
    
    # Set column widths
    tree.column("date", width=150)
    tree.column("type", width=100)
    tree.column("price", width=100)
    tree.column("shares", width=100)
    tree.column("value", width=100)
    tree.column("pnl", width=100)
    
    # Add a scrollbar
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # Pack treeview and scrollbar
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Insert trades data
    for i, row in trades.iterrows():
        try:
            trade_type = row['trade_type'].capitalize()
            trade_price = f"${row['trade_price']:.2f}" if pd.notna(row['trade_price']) else ""
            shares = f"{row['shares']:.4f}" if pd.notna(row['shares']) else ""
            value = f"${row['shares'] * row['price']:.2f}" if pd.notna(row['shares']) and pd.notna(row['price']) else ""
            pnl = f"${row['pnl']:.2f}" if pd.notna(row['pnl']) else ""
            
            # Set row colors based on trade type
            trade_tag = "buy" if trade_type.lower() == "buy" else \
                        "sell" if trade_type.lower() == "sell" else \
                        "profit" if trade_type.lower() == "take_profit" else \
                        "loss" if trade_type.lower() == "stop_loss" else ""
            
            tree.insert("", "end", values=(row.name, trade_type, trade_price, shares, value, pnl), tags=(trade_tag,))
        except Exception as e:
            logging.error(f"Error adding trade row: {e}")
    
    # Configure row tags for colors
    tree.tag_configure("buy", background="#e6ffe6")  # Light green
    tree.tag_configure("sell", background="#ffe6e6")  # Light red
    tree.tag_configure("profit", background="#ccffcc")  # Medium green
    tree.tag_configure("loss", background="#ffcccc")  # Medium red

def create_interactive_tab(parent, backtester):
    """
    Create interactive charts tab using Plotly
    
    Args:
        parent: Parent frame
        backtester: StrategyBacktester instance
    """
    if not PLOTLY_AVAILABLE:
        ttk.Label(parent, text="Plotly is not available. Install plotly package for interactive charts.").pack(expand=True)
        return
    
    # Get plotly figures
    figures = backtester.plot_interactive_performance()
    
    if not figures:
        ttk.Label(parent, text="Error creating interactive charts.").pack(expand=True)
        return
    
    # Create a sub-notebook for different interactive charts
    sub_notebook = ttk.Notebook(parent)
    sub_notebook.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Create frames for sub-tabs
    performance_frame = ttk.Frame(sub_notebook)
    price_frame = ttk.Frame(sub_notebook)
    
    # Add sub-tabs
    sub_notebook.add(performance_frame, text="Performance")
    sub_notebook.add(price_frame, text="Price Chart")
    
    # Convert plotly figures to HTML
    performance_html = figures['performance'].to_html(
        full_html=False, 
        include_plotlyjs='cdn'
    )
    
    price_html = figures['price'].to_html(
        full_html=False, 
        include_plotlyjs='cdn'
    )
    
    # Display HTML using webview if available, otherwise show a message
    try:
        import webview
        
        # Create a function to show the webview
        def show_webview(html, title):
            webview.create_window(title, html=html)
            webview.start()
        
        # Create buttons to open webviews
        ttk.Button(
            performance_frame,
            text="Open Performance Chart in Browser",
            command=lambda: show_webview(performance_html, "Strategy Performance")
        ).pack(pady=20)
        
        ttk.Label(
            performance_frame,
            text="Click the button above to open an interactive performance chart in a separate window.",
            wraplength=400
        ).pack(pady=10)
        
        ttk.Button(
            price_frame,
            text="Open Price Chart in Browser",
            command=lambda: show_webview(price_html, "Price Chart with Signals")
        ).pack(pady=20)
        
        ttk.Label(
            price_frame,
            text="Click the button above to open an interactive price chart with buy/sell signals in a separate window.",
            wraplength=400
        ).pack(pady=10)
        
    except ImportError:
        # If webview is not available, provide instructions to install
        ttk.Label(
            performance_frame,
            text="The pywebview package is required for interactive charts.\n\nInstall it with: pip install pywebview",
            wraplength=400
        ).pack(expand=True)
        
        ttk.Label(
            price_frame,
            text="The pywebview package is required for interactive charts.\n\nInstall it with: pip install pywebview",
            wraplength=400
        ).pack(expand=True)


def run_ml_optimization(app_config, data_handler):
    """
    Run machine learning strategy optimization
    
    Args:
        app_config: Application configuration
        data_handler: DataHandler with historical data
    """
    # Check if ML optimization is enabled
    if not app_config.strategy.enable_ml_optimization:
        messagebox.showinfo("ML Optimization", "Machine learning optimization is not enabled. "
                            "Please enable it in the ML Optimization tab first.")
        return
    
    # Check if we have the necessary data
    if data_handler is None or data_handler.df_train_raw is None or data_handler.df_train_raw.empty:
        messagebox.showerror("Error", "No data available for optimization. Please load data first.")
        return
    
    try:
        # Import the strategy module
        from stock_predictor.strategy_implementation import StrategyBacktester
        
        # Create a progress window
        progress_window = tk.Toplevel()
        progress_window.title("Running ML Optimization")
        progress_window.geometry("400x200")
        progress_window.resizable(False, False)
        
        # Center the window
        progress_window.update_idletasks()
        width = progress_window.winfo_width()
        height = progress_window.winfo_height()
        x = (progress_window.winfo_screenwidth() // 2) - (width // 2)
        y = (progress_window.winfo_screenheight() // 2) - (height // 2)
        progress_window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Add progress information
        progress_label = ttk.Label(progress_window, text="Training ML model...", font=("Arial", 12))
        progress_label.pack(pady=20)
        
        progress_info = tk.StringVar(value="Initializing...")
        info_label = ttk.Label(progress_window, textvariable=progress_info)
        info_label.pack(pady=5)
        
        # Add a progress bar
        progress_bar = ttk.Progressbar(progress_window, mode="indeterminate")
        progress_bar.pack(fill="x", padx=20, pady=10)
        progress_bar.start(10)
        
        # Function to run ML optimization in a separate thread
        def run_optimization_thread():
            try:
                # Update progress
                progress_info.set("Preparing data...")
                
                # Initialize the backtester with ML strategy
                config_copy = app_config.strategy
                config_copy.strategy_type = "ml_optimized"
                
                backtester = StrategyBacktester(config_copy, data_handler)
                
                # Train the ML model
                progress_info.set("Training ML model...")
                df = data_handler.df_train_raw.copy()
                backtester._train_ml_model(df)
                
                # Check if model was created successfully
                if backtester.ml_model is None:
                    progress_window.destroy()
                    messagebox.showerror("Training Error", "Failed to train ML model. Check logs for details.")
                    return
                
                # Run backtest with the trained model
                progress_info.set("Running backtest with trained model...")
                metrics, results = backtester.backtest()
                
                # Close progress window
                progress_window.destroy()
                
                # Display results
                display_strategy_results(app_config, backtester, metrics, results)
                
            except Exception as e:
                logging.error(f"Error in ML optimization: {e}")
                logging.error(traceback.format_exc())
                progress_window.destroy()
                messagebox.showerror("Optimization Error", f"Error running ML optimization: {str(e)}")
        
        # Start the optimization thread
        threading.Thread(target=run_optimization_thread, daemon=True).start()
        
    except ImportError as e:
        messagebox.showerror("Module Error", f"Required module not found: {str(e)}")
    except Exception as e:
        logging.error(f"Error in ML optimization: {e}")
        logging.error(traceback.format_exc())
        messagebox.showerror("Optimization Error", f"Error running ML optimization: {str(e)}")


def run_strategy_comparison(app_config, data_handler):
    """
    Run a comparison of multiple trading strategies
    
    Args:
        app_config: Application configuration
        data_handler: DataHandler with historical data
    """
    # Check if we have the necessary data
    if data_handler is None or data_handler.df_train_raw is None or data_handler.df_train_raw.empty:
        messagebox.showerror("Error", "No data available for comparison. Please load data first.")
        return
    
    try:
        # Import the strategy module
        from stock_predictor.strategy_implementation import StrategyBacktester
        
        # Define strategies to compare
        strategies_to_compare = [
            "buy_and_hold",
            "moving_average_crossover",
            "rsi_based",
            "macd_based",
            "bollinger_bands"
        ]
        
        # Ask user which strategies to compare
        comparison_window = tk.Toplevel()
        comparison_window.title("Strategy Comparison")
        comparison_window.geometry("400x300")
        comparison_window.resizable(False, False)
        
        # Center the window
        comparison_window.update_idletasks()
        width = comparison_window.winfo_width()
        height = comparison_window.winfo_height()
        x = (comparison_window.winfo_screenwidth() // 2) - (width // 2)
        y = (comparison_window.winfo_screenheight() // 2) - (height // 2)
        comparison_window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Add instructions
        ttk.Label(comparison_window, text="Select strategies to compare:", font=("Arial", 12)).pack(pady=10)
        
        # Strategy checkboxes
        strategy_vars = {}
        for strategy in strategies_to_compare:
            var = tk.BooleanVar(value=True)
            strategy_vars[strategy] = var
            ttk.Checkbutton(
                comparison_window,
                text=strategy.replace("_", " ").title(),
                variable=var
            ).pack(anchor="w", padx=20, pady=5)
        
        # Add ML strategy option if enabled
        ml_var = None
        if app_config.strategy.enable_ml_optimization:
            ml_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                comparison_window,
                text="ML-Optimized Strategy",
                variable=ml_var
            ).pack(anchor="w", padx=20, pady=5)
        
        # Add buttons
        button_frame = ttk.Frame(comparison_window)
        button_frame.pack(pady=20)
        
        def on_cancel():
            comparison_window.destroy()
        
        def on_compare():
            # Get selected strategies
            selected = [s for s, v in strategy_vars.items() if v.get()]
            if ml_var is not None and ml_var.get():
                selected.append("ml_optimized")
            
            if not selected:
                messagebox.showwarning("No Selection", "Please select at least one strategy to compare.")
                return
            
            comparison_window.destroy()
            
            # Run the comparison
            run_selected_comparison(app_config, data_handler, selected)
        
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="left", padx=10)
        ttk.Button(button_frame, text="Compare", command=on_compare).pack(side="left", padx=10)
        
    except ImportError as e:
        messagebox.showerror("Module Error", f"Required module not found: {str(e)}")
    except Exception as e:
        logging.error(f"Error setting up strategy comparison: {e}")
        logging.error(traceback.format_exc())
        messagebox.showerror("Comparison Error", f"Error setting up comparison: {str(e)}")

def run_selected_comparison(app_config, data_handler, strategies):
    """
    Run comparison of selected strategies
    
    Args:
        app_config: Application configuration
        data_handler: DataHandler with historical data
        strategies: List of strategy names to compare
    """
    try:
        # Import the strategy module
        from stock_predictor.strategy_implementation import StrategyBacktester
        
        # Create a progress window
        progress_window = tk.Toplevel()
        progress_window.title("Running Strategy Comparison")
        progress_window.geometry("400x200")
        progress_window.resizable(False, False)
        
        # Center the window
        progress_window.update_idletasks()
        width = progress_window.winfo_width()
        height = progress_window.winfo_height()
        x = (progress_window.winfo_screenwidth() // 2) - (width // 2)
        y = (progress_window.winfo_screenheight() // 2) - (height // 2)
        progress_window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Add progress information
        progress_label = ttk.Label(progress_window, text="Comparing strategies...", font=("Arial", 12))
        progress_label.pack(pady=20)
        
        progress_info = tk.StringVar(value=f"Preparing to compare {len(strategies)} strategies")
        info_label = ttk.Label(progress_window, textvariable=progress_info)
        info_label.pack(pady=5)
        
        # Add a progress bar
        progress_bar = ttk.Progressbar(progress_window, mode="determinate", maximum=len(strategies))
        progress_bar.pack(fill="x", padx=20, pady=10)
        
        # Function to run comparison in a separate thread
        def run_comparison_thread():
            results = []
            
            for i, strategy_name in enumerate(strategies):
                try:
                    # Update progress
                    progress_window.after(0, lambda: progress_info.set(f"Testing strategy: {strategy_name}"))
                    progress_window.after(0, lambda: progress_bar.configure(value=i))
                    
                    # Create config copy with current strategy
                    config_copy = app_config.strategy
                    config_copy.strategy_type = strategy_name
                    
                    # Initialize backtester
                    backtester = StrategyBacktester(config_copy, data_handler)
                    
                    # Run backtest
                    metrics, _ = backtester.backtest()
                    
                    # Store results
                    results.append({
                        'strategy': strategy_name,
                        'metrics': metrics
                    })
                    
                except Exception as e:
                    logging.error(f"Error testing strategy {strategy_name}: {e}")
                    logging.error(traceback.format_exc())
            
            # Close progress window
            progress_window.destroy()
            
            # Display comparison results
            if results:
                display_comparison_results(app_config, results)
            else:
                messagebox.showerror("Comparison Error", "No strategies could be successfully tested.")
        
        # Start the comparison thread
        threading.Thread(target=run_comparison_thread, daemon=True).start()
        
    except Exception as e:
        logging.error(f"Error in strategy comparison: {e}")
        logging.error(traceback.format_exc())
        progress_window.destroy()
        messagebox.showerror("Comparison Error", f"Error running comparison: {str(e)}")

def display_comparison_results(app_config, results):
    """
    Display comparison results
    
    Args:
        app_config: Application configuration
        results: List of dictionaries with strategy results
    """
    # Create results window
    results_window = tk.Toplevel()
    results_window.title("Strategy Comparison Results")
    results_window.geometry("1000x700")
    
    # Create a frame for the table
    table_frame = ttk.LabelFrame(results_window, text="Performance Comparison")
    table_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Create a treeview for the results
    columns = ("strategy", "total_return", "annual_return", "sharpe", "sortino", "max_dd", "win_rate", "profit_factor", "trades")
    tree = ttk.Treeview(table_frame, columns=columns, show="headings")
    
    # Configure columns
    column_titles = {
        "strategy": "Strategy",
        "total_return": "Total Return (%)",
        "annual_return": "Annual Return (%)",
        "sharpe": "Sharpe Ratio",
        "sortino": "Sortino Ratio",
        "max_dd": "Max Drawdown (%)",
        "win_rate": "Win Rate (%)",
        "profit_factor": "Profit Factor",
        "trades": "Trades"
    }
    
    for col, title in column_titles.items():
        tree.heading(col, text=title)
        tree.column(col, width=100, anchor="center")
    
    # Adjust the strategy column width
    tree.column("strategy", width=150, anchor="w")
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # Pack treeview and scrollbar
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Insert data rows
    for result in results:
        strategy_name = result['strategy'].replace("_", " ").title()
        metrics = result['metrics']
        
        tree.insert("", "end", values=(
            strategy_name,
            f"{metrics['total_return']:.2f}",
            f"{metrics['annualized_return']:.2f}",
            f"{metrics['sharpe_ratio']:.2f}",
            f"{metrics['sortino_ratio']:.2f}",
            f"{metrics['max_drawdown']:.2f}",
            f"{metrics['win_rate']:.2f}",
            f"{metrics['profit_factor']:.2f}",
            f"{metrics['trades']}"
        ))
    
    # Create visualization frame
    viz_frame = ttk.LabelFrame(results_window, text="Performance Visualization")
    viz_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(9, 4))
    
    # Extract data for plotting
    strategies = [r['strategy'].replace("_", " ").title() for r in results]
    returns = [r['metrics']['total_return'] for r in results]
    
    # Create a bar chart
    bars = ax.bar(strategies, returns, color='skyblue')
    
    # Add value labels on the bars
    for bar, value in zip(bars, returns):
        height = bar.get_height()
        label_y_pos = height if height >= 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2., label_y_pos,
                f'{value:.1f}%', ha='center', va='bottom', rotation=0)
    
    # Customize the chart
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Strategy Performance Comparison')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Display the figure
    canvas = FigureCanvasTkAgg(fig, master=viz_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    # Add export button
    def export_comparison():
        """Export comparison results to CSV"""
        from tkinter import filedialog
        import os
        import pandas as pd
        
        # Ask for file path
        file_path = filedialog.asksaveasfilename(
            title="Save Comparison Results",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            defaultextension=".csv"
        )
        
        if not file_path:
            return
        
        try:
            # Create DataFrame from results
            df = pd.DataFrame([
                {
                    'Strategy': r['strategy'].replace("_", " ").title(),
                    'Total Return (%)': r['metrics']['total_return'],
                    'Annual Return (%)': r['metrics']['annualized_return'],
                    'Sharpe Ratio': r['metrics']['sharpe_ratio'],
                    'Sortino Ratio': r['metrics']['sortino_ratio'],
                    'Max Drawdown (%)': r['metrics']['max_drawdown'],
                    'Win Rate (%)': r['metrics']['win_rate'],
                    'Profit Factor': r['metrics']['profit_factor'],
                    'Trades': r['metrics']['trades']
                }
                for r in results
            ])
            
            # Export to CSV
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Export Successful", f"Comparison results exported to:\n{file_path}")
            
        except Exception as e:
            logging.error(f"Error exporting comparison results: {e}")
            messagebox.showerror("Export Error", f"Error exporting results: {str(e)}")
    
    # Add export button at the bottom
    export_button = ttk.Button(results_window, text="Export Results", command=export_comparison)
    export_button.pack(side="bottom", pady=10)