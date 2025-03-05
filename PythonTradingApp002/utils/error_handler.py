"""
Error Handling Module for Stock Prediction Application

This module provides standardized error handling and validation functionality
to improve robustness and provide better user feedback.
"""

import logging
import traceback
import threading
import tkinter as tk
from tkinter import messagebox
from typing import Dict, List, Tuple, Any, Callable, Optional
import functools


class ErrorHandler:
    """Class for standardized error handling"""
    
    def __init__(self, root=None):
        """
        Initialize the error handler
        
        Args:
            root: Root window for displaying messages
        """
        self.root = root
        self.error_log = []
        self._setup_logger()
    
    def _setup_logger(self):
        """Set up the logger with custom error handling"""
        # Create a custom handler that logs to self.error_log
        class ErrorLogHandler(logging.Handler):
            def __init__(self, error_log):
                super().__init__()
                self.error_log = error_log
            
            def emit(self, record):
                if record.levelno >= logging.ERROR:
                    self.error_log.append({
                        'level': record.levelname,
                        'message': record.getMessage(),
                        'timestamp': record.created,
                        'traceback': record.exc_text if hasattr(record, 'exc_text') and record.exc_text else None
                    })
        
        # Get the root logger
        logger = logging.getLogger()
        
        # Add our custom handler
        handler = ErrorLogHandler(self.error_log)
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)
    
    def show_error(self, title, message, detail=None):
        """
        Show an error message to the user
        
        Args:
            title: Error title
            message: Error message
            detail: Detailed error information
        """
        if self.root:
            # Create error dialog in the main thread
            if threading.current_thread() is threading.main_thread():
                self._show_error_dialog(title, message, detail)
            else:
                # Schedule dialog in main thread
                if hasattr(self.root, 'after'):
                    self.root.after(0, lambda: self._show_error_dialog(title, message, detail))
                else:
                    logging.error(f"Error: {title} - {message}")
        else:
            # No root window, just log the error
            logging.error(f"Error: {title} - {message}")
            if detail:
                logging.error(f"Detail: {detail}")
    
    def _show_error_dialog(self, title, message, detail=None):
        """Show error dialog in the main thread"""
        try:
            if hasattr(messagebox, 'showerror'):
                full_message = message
                if detail:
                    full_message += f"\n\nDetails: {detail}"
                messagebox.showerror(title, full_message)
            else:
                logging.error(f"Error: {title} - {message}")
                if detail:
                    logging.error(f"Detail: {detail}")
        except Exception as e:
            logging.error(f"Error showing error dialog: {e}")
    
    def log_error(self, error, context=None):
        """
        Log an error
        
        Args:
            error: Error object or message
            context: Optional context information
        """
        if isinstance(error, Exception):
            error_message = str(error)
            error_traceback = traceback.format_exc()
        else:
            error_message = str(error)
            error_traceback = None
        
        context_str = f" [{context}]" if context else ""
        logging.error(f"Error{context_str}: {error_message}")
        
        if error_traceback:
            logging.error(f"Traceback: {error_traceback}")
    
    def get_error_log(self):
        """
        Get the error log
        
        Returns:
            list: List of logged errors
        """
        return self.error_log.copy()
    
    def get_last_error(self):
        """
        Get the last error
        
        Returns:
            dict: Last error or None
        """
        if self.error_log:
            return self.error_log[-1]
        return None


# Global error handler instance
_error_handler = None

def get_error_handler(root=None):
    """
    Get the global error handler
    
    Args:
        root: Root window for displaying messages
        
    Returns:
        ErrorHandler: Global error handler instance
    """
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler(root)
    elif root is not None and _error_handler.root is None:
        _error_handler.root = root
    return _error_handler


def handle_errors(show_dialog=True, context=None):
    """
    Decorator for handling errors in functions
    
    Args:
        show_dialog: Whether to show error dialog
        context: Optional context information
        
    Returns:
        function: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = get_error_handler()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.log_error(e, context)
                if show_dialog:
                    error_handler.show_error(
                        title="Error",
                        message=f"An error occurred: {str(e)}",
                        detail=f"Function: {func.__name__}, Context: {context}"
                    )
                # Re-raise for higher-level handlers
                raise
        return wrapper
    return decorator


def log_errors(context=None):
    """
    Decorator for logging errors without showing dialog
    
    Args:
        context: Optional context information
        
    Returns:
        function: Decorated function
    """
    return handle_errors(show_dialog=False, context=context)


def catch_and_log_errors(show_dialog=True, context=None):
    """
    Decorator for catching and logging errors without re-raising
    
    Args:
        show_dialog: Whether to show error dialog
        context: Optional context information
        
    Returns:
        function: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = get_error_handler()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.log_error(e, context)
                if show_dialog:
                    error_handler.show_error(
                        title="Error",
                        message=f"An error occurred: {str(e)}",
                        detail=f"Function: {func.__name__}, Context: {context}"
                    )
                # Return None instead of re-raising
                return None
        return wrapper
    return decorator


def run_with_error_handling(func, *args, show_dialog=True, context=None, **kwargs):
    """
    Run a function with error handling
    
    Args:
        func: Function to run
        *args: Function arguments
        show_dialog: Whether to show error dialog
        context: Optional context information
        **kwargs: Function keyword arguments
        
    Returns:
        Any: Function result or None
    """
    error_handler = get_error_handler()
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_handler.log_error(e, context)
        if show_dialog:
            error_handler.show_error(
                title="Error",
                message=f"An error occurred: {str(e)}",
                detail=f"Function: {func.__name__}, Context: {context}"
            )
        # Return None on error
        return None


class ErrorAwareThread(threading.Thread):
    """Thread class that catches and logs errors"""
    
    def __init__(self, target=None, args=(), kwargs=None, show_dialog=True, context=None, daemon=True):
        """
        Initialize the thread
        
        Args:
            target: Target function
            args: Function arguments
            kwargs: Function keyword arguments
            show_dialog: Whether to show error dialog
            context: Optional context information
            daemon: Whether the thread is a daemon
        """
        if kwargs is None:
            kwargs = {}
        
        self.target_func = target
        self.target_args = args
        self.target_kwargs = kwargs
        self.show_dialog = show_dialog
        self.context = context
        self.error = None
        self.result = None
        
        super().__init__(target=self._target_with_error_handling, daemon=daemon)
    
    def _target_with_error_handling(self):
        """Target function with error handling"""
        if self.target_func:
            try:
                self.result = self.target_func(*self.target_args, **self.target_kwargs)
            except Exception as e:
                self.error = e
                error_handler = get_error_handler()
                error_handler.log_error(e, self.context)
                if self.show_dialog:
                    error_handler.show_error(
                        title="Error in Thread",
                        message=f"An error occurred: {str(e)}",
                        detail=f"Context: {self.context}"
                    )
    
    def get_result(self):
        """
        Get the result of the thread
        
        Returns:
            Any: Thread result or None if error occurred
        """
        if self.error:
            return None
        return self.result
    
    def get_error(self):
        """
        Get the error that occurred in the thread
        
        Returns:
            Exception: Error or None
        """
        return self.error


class ErrorHandlingEvent:
    """Event handler that catches and logs errors"""
    
    def __init__(self, handler, show_dialog=True, context=None):
        """
        Initialize the event handler
        
        Args:
            handler: Event handler function
            show_dialog: Whether to show error dialog
            context: Optional context information
        """
        self.handler = handler
        self.show_dialog = show_dialog
        self.context = context
    
    def __call__(self, *args, **kwargs):
        """Call the event handler with error handling"""
        error_handler = get_error_handler()
        try:
            return self.handler(*args, **kwargs)
        except Exception as e:
            error_handler.log_error(e, self.context)
            if self.show_dialog:
                error_handler.show_error(
                    title="Error in Event Handler",
                    message=f"An error occurred: {str(e)}",
                    detail=f"Context: {self.context}"
                )
            # Return None on error
            return None


def validate_file_path(file_path, must_exist=True):
    """
    Validate a file path
    
    Args:
        file_path: File path to validate
        must_exist: Whether the file must exist
        
    Returns:
        bool: Whether the file path is valid
        
    Raises:
        ValueError: If the file path is invalid
    """
    import os
    
    if not file_path:
        raise ValueError("File path cannot be empty")
    
    if must_exist and not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")
    
    return True


def validate_gpu_availability():
    """
    Validate that GPUs are available and working
    
    Returns:
        Tuple[bool, str]: Whether GPUs are available and status message
        
    Raises:
        RuntimeError: If GPU initialization fails
    """
    try:
        import tensorflow as tf
        
        # List physical devices
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            return False, "No GPUs found"
        
        # Check if GPUs can be used
        for gpu in gpus:
            try:
                # Try to configure memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                return False, f"GPU {gpu.name} initialization failed: {e}"
        
        # Try to create a small tensor on GPU to verify it's working
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                tf.debugging.check_numerics(c, "GPU test operation failed")
        except Exception as e:
            return False, f"GPU test operation failed: {e}"
        
        return True, f"Found {len(gpus)} GPUs: {', '.join(gpu.name for gpu in gpus)}"
    except Exception as e:
        raise RuntimeError(f"GPU initialization failed: {e}")


def validate_data_has_required_columns(df, required_columns):
    """
    Validate that a dataframe has the required columns
    
    Args:
        df: Dataframe to validate
        required_columns: List of required column names
        
    Returns:
        bool: Whether the dataframe has the required columns
        
    Raises:
        ValueError: If the dataframe is missing required columns
    """
    if df is None:
        raise ValueError("Dataframe is None")
    
    if df.empty:
        raise ValueError("Dataframe is empty")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    return True


def validate_at_least_n_samples(df, n):
    """
    Validate that a dataframe has at least n samples
    
    Args:
        df: Dataframe to validate
        n: Minimum number of samples
        
    Returns:
        bool: Whether the dataframe has at least n samples
        
    Raises:
        ValueError: If the dataframe has fewer than n samples
    """
    if df is None:
        raise ValueError("Dataframe is None")
    
    if len(df) < n:
        raise ValueError(f"Dataframe has {len(df)} samples, but at least {n} are required")
    
    return True