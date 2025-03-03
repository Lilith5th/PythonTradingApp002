import queue
import threading
import logging
import tkinter as tk

# Global queue for GUI operations
gui_queue = queue.Queue()

# Flag to indicate if the queue processor is running
queue_processor_running = False

def process_gui_queue():
    """Process GUI operations from the queue"""
    global queue_processor_running
    
    try:
        while not gui_queue.empty():
            func, args, kwargs = gui_queue.get(block=False)
            try:
                func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in queued GUI operation: {e}")
            finally:
                gui_queue.task_done()
                
        # Schedule next check
        if queue_processor_running:
            tk._default_root.after(100, process_gui_queue)
            
    except Exception as e:
        logging.error(f"Error processing GUI queue: {e}")
        # Try to reschedule anyway
        if queue_processor_running:
            tk._default_root.after(100, process_gui_queue)

def start_queue_processor():
    """Start the GUI queue processor"""
    global queue_processor_running
    
    if not queue_processor_running:
        queue_processor_running = True
        
        # Ensure we have a default root window
        if tk._default_root is None:
            # Create a hidden root window
            root = tk.Tk()
            root.withdraw()
            
        # Start processing the queue
        tk._default_root.after(100, process_gui_queue)
        
def stop_queue_processor():
    """Stop the GUI queue processor"""
    global queue_processor_running
    queue_processor_running = False

def run_in_main_thread(func, *args, **kwargs):
    """
    Queue a function to be run in the main thread
    
    Args:
        func: Function to run
        *args, **kwargs: Arguments to the function
    """
    if threading.current_thread() is threading.main_thread():
        # Already in main thread, run directly
        return func(*args, **kwargs)
    else:
        # Queue for execution in main thread
        gui_queue.put((func, args, kwargs))
        return None  # Can't return a result when queuing