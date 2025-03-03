import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
    from gui_wrapper_gpu import create_gui
    print("Running on GPU")
else:
    from gui_wrapper_cpu import create_gui
    print("Running on CPU")

from stock_predictor import forecast_module
app_config = forecast_module.AppConfig()
create_gui(app_config)