def initialize_app():
    """Initialize the application with proper configuration."""
    from config import AppConfig
    
    # Create the app configuration with all required sections
    app_config = AppConfig()
    
    # Ensure all required sections exist
    if not hasattr(app_config, 'root_widgets'):
        from config import RootWidgetsConfig
        app_config.root_widgets = RootWidgetsConfig()
    
    if not hasattr(app_config, 'preferences'):
        from config import PreferencesConfig
        app_config.preferences = PreferencesConfig()
    
    if not hasattr(app_config, 'features'):
        from config import FeaturesConfig
        app_config.features = FeaturesConfig()
    
    if not hasattr(app_config, 'advanced_prediction'):
        from config import AdvancedPredictionConfig
        app_config.advanced_prediction = AdvancedPredictionConfig()
    
    # Make sure learning.size_layer is within valid range
    if app_config.learning.size_layer > 10:
        app_config.learning.size_layer = 10
        print("Warning: learning.size_layer adjusted to maximum value of 10")
    
    return app_config

# Then in your main function or wherever you initialize the app:
def main():
    app_config = initialize_app()
    