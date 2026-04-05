from src.advanced_forecasting import main as run_pipeline


def run_forecast():
    """
    Runs the full forecasting pipeline and returns predictions
    """

    try:
        # Run your existing pipeline
        result = run_pipeline()

        # 🔥 IMPORTANT: Ensure pipeline returns something usable
        if result is None:
            return ["Forecast generated successfully (no explicit return from pipeline)"]

        # If result is numpy array / pandas series
        try:
            return list(result)
        except Exception:
            return [str(result)]

    except Exception as e:
        return [f"Error in forecasting pipeline: {str(e)}"]