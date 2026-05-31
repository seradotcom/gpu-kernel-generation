import wandb
import time
import core.config as config
from typing import Any, Dict, Optional

class MLOpsTracker:
    """
    Wrapper class for Weights & Biases.
    Handles the abstraction of initialization and logging of metrics and artifacts.
    """
    
    def __init__(self, job_type: str = "compilation-test"):
        """
        Initializes a new Run in W&B.
        
        Args:
            job_type (str): Label to categorize the execution in the dashboard.
        """
        self.run = wandb.init(
            project=config.WANDB_PROJECT_NAME,
            entity=config.WANDB_ENTITY,
            job_type=job_type,
            config={
                "model_kimi": config.MODEL_KIMI,
                "model_gemini": config.MODEL_GEMINI,
                "generation_params": config.GENERATION_PARAMS
            }
        )
        self._start_time = None

    def start_timer(self):
        """
        Starts the chronometer to measure LLM latency.
        """
        self._start_time = time.time()

    def get_elapsed_time_ms(self) -> float:
        """
        Returns the elapsed time since start_timer in milliseconds.

        Returns:
            float: Elapsed time in ms.
        """
        if self._start_time is None:
            return 0.0
        return (time.time() - self._start_time) * 1000.0

    def log_iteration(
        self, 
        iteration_idx: int, 
        prompt: str, 
        raw_json: str, 
        mlir_code: str, 
        success: bool,
        error_msg: Optional[str] = None
    ):
        """
        Logs an iteration of the logic loop in W&B.
        
        Logs the latency, success state, and saves the texts as HTML tables
        so they are easily readable in the W&B interface.
        
        Args:
            iteration_idx (int): Current attempt number in the logic loop.
            prompt (str): The full prompt sent to the model.
            raw_json (str): The raw response from the model.
            mlir_code (str): The generated MLIR code (or empty string if parsing failed).
            success (bool): Whether module.operation.verify() was successful.
            error_msg (str, optional): MLIR error message if there was a failure.
        """
        latency_ms = self.get_elapsed_time_ms()
        
        metrics = {
            "iteration": iteration_idx,
            "latency_ms": latency_ms,
            "verification_success": int(success) # 1 if success, 0 if failure
        }
        
        # Log as a table to allow long text reading in W&B
        table = wandb.Table(columns=["Iteration", "Success", "Prompt", "JSON Output", "MLIR Output", "Error"])
        table.add_data(iteration_idx, success, prompt, raw_json, mlir_code, str(error_msg))
        
        metrics["iteration_details"] = table
        
        wandb.log(metrics)

    def finish(self):
        """
        Finalizes the Run and synchronizes with W&B servers.
        """
        if self.run:
            self.run.finish()
