# 文件路径: src/utils/logger_utils.py
import logging
import os
from datetime import datetime
from hydra.core.hydra_config import HydraConfig


_logger_instance = None

def get_rank_zero_logger():
    
    global _logger_instance

    
    if _logger_instance is None:
        try:
            
            hydra_output_dir = HydraConfig.get().run.dir
            log_dir = os.path.join(hydra_output_dir, "logs")
        except Exception:
            
            log_dir = "logs"

        
        logger = logging.getLogger("MyProjectLogger")
        logger.setLevel(logging.INFO)
        
        
        logger.propagate = False

        
        if not logger.hasHandlers():
            
            if os.environ.get("LOCAL_RANK", "0") == "0":
                os.makedirs(log_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_file = os.path.join(log_dir, f"training_log_{timestamp}.log")

                
                file_handler = logging.FileHandler(log_file, mode='w')
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
                logger.info(f"logfile: {log_file}")
            else:
                
                logger.addHandler(logging.NullHandler())
        
        _logger_instance = logger

    return _logger_instance