import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """Formats a detailed tracking error message summary."""

    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    error_message = (
        f"Execution script error caught in [{file_name}] "
        f"at line number [{exc_tb.tb_lineno}] "
        f"with message detail: [{str(error)}]"
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        
    def __str__(self):
        return self.error_message