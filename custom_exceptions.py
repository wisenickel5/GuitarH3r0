class TranscriptParsingError(Exception):
    def __init__(self, message, original_exception: Exception = None, transcript_path: str = None):
        super().__init__(message)
        self.original_exception = original_exception
        self.transcript_path = transcript_path

    def __str__(self):
        details = ""
        if self.transcript_path is not None:
            details += f" for file {self.transcript_path}"
            return (f"{super().__str__()} {details} \n"
                    f"Original Exception: {self.original_exception}")
