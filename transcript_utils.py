import pandas as pd
from os.path import isfile
from typing import List, Tuple, Any

from custom_exceptions import TranscriptParsingError


def get_transcript_data(transcript_path: str) -> pd.DataFrame:
    try:
        if isfile(transcript_path):
            transcript_df = pd.read_csv(transcript_path, skiprows=6, delimiter="\t", comment=None)
            transcript_df.columns = ["MediaFilename", "Channel", "Type", "Phrase", "Score", "StartTimeCs", "EndTimeCs"]
            return transcript_df
        else:
            raise FileNotFoundError("Transcript file not found!")
    except Exception as e:
        raise TranscriptParsingError("An error occurred while parsing the transcript",
                                     original_exception=e,
                                     transcript_path=transcript_path) from e


def get_transcript_turns(transcript_df: pd.DataFrame) -> Tuple[List[str], List[int | Any]]:
    phrases_per_turn = []
    speaker_per_turn = []
    current_phrase_parts = []
    current_speaker = 0
    try:
        for idx, row in transcript_df[['Channel', 'Phrase']].iterrows():
            speaker = row['Channel']
            phrase = row['Phrase']

            # If it's a different speaker, and it's not our first pass, reset our tracking variables
            if current_speaker != speaker:
                phrases_per_turn.append(' '.join(current_phrase_parts) + " ") if len(current_phrase_parts) >= 1 else None
                speaker_per_turn.append(current_speaker) if len(current_phrase_parts) >= 1 else None
                current_speaker = speaker
                current_phrase_parts = []  # Reset current_phrase_parts

        # Account for the final phrase
        if current_phrase_parts:
            phrases_per_turn.append(' '.join(current_phrase_parts) + " ")
            speaker_per_turn.append(current_speaker)

        return phrases_per_turn, speaker_per_turn
    except Exception as e:
        raise TranscriptParsingError("An error occurred while parsing the transcript",
                                     original_exception=e) from e
