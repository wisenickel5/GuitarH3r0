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

            current_phrase_parts.append(phrase)

        # Account for the final phrase
        if current_phrase_parts:
            phrases_per_turn.append(' '.join(current_phrase_parts) + " ")
            speaker_per_turn.append(current_speaker)

        return phrases_per_turn, speaker_per_turn
    except Exception as e:
        raise TranscriptParsingError("An error occurred while parsing the transcript",
                                     original_exception=e) from e


def create_transcript_subsets(phrases_per_turn: List[str], speaker_per_turn: List[int]):
    """

    :param phrases_per_turn:
    :param speaker_per_turn:
    :return:
    """
    # [Set of 5-6 turns, each subset of turns ends with the customer as the speaker.]
    # 5-6 turns should provide enough context in the conversation for ChatGPT to provide a reasonable response
    transcript_subsets = []
    subset = {"interaction": [], "actual_agent_response": None}
    counter = 0
    for turn, speaker in zip(phrases_per_turn, speaker_per_turn):
        if counter <= 4:
            subset["interaction"].append((turn, speaker))
            # If the last turn for this subset ends with the agent speaking, extend the count by 1
            if counter == 4 and speaker == 0:
                # This ensures that all turn subsets will end with a customer response/request
                counter -= 1
            counter += 1  # Increment counter
        else:
            # This turn will always contain the agents response to the previous turn.
            subset["actual_agent_response"] = turn
            transcript_subsets.append(subset)  # A full subset was created, append it to the larger result
            # Reset the subset array and fill it with this iterations turn
            subset = {"interaction": [(turn, speaker)], "actual_agent_response": None}
            counter = 0  # Reset counter

    return transcript_subsets


def extract_agent_responses(transcript_subsets):
    return [i["actual_agent_response"] for i in transcript_subsets]



def convert_subsets_to_messages(transcript_subsets):
    all_transcript_messages = []
    for subset in transcript_subsets:
        subset_messages = []
        for turn, speaker in subset['interaction']:
            if speaker == 0:
                subset_messages.append({"role": "system", "content": turn})
            elif speaker == 1:
                subset_messages.append({"role": "user", "content": turn})

        all_transcript_messages.append(subset_messages)

    return all_transcript_messages
