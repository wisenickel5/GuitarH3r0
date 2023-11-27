import pandas as pd
from os.path import isfile
from typing import List, Tuple, Any

from custom_exceptions import TranscriptParsingError


def get_transcript_data(transcript_path: str) -> pd.DataFrame:
    """
    Read the transcript data from a file into a pandas DataFrame
    :param transcript_path:
    :return: DataFrame
    """
    try:
        # Check if the file exists at the specified path
        if isfile(transcript_path):
            # Read the CSV file, skipping the first 6 rows, using tab as the delimiter
            transcript_df = pd.read_csv(transcript_path, skiprows=6, delimiter="\t", comment=None)
            # Assign column names to the DataFrame for clarity
            transcript_df.columns = ["MediaFilename", "Channel", "Type", "Phrase", "Score", "StartTimeCs", "EndTimeCs"]
            return transcript_df
        else:
            # Raise a FileNotFoundError if the transcript file does not exist
            raise FileNotFoundError("Transcript file not found!")
    except Exception as e:
        # Wrap any exception in a custom TranscriptParsingError for better error handling upstream
        raise TranscriptParsingError("An error occurred while parsing the transcript",
                                     original_exception=e,
                                     transcript_path=transcript_path) from e


def get_transcript_turns(transcript_df: pd.DataFrame) -> Tuple[List[str], List[int | Any]]:
    """
    Parse the transcript DataFrame and extract phrases and speaker turns.
    :param transcript_df: pd.DataFrame
    :return: turns: Tuple[List[str], List[int | Any]]
    """
    # Initialize lists to store phrases and speakers for each turn
    phrases_per_turn = []
    speaker_per_turn = []
    current_phrase_parts = []
    current_speaker = 0
    # Try to iterate over the DataFrame rows and handle any exceptions
    try:
        for idx, row in transcript_df[['Channel', 'Phrase']].iterrows():
            speaker = row['Channel']
            phrase = row['Phrase']

            # If the speaker changes (indicating a turn in the conversation), reset tracking variables
            if current_speaker != speaker:
                # Only append a phrase if the current phrase parts are not empty
                phrases_per_turn.append(' '.join(current_phrase_parts) + " ") if current_phrase_parts else None
                speaker_per_turn.append(current_speaker) if current_phrase_parts else None
                current_speaker = speaker
                current_phrase_parts = []  # Reset the list of current phrase parts

            # Add the current phrase to the list of parts for this turn
            current_phrase_parts.append(phrase)

        # After the loop, account for the final phrase
        if current_phrase_parts:
            phrases_per_turn.append(' '.join(current_phrase_parts) + " ")
            speaker_per_turn.append(current_speaker)

        return phrases_per_turn, speaker_per_turn
    except Exception as e:
        # Wrap any exception in a custom TranscriptParsingError for better error handling upstream
        raise TranscriptParsingError("An error occurred while parsing the transcript",
                                     original_exception=e) from e


def create_transcript_subsets(phrases_per_turn: List[str], speaker_per_turn: List[int]):
    """
    Create a list of interactions from the list of turns and speakers given.
    :param phrases_per_turn:
    :param speaker_per_turn:
    :return:
    """
    # Initialize variables for constructing subsets of the transcript
    transcript_subsets = []
    subset = {"interaction": [], "actual_agent_response": None}
    counter = 0
    # Iterate over turns and speakers to group phrases into subsets
    for turn, speaker in zip(phrases_per_turn, speaker_per_turn):
        # Construct subsets of 5-6 turns, ensuring the last turn is the customer's (speaker == 0)
        if counter <= 4:
            subset["interaction"].append((turn, speaker))
            if counter == 4 and speaker == 0:
                # If the 5th turn is the customer's, prepare to add one more agent turn
                counter -= 1
            counter += 1
        else:
            # Once 5-6 turns are collected, save the last agent response and start a new subset
            subset["actual_agent_response"] = turn
            transcript_subsets.append(subset)
            subset = {"interaction": [(turn, speaker)], "actual_agent_response": None}
            counter = 0

    return transcript_subsets


def extract_agent_responses(transcript_subsets):
    """
    Extract actual agent responses from the transcript subsets (Interactions)
    :param transcript_subsets:
    :return:
    """
    # Return a list of actual agent responses from each subset
    return [i["actual_agent_response"] for i in transcript_subsets]


def convert_subsets_to_messages(transcript_subsets):
    """
    Convert transcript subsets into a format suitable for requests made to OpenAI API
    :param transcript_subsets:
    :return:
    """
    # Initialize a list to store all messages from the transcript
    all_transcript_messages = []
    # Iterate over each subset to format the messages
    for subset in transcript_subsets:
        subset_messages = []
        # Assign roles and content to each message based on the speaker
        for turn, speaker in subset['interaction']:
            if speaker == 0:
                subset_messages.append({"role": "system", "content": turn})
            elif speaker == 1:
                subset_messages.append({"role": "user", "content": turn})

        # Add the formatted messages from this subset to the overall list
        all_transcript_messages.append(subset_messages)

    return all_transcript_messages
