import json

import pandas as pd


class Data:
    """
    A class to load and process training data and related questions
    for multiple-choice tasks involving questions and answers.
    """

    def __init__(self, train_data_path: str = None, questions_path: str = None):
        """
        Initializes the Data object and loads the dataset from disk.
        """
        self.train_data_path = train_data_path
        self.questions_path = questions_path
        self.data = self.load()

    def load(self) -> pd.DataFrame:
        """
        Loads training and question data from CSV and Parquet files.

        - Reads the training data from a CSV file.
        - Reads the questions, choices, and answers from a Parquet file.
        - Groups questions by ID, aggregating the questions, choices, and answers into lists.
        - Merges the questions with the training data on the 'id' field.
        """
        self.train_data = pd.read_csv(self.train_data_path)
        self.questions = pd.read_parquet(self.questions_path)

        questions = self.questions.groupby("id").agg(
            {"question": list, "choices": list, "answer": list}
        )

        data = pd.merge(self.train_data, questions, left_on="id", right_index=True)
        return data

    def get_train_csv(self) -> pd.DataFrame:
        """
        Returns the raw training data from the CSV file.

        Returns:
             pd.DataFrame: The training dataset.
        """
        return self.train_data

    def get_questions(self) -> pd.DataFrame:
        """
        Returns the raw questions data from the Parquet file.

        Returns:
             pd.DataFrame: The questions dataset.
        """
        return self.questions

    def get_data(self) -> pd.DataFrame:
        """
        Returns the merged dataset containing training data and questions.

        Returns:
             pd.DataFrame: The merged dataset.
        """
        return self.data
    
    def get_description_by_id(self, idx: str) -> str:
        """
        Returns the description corresponding to a specific ID from the training dataset.

        Parameters:
            idx (str): The ID of the data entry to retrieve the description for.

        Returns:
            str: The description of the entry with the given ID. 
                Returns an empty string if the ID is not found.
        """
        matched_rows = self.train_data[self.train_data["id"] == idx]
        if not matched_rows.empty:
            return matched_rows["description"].values[0]
        return ""
        

    def get_solution(self, idx: str) -> pd.DataFrame:
        """
        Returns the formatted question, choices, and answer for a specific data sample by ID.

        Args:
             idx (str): The ID of the sample to retrieve.

        Returns:
             pd.DataFrame: A single-row DataFrame containing the ID, question, choices, and answer
                            as JSON-formatted strings.
        """
        row = self.data[self.data["id"] == idx].iloc[0]

        idx, question, choices, answer = (
            row["id"],
            row["question"],
            row["choices"],
            row["answer"],
        )

        choices = [choice.tolist() for choice in choices]
        choices = json.dumps(choices, ensure_ascii=False)

        answer = list(map(str, answer)) if isinstance(answer, list) else [str(answer)]

        solution = pd.DataFrame(
            {
                "id": [idx],
                "question": [json.dumps(question)],
                "choices": choices,
                "answer": [json.dumps(answer)],
            }
        )

        return solution
