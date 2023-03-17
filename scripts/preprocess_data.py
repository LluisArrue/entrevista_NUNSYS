from argparse import ArgumentParser
from os import makedirs
from os.path import join
import pandas as pd
import numpy as np
import re
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def clear_data(text: str) -> str:
    """
    Remove links, html artifacts, non-alphanumeric characters, and extra spaces from text.
    """
    patterns = (
        r'https?\S+',  # remove links
        r'&\w+;',  # remove html artifacts
        r'[^A-Za-z0-9]',  # remove all but alphanumeric characters
        r'\s{2,}'  # remove two or more than two spaces together
    )
    for patt in patterns:
        text = re.sub(patt, ' ', text)

    return text


def get_data(path: str, val_size: Optional[float] = 0, test_size: Optional[float] = 0,
             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from a CSV file and split into train, test, and validation sets.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f'Error loading CSV file: {e}')

    df['text'] = df['text'].apply(clear_data)

    if not test_size and not val_size:
        # No splitting required, return entire dataset as train set
        return df, pd.DataFrame(columns=['text', 'target']), pd.DataFrame(columns=['text', 'target'])

    if not test_size:
        # Split only into train and val
        train_df, val_df = train_test_split(df, test_size=val_size, random_state=random_state)
        return train_df, pd.DataFrame(columns=['text', 'target']), val_df

    if not val_size:
        # Split only into train and test
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_df, test_df, pd.DataFrame(columns=['text', 'target'])

    # Split into train, test, and val
    train_df, testval_df = train_test_split(df, test_size=test_size + val_size, random_state=random_state)
    test_df, val_df = train_test_split(testval_df, test_size=val_size / (test_size + val_size), random_state=random_state)

    return train_df, test_df, val_df


def encode_texts(X: List[str], model: SentenceTransformer, batch_size: int = 4) -> np.ndarray:
    """
    Encode a list of texts using a SentenceTransformer model, batching the encoding to avoid running out of memory.
    """
    result = None

    for i in tqdm(range(0, len(X), batch_size)):
        batch = X[i:i + batch_size]
        batch_result = model.encode(batch)
        if result is None:
            result = batch_result
        else:
            result = np.concatenate(
                (result, batch_result),
                0
            )

    return result


def preprocess_data(path, outpath, model_id: str, val_size: float = 0., test_size: float = 0., random_state=42,
                    bs: int = 8) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Preprocesses data by encoding text data using a SentenceTransformer model and splitting it into train, test, and validation sets.

    Args:
        path (str): Path to the input CSV file
        outpath (str): Path to the folder to save the results. By default, results are not saved.
        model_id (str): ID of the model from the SentenceTransformer library (https://www.sbert.net/docs/pretrained_models.html)
        val_size (float): Size of the validation set as a percentage of the total dataset. Default is 0.
        test_size (float): Size of the test set as a percentage of the total dataset. Default is 0.
        random_state (int): Random seed to use for splitting the dataset. Default is 42.
        bs (int): Batch size to use for encoding the texts with the SentenceTransformer model. Default is 8.

    Returns:
        transformed_data (list): A list of tuples, each containing the encoded texts and corresponding targets
                                 for the train, test, and validation sets, in that order.
    """
    if not 0 <= val_size < 1:
        raise ValueError('Val size must be between 0 and 1')
    if not 0 <= test_size < 1:
        raise ValueError('Test size must be between 0 and 1')

    if outpath:
        # Create the output directory if it does not exist
        makedirs(outpath, exist_ok=True)

    # Load the data and preprocess the text column
    datasets = get_data(path, val_size, test_size, random_state)
    model = SentenceTransformer(model_id)

    savenames = ('train', 'test', 'val')
    transformed_data = []
    for df, savename in zip(datasets, savenames):
        if not len(df):
            continue
        X = df['text'].values
        X = encode_texts(X, model, bs)

        y = df['target'].values.reshape(-1, 1).astype(np.float32)

        transformed_data.append((X, y))

        # Save the encoded texts and targets to disk, if an output directory is provided
        if outpath:
            np.save(join(outpath, f'X_{savename}.npy'), X)
            np.save(join(outpath, f'y_{savename}.npy'), y)

    return transformed_data


def get_parser():
    parser = ArgumentParser()

    parser.add_argument('-p', '--path', required=True, type=str, help='Path to the input CSV file')
    parser.add_argument('-o', '--outpath', default='.', type=str, help='path to the folder to save the results. By '
                                                                       'default, results are saved in the execution '
                                                                       'folder')
    parser.add_argument('-m', '--model', default='all-MiniLM-L6-v2', type=str, help='id of the model from the '
                                                                                    'SentenceTransformer library '
                                                                                    '(https://www.sbert.net/docs/pretrained_models.html)')
    parser.add_argument('-t', '--test', default=0., type=float, help='test size')
    parser.add_argument('-v', '--val', default=0., type=float, help='val size')

    return parser


def main():
    args = get_parser().parse_args()

    preprocess_data(args.path, args.outpath, args.model, args.val, args.test)


if __name__ == '__main__':
    main()


