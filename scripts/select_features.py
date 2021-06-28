import pandas as pd


if __name__ == '__main__':

    hmdb_fingerprints = pd.read_csv("../data/mol_key_test/hmdb_fingerprints.csv", dtype=int, header=None, index_col=False)
    zinc_fingerprints = pd.read_csv("../data/mol_key_test/zinc_fingerprints.csv", dtype=int, header=None, index_col=False)

    hmdb_num_rows, hmdb_num_cols = hmdb_fingerprints.shape
    print(hmdb_num_rows)
    hmdb_initial_index = hmdb_fingerprints.index

    hmdb_fingerprints.columns = range(0, hmdb_num_cols)
    print(hmdb_fingerprints)

    zinc_num_rows, zinc_num_cols = zinc_fingerprints.shape
    print(zinc_num_rows)
    zinc_initial_index = zinc_fingerprints.index

    assert zinc_num_cols == hmdb_num_cols

    zinc_fingerprints.columns = range(0, hmdb_num_cols)
    print(zinc_fingerprints)

    assert all(hmdb_fingerprints.columns == zinc_fingerprints.columns)

    # remove unbalanced features
    unbalanced = 0.1
    cols_to_remove = []
    for i, cname in enumerate(hmdb_fingerprints):

        n_unique = hmdb_fingerprints[cname].nunique()
        if n_unique == 1:
            cols_to_remove.append(i)

        elif n_unique == 2:
            balance_table = hmdb_fingerprints[cname].value_counts()
            balance = balance_table[0] / (balance_table[1] + balance_table[0])

            if balance > (1 - unbalanced) or balance < unbalanced:
                cols_to_remove.append(i)
        else:
            assert False

    hmdb_fingerprints.drop(cols_to_remove, axis=1, inplace=True)
    print(hmdb_fingerprints)
    zinc_fingerprints.drop(cols_to_remove, axis=1, inplace=True)
    print(zinc_fingerprints)

    hmdb_num_rows_processed, hmdb_num_cols_processed = hmdb_fingerprints.shape
    print(hmdb_num_cols_processed)
    assert hmdb_num_rows_processed == hmdb_num_rows
    assert all(hmdb_fingerprints.index == hmdb_initial_index)

    zinc_num_rows_processed, zinc_num_cols_processed = zinc_fingerprints.shape
    print(zinc_num_cols_processed)
    assert zinc_num_rows_processed == zinc_num_rows
    assert all(zinc_fingerprints.index == zinc_initial_index)

    assert all(hmdb_fingerprints.columns == zinc_fingerprints.columns)

    hmdb_fingerprints.to_csv("../data/mol_key_test/hmdb_fingerprints_processed.csv", header=False, index=False)
    zinc_fingerprints.to_csv("../data/mol_key_test/zinc_fingerprints_processed.csv", header=False, index=False)
