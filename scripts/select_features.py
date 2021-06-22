import pandas as pd


if __name__ == '__main__':
    fingerprints = pd.read_csv("../data/mol_key_test/fingerprints.csv", dtype=int, header=None, index_col=False)

    num_rows, num_cols = fingerprints.shape
    print(num_rows)
    initial_index = fingerprints.index

    fingerprints.columns = range(0, num_cols)
    print(fingerprints)

    # remove unbalanced features
    unbalanced = 0.1
    cols_to_remove = []
    for i, cname in enumerate(fingerprints):

        n_unique = fingerprints[cname].nunique()
        if n_unique == 1:
            cols_to_remove.append(i)

        elif n_unique == 2:
            balance_table = fingerprints[cname].value_counts()
            balance = balance_table[0] / (balance_table[1] + balance_table[0])

            if balance > (1 - unbalanced) or balance < unbalanced:
                cols_to_remove.append(i)
        else:
            assert False

    fingerprints.drop(cols_to_remove, axis=1, inplace=True)
    print(fingerprints)

    # remove redundant features
    fingerprints = fingerprints.T.drop_duplicates().T
    print(fingerprints)

    num_rows_processed, num_cols_processed = fingerprints.shape
    print(num_cols_processed)
    assert num_rows_processed == num_rows
    assert all(fingerprints.index == initial_index)

    fingerprints.to_csv("../data/mol_key_test/fingerprints_processed.csv", header=False, index=False)
