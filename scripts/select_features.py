import pandas as pd


if __name__ == '__main__':

    fingerprint_names = ["hmdb", "zinc", "chembl"]

    fingerprints = [pd.read_csv("../data/mol_key_test/hmdb_fingerprints.csv", dtype=int, header=None, index_col=False),
                    pd.read_csv("../data/mol_key_test/zinc_fingerprints.csv", dtype=int, header=None, index_col=False),
                    pd.read_csv("../data/mol_key_test/chembl_fingerprints.csv", dtype=int, header=None, index_col=False)]

    num_rows = [None, None, None]
    num_cols = [None, None, None]
    initial_index = [None, None, None]

    for i in range(len(fingerprints)):
        num_rows[i], num_cols[i] = fingerprints[i].shape
        print(num_rows[i])
        initial_index[i] = fingerprints[i].index

        fingerprints[i].columns = range(0, num_cols[i])
        print(fingerprints[i])

        assert all(fingerprints[i].columns == fingerprints[0].columns)

    # remove unbalanced features
    unbalanced = 0.1
    cols_to_remove = []
    for i, cname in enumerate(fingerprints[0]):

        n_unique = fingerprints[0][cname].nunique()
        if n_unique == 1:
            cols_to_remove.append(i)

        elif n_unique == 2:
            balance_table = fingerprints[0][cname].value_counts()
            balance = balance_table[0] / (balance_table[1] + balance_table[0])

            if balance > (1 - unbalanced) or balance < unbalanced:
                cols_to_remove.append(i)
        else:
            assert False

    for i in range(len(fingerprints)):
        fingerprints[i].drop(cols_to_remove, axis=1, inplace=True)
        print(fingerprints[i])

        num_rows_processed, num_cols_processed = fingerprints[i].shape
        print(num_cols_processed)
        assert num_rows_processed == num_rows[i]
        assert all(fingerprints[i].index == initial_index[i])

        assert all(fingerprints[i].columns == fingerprints[0].columns)

        fingerprints[i].to_csv("../data/mol_key_test/" + fingerprint_names[i] + "_fingerprints_processed.csv", header=False, index=False)
