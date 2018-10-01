from algo.tree.info_gain import InfoGain
import pandas as pd
from pathlib import Path
import pytest
from pandas.util.testing import assert_frame_equal
from pandas.util.testing import assert_series_equal


@pytest.fixture(scope="function")
def csv_file(tmpdir_factory):
    df = pd.DataFrame(data={"A": [4.8, 5. , 5., 5.2, 5.2, 4.7, 4.8, 5.4, 7., 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9],
                            "B": [3.4, 3. , 3.4, 3.5, 3.4, 3.2, 3.1, 3.4, 3.2, 3.2, 3.1, 2.3, 2.8, 2.8, 3.3, 2.4],
                            "C": [1.9, 1.6, 1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 4.7, 4.5, 4.9, 4., 4.6, 4.5, 4.7, 3.3],
                            "D": [0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0],
                            "E": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]})
    csv_source = tmpdir_factory.mktemp('data').join('data.csv')
    df.to_csv(str(csv_source), index=False)

    return csv_source, df


@pytest.fixture(scope="function")
def train_x_y(csv_file):
    df = csv_file[1]
    train_y = df["E"]
    train_x = df.drop(labels="E", axis=1)

    return train_x, train_y


def test_init_correctly_with_split_mode_equals_to_random(train_x_y):
    info_gain = InfoGain(train_x_y[0], train_x_y[1], split_mode="random")
    assert_frame_equal(info_gain.train_x, train_x_y[0])
    assert_series_equal(info_gain.train_y, train_x_y[1])
    assert set(info_gain.split_threshold.keys()) == set(info_gain.train_x.columns)
    assert info_gain.target_entropy == 1.0  # since we are using p*log2(p) to calculate entropy and in the test data,
    # p=1/2


def test_init_exit_with_wrong_split_mode(train_x_y, capfd):
    with pytest.raises(SystemExit):
        info_gain = InfoGain(train_x_y[0], train_x_y[1], split_mode="xxxx")
    out, _ = capfd.readouterr()
    assert "Error: split mode" in out


def test_init_correctly_with_split_threshold_is_given(train_x_y):
    split_threshold = {"A": 5., "B": 3., "C": 4.2, "D": 1.4}
    info_gain = InfoGain(train_x_y[0], train_x_y[1], split_threshold=split_threshold)
    assert_frame_equal(info_gain.train_x, train_x_y[0])
    assert_series_equal(info_gain.train_y, train_x_y[1])
    assert info_gain.split_threshold == split_threshold
    assert info_gain.target_entropy == 1.0  # since we are using p*log2(p) to calculate entropy and in the test data,
    # p=1/2


def test_init_exit_if_neither_split_mode_nor_split_threshold_is_given(train_x_y, capfd):
    with pytest.raises(SystemExit):
        info_gain = InfoGain(train_x_y[0], train_x_y[1])

    out, _ = capfd.readouterr()
    assert "Error: if split_mode is not given" in out


def test_info_gain_can_give_the_correct_information_gain(train_x_y):
    split_threshold = {"A": 5., "B": 3., "C": 4.2, "D": 1.4}
    info_gain = InfoGain(train_x_y[0], train_x_y[1], split_threshold=split_threshold)
    # the answer is from http://dataaspirant.com/2017/01/30/how-decision-tree-algorithm-works/
    # note that the information gain for 'B' is wrong on the website
    answer = {'A': 0.062278901396852215, 'B': 0.31127812445913283, 'C': 0.54879494069539858, 'D': 0.41882123107207492}
    assert info_gain.information_gain == pytest.approx(answer)






