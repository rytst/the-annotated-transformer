import unittest
import altair as alt
import polars as pl
import torch
import transformer as t


def example_mask() -> None:
    LS_data = pl.concat(
        [
            pl.DataFrame(
                {
                    "Subsequent Mask": t.subsequent_mask(20)[0][x, y],
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    chart = (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
    )

    chart.save("img/mask.png")
    return


def positional_encoding() -> None:
    pe = t.PositionalEncoding(d_model=20, dropout=0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pl.concat(
        [
            pl.DataFrame(
                {
                    "embedding": y[0, :, dim].numpy(),
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    chart = (
        alt.Chart(data)
        .mark_line()
        .properties(width=880)
        .encode(x="position", y="embedding", color="dimension:N")
    )

    chart.save("img/pe.png")
    return


class TestSubsequentMask(unittest.TestCase):
    def test_subsequent_mask(self) -> None:
        example_mask()
        return


class TestPositionalEncoding(unittest.TestCase):
    def test_positional_encoding(self) -> None:
        positional_encoding()
        return


if __name__ == "__main__":
    unittest.main()
