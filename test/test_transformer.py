import unittest
import altair as alt
import polars as pl
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


class TestSubsequentMask(unittest.TestCase):
    def test_subsequent_mask(self) -> None:
        example_mask()


if __name__ == "__main__":
    unittest.main()
