import unittest

from app.services.checkerboard_pdf import CheckerboardSpec, generate_checkerboard_pdf


class CheckerboardPdfTests(unittest.TestCase):
    def test_pdf_header_and_nonempty(self):
        spec = CheckerboardSpec(inside_cols=9, inside_rows=6, square_size_m=0.025)
        data = generate_checkerboard_pdf(spec)
        self.assertGreater(len(data), 1000)
        self.assertTrue(data.startswith(b"%PDF-1.4"))

    def test_uses_inside_corner_logic(self):
        spec = CheckerboardSpec(inside_cols=1, inside_rows=1, square_size_m=0.01)
        data = generate_checkerboard_pdf(spec)
        # 1x1 inside corners => 2x2 squares.
        # Black squares for 2x2 pattern = 2 rectangles.
        self.assertEqual(data.count(b" re f"), 3)  # +1 for white page fill

    def test_invalid_spec_rejected(self):
        with self.assertRaises(ValueError):
            generate_checkerboard_pdf(
                CheckerboardSpec(inside_cols=0, inside_rows=6, square_size_m=0.025)
            )
        with self.assertRaises(ValueError):
            generate_checkerboard_pdf(
                CheckerboardSpec(inside_cols=9, inside_rows=6, square_size_m=0.0)
            )


if __name__ == "__main__":
    unittest.main()
